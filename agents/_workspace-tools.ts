/**
 * Sandbox-backed coding workspace tools.
 *
 * Sandbox /tmp is NOT persistent across requests, so every write is mirrored
 * into Blob (`projects/<conversation>/workspace/...`) immediately; the next
 * turn restores the workspace from that mirror before doing anything. A
 * pure-text heartbeat never touches these; they only run when the agent
 * decides to call a workspace_* tool (which is exactly when the sandbox
 * spins up).
 */
import {
  clampText,
  safeSegment,
  type MakersContext,
  type SandboxLike,
} from './_shared.ts'
import { applyPatch, type PatchFs } from './_apply-patch.ts'
import { getBlobStore, normalizeBlobKey, scopedBlobKey, textFromSandboxRead } from './_blob-tools.ts'

const IGNORED_DIRECTORIES = new Set([
  '.git', '.next', '.cache', '.turbo', '.vite',
  'node_modules', 'dist', 'build', 'coverage', '__pycache__',
])

const IGNORED_FILES = new Set(['.DS_Store', 'preview'])

const TEXT_PREVIEW_LIMIT = 256 * 1024
const MAX_LIST_ITEMS = 400
const COMMAND_OUTPUT_LIMIT = 20_000

export interface WorkspaceItem {
  path: string
  name: string
  type: 'file' | 'directory'
  depth: number
  size?: number
}

export function workspaceRoot(conversationId: string): string {
  return `projects/${safeSegment(conversationId)}/workspace`
}

function requireSandbox(context: MakersContext): SandboxLike {
  if (!context.sandbox) {
    throw new Error('Sandbox is not available in this turn. workspace_* tools need a sandbox; prefer pure text or blob_* tools.')
  }
  return context.sandbox
}

export function normalizeWorkspacePath(value: string): string | null {
  const raw = value.trim().replaceAll('\\', '/').replace(/^\.\//, '')
  if (!raw || raw.startsWith('/') || raw.includes('\0')) return null
  const parts = raw.split('/')
  if (parts.some((part) => !part || part === '.' || part === '..')) return null
  return parts.join('/')
}

/** Ensure the sandbox workspace dir exists and restore from the Blob mirror. */
export async function ensureWorkspace(context: MakersContext, conversationId: string): Promise<string> {
  const sandbox = requireSandbox(context)
  const root = workspaceRoot(conversationId)
  await sandbox.files.makeDir(root)
  await restoreWorkspaceFromBlob(context, conversationId, root)
  return root
}

/** Rehydrate the sandbox workspace from the Blob mirror (write-back cache). */
export async function restoreWorkspaceFromBlob(context: MakersContext, conversationId: string, root: string): Promise<number> {
  const sandbox = requireSandbox(context)
  const store = await getBlobStore()
  const prefix = `projects/${safeSegment(conversationId)}/workspace/`
  const { blobs } = await store.list({ prefix })
  let restored = 0
  for (const blob of blobs) {
    const relative = normalizeBlobKey(blob.key.slice(prefix.length))
    if (!relative) continue
    const raw = await store.get(blob.key)
    if (typeof raw !== 'string') continue
    const parent = relative.split('/').slice(0, -1).join('/')
    if (parent) await sandbox.files.makeDir(`${root}/${parent}`)
    await sandbox.files.write(`${root}/${relative}`, raw)
    restored += 1
  }
  return restored
}

/** Persist the sandbox workspace back to Blob (call after every play turn). */
export async function snapshotWorkspaceToBlob(context: MakersContext, conversationId: string, root: string): Promise<number> {
  const sandbox = requireSandbox(context)
  const store = await getBlobStore()
  const paths = await listSandboxFiles(context, root)
  let saved = 0
  for (const path of paths) {
    const raw = await sandbox.files.read(`${root}/${path}`)
    const content = textFromSandboxRead(raw)
    await store.set(scopedBlobKey(conversationId, `workspace/${path}`), content)
    saved += 1
  }
  return saved
}

export async function listSandboxFiles(context: MakersContext, root: string): Promise<string[]> {
  const sandbox = requireSandbox(context)
  const result = await sandbox.commands.run(
    `find . -type f -not -path '*/.*' -print`,
    { cwd: root, timeout: 20 },
  )
  if (result.exitCode !== 0) throw new Error(String(result.stderr || result.stdout || 'find failed'))
  return String(result.stdout ?? '')
    .split('\n')
    .map((line) => line.trim().replace(/^\.\//, ''))
    .filter((line) => line.length > 0)
    .filter((line) => !line.split('/').some((segment) => IGNORED_DIRECTORIES.has(segment)))
    .filter((line) => !IGNORED_FILES.has(line.split('/').pop() ?? ''))
}

export async function listWorkspaceFiles(context: MakersContext, conversationId: string, root: string): Promise<WorkspaceItem[]> {
  const sandbox = requireSandbox(context)
  const ignored = [...IGNORED_DIRECTORIES]
    .map((directory) => `-path './${directory}'`)
    .join(' -o ')
  const expression = `find . \\( ${ignored} \\) -prune -o -maxdepth 6`
  const result = await sandbox.commands.run(
    `${expression} \\( -type f -printf 'f\\t%s\\t%p\\n' \\) \\( -type d -printf 'd\\t0\\t%p\\n' \\) 2>/dev/null`,
    { cwd: root, timeout: 30 },
  )
  if (result.exitCode !== 0) throw new Error(String(result.stderr || result.stdout || 'list failed'))
  return String(result.stdout ?? '')
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const [kind = '', sizeRaw = '', ...pathParts] = line.split('\t')
      const rawPath = pathParts.join('\t').replace(/^\.\//, '')
      return { kind, sizeRaw, rawPath }
    })
    .filter((item) => item.rawPath && item.rawPath !== '.')
    .slice(0, MAX_LIST_ITEMS)
    .map((item) => ({
      path: item.rawPath,
      name: item.rawPath.split('/').pop() || item.rawPath,
      type: item.kind === 'd' ? 'directory' as const : 'file' as const,
      depth: item.rawPath.split('/').length - 1,
      ...(item.kind === 'f' && Number.isFinite(Number(item.sizeRaw)) ? { size: Number(item.sizeRaw) } : {}),
    }))
}

export async function readWorkspaceFile(context: MakersContext, conversationId: string, root: string, requestedPath: string): Promise<{ path: string; content: string; truncated: boolean }> {
  const path = normalizeWorkspacePath(requestedPath)
  if (!path) throw new Error(`Invalid workspace path: '${requestedPath}'.`)
  const sandbox = requireSandbox(context)
  const raw = await sandbox.files.read(`${root}/${path}`)
  const content = textFromSandboxRead(raw)
  const encoded = new TextEncoder().encode(content)
  const truncated = encoded.byteLength > TEXT_PREVIEW_LIMIT
  return {
    path,
    content: truncated
      ? new TextDecoder().decode(encoded.slice(0, TEXT_PREVIEW_LIMIT))
      : content,
    truncated,
  }
}

export async function writeWorkspaceFile(context: MakersContext, conversationId: string, root: string, requestedPath: string, content: string): Promise<{ path: string; bytes: number }> {
  const path = normalizeWorkspacePath(requestedPath)
  if (!path) throw new Error(`Invalid workspace path: '${requestedPath}'.`)
  const sandbox = requireSandbox(context)
  const parent = path.split('/').slice(0, -1).join('/')
  if (parent) await sandbox.files.makeDir(`${root}/${parent}`)
  await sandbox.files.write(`${root}/${path}`, content)
  const store = await getBlobStore()
  await store.set(scopedBlobKey(conversationId, `workspace/${path}`), content)
  return { path, bytes: new TextEncoder().encode(content).byteLength }
}

/** Build a PatchFs adapter that reads/writes sandbox files and mirrors to Blob. */
function sandboxPatchFs(context: MakersContext, conversationId: string, root: string): PatchFs {
  const sandbox = requireSandbox(context)
  return {
    read: async (path) => textFromSandboxRead(await sandbox.files.read(`${root}/${path}`)),
    write: async (path, content) => {
      const parent = path.split('/').slice(0, -1).join('/')
      if (parent) await sandbox.files.makeDir(`${root}/${parent}`)
      await sandbox.files.write(`${root}/${path}`, content)
      const store = await getBlobStore()
      await store.set(scopedBlobKey(conversationId, `workspace/${path}`), content)
    },
    delete: async (path) => {
      await sandbox.files.remove(`${root}/${path}`)
      const store = await getBlobStore()
      await store.delete(scopedBlobKey(conversationId, `workspace/${path}`))
    },
    exists: async (path) => sandbox.files.exists(`${root}/${path}`),
    makeDir: async (path) => {
      if (path) await sandbox.files.makeDir(`${root}/${path}`)
    },
  }
}

export async function editWorkspaceFiles(context: MakersContext, conversationId: string, root: string, input: string): Promise<{ summary: string[]; text: string }> {
  const result = await applyPatch(input, sandboxPatchFs(context, conversationId, root))
  const flattened = result.summary.added
    .concat(result.summary.modified)
    .concat(result.summary.deleted)
    .concat(result.summary.moved)
  return { summary: flattened, text: result.text }
}

export async function runWorkspaceCommand(context: MakersContext, conversationId: string, root: string, command: string, timeout = 60): Promise<{ command: string; stdout: string; stderr: string; exitCode: number }> {
  if (!command.trim()) throw new Error('Command must not be empty.')
  const sandbox = requireSandbox(context)
  const safeTimeout = Math.min(Math.max(Math.round(timeout), 1), 300)
  const result = await sandbox.commands.run(command, { cwd: root, timeout: safeTimeout })
  return {
    command,
    stdout: clampText(String(result.stdout ?? ''), COMMAND_OUTPUT_LIMIT),
    stderr: clampText(String(result.stderr ?? ''), COMMAND_OUTPUT_LIMIT),
    exitCode: Number(result.exitCode),
  }
}
