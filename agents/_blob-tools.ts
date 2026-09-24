/**
 * Blob-backed persistence for alive.
 *
 * Everything persistent lives in one strong-consistency Blob namespace
 * (`getStore({ name: BLOB_STORE_NAME, consistency: 'strong' })`, IRON RULE).
 * Sandbox /tmp is NOT persistent, so workspace mirrors and memory files always
 * live here and are written through synchronously.
 *
 * The `@edgeone/pages-blob` SDK is imported lazily at first use so local tests
 * can inject a mock store via `injectBlobStoreForTesting`.
 */
import {
  BLOB_STORE_NAME,
  safeSegment,
  type MakersContext,
} from './_shared.ts'
import { applyPatch, type PatchFs } from './_apply-patch.ts'

export interface BlobStore {
  get(key: string, options?: { type?: 'text' | 'json'; consistency?: 'strong' | 'eventual' }): Promise<unknown>
  set(key: string, value: string | ArrayBuffer | Blob | ReadableStream, options?: { onlyIfNew?: boolean }): Promise<void>
  setJSON(key: string, value: unknown, options?: { onlyIfNew?: boolean }): Promise<void>
  delete(key: string): Promise<void>
  list(options?: { prefix?: string; consistency?: 'strong' | 'eventual' }): Promise<{ blobs: Array<{ key: string; etag: string }>; cursor?: string }>
}

declare module '@edgeone/pages-blob' {
  export function getStore(options: { name: string; consistency: 'strong' | 'eventual' }): BlobStore
}

let cachedStore: BlobStore | null = null
let injectedStore: BlobStore | null = null

/** Test seam: replace the Blob store with an in-memory fake. */
export function injectBlobStoreForTesting(store: BlobStore | null): void {
  injectedStore = store
  cachedStore = null
}

/** Lazily create (once per instance) a strong-consistency store. */
export async function getBlobStore(): Promise<BlobStore> {
  if (injectedStore) return injectedStore
  if (cachedStore) return cachedStore
  try {
    const mod = (await import('@edgeone/pages-blob')) as { getStore: (opts: { name: string; consistency: 'strong' | 'eventual' }) => BlobStore }
    cachedStore = mod.getStore({ name: BLOB_STORE_NAME, consistency: 'strong' })
    return cachedStore
  } catch (error) {
    throw new Error(
      `Blob store unavailable: ${error instanceof Error ? error.message : String(error)}. ` +
        'Ensure @edgeone/pages-blob is installed and running inside Makers Functions.',
    )
  }
}

/**
 * Normalize an arbitrary blob key into a safe relative key.
 * Returns null when the key escapes (absolute path, `..`, empty segment).
 */
export function normalizeBlobKey(value: string): string | null {
  const raw = value.trim().replaceAll('\\', '/')
  if (!raw || raw.startsWith('/') || raw.includes('\0')) return null
  const parts = raw.split('/').filter((part) => part.length > 0)
  if (parts.length === 0 || parts.some((part) => part === '.' || part === '..')) return null
  return parts.join('/')
}

/** Scope a blob key to a conversation, except agent-global memory keys. */
export function scopedBlobKey(conversationId: string, key: string): string {
  const normalized = normalizeBlobKey(key)
  if (!normalized) throw new Error(`Invalid blob key: '${key}'.`)
  if (normalized.startsWith('memory/')) return normalized
  return `projects/${safeSegment(conversationId)}/${normalized}`
}

export async function readBlobText(context: MakersContext, conversationId: string, key: string): Promise<string | null> {
  const store = await getBlobStore()
  const value = await store.get(scopedBlobKey(conversationId, key))
  return typeof value === 'string' ? value : null
}

export async function writeBlobText(context: MakersContext, conversationId: string, key: string, content: string): Promise<void> {
  const store = await getBlobStore()
  await store.set(scopedBlobKey(conversationId, key), content)
}

export async function deleteBlobKey(context: MakersContext, conversationId: string, key: string): Promise<void> {
  const store = await getBlobStore()
  await store.delete(scopedBlobKey(conversationId, key))
}

export async function listBlobKeys(context: MakersContext, conversationId: string, prefix: string): Promise<string[]> {
  const store = await getBlobStore()
  const scopedPrefix = prefix ? scopedBlobKey(conversationId, prefix) : `projects/${safeSegment(conversationId)}/`
  const { blobs } = await store.list({ prefix: scopedPrefix })
  return blobs.map((blob) => blob.key)
}

/* ------------------------------------------------------------------ */
/* Tools (registered in _tools.ts)                                    */
/* ------------------------------------------------------------------ */

export async function blobReadTool(context: MakersContext, conversationId: string, args: Record<string, unknown>): Promise<{ content: string; isError?: boolean }> {
  try {
    const key = typeof args.key === 'string' ? args.key : ''
    if (!key) return { content: 'blob_read requires "key".', isError: true }
    const text = await readBlobText(context, conversationId, key)
    if (text === null) return { content: `Blob key '${key}' not found.`, isError: true }
    return { content: text.slice(0, 8_000) }
  } catch (error) {
    return { content: error instanceof Error ? error.message : String(error), isError: true }
  }
}

export async function blobWriteTool(context: MakersContext, conversationId: string, args: Record<string, unknown>): Promise<{ content: string; isError?: boolean }> {
  try {
    const key = typeof args.key === 'string' ? args.key : ''
    const content = typeof args.content === 'string' ? args.content : ''
    if (!key) return { content: 'blob_write requires "key".', isError: true }
    await writeBlobText(context, conversationId, key, content)
    return { content: `Wrote ${key} (${new TextEncoder().encode(content).byteLength} bytes).` }
  } catch (error) {
    return { content: error instanceof Error ? error.message : String(error), isError: true }
  }
}

/** blob_edit: apply a single-file apply_patch envelope against a blob key. */
export async function blobEditTool(context: MakersContext, conversationId: string, args: Record<string, unknown>): Promise<{ content: string; isError?: boolean }> {
  try {
    const key = typeof args.key === 'string' ? args.key : ''
    const input = typeof args.input === 'string' ? args.input : ''
    if (!key) return { content: 'blob_edit requires "key".', isError: true }
    if (!input) return { content: 'blob_edit requires "input" patch text.', isError: true }

    const scoped = scopedBlobKey(conversationId, key)
    const store = await getBlobStore()
    const existing = (await store.get(scoped)) as string | null
    if (existing === null) return { content: `Blob key '${key}' not found.`, isError: true }

    const touched = new Set<string>()
    const fs: PatchFs = {
      read: async () => existing,
      write: async (_path, content) => {
        await store.set(scoped, content)
      },
      delete: async () => {
        await store.delete(scoped)
      },
      exists: async () => true,
      makeDir: async () => undefined,
    }
    const result = await applyPatch(input, fs)
    for (const bucket of ['added', 'modified', 'deleted', 'moved'] as const) {
      for (const path of result.summary[bucket]) touched.add(path)
    }
    if (touched.size > 1) {
      throw new Error('blob_edit supports exactly one file per patch; split the patch per blob key.')
    }
    return { content: result.text }
  } catch (error) {
    return { content: error instanceof Error ? error.message : String(error), isError: true }
  }
}

export async function blobListTool(context: MakersContext, conversationId: string, args: Record<string, unknown>): Promise<{ content: string; isError?: boolean }> {
  try {
    const prefix = typeof args.prefix === 'string' ? args.prefix : ''
    const keys = await listBlobKeys(context, conversationId, prefix)
    return { content: keys.length === 0 ? 'No blobs found.' : keys.join('\n') }
  } catch (error) {
    return { content: error instanceof Error ? error.message : String(error), isError: true }
  }
}

export async function blobDeleteTool(context: MakersContext, conversationId: string, args: Record<string, unknown>): Promise<{ content: string; isError?: boolean }> {
  try {
    const key = typeof args.key === 'string' ? args.key : ''
    if (!key) return { content: 'blob_delete requires "key".', isError: true }
    await deleteBlobKey(context, conversationId, key)
    return { content: `Deleted ${key}.` }
  } catch (error) {
    return { content: error instanceof Error ? error.message : String(error), isError: true }
  }
}

/** Copy a blob key into the sandbox at `<root>/<sandbox_path>`. */
export async function blobToSandboxTool(context: MakersContext, conversationId: string, root: string, args: Record<string, unknown>): Promise<{ content: string; isError?: boolean }> {
  try {
    const key = typeof args.key === 'string' ? args.key : ''
    const sandboxPath = typeof args.sandbox_path === 'string' && args.sandbox_path.trim()
      ? args.sandbox_path.trim()
      : key
    if (!key) return { content: 'blob_to_sandbox requires "key".', isError: true }
    const text = await readBlobText(context, conversationId, key)
    if (text === null) return { content: `Blob key '${key}' not found.`, isError: true }
    const normalized = normalizeBlobKey(sandboxPath)
    if (!normalized) return { content: `Invalid sandbox path '${sandboxPath}'.`, isError: true }
    const parent = normalized.split('/').slice(0, -1).join('/')
    if (parent) await context.sandbox?.files.makeDir(`${root}/${parent}`)
    await context.sandbox?.files.write(`${root}/${normalized}`, text)
    return { content: `Copied blob '${key}' -> sandbox ${normalized}.` }
  } catch (error) {
    return { content: error instanceof Error ? error.message : String(error), isError: true }
  }
}

/** Copy a sandbox file into Blob at `<conversation scope>/<key>`. */
export async function blobFromSandboxTool(context: MakersContext, conversationId: string, root: string, args: Record<string, unknown>): Promise<{ content: string; isError?: boolean }> {
  try {
    const key = typeof args.key === 'string' ? args.key : ''
    const sandboxPath = typeof args.sandbox_path === 'string' && args.sandbox_path.trim()
      ? args.sandbox_path.trim()
      : key
    if (!key) return { content: 'blob_from_sandbox requires "key".', isError: true }
    const normalized = normalizeBlobKey(sandboxPath)
    if (!normalized) return { content: `Invalid sandbox path '${sandboxPath}'.`, isError: true }
    const raw = await context.sandbox?.files.read(`${root}/${normalized}`)
    const content = textFromSandboxRead(raw)
    await writeBlobText(context, conversationId, key, content)
    return { content: `Copied sandbox ${normalized} -> blob '${key}'.` }
  } catch (error) {
    return { content: error instanceof Error ? error.message : String(error), isError: true }
  }
}

export function textFromSandboxRead(value: unknown): string {
  if (typeof value === 'string') return value
  if (value instanceof Uint8Array) return new TextDecoder().decode(value)
  if (value instanceof ArrayBuffer) return new TextDecoder().decode(new Uint8Array(value))
  if (value && typeof value === 'object' && 'content' in value && typeof (value as { content?: unknown }).content === 'string') {
    return (value as { content: string }).content
  }
  return String(value ?? '')
}
