/**
 * Shared in-memory mocks for the alive test suite.
 *
 * These fakes mirror the platform surface (`StoreLike`, `SandboxLike`,
 * `BlobStore`) so endpoint tests can run without any external service.
 * Everything is kept in plain Maps and reset per-test by the caller.
 */
import type { Env, MakersContext, SandboxLike, StoreLike, StoreMessage } from '../agents/_shared.ts'
import type { BlobStore } from '../agents/_blob-tools.ts'

export interface MockStore extends StoreLike {
  /**
   * Persistent KV isolated per conversation: `conversationId -> key -> value`.
   * This mirrors the platform model where `store.state` may be scoped to the
   * current conversation; callers pass the conversationId explicitly and the
   * agent also splits SELF/PLAY into distinct keys, so isolation holds under
   * either platform behaviour.
   */
  stateMap: Map<string, Map<string, unknown>>
  /** Flat log of every appended message (for assertions). */
  messageLog: StoreMessage[]
  /** Messages grouped by conversation id. */
  logs: Map<string, StoreMessage[]>
  addMessage(
    conversationId: string,
    message: { role: string; content: string; metadata?: Record<string, unknown> },
  ): void
}

/**
 * Conversation-scoped persistent KV + message log, backed by plain Maps.
 * Seed with `{ [conversationId]: { [key]: value } }`.
 */
export function makeMockStore(initialState: Record<string, Record<string, unknown>> = {}): MockStore {
  const stateMap = new Map<string, Map<string, unknown>>()
  for (const [conversationId, entries] of Object.entries(initialState)) {
    stateMap.set(conversationId, new Map(Object.entries(entries)))
  }
  const logs = new Map<string, StoreMessage[]>()
  const messageLog: StoreMessage[] = []
  let seq = 0

  const addMessage: MockStore['addMessage'] = (conversationId, message) => {
    seq += 1
    const entry: StoreMessage = {
      id: String(seq),
      role: message.role,
      content: message.content,
      metadata: message.metadata,
      createdAt: new Date().toISOString(),
    }
    messageLog.push(entry)
    const list = logs.get(conversationId) ?? []
    list.push(entry)
    logs.set(conversationId, list)
  }

  /** State scoped to one conversation (creates the bucket on first write). */
  const scoped = (conversationId: string | undefined): Map<string, unknown> => {
    const id = conversationId?.trim() ? conversationId : 'global'
    let bucket = stateMap.get(id)
    if (!bucket) {
      bucket = new Map<string, unknown>()
      stateMap.set(id, bucket)
    }
    return bucket
  }

  return {
    state: {
      get: async <T>(key: string, conversationId?: string): Promise<T | null> => {
        const bucket = scoped(conversationId)
        return bucket.has(key) ? (bucket.get(key) as T) : null
      },
      set: async (key: string, value: unknown, conversationId?: string): Promise<void> => {
        scoped(conversationId).set(key, value)
      },
      delete: async (key: string, conversationId?: string): Promise<void> => {
        scoped(conversationId).delete(key)
      },
    },
    appendMessage: async (opts): Promise<void> => {
      addMessage(opts.conversationId, { role: opts.role, content: opts.content, metadata: opts.metadata })
    },
    getMessages: async (opts): Promise<StoreMessage[]> => {
      const list = (logs.get(opts.conversationId) ?? []).filter((message) => message.role !== 'system')
      const ordered = opts.order === 'desc' ? [...list].reverse() : [...list]
      return ordered.slice(0, opts.limit ?? 50)
    },
    deleteMessage: async (opts) => {
      const list = logs.get(opts.conversationId)
      if (list) {
        const index = list.findIndex((message) => message.id === opts.id)
        if (index >= 0) list.splice(index, 1)
      }
      const logIndex = messageLog.findIndex((message) => message.id === opts.id)
      if (logIndex >= 0) messageLog.splice(logIndex, 1)
    },
    getConversation: async (id: string) => ({ metadata: { id } }),
    updateConversation: async () => undefined,
    stateMap,
    messageLog,
    logs,
    addMessage,
  }
}

export interface MockSandbox extends SandboxLike {
  filesMap: Map<string, string>
}

/**
 * In-memory sandbox backed by a readable/writable Map filesystem. `files.list`
 * returns the matching keys and `commands.run` simulates GNU `find` over the
 * map (excluding hidden dot-paths the way real find does), so the real
 * `restoreWorkspaceFromBlob` / `snapshotWorkspaceToBlob` / `listWorkspaceFiles`
 * / `listSandboxFiles` code paths are exercised instead of a fake empty stdout.
 */
export function makeMockSandbox(initial: Record<string, string> = {}): MockSandbox {
  const filesMap = new Map<string, string>(Object.entries(initial))

  const relativeTo = (path: string, cwd: string): string =>
    path.startsWith(`${cwd}/`) ? path.slice(cwd.length + 1) : ''

  /** Real find hides dot-paths via its not-hidden-path predicate. */
  const isFindHidden = (relative: string): boolean =>
    relative.split('/').some((segment) => segment.startsWith('.'))

  /** Emit `find` output for the cwd, either plain paths or `-printf` records. */
  const simulateFind = (cwd: string, printf: boolean): string => {
    const lines: string[] = []
    for (const [path, content] of filesMap) {
      const relative = relativeTo(path, cwd)
      if (!relative || isFindHidden(relative)) continue
      if (printf) {
        lines.push(`f\t${new TextEncoder().encode(content).byteLength}\t${relative}`)
      } else {
        lines.push(relative)
      }
    }
    return lines.join('\n')
  }

  return {
    commands: {
      run: async (cmd, opts) => {
        if (!cmd.includes('find')) return { stdout: '', stderr: '', exitCode: 0 }
        const cwd = opts?.cwd ?? '/'
        return { stdout: simulateFind(cwd, cmd.includes('-printf')), stderr: '', exitCode: 0 }
      },
    },
    files: {
      read: async (path) => filesMap.get(path) ?? null,
      write: async (path, content) => {
        filesMap.set(path, content)
      },
      list: async (path) => [...filesMap.keys()].filter((key) => key === path || key.startsWith(`${path}/`)),
      makeDir: async () => undefined,
      exists: async (path) => filesMap.has(path),
      remove: async (path) => {
        filesMap.delete(path)
      },
    },
    filesMap,
  }
}

export interface MockBlobStore extends BlobStore {
  blobMap: Map<string, string>
}

/** Strong-consistency Blob fake: text-only, prefix-listable. */
export function makeMockBlobStore(initial: Record<string, string> = {}): MockBlobStore {
  const blobMap = new Map<string, string>(Object.entries(initial))
  return {
    get: async (key) => (blobMap.has(key) ? blobMap.get(key) : null),
    set: async (key, value) => {
      blobMap.set(key, typeof value === 'string' ? value : '')
    },
    setJSON: async (key, value) => {
      blobMap.set(key, JSON.stringify(value))
    },
    delete: async (key) => {
      blobMap.delete(key)
    },
    list: async (opts) => {
      const prefix = opts?.prefix ?? ''
      const blobs = [...blobMap.entries()]
        .filter(([key]) => key.startsWith(prefix))
        .map(([key]) => ({ key, etag: '' }))
      return { blobs }
    },
    blobMap,
  }
}

export interface MakeContextOptions {
  store?: StoreLike
  sandbox?: SandboxLike
  env?: Env
  conversation_id?: string
  body?: unknown
  headers?: Record<string, string | string[] | undefined>
  url?: string
  abortActiveRun?: (conversationId: string) => Promise<{ aborted?: boolean } | undefined>
}

/** Build a minimal `MakersContext` with a live AbortSignal for the request. */
export function makeContext(options: MakeContextOptions = {}): MakersContext {
  return {
    env: options.env ?? {},
    conversation_id: options.conversation_id,
    request: {
      body: options.body,
      headers: options.headers ?? {},
      signal: new AbortController().signal,
      ...(options.url ? { url: options.url } : {}),
    },
    store: options.store,
    sandbox: options.sandbox,
    utils: options.abortActiveRun ? { abortActiveRun: options.abortActiveRun } : undefined,
  }
}

/** Minimal AI gateway env that passes `requireGatewayEnv`. */
export function gatewayEnv(): Env {
  return {
    AI_GATEWAY_API_KEY: 'test-key',
    AI_GATEWAY_BASE_URL: 'https://gateway.test',
    AI_GATEWAY_MODEL: '@makers/deepseek-v4-flash',
  }
}
