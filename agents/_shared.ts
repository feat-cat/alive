/**
 * Shared primitives for the alive project: logger, SSE helpers, platform
 * context types, and the fixed conversation ids used by autonomous turns.
 *
 * Everything here is pure / runtime-agnostic so it can be unit-tested with a
 * mock `context` and does not touch `process.env`.
 */

/**
 * Fixed conversation used by every turn (heartbeat + chat share ONE history —
 * the agent's private thoughts and its conversations with the user live in the
 * same message stream, so they are one continuous self).
 */
export const SELF_ID = 'eo-self'

/** Blob store namespace shared by memory + workspace mirrors. */
export const BLOB_STORE_NAME = 'alive'

/** Default model when AI_GATEWAY_MODEL is not configured. */
export const DEFAULT_MODEL = '@makers/deepseek-v4-flash'

/** Maximum tool turns in a single LLM loop (platform loop-cap rule #10). */
export const DEFAULT_MAX_TURNS = 5

/**
 * Single heartbeat turn budget. Set to 100s (not the 120s task ceiling) so the
 * remaining ~20s is reserved for the final workspace snapshot, state update
 * and response serialization inside the platform timeout.
 */
export const PLAY_TURN_TIMEOUT_MS = 100_000

/** Short LLM call timeout for a single gateway request. */
export const LLM_TIMEOUT_MS = 90_000

export interface Env {
  AI_GATEWAY_API_KEY?: string
  AI_GATEWAY_BASE_URL?: string
  AI_GATEWAY_MODEL?: string
  TAVILY_API_KEY?: string
  [key: string]: string | undefined
}

/**
 * Persistent KV that MAY be conversation-scoped. The real platform either
 * scopes `store.state` to the current conversation implicitly or keys it by a
 * global string; callers pass the explicit `conversationId` anyway so the two
 * behaviours are indistinguishable. `_state.ts` additionally splits SELF vs
 * PLAY into distinct keys (`agent_state_self` / `agent_state_play`) so the two
 * autonomous personas never overwrite each other under either platform model.
 */
export interface StoreState {
  get<T>(key: string, conversationId?: string): Promise<T | null>
  set(key: string, value: unknown, conversationId?: string): Promise<void>
  delete(key: string, conversationId?: string): Promise<void>
}

export interface StoreMessage {
  id?: string
  role: string
  content: string
  metadata?: Record<string, unknown>
  createdAt?: string
}

export interface StoreLike {
  state: StoreState
  appendMessage(opts: {
    conversationId: string
    role: string
    content: string
    metadata?: Record<string, unknown>
  }): Promise<unknown>
  getMessages(opts: {
    conversationId: string
    limit?: number
    order?: 'asc' | 'desc'
  }): Promise<StoreMessage[]>
  deleteMessage(opts: {
    conversationId: string
    id: string
  }): Promise<unknown>
  getConversation(id: string): Promise<{ metadata?: Record<string, unknown> } | null>
  updateConversation(id: string, opts: { metadata: Record<string, unknown> }): Promise<unknown>
}

export interface SandboxCommandResult {
  stdout: unknown
  stderr: unknown
  exitCode: number
}

export interface SandboxLike {
  commands: {
    run(cmd: string, opts?: { cwd?: string; timeout?: number }): Promise<SandboxCommandResult>
  }
  files: {
    read(path: string): Promise<unknown>
    write(path: string, content: string): Promise<unknown>
    list(path: string): Promise<unknown>
    makeDir(path: string): Promise<unknown>
    exists(path: string): Promise<boolean>
    remove(path: string): Promise<unknown>
  }
}

export interface MakersContext {
  env: Env
  conversation_id?: string
  request?: {
    body?: unknown
    headers?: Record<string, string | string[] | undefined>
    signal?: AbortSignal
    url?: string
  }
  store?: StoreLike
  sandbox?: SandboxLike
  utils?: {
    abortActiveRun?: (conversationId: string) => Promise<{ aborted?: boolean } | undefined>
  }
  [key: string]: unknown
}

/** Platform entry passes an opaque `any`; narrow it once at the boundary. */
export function asMakersContext(value: unknown): MakersContext {
  return value as MakersContext
}

/**
 * Read a string env var from the platform-injected context (never process.env).
 * Returns `''` when unset / non-string / whitespace-only.
 */
export function envString(context: MakersContext, key: string): string {
  const value = context.env?.[key]
  return typeof value === 'string' ? value.trim() : ''
}

export function createLogger(name: string): {
  log: (...args: unknown[]) => void
  error: (...args: unknown[]) => void
} {
  return {
    log: (...args: unknown[]) => console.log(`[${name}][${new Date().toISOString()}]`, ...args),
    error: (...args: unknown[]) => console.error(`[${name}][${new Date().toISOString()}]`, ...args),
  }
}

/** SSE event envelope (platform recommended convention). */
export function sseEvent(data: Record<string, unknown>): string {
  return `data: ${JSON.stringify(data)}\n\n`
}

/** Standard SSE end-of-stream sentinel: `data: [DONE]\n\n`. */
export function sseDone(): string {
  return 'data: [DONE]\n\n'
}

/**
 * Wrap an async generator of SSE frames into a `text/event-stream` Response.
 * A 5s `ping` frame keeps proxies from closing idle streams; generator errors
 * become an `error_message` frame (AbortError / aborted signals are silent).
 * A client disconnect (`cancel()`) sets an internal flag that stops the
 * generator loop, and every `controller.enqueue` in the error path is wrapped
 * in try/catch — an enqueue on an already-cancelled controller throws and must
 * never escape the stream. CORS is applied like `jsonOk`/`jsonError` so a web
 * frontend can consume the stream cross-origin.
 */
export function createSSEResponse(
  generator: (signal?: AbortSignal) => AsyncGenerator<string>,
  signal?: AbortSignal,
): Response {
  const encoder = new TextEncoder()
  let cancelled = false
  const readableStream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const heartbeat = setInterval(() => {
        if (cancelled) return
        try {
          controller.enqueue(encoder.encode(sseEvent({ type: 'ping', ts: Date.now() })))
        } catch {
          /* stream closed */
        }
      }, 5_000)
      try {
        for await (const chunk of generator(signal)) {
          if (signal?.aborted || cancelled) break
          controller.enqueue(encoder.encode(chunk))
        }
      } catch (error) {
        const err = error as Error
        if (err.name !== 'AbortError' && !signal?.aborted && !cancelled) {
          try {
            controller.enqueue(encoder.encode(sseEvent({ type: 'error_message', content: err.message })))
          } catch {
            /* client disconnected while the error frame was enqueued */
          }
        }
      } finally {
        clearInterval(heartbeat)
        try {
          controller.close()
        } catch {
          /* already closed */
        }
      }
    },
    cancel() {
      cancelled = true
    },
  })
  return new Response(readableStream, {
    status: 200,
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'X-Accel-Buffering': 'no',
      ...CORS_HEADERS,
    },
  })
}

/** Replace any character not safe for a blob-key/workspace segment. */
export function safeSegment(value: string): string {
  const clean = value.replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 80)
  return clean.length > 0 ? clean : 'default'
}

export function nowIso(): string {
  return new Date().toISOString()
}

/** Conversation id for a request: header/body/conversation_id, else a fallback. */
export function resolveConversationId(context: MakersContext, fallback: string): string {
  const header = readHeader(context, 'makers-conversation-id')
  const bodyConv = bodyValue(context, 'conversation_id')
  const direct = context.conversation_id
  const candidate = header || bodyConv || direct
  return candidate && candidate.trim() ? candidate.trim() : fallback
}

export function readHeader(context: MakersContext, name: string): string {
  const value = context.request?.headers?.[name]
  if (Array.isArray(value)) return String(value[0] ?? '')
  return typeof value === 'string' ? value : ''
}

export function bodyValue(context: MakersContext, key: string): string {
  const body = context.request?.body
  if (!body || typeof body !== 'object') return ''
  const value = (body as Record<string, unknown>)[key]
  return typeof value === 'string' ? value : ''
}

/**
 * Optional bearer-token auth. Reads `ALIVE_AUTH_TOKEN` from the platform
 * context (never `process.env`); trimmed, `''` when unset or whitespace-only.
 */
export function authToken(context: MakersContext): string {
  return envString(context, 'ALIVE_AUTH_TOKEN')
}

/**
 * Enforce optional auth on a request. When `ALIVE_AUTH_TOKEN` is unset (or
 * empty) the endpoint stays open (local dev) and this returns `null`. When set,
 * the request must carry `Authorization: Bearer <token>` with the exact token
 * value (case-sensitive comparison); otherwise a 401 JSON response is returned.
 */
export function requireAuth(context: MakersContext): Response | null {
  const token = authToken(context)
  if (!token) return null
  // Headers reach us as a plain object; the runtime may deliver the key in
  // either casing, so accept both. `readHeader` handles array values too.
  const header = readHeader(context, 'authorization') || readHeader(context, 'Authorization')
  // KNOWN LIMITATION: this is not a timing-safe comparison. For the single
  // tenant / self-use / free-tier threat model this is acceptable. If it ever
  // needs hardening, hash the token (e.g. SHA-256) and the candidate before a
  // constant-time compare (`crypto.timingSafeEqual`). No logic change here.
  if (header === `Bearer ${token}`) return null
  return jsonError(401, 'Unauthorized')
}

/**
 * Common CORS headers on every JSON response so the web frontend can call the
 * endpoints cross-origin. The object is shared (const, never mutated) — jsonOk
 * and jsonError both spread it into a fresh headers map per response.
 */
const CORS_HEADERS = {
  'access-control-allow-origin': '*',
  'access-control-allow-methods': 'GET,POST,OPTIONS',
  'access-control-allow-headers': 'content-type,authorization',
} as const

export function jsonError(status: number, message: string, extra?: Record<string, unknown>): Response {
  return Response.json({ ok: false, error: message, ...extra }, {
    status,
    headers: { 'cache-control': 'no-store', ...CORS_HEADERS },
  })
}

export function jsonOk(payload: Record<string, unknown>): Response {
  return Response.json({ ok: true, ...payload }, {
    headers: { 'cache-control': 'no-store', ...CORS_HEADERS },
  })
}

/** Map any thrown error to a stable JSON error response (rule #11: never crash). */
export function errorResponse(error: unknown): Response {
  const message = error instanceof Error ? error.message : String(error)
  const aborted = error instanceof Error && (error.name === 'AbortError' || /aborted/i.test(message))
  return jsonError(aborted ? 499 : 500, aborted ? 'Request aborted' : message)
}

/** Keep the first N bytes of a string (model context budget). */
export function clampText(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text
  return `${text.slice(0, maxChars)}…[truncated]`
}
