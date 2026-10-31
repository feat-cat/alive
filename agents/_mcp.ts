/**
 * MCP (Model Context Protocol) client — Streamable HTTP transport.
 *
 * Lets alive reach ANY public HTTP MCP server and inject its tools into the
 * agent's tool registry, so the tool surface is "infinitely extensible": point
 * `ALIVE_MCP_SERVERS` at one or more MCP server root URLs and each one's
 * `tools/list` becomes a batch of alive tools.
 *
 * PURE OUTBOUND HTTP — this module NEVER touches `context.sandbox`, `/tmp`,
 * `runCode`/`commands`, or `process.env`. All config comes from
 * `context.env.ALIVE_MCP_SERVERS` (or an injected test seam). Failures are
 * best-effort: an unreachable / misconfigured server is skipped and never
 * blocks the agent's 20 handwritten tools.
 *
 * Wire shape (Streamable HTTP, 2025-03-26):
 *   POST {url}  Content-Type: application/json   body = JSON-RPC 2.0
 *   initialize  -> Mcp-Session-Id response header (carried on later requests)
 *   notifications/initialized (notification, no id, no response expected)
 *   tools/list  -> { result: { tools: [{ name, description?, inputSchema? }] } }
 *   tools/call  -> { result: { content: [{ type: 'text', text }], isError? } }
 */
import type { MakersContext } from './_shared.ts'

/** MCP protocol version negotiated during `initialize`. */
export const MCP_PROTOCOL_VERSION = '2025-03-26'

/** Per-request network timeout (ms). Injected via a shorter value in tests. */
export const MCP_DEFAULT_TIMEOUT_MS = 15_000

/** Single MCP tool-result truncation cap (chars). */
export const MCP_TOOL_RESULT_MAX = 4_000

/** Env var: comma-separated MCP server root URLs, e.g. https://a/x/mcp,https://b/y/mcp. */
export const MCP_ENV_VAR = 'ALIVE_MCP_SERVERS'

export interface McpServerConfig {
  /** Full server root URL (the value after the comma in ALIVE_MCP_SERVERS). */
  url: string
  /** Short server id used to prefix tool names (defaults to the URL host). */
  name: string
}

/** One MCP tool mapped to alive's `LlmToolDef` shape (name/description/parameters). */
export interface McpTool {
  name: string
  description: string
  parameters: Record<string, unknown>
}

/** Result of one `tools/call`: text content plus the server's error flag. */
export interface McpCallResult {
  content: string
  isError?: boolean
}

/** A connected MCP server: its discovered tools plus a call dispatcher. */
export interface McpClient {
  config: McpServerConfig
  tools: McpTool[]
  /**
   * Execute one MCP tool by its ORIGINAL (unprefixed) name. Returns the text
   * content (truncated); `isError: true` when the server flagged the call.
   * Throws `McpError` on transport / JSON-RPC / non-2xx failures.
   */
  call(name: string, args: Record<string, unknown>, signal?: AbortSignal): Promise<McpCallResult>
}

/** A merged MCP tool definition: the `LlmToolDef` shape plus a runner. */
export interface McpToolDef {
  name: string
  description: string
  parameters: Record<string, unknown>
  run: (args: Record<string, unknown>, signal?: AbortSignal) => Promise<{ content: string; isError?: boolean }>
}

/** Error for any MCP transport / JSON-RPC / HTTP failure (short message). */
export class McpError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'McpError'
  }
}

/** Short human-readable summary of any thrown value. */
export function mcpErrorSummary(error: unknown): string {
  if (error instanceof Error) return error.message
  return String(error)
}

/**
 * Parse `ALIVE_MCP_SERVERS`. Empty / undefined -> `[]`. Comma-separated; each
 * segment is trimmed, validated as an http(s) URL, and invalid segments are
 * skipped (best-effort). `name` defaults to the URL host.
 */
export function parseMcpServers(envValue: string | undefined): McpServerConfig[] {
  if (typeof envValue !== 'string') return []
  const configs: McpServerConfig[] = []
  for (const raw of envValue.split(',')) {
    const url = raw.trim()
    if (!url) continue
    let hostname = ''
    try {
      const parsed = new URL(url)
      if (parsed.protocol !== 'https:' && parsed.protocol !== 'http:') continue
      hostname = parsed.hostname
    } catch {
      continue // invalid URL — best-effort skip
    }
    configs.push({ url, name: hostname || 'mcp' })
  }
  return configs
}

let rpcSeq = 0
function nextRpcId(): number {
  rpcSeq += 1
  return rpcSeq
}

/**
 * One JSON-RPC POST. Always sends an id and expects a `result` (request). Reads
 * the `Mcp-Session-Id` response header into `sessionId`. Throws `McpError` on
 * non-2xx, network failure, timeout, JSON-RPC `error`, or a missing result.
 */
export async function mcpRpc(
  baseUrl: string,
  method: string,
  params: object | undefined,
  sessionId: string | undefined,
  signal?: AbortSignal,
  timeoutMs: number = MCP_DEFAULT_TIMEOUT_MS,
): Promise<{ result: unknown; sessionId?: string }> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  if (signal?.aborted) controller.abort()
  const onAbort = () => controller.abort()
  signal?.addEventListener('abort', onAbort, { once: true })

  let response: Response
  try {
    response = await fetch(baseUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(sessionId ? { 'Mcp-Session-Id': sessionId } : {}),
      },
      body: JSON.stringify({
        jsonrpc: '2.0',
        id: nextRpcId(),
        method,
        ...(params !== undefined ? { params } : {}),
      }),
      signal: controller.signal,
    })
  } catch (error) {
    const message = controller.signal.aborted
      ? `MCP ${method} to ${baseUrl} timed out after ${timeoutMs}ms`
      : `MCP ${method} to ${baseUrl} failed: ${error instanceof Error ? error.message : String(error)}`
    throw new McpError(message)
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', onAbort)
  }

  if (!response.ok) {
    const bodyText = await response.text().catch(() => '')
    const snippet = bodyText.trim().slice(0, 300)
    const status = response.statusText ? ` (${response.statusText})` : ''
    const suffix = snippet ? `: ${snippet}` : ''
    throw new McpError(`MCP ${method} to ${baseUrl} returned HTTP ${response.status}${status}${suffix}`)
  }

  const nextSessionId = response.headers.get('Mcp-Session-Id') ?? undefined
  let payload: unknown
  try {
    payload = await response.json()
  } catch {
    throw new McpError(`MCP ${method} to ${baseUrl} returned a non-JSON response`)
  }

  const rpc = payload as { result?: unknown; error?: { code?: unknown; message?: unknown } }
  if (rpc.error) {
    const code = typeof rpc.error.code === 'number' ? ` (code ${rpc.error.code})` : ''
    const message = typeof rpc.error.message === 'string' && rpc.error.message ? rpc.error.message : 'unknown error'
    throw new McpError(`MCP ${method} error${code}: ${message}`)
  }
  if (rpc.result === undefined) {
    throw new McpError(`MCP ${method} returned no result`)
  }
  return { result: rpc.result, sessionId: nextSessionId }
}

/**
 * Fire-and-forget JSON-RPC notification (no id, no response expected). Used for
 * `notifications/initialized`. Failures are ignored by the caller.
 */
async function sendNotification(
  url: string,
  method: string,
  sessionId: string | undefined,
  signal: AbortSignal | undefined,
  timeoutMs: number,
): Promise<void> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  if (signal?.aborted) controller.abort()
  const onAbort = () => controller.abort()
  signal?.addEventListener('abort', onAbort, { once: true })
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(sessionId ? { 'Mcp-Session-Id': sessionId } : {}),
      },
      body: JSON.stringify({ jsonrpc: '2.0', method }),
      signal: controller.signal,
    })
    // A notification has no expected result body, but draining it lets the
    // connection be reused. Non-2xx is surfaced as an error (ignored upstream).
    if (!response.ok) {
      throw new McpError(`MCP notification ${method} to ${url} returned HTTP ${response.status}`)
    }
    await response.text().catch(() => '')
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', onAbort)
  }
}

function truncateMcpText(text: string): string {
  if (text.length <= MCP_TOOL_RESULT_MAX) return text
  return `${text.slice(0, MCP_TOOL_RESULT_MAX)}…(truncated)`
}

async function callMcpTool(
  url: string,
  sessionId: string | undefined,
  name: string,
  args: Record<string, unknown>,
  signal: AbortSignal | undefined,
  timeoutMs: number,
): Promise<McpCallResult> {
  const params: Record<string, unknown> = { name }
  if (args && Object.keys(args).length > 0) params.arguments = args
  const rpc = await mcpRpc(url, 'tools/call', params, sessionId, signal, timeoutMs)
  const callResult = (rpc.result ?? {}) as { content?: unknown; isError?: unknown }
  const parts: string[] = []
  if (Array.isArray(callResult.content)) {
    for (const item of callResult.content) {
      if (item && typeof item === 'object' && (item as { type?: unknown }).type === 'text') {
        const text = (item as { text?: unknown }).text
        if (typeof text === 'string') parts.push(text)
      }
    }
  }
  const text = truncateMcpText(parts.join('\n'))
  const isError = callResult.isError === true
  return isError ? { content: text, isError: true } : { content: text }
}

/**
 * Connect to one MCP server: `initialize` (captures `Mcp-Session-Id`), the
 * `notifications/initialized` notification (failures ignored), then `tools/list`
 * mapped to `McpTool[]`. Throws `McpError` on any hard failure — the caller
 * (`discoverMcpClients`) decides whether to skip it.
 */
export async function connectMcpServer(
  config: McpServerConfig,
  signal?: AbortSignal,
  timeoutMs: number = MCP_DEFAULT_TIMEOUT_MS,
): Promise<McpClient> {
  const init = await mcpRpc(
    config.url,
    'initialize',
    {
      protocolVersion: MCP_PROTOCOL_VERSION,
      capabilities: {},
      clientInfo: { name: 'alive', version: '0.1.0' },
    },
    undefined,
    signal,
    timeoutMs,
  )
  const sessionId = init.sessionId

  // Notification is fire-and-forget: a failure here must not abort the connect
  // (some servers 404 it or don't respond). Awaiting it (with ignore) ensures
  // the server processes `initialized` before `tools/list`.
  try {
    await sendNotification(config.url, 'notifications/initialized', sessionId, signal, timeoutMs)
  } catch {
    /* best-effort */
  }

  const list = await mcpRpc(config.url, 'tools/list', undefined, sessionId, signal, timeoutMs)
  const listResult = (list.result ?? {}) as { tools?: unknown }
  const rawTools = Array.isArray(listResult.tools) ? (listResult.tools as unknown[]) : []
  const tools: McpTool[] = []
  for (const raw of rawTools) {
    if (!raw || typeof raw !== 'object') continue
    const t = raw as { name?: unknown; description?: unknown; inputSchema?: unknown }
    const name = typeof t.name === 'string' && t.name.trim() ? t.name.trim() : ''
    if (!name) continue
    tools.push({
      name,
      description: typeof t.description === 'string' ? t.description : '',
      parameters: (t.inputSchema && typeof t.inputSchema === 'object' ? t.inputSchema : {}) as Record<string, unknown>,
    })
  }

  const client: McpClient = {
    config,
    tools,
    // NOTE: `call` must NOT capture the connect-time signal — the cached client
    // outlives the turn, and a stale aborted signal would poison every later
    // call. Each call receives its own signal (the running turn's) instead.
    call: (name, args, callSignal) => callMcpTool(config.url, sessionId, name, args, callSignal, timeoutMs),
  }
  return client
}

/** Per-instance cache: server URL -> connected client (never reconnects per request). */
const mcpClientCache = new Map<string, McpClient>()
/** In-flight connect promises, so concurrent discovery of one URL dedupes. */
const mcpInFlight = new Map<string, Promise<McpClient>>()

/** Cached clients currently known to this instance (sync read for buildTools). */
export function cachedMcpClients(): McpClient[] {
  return [...mcpClientCache.values()]
}

/** Test seam: drop the per-instance cache + in-flight connects. */
export function clearMcpCacheForTesting(): void {
  mcpClientCache.clear()
  mcpInFlight.clear()
}

/**
 * Discover MCP clients from `context.env.ALIVE_MCP_SERVERS`. Best-effort: a
 * server that fails to connect is logged and skipped; the others still return.
 * Never throws. Uses the per-instance cache so repeated calls don't reconnect.
 */
export async function discoverMcpClients(
  context: MakersContext,
  signal?: AbortSignal,
  timeoutMs: number = MCP_DEFAULT_TIMEOUT_MS,
): Promise<McpClient[]> {
  const envValue = context.env?.[MCP_ENV_VAR]
  const configs = parseMcpServers(typeof envValue === 'string' ? envValue : undefined)
  const clients: McpClient[] = []
  for (const config of configs) {
    const cached = mcpClientCache.get(config.url)
    if (cached) {
      clients.push(cached)
      continue
    }
    let connecting = mcpInFlight.get(config.url)
    if (!connecting) {
      connecting = connectMcpServer(config, signal, timeoutMs)
      mcpInFlight.set(config.url, connecting)
      // Clean the in-flight slot when it settles (only if it's still ours).
      void connecting
        .then(() => {
          if (mcpInFlight.get(config.url) === connecting) mcpInFlight.delete(config.url)
        })
        .catch(() => {
          if (mcpInFlight.get(config.url) === connecting) mcpInFlight.delete(config.url)
        })
    }
    try {
      const client = await connecting
      mcpClientCache.set(config.url, client)
      clients.push(client)
    } catch (error) {
      console.log(`[mcp] connect ${config.url} failed: ${mcpErrorSummary(error)}`)
    }
  }
  return clients
}

/**
 * Turn MCP clients into alive tool definitions. Every tool is namespaced with
 * `<serverName>_<toolName>` so MCP tools can never shadow the handwritten ones
 * (e.g. a server's `web_search` becomes `<server>_web_search`). The runner
 * never throws: transport failures become `{ content, isError: true }`.
 */
export function buildMcpTools(clients: McpClient[]): McpToolDef[] {
  const defs: McpToolDef[] = []
  for (const client of clients) {
    const serverName = client.config.name || 'mcp'
    for (const tool of client.tools) {
      const fullName = `${serverName}_${tool.name}`
      defs.push({
        name: fullName,
        description: tool.description,
        parameters: tool.parameters,
        run: async (args, signal) => {
          try {
            return await client.call(tool.name, args, signal)
          } catch (error) {
            return { content: `MCP 工具 ${fullName} 调用失败: ${mcpErrorSummary(error)}`, isError: true }
          }
        },
      })
    }
  }
  return defs
}
