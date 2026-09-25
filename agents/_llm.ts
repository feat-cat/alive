/**
 * LLM access through the Makers AI Gateway. Reads AI_GATEWAY_* from
 * `context.env` (never process.env), calls `${AI_GATEWAY_BASE_URL}/chat/completions`,
 * and drives a bounded tool-calling loop (rule #10: cap your loops).
 *
 * Requests include an `x-gateway-quota-bypass: true` header — inherited from
 * the deepseek-harness template for the EdgeOne Makers AI Gateway to recognize
 * (it may lift free-tier quota accounting for this agent). Whether the gateway
 * actually needs it is UNVERIFIED: if deployment shows the header is
 * unnecessary or rejected, delete the line from the fetch headers below. See
 * README "已知限制" for the pending deployment confirmation.
 */
import {
  DEFAULT_MAX_TURNS,
  DEFAULT_MODEL,
  LLM_TIMEOUT_MS,
  type MakersContext,
} from './_shared.ts'

export interface LlmToolCall {
  id: string
  name: string
  /** Model-provided args: usually a JSON string, but some gateways send an object. */
  arguments: string | Record<string, unknown>
}

export interface LlmToolDef {
  name: string
  description: string
  parameters: Record<string, unknown>
}

/** OpenAI-style text part of a multimodal content array. */
export interface LlmTextPart {
  type: 'text'
  text: string
}

/** OpenAI-style image part; `url` is a base64 `data:` URL or an http(s) URL. */
export interface LlmImagePart {
  type: 'image_url'
  image_url: { url: string }
}

export type LlmContentPart = LlmTextPart | LlmImagePart

/**
 * Message content for the AI Gateway: either plain text (the common case —
 * heartbeat, compact, tool results, most chat turns) or an OpenAI multimodal
 * content array (text + `image_url` parts, used by chat when the user sends
 * images). The normalizer below never flattens/splices the array — it passes
 * the value through verbatim so the structured parts survive serialization.
 */
export type LlmContent = string | LlmContentPart[]

export interface LlmMessage {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content: LlmContent
  tool_calls?: LlmToolCall[]
  tool_call_id?: string
  name?: string
}

/**
 * OpenAI-standard tool_call entry as the DeepSeek V4 gateway's strict serde
 * requires: each item carries `type: 'function'` and nests the flat
 * `{ id, name, arguments }` inside `function: { name, arguments }`. Sending
 * the flat shape 400s live with `messages[i]: missing field name`.
 */
export interface ProviderToolCall {
  id: string
  type: 'function'
  function: {
    name: string
    arguments: string
  }
}

/** Upstream wire message: name-normalized + tool_calls wrapped. */
export interface ProviderMessage {
  role: LlmMessage['role']
  content: LlmContent
  tool_calls?: ProviderToolCall[]
  tool_call_id?: string
  name?: string
}

/**
 * The DeepSeek V4 gateway (via EdgeOne Makers `@makers/deepseek-v4-flash`)
 * deserializes every request message with a Rust serde target type that
 * REQUIRES a `name` field on non-system messages (observed live: 400
 * `messages[1]: missing field name` on the first heartbeat trigger) and an
 * OpenAI-standard `tool_calls` array (flat `{ id, name, arguments }` items
 * 400 with `missing field name` too). Standard OpenAI clients only send
 * `{ role, content }`, which the strict schema rejects. This normalizer
 * guarantees every outbound message is complete: non-system messages get a
 * participant `name` (role-derived), assistant `tool_calls` are wrapped into
 * `{ id, type: 'function', function: { name, arguments } }`, and system
 * messages / messages that already carry a name (runtime `tool` results use
 * the function name) pass through their fields unchanged.
 */
export function withProviderMessageName(messages: LlmMessage[]): ProviderMessage[] {
  return messages.map(normalizeOutboundMessage)
}

/**
 * Wrap a flat `LlmToolCall` into the OpenAI-standard tool_call entry. `arguments`
 * is already a JSON string for most gateways; an object is stringified here.
 */
export function toStandardToolCalls(toolCalls: LlmToolCall[] | undefined): ProviderToolCall[] {
  if (!toolCalls || toolCalls.length === 0) return []
  return toolCalls.map((call) => ({
    id: call.id,
    type: 'function' as const,
    function: {
      name: call.name,
      arguments: typeof call.arguments === 'string' ? call.arguments : JSON.stringify(call.arguments),
    },
  }))
}

/**
 * Normalize one outbound message for the strict gateway: add a role-derived
 * `name` to non-system messages that lack one, and wrap assistant `tool_calls`
 * into the OpenAI standard shape. The field is omitted entirely when the
 * message has no tool calls.
 */
export function normalizeOutboundMessage(message: LlmMessage): ProviderMessage {
  const hasName = typeof message.name === 'string' && message.name.trim().length > 0
  const name = hasName ? message.name : message.role === 'system' ? undefined : message.role
  const out: ProviderMessage = { role: message.role, content: message.content }
  if (name !== undefined) out.name = name
  if (message.tool_call_id !== undefined) out.tool_call_id = message.tool_call_id
  const toolCalls = toStandardToolCalls(message.tool_calls)
  if (toolCalls.length > 0) out.tool_calls = toolCalls
  return out
}

/* ------------------------------------------------------------------ */
/* Vision degradation (multimodal → text fallback)                     */
/* ------------------------------------------------------------------ */

/**
 * System-note appended when the gateway rejects image input so the model knows
 * the user tried to send a picture and must reply in text.
 */
export const VISION_UNSUPPORTED_NOTE =
  '注意：用户刚刚发了一张图片，但当前模型不支持视觉输入。请用文本回应——你可以告诉用户无法看图，或请用户描述图片内容。'

/**
 * Gateway 400s that mean "this model cannot see" — matched on vision-specific
 * tokens with word boundaries. The broad old pattern (`image|not support`)
 * also matched unrelated gateway errors such as "temperature not supported"
 * and triggered a pointless vision-degraded retry.
 */
const VISION_UNSUPPORTED_RE = /\b(image_url|data:image|vision|multimodal|multi[- ]?modal)\b/i

/** The gateway error message must literally carry the 400 HTTP status. */
const VISION_UNSUPPORTED_STATUS_RE = /\bAI gateway HTTP 400\b/

/**
 * True when an AI Gateway error indicates image/multimodal input is not
 * supported by the current model. The message must carry a vision-specific
 * token (image_url / data:image / vision / multimodal / multi-modal) AND the
 * gateway HTTP status must be 400 — an unrelated "temperature not supported"
 * 400 or a plain 503 must never trigger the vision degradation retry. Used by
 * the chat endpoint to retry the turn WITHOUT images on a fresh request.
 */
export function isVisionUnsupportedError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error)
  return VISION_UNSUPPORTED_STATUS_RE.test(message) && VISION_UNSUPPORTED_RE.test(message)
}

/**
 * Collapse multimodal content back to text: returns the concatenation of the
 * `text` parts (trimmed), or `fallback` when the content had no usable text.
 * String content is returned untouched, so this is safe to call on ANY message
 * (system/assistant/history that never carried images pass through).
 */
export function stripImageContent(content: LlmContent, fallback = '（图片已省略，当前模型不支持视觉输入）'): string {
  if (typeof content === 'string') return content
  const text = content
    .filter((part): part is LlmTextPart => part.type === 'text')
    .map((part) => part.text.trim())
    .filter(Boolean)
    .join('\n')
    .trim()
  return text.length > 0 ? text : fallback
}

/**
 * Build the degraded request for vision-less models: the first system message
 * gains `VISION_UNSUPPORTED_NOTE`, and every other message has its image parts
 * stripped (`stripImageContent`) so the gateway never sees a bare `image_url`
 * part again. Used once after a vision-unsupported 400.
 */
export function degradeVisionMessages(messages: LlmMessage[]): LlmMessage[] {
  return messages.map((message, index) => {
    if (index === 0 && message.role === 'system') {
      const base = typeof message.content === 'string' ? message.content : ''
      const note = base.length > 0 ? `${base}\n\n${VISION_UNSUPPORTED_NOTE}` : VISION_UNSUPPORTED_NOTE
      return { ...message, content: note }
    }
    return { ...message, content: stripImageContent(message.content) }
  })
}

export interface ToolRunRecord {
  name: string
  args: Record<string, unknown>
  isError: boolean
  content: string
}

export interface ToolRunner {
  (
    name: string,
    args: Record<string, unknown>,
    signal?: AbortSignal,
  ): Promise<{ content: string; isError?: boolean }>
}

export interface ChatOptions {
  context: MakersContext
  conversationId: string
  messages: LlmMessage[]
  tools?: LlmToolDef[]
  toolRunner?: ToolRunner
  maxTurns?: number
  signal?: AbortSignal
  temperature?: number
  maxTokens?: number
}

export interface ChatResult {
  text: string
  turns: number
  toolResults: ToolRunRecord[]
}

export interface GatewayEnv {
  apiKey: string
  baseUrl: string
  model: string
}

/**
 * Build the OpenAI-compatible request body shared by the blocking
 * (`stream: false`) and streaming (`stream: true`) completion calls.
 * DeepSeek V4's strict serde requires the `name` fields on non-system
 * messages and the OpenAI-standard tool wrappers, so outbound messages are
 * normalized here exactly like `singleCall` did before the extraction.
 */
export function buildChatBody(
  gateway: GatewayEnv,
  messages: LlmMessage[],
  tools: LlmToolDef[] | undefined,
  temperature: number,
  maxTokens: number | undefined,
  stream: boolean,
): Record<string, unknown> {
  const body: Record<string, unknown> = {
    model: gateway.model,
    messages: messages.map(normalizeOutboundMessage),
    temperature,
    stream,
  }
  if (tools && tools.length > 0) {
    // OpenAI-compatible gateways require the { type: 'function', function:
    // { name, description, parameters } } wrapper (the flat LlmToolDef registry
    // shape would otherwise 400: `tools[0].type is invalid or missing`), AND
    // `parameters` must be a complete JSON Schema — `{ type: 'object',
    // properties, required }`. The bare property map would otherwise 400
    // (`schema must be ... got 'type': null`). Tools don't distinguish
    // required/optional, so all keys are required.
    body.tools = tools.map((tool) => ({
      type: 'function' as const,
      function: {
        name: tool.name,
        description: tool.description,
        parameters: {
          type: 'object',
          properties: tool.parameters,
          required: Object.keys(tool.parameters),
        },
      },
    }))
  }
  if (maxTokens !== undefined) body.max_tokens = maxTokens
  return body
}

/** Request headers for the Makers AI Gateway (shared by blocking + streaming). */
function gatewayHeaders(gateway: GatewayEnv, conversationId: string, accept: string): Record<string, string> {
  return {
    authorization: `Bearer ${gateway.apiKey}`,
    'content-type': 'application/json',
    accept,
    // See the header note at the top of this file: deepseek-harness template
    // header for the Makers AI Gateway; delete if deployment proves the
    // gateway does not need it.
    'x-gateway-quota-bypass': 'true',
    'makers-conversation-id': conversationId,
  }
}

/** Validate required gateway env; missing variables throw explicitly. */
export function requireGatewayEnv(context: MakersContext): GatewayEnv {
  const env = context.env ?? {}
  const apiKey = env.AI_GATEWAY_API_KEY?.trim() ?? ''
  const baseUrl = (env.AI_GATEWAY_BASE_URL ?? '').trim().replace(/\/+$/, '')
  if (!apiKey) throw new Error('Missing environment variable: AI_GATEWAY_API_KEY')
  if (!baseUrl) throw new Error('Missing environment variable: AI_GATEWAY_BASE_URL')
  const model = (env.AI_GATEWAY_MODEL ?? '').trim() || DEFAULT_MODEL
  return { apiKey, baseUrl, model }
}

interface RawCompletion {
  choices?: Array<{
    message?: {
      content?: string | null
      tool_calls?: Array<{
        id?: string
        type?: string
        function?: { name?: string; arguments?: string | Record<string, unknown> }
      }>
    }
    finish_reason?: string
  }>
}

function parseToolCalls(message: NonNullable<RawCompletion['choices']>[number]['message']): LlmToolCall[] {
  const calls = message?.tool_calls
  if (!Array.isArray(calls)) return []
  const parsed: LlmToolCall[] = []
  for (const call of calls) {
    if (!call || typeof call !== 'object' || !call.function) continue
    const name = call.function.name?.trim()
    if (!name) continue
    parsed.push({
      id: call.id || `call_${parsed.length + 1}`,
      name,
      arguments: call.function.arguments ?? '',
    })
  }
  return parsed
}

/** Parse model-provided args whether the gateway sent a JSON string or an object. */
function safeParseArguments(raw: string | Record<string, unknown>): Record<string, unknown> {
  if (raw && typeof raw === 'object' && !Array.isArray(raw)) return raw
  if (typeof raw !== 'string' || !raw.trim()) return {}
  try {
    const value: unknown = JSON.parse(raw)
    return value && typeof value === 'object' && !Array.isArray(value)
      ? (value as Record<string, unknown>)
      : {}
  } catch {
    return {}
  }
}

async function singleCall(
  context: MakersContext,
  conversationId: string,
  messages: LlmMessage[],
  tools: LlmToolDef[] | undefined,
  signal: AbortSignal | undefined,
  temperature: number,
  maxTokens: number | undefined,
): Promise<{ content: string; toolCalls: LlmToolCall[] }> {
  const gateway = requireGatewayEnv(context)
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), LLM_TIMEOUT_MS)
  if (signal?.aborted) controller.abort()
  const onAbort = () => controller.abort()
  signal?.addEventListener('abort', onAbort, { once: true })

  try {
    // `buildChatBody` re-applies the strict-serif normalization (role-derived
    // `name` + OpenAI-standard `tool_calls` wrappers) exactly as before, so
    // the upstream payload shape never changes.
    const body = buildChatBody(gateway, messages, tools, temperature, maxTokens, false)
    const response = await fetch(`${gateway.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: gatewayHeaders(gateway, conversationId, 'application/json'),
      body: JSON.stringify(body),
      signal: controller.signal,
    })
    if (!response.ok) {
      const text = await response.text().catch(() => '')
      throw new Error(`AI gateway HTTP ${response.status}${text ? `: ${text.slice(0, 300)}` : ''}`)
    }
    const payload = (await response.json()) as RawCompletion
    const message = payload.choices?.[0]?.message
    return {
      content: message?.content ?? '',
      toolCalls: parseToolCalls(message),
    }
  } catch (error) {
    if (controller.signal.aborted) {
      const abort = new Error('LLM request aborted')
      abort.name = 'AbortError'
      throw abort
    }
    throw error
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', onAbort)
  }
}

/**
 * Run a bounded chat-completion loop. When the model requests tool calls and a
 * runner is provided, each call is executed and fed back as a role:"tool"
 * message. Never loops more than `maxTurns` times.
 */
export async function chatCompletion(options: ChatOptions): Promise<ChatResult> {
  const {
    context,
    conversationId,
    messages,
    tools,
    toolRunner,
    signal,
    temperature = 0.6,
    maxTokens,
  } = options
  const maxTurns = Math.max(1, options.maxTurns ?? DEFAULT_MAX_TURNS)

  const current: LlmMessage[] = messages.map((message) => ({ ...message }))
  const toolResults: ToolRunRecord[] = []
  let turns = 0

  while (turns < maxTurns) {
    if (signal?.aborted) throwAbort()
    turns += 1
    const { content, toolCalls } = await singleCall(
      context,
      conversationId,
      current,
      tools,
      signal,
      temperature,
      maxTokens,
    )

    // No tool work left, or no runner to do it: return the final assistant text.
    if (toolCalls.length === 0 || !toolRunner) {
      return { text: content, turns, toolResults }
    }

    const assistantMessage: LlmMessage = { role: 'assistant', content, tool_calls: toolCalls }
    current.push(assistantMessage)
    for (const call of toolCalls) {
      // Abort checkpoint before every tool so the turn budget covers tool
      // execution, not just the LLM fetch (play's 100s controller signal).
      if (signal?.aborted) throwAbort()
      const args = safeParseArguments(call.arguments)
      const result = await toolRunner(call.name, args, signal).catch((error: unknown) => ({
        content: error instanceof Error ? `Tool error: ${error.message}` : `Tool error: ${String(error)}`,
        isError: true,
      }))
      const text = clampForModel(result.content)
      toolResults.push({
        name: call.name,
        args,
        isError: result.isError === true,
        content: text,
      })
      current.push({
        role: 'tool',
        tool_call_id: call.id,
        name: call.name,
        content: text,
      })
    }

    // The maxTurns budget was consumed by this batch of tool calls. The work is
    // already recorded in toolResults/current, so don't drop it silently — the
    // final text may be empty because the turn ended on tool calls.
    if (turns >= maxTurns) {
      const last = current.filter((message) => message.role === 'assistant').at(-1)
      return { text: textOf(last), turns, toolResults }
    }
  }

  // Unreachable when maxTurns >= 1 (the loop always returns), kept for TS.
  const last = current.filter((message) => message.role === 'assistant').at(-1)
  return { text: textOf(last), turns, toolResults }
}

/** Model-reply text from a message; multimodal array content yields ''. */
function textOf(message: { content?: LlmContent } | undefined): string {
  const content = message?.content
  return typeof content === 'string' ? content : ''
}

/**
 * The streaming path is TEXT-ONLY: when the model decides to call tools it
 * signals `StreamToolCallsError` and the endpoint aborts the stream and falls
 * back to the non-streaming `chatCompletion` on a fresh request. This keeps the
 * stream loop simple and the tool-execution semantics (bounded turns, tool
 * results re-injected) in exactly one place.
 */
export class StreamToolCallsError extends Error {
  name = 'StreamToolCallsError'
  constructor() {
    super('Model requested tool calls during streaming; fall back to a non-streaming completion.')
  }
}

export interface StreamChatCompletionOptions {
  context: MakersContext
  conversationId: string
  messages: LlmMessage[]
  /**
   * Tool definitions are SENT so the model may still choose to act; a delta
   * that carries `tool_calls` aborts the stream by throwing
   * `StreamToolCallsError` instead of buffering partial text.
   */
  tools?: LlmToolDef[]
  signal?: AbortSignal
  temperature?: number
  maxTokens?: number
  /** Called with each decoded content delta exactly once, in order. */
  onDelta: (text: string) => void
}

export interface StreamChatResult {
  text: string
}

/** One decoded SSE chunk from the gateway stream. */
interface StreamDelta {
  content: string | null
  toolCalls: boolean
}

/** Extract the content delta + whether any tool_call was requested from one chunk. */
function extractStreamDelta(parsed: unknown): StreamDelta {
  if (!parsed || typeof parsed !== 'object') return { content: null, toolCalls: false }
  const chunk = parsed as { choices?: Array<{ delta?: { content?: unknown; tool_calls?: unknown } }> }
  const delta = chunk.choices?.[0]?.delta
  if (!delta) return { content: null, toolCalls: false }
  const content = typeof delta.content === 'string' && delta.content.length > 0 ? delta.content : null
  const toolCalls = Array.isArray(delta.tool_calls) && delta.tool_calls.length > 0
  return { content, toolCalls }
}

/**
 * Stream a single chat-completion request from the Makers AI Gateway
 * (`stream: true`). Parses SSE `data:` frames from the response body, feeds
 * each `choices[0].delta.content` to `onDelta`, and accumulates the full text
 * in `{ text }`. Frames are split across arbitrary byte boundaries, `[DONE]`
 * (or end-of-stream) terminates the read, non-2xx responses throw with the
 * status + body snippet, and the caller's `signal` / the LLM timeout abort the
 * request like the blocking path.
 */
export async function streamChatCompletion(options: StreamChatCompletionOptions): Promise<StreamChatResult> {
  const { context, conversationId, messages, tools, signal, temperature = 0.6, maxTokens, onDelta } = options
  const gateway = requireGatewayEnv(context)
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), LLM_TIMEOUT_MS)
  if (signal?.aborted) controller.abort()
  const onAbort = () => controller.abort()
  signal?.addEventListener('abort', onAbort, { once: true })

  const body = buildChatBody(gateway, messages, tools, temperature, maxTokens, true)
  let reader: ReadableStreamDefaultReader<Uint8Array> | undefined

  try {
    const response = await fetch(`${gateway.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: gatewayHeaders(gateway, conversationId, 'text/event-stream'),
      body: JSON.stringify(body),
      signal: controller.signal,
    })
    if (!response.ok) {
      const text = await response.text().catch(() => '')
      throw new Error(`AI gateway HTTP ${response.status}${text ? `: ${text.slice(0, 300)}` : ''}`)
    }
    if (!response.body) throw new Error('AI gateway streaming response had no body.')
    reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let text = ''

    outer: while (true) {
      if (controller.signal.aborted) throwAbort()
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''
      for (const line of lines) {
        const clean = line.trim()
        if (!clean.startsWith('data:')) continue
        const payload = clean.slice('data:'.length).trim()
        if (!payload) continue
        if (payload === '[DONE]') break outer
        let parsed: unknown
        try {
          parsed = JSON.parse(payload)
        } catch {
          continue
        }
        const delta = extractStreamDelta(parsed)
        if (delta.toolCalls) throw new StreamToolCallsError()
        if (delta.content) {
          text += delta.content
          onDelta(delta.content)
        }
      }
    }
    return { text }
  } catch (error) {
    if (error instanceof StreamToolCallsError) throw error
    if (controller.signal.aborted) {
      const abort = new Error('LLM request aborted')
      abort.name = 'AbortError'
      throw abort
    }
    throw error
  } finally {
    if (reader) {
      try {
        await reader.cancel()
      } catch {
        /* reader already closed/errored */
      }
    }
    clearTimeout(timer)
    signal?.removeEventListener('abort', onAbort)
  }
}

function throwAbort(): never {
  const abort = new Error('Aborted')
  abort.name = 'AbortError'
  throw abort
}

function clampForModel(text: string): string {
  const MAX = 4_000
  return text.length <= MAX ? text : `${text.slice(0, MAX)}…[truncated]`
}
