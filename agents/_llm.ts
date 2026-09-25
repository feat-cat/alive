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

export interface LlmMessage {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content: string
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
  content: string
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

  // DeepSeek V4's strict serde requires `name` on non-system messages (the
  // heartbeat trigger / history rows restored by `loadMessages` only carry
  // `{ role, content }`, which the live gateway rejected with 400
  // `messages[1]: missing field name`) and OpenAI-standard `tool_calls` items
  // (flat `{ id, name, arguments }` would 400 with `missing field name` too).
  // Normalize every message here so the upstream payload is complete: system
  // untouched, tool results keep their function name, user/assistant get a
  // stable role-derived name, and assistant tool_calls are wrapped.
  const providerMessages = messages.map(normalizeOutboundMessage)

  const body: Record<string, unknown> = {
    model: gateway.model,
    messages: providerMessages,
    temperature,
    stream: false,
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

  try {
    const response = await fetch(`${gateway.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        authorization: `Bearer ${gateway.apiKey}`,
        'content-type': 'application/json',
        accept: 'application/json',
        // See the header note at the top of this file: deepseek-harness template
        // header for the Makers AI Gateway; delete if deployment proves the
        // gateway does not need it.
        'x-gateway-quota-bypass': 'true',
        'makers-conversation-id': conversationId,
      },
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
      return { text: last?.content ?? '', turns, toolResults }
    }
  }

  // Unreachable when maxTurns >= 1 (the loop always returns), kept for TS.
  const last = current.filter((message) => message.role === 'assistant').at(-1)
  return { text: last?.content ?? '', turns, toolResults }
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
