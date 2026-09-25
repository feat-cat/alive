/**
 * POST /chat — conversation endpoint with SSE streaming (typewriter effect).
 *
 * Flow (streaming, the default): append user message → build a persona (wall
 * clock + MEMORY.md "self") → feed the model a STANDARD messages array
 * ([system(persona), ...history] from the store via `loadMessages`) → try
 * `streamChatCompletion` (AI Gateway `stream: true`). Every text delta is
 * emitted as an SSE `ai_response` event while the full text accumulates; after
 * the stream closes the assistant reply is persisted (store + chatlog archive).
 *
 * Images: the request body may carry `images: string[]` (base64 `data:` URLs).
 * They are assembled into an OpenAI multimodal content array
 * (`[{ type:'text', text }, { type:'image_url', ... }]`) and persisted to
 * history as a JSON string with `metadata.kind:'image-user'`, so later turns
 * replay the picture to vision-capable models.
 *
 * Vision degradation: `@makers/deepseek-v4-flash` may not support image input.
 * If the gateway rejects the multimodal request with a 400 whose text mentions
 * image/vision/multimodal, the turn is retried ONCE without images: the system
 * prompt gains `VISION_UNSUPPORTED_NOTE` and every content array is stripped
 * back to text. A non-image 400 is thrown as-is.
 *
 * Tool fallback: chat keeps the FULL tool registry (blob + diary + chatlog +
 * workspace + search, same as heartbeat) so the agent can still act while
 * chatting — the tool definitions are sent on the streaming call too. If the
 * model chooses to call a tool, the first tool_call delta aborts the stream via
 * `StreamToolCallsError`, and the endpoint falls back to the non-streaming
 * `chatCompletion` loop which executes the tools; the completed reply is then
 * sent as ONE `ai_response` SSE event (`streamed:false`) before `[DONE]`, so
 * the transport stays consistently SSE for streaming clients.
 *
 * JSON compat: `?stream=false` (query) or `{ "stream": false }` (body) returns
 * the original one-shot JSON envelope via `runChat`.
 */
import {
  SELF_ID,
  asMakersContext,
  createSSEResponse,
  errorResponse,
  jsonOk,
  nowIso,
  requireAuth,
  resolveConversationId,
  sseDone,
  sseEvent,
  type MakersContext,
} from './_shared.ts'
import {
  StreamToolCallsError,
  chatCompletion,
  degradeVisionMessages,
  isVisionUnsupportedError,
  streamChatCompletion,
  type LlmContent,
  type LlmContentPart,
  type LlmMessage,
} from './_llm.ts'
import { buildPersona, humanNowText } from './_persona.ts'
import {
  clampMemoryForContext,
  ensureMemorySeed,
  loadMessages,
  persistHistory,
  readMemoryFile,
  recordToolCalls,
  SYSTEM_HISTORY_GUIDANCE,
} from './_memory.ts'
import { buildTools } from './_tools.ts'

const CHAT_MAX_TURNS = 3
const CHAT_TEMPERATURE = 0.7
const CHAT_MAX_TOKENS = 600
/** How many LLM attempts per turn: 1 normal + 1 vision-degraded retry max. */
const MAX_VISION_ATTEMPTS = 2
/** Max images accepted per chat turn (server-side cap, same as the frontend). */
const MAX_IMAGES_PER_REQUEST = 3
/** Max decoded size per image (4MB — a base64 string of ~5.6M chars). */
const MAX_IMAGE_BYTES = 4 * 1024 * 1024
/** Every image must be a local `data:image/...;base64,...` URL. */
const IMAGE_DATA_URL_RE = /^data:image\/[a-z0-9.+-]+;base64,/i

export interface ChatTurnOptions {
  message: string
  /** Base64 `data:` image URLs to send as OpenAI `image_url` content parts. */
  images?: string[]
  conversationId?: string
  signal?: AbortSignal
}

export type ChatResult = {
  reply: string
  conversationId: string
  now: string
}

export async function runChat(
  context: MakersContext,
  options: ChatTurnOptions,
): Promise<ChatResult> {
  const { message, images: imagesOption = [], conversationId: conversationIdOption, signal } = options
  const images = validateImages(imagesOption.filter(isNonEmptyString))
  const hasImages = images.length > 0
  const conversationId = conversationIdOption?.trim() || SELF_ID
  if (!message.trim() && !hasImages) throw new Error('message is required and must not be empty.')

  const store = context.store
  if (!store) throw new Error('Store is not available in this context.')

  await persistUserMessage(context, conversationId, message, images)

  // The persona is just the wall clock + MEMORY.md (the AI's self). Seeding +
  // reading MEMORY.md is best-effort so a Blob-less first turn still works.
  const memory = await readMemoryWithSeedSafe(context)
  const persona = buildPersona({
    nowText: humanNowText(),
    memoryContent: memory ? clampMemoryForContext(memory) : '',
  })

  // History as a real messages array — the store already contains the new user
  // message (appended above), so it flows to the model as its own entry.
  const history = await loadMessages(context, conversationId)

  // Full tool registry like heartbeat: the model can chat AND act (blob_*,
  // diary_*, chatlog_*, workspace_*, web_search). No 100s turn budget here —
  // only the request AbortSignal bounds the loop/tools.
  const tools = buildTools({ context, conversationId, signal })
  const baseMessages: LlmMessage[] = [
    { role: 'system', content: `${persona}\n\n${SYSTEM_HISTORY_GUIDANCE}` },
    ...history,
  ]
  const result = await chatCompletionWithVisionFallback(
    {
      context,
      conversationId,
      messages: baseMessages,
      tools: tools.definitions,
      toolRunner: tools.run,
      maxTurns: CHAT_MAX_TURNS,
      signal,
      temperature: CHAT_TEMPERATURE,
      maxTokens: CHAT_MAX_TOKENS,
    },
    hasImages,
  )

  // Persist tool calls into the history before the final assistant reply.
  await recordToolCalls(context, conversationId, result.toolResults)

  const reply = result.text.trim() || '（没有回复）'
  await persistHistory(context, conversationId, 'assistant', reply)

  return { reply, conversationId, now: nowIso() }
}

/** Run `chatCompletion`, retrying once WITHOUT images when the gateway rejects them. */
async function chatCompletionWithVisionFallback(
  options: Parameters<typeof chatCompletion>[0],
  hasImages: boolean,
): Promise<Awaited<ReturnType<typeof chatCompletion>>> {
  try {
    return await chatCompletion(options)
  } catch (error) {
    if (hasImages && isVisionUnsupportedError(error)) {
      return await chatCompletion({ ...options, messages: degradeVisionMessages(options.messages) })
    }
    throw error
  }
}

/**
 * Producer/consumer bridge between `streamChatCompletion`'s `onDelta` callback
 * (a plain function — it cannot `yield`) and the async-generator of SSE
 * frames. `push` never blocks the producer; `next` resolves as soon as a delta
 * is available or the producer closes.
 */
class DeltaBuffer {
  private queue: string[] = []
  private resolver: (() => void) | null = null
  private closed = false

  push(text: string): void {
    this.queue.push(text)
    this.signal()
  }

  close(): void {
    this.closed = true
    this.signal()
  }

  private signal(): void {
    if (this.resolver) {
      const resolver = this.resolver
      this.resolver = null
      resolver()
    }
  }

  async next(): Promise<{ done: boolean; value?: string }> {
    while (this.queue.length === 0) {
      if (this.closed) return { done: true }
      await new Promise<void>((resolve) => {
        this.resolver = resolve
      })
    }
    return { done: false, value: this.queue.shift() }
  }
}

/**
 * Async-generator of SSE frames for the streaming path. Shared setup with the
 * blocking `runChat`, then either live text deltas (`streamed:true`) or —
 * after a `StreamToolCallsError` — one complete reply (`streamed:false`)
 * produced by the non-streaming tool loop. The full assistant reply is ALWAYS
 * persisted so history stays complete regardless of which branch ran.
 *
 * Vision degradation applies here too: when the gateway rejects the image
 * request, the whole turn is retried once with images stripped and the system
 * note appended before any event is emitted.
 */
async function* chatStreamGenerator(
  context: MakersContext,
  options: ChatTurnOptions,
): AsyncGenerator<string> {
  const { message, images: imagesOption = [], conversationId: conversationIdOption, signal } = options
  const images = validateImages(imagesOption.filter(isNonEmptyString))
  const hasImages = images.length > 0
  const conversationId = conversationIdOption?.trim() || SELF_ID
  if (!message.trim() && !hasImages) throw new Error('message is required and must not be empty.')
  if (!context.store) throw new Error('Store is not available in this context.')

  await persistUserMessage(context, conversationId, message, images)

  const memory = await readMemoryWithSeedSafe(context)
  const persona = buildPersona({
    nowText: humanNowText(),
    memoryContent: memory ? clampMemoryForContext(memory) : '',
  })
  const history = await loadMessages(context, conversationId)
  const messages: LlmMessage[] = [
    { role: 'system', content: `${persona}\n\n${SYSTEM_HISTORY_GUIDANCE}` },
    ...history,
  ]
  const tools = buildTools({ context, conversationId, signal })

  for (let attempt = 1; attempt <= MAX_VISION_ATTEMPTS; attempt += 1) {
    try {
      yield* streamChatTurn(context, conversationId, attempt === 1 ? messages : degradeVisionMessages(messages), tools, signal)
      return
    } catch (error) {
      // Only retry when this turn actually carried images AND the gateway says
      // the model cannot see. Anything else propagates to an error_message frame.
      if (attempt >= MAX_VISION_ATTEMPTS || !hasImages || !isVisionUnsupportedError(error)) throw error
    }
  }
}

/** One full attempt of the streaming turn: live deltas or the tool-loop reply + persistence. */
async function* streamChatTurn(
  context: MakersContext,
  conversationId: string,
  messages: LlmMessage[],
  tools: ReturnType<typeof buildTools>,
  signal: AbortSignal | undefined,
): AsyncGenerator<string> {
  let fullText = ''
  try {
    // Drive the gateway stream from a side task and drain deltas into SSE
    // frames as they arrive (no `yield` allowed inside `onDelta`, hence the
    // buffer bridge). Stream errors are captured and rethrown out of the
    // generator so `createSSEResponse` maps them to an `error_message` frame.
    const buffer = new DeltaBuffer()
    let streamError: unknown = null
    const streamTask = (async () => {
      try {
        await streamChatCompletion({
          context,
          conversationId,
          messages,
          tools: tools.definitions,
          signal,
          temperature: CHAT_TEMPERATURE,
          maxTokens: CHAT_MAX_TOKENS,
          onDelta: (delta) => buffer.push(delta),
        })
      } catch (error) {
        streamError = error
      } finally {
        buffer.close()
      }
    })()

    for (;;) {
      const { done, value } = await buffer.next()
      if (done) break
      fullText += value
      yield sseEvent({ type: 'ai_response', content: value, streamed: true })
    }
    await streamTask
    if (streamError) throw streamError
  } catch (error) {
    if (!(error instanceof StreamToolCallsError)) throw error
    // The model wants to ACT: abort the stream and run the full bounded tool
    // loop non-streaming, then deliver the final reply in one event.
    const result = await chatCompletion({
      context,
      conversationId,
      messages,
      tools: tools.definitions,
      toolRunner: tools.run,
      maxTurns: CHAT_MAX_TURNS,
      signal,
      temperature: CHAT_TEMPERATURE,
      maxTokens: CHAT_MAX_TOKENS,
    })
    await recordToolCalls(context, conversationId, result.toolResults)
    const reply = result.text.trim() || '（没有回复）'
    await persistHistory(context, conversationId, 'assistant', reply)
    yield sseEvent({ type: 'ai_response', content: reply, streamed: false })
    yield sseDone()
    return
  }

  const reply = fullText.trim() || '（没有回复）'
  await persistHistory(context, conversationId, 'assistant', reply)
  yield sseDone()
}

/** Best-effort MEMORY.md seed + read: Blob unavailable degrades to ''. */
async function readMemoryWithSeedSafe(context: MakersContext): Promise<string> {
  try {
    await ensureMemorySeed(context)
    return await readMemoryFile(context)
  } catch {
    return ''
  }
}

/**
 * Persist the incoming chat turn as the history/archive user row. With images
 * the content is serialized to a JSON string (store is string-typed) and
 * tagged `metadata.kind:'image-user'` so `loadMessages` restores the content
 * array on later turns.
 */
async function persistUserMessage(
  context: MakersContext,
  conversationId: string,
  message: string,
  images: string[],
): Promise<void> {
  const content = buildUserContent(message, images)
  const storeContent = typeof content === 'string' ? content : JSON.stringify(content)
  const hasImages = images.length > 0
  await persistHistory(
    context,
    conversationId,
    'user',
    storeContent,
    hasImages ? { metadata: { kind: 'image-user', hasImage: true } } : undefined,
  )
}

/** One OpenAI-ready content value for the incoming user turn. */
function buildUserContent(message: string, images: string[]): LlmContent {
  if (images.length === 0) return message
  const parts: LlmContentPart[] = []
  if (message.trim()) parts.push({ type: 'text', text: message })
  for (const url of images) parts.push({ type: 'image_url', image_url: { url } })
  return parts
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0
}

export async function onRequest(context: any): Promise<Response> {
  const ctx = asMakersContext(context)
  const denied = requireAuth(ctx)
  if (denied) return denied
  const signal = ctx.request?.signal
  try {
    const message = messageValue(ctx)
    const images = validateImages(imagesValue(ctx))
    if (!message.trim() && images.length === 0) throw new Error('message is required and must not be empty.')
    const conversationId = resolveConversationId(ctx, SELF_ID)
    if (wantsJson(ctx)) {
      const result = await runChat(ctx, { message, images, conversationId, signal })
      return jsonOk(result)
    }
    return createSSEResponse(
      (sseSignal) => chatStreamGenerator(ctx, { message, images, conversationId, signal: sseSignal }),
      signal,
    )
  } catch (error) {
    return errorResponse(error)
  }
}

function messageValue(context: MakersContext): string {
  const body = context.request?.body
  if (!body || typeof body !== 'object') return ''
  const value = (body as Record<string, unknown>).message
  return typeof value === 'string' ? value : ''
}

function imagesValue(context: MakersContext): string[] {
  const body = context.request?.body
  if (!body || typeof body !== 'object') return []
  const value = (body as Record<string, unknown>).images
  if (!Array.isArray(value)) return []
  return value.filter(isNonEmptyString)
}

/**
 * Validate the `images` payload before it reaches the store/gateway (P2-2):
 * each entry must be a local `data:image/...;base64,...` URL (remote http/https
 * links are rejected — SSRF / cost transfer), at most MAX_IMAGES_PER_REQUEST,
 * and each decoded to ≤ MAX_IMAGE_BYTES. Throws a clear, user-facing error on
 * any violation. The same check runs in `runChat`, the stream generator and
 * the endpoint so no entry point can bypass it.
 */
function validateImages(images: string[]): string[] {
  const valid: string[] = []
  for (const url of images) {
    if (!IMAGE_DATA_URL_RE.test(url)) {
      throw new Error('图片必须是 data:image/...;base64 数据（不支持远程 URL）')
    }
    const b64 = url.slice(url.indexOf(',') + 1)
    if (base64ByteLength(b64) > MAX_IMAGE_BYTES) {
      throw new Error('单张图片不能超过 4MB')
    }
    valid.push(url)
  }
  if (valid.length > MAX_IMAGES_PER_REQUEST) {
    throw new Error(`一次最多发送 ${MAX_IMAGES_PER_REQUEST} 张图片（收到 ${valid.length} 张）`)
  }
  return valid
}

/** Approximate decoded byte length of a base64 payload (padding-aware). */
function base64ByteLength(b64: string): number {
  const padding = b64.endsWith('==') ? 2 : b64.endsWith('=') ? 1 : 0
  return Math.max(0, Math.floor((b64.length * 3) / 4) - padding)
}

/**
 * JSON mode opt-out: `?stream=false`/`?stream=0` (query) or
 * `{ "stream": false }` (body) returns the original one-shot JSON envelope.
 * Everything else streams SSE.
 */
function wantsJson(context: MakersContext): boolean {
  const url = context.request?.url ?? ''
  const flag = new URL(url, 'http://local').searchParams.get('stream')
  if (flag === 'false' || flag === '0') return true
  const body = context.request?.body
  if (body && typeof body === 'object' && !Array.isArray(body)) {
    if ((body as Record<string, unknown>).stream === false) return true
  }
  return false
}