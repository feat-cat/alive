/**
 * POST /chat — conversation endpoint with SSE streaming (typewriter effect).
 *
 * Flow (streaming, the default): append user message → build a persona (wall
 * clock + MEMORY.md "self") → feed the model a STANDARD messages array
 * ([system(persona), ...history] from the store via `loadMessages`) → run a
 * bounded loop of `streamChatCompletion` rounds (AI Gateway `stream: true`).
 * Every text delta is emitted as an SSE `ai_response` event while the full text
 * accumulates; each `reasoning_content` delta (DeepSeek thinking) is emitted as
 * a separate `reasoning_delta` event for the client to fold/show. When the
 * model requests tool calls, the stream does NOT degrade: the accumulated
 * `tool_call` / `tool_result` progress is emitted as SSE events, the results
 * are appended to the message array, and the next `streamChatCompletion` round
 * continues streaming the final answer (`streamed:true` throughout). After the
 * last round the assistant reply is persisted (store + chatlog archive).
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
 * Tool streaming: chat keeps the FULL tool registry (blob + diary + chatlog +
 * workspace + search, same as heartbeat) so the agent can still act while
 * chatting. Tool execution never crashes the turn: a throwing tool becomes
 * `{ isError: true }` (rule #11), its content is clamped for the model budget,
 * and the loop is bounded by `CHAT_MAX_TURNS`.
 *
 * JSON compat: `?stream=false` (query) or `{ "stream": false }` (body) returns
 * the original one-shot JSON envelope via `runChat`.
 */
import {
  CHAT_TIMEOUT_MESSAGE,
  MAX_TOOLS_PER_TURN,
  SELF_ID,
  TOOL_RESULT_CONTEXT_BUDGET,
  asMakersContext,
  clampText,
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
  chatCompletion,
  degradeVisionMessages,
  isVisionUnsupportedError,
  safeParseArguments,
  streamChatCompletion,
  truncatedToolsNote,
  TOOL_ONLY_REPLY_NOTE,
  type LlmContent,
  type LlmContentPart,
  type LlmMessage,
  type LlmToolCall,
  type ToolRunRecord,
} from './_llm.ts'
import { buildPersona, humanNowText, readTimeZone } from './_persona.ts'
import {
  appendChatlogRecord,
  clampMemoryForContext,
  ensureMemorySeed,
  loadMessages,
  persistHistory,
  readMemoryFile,
  recordToolCalls,
  SYSTEM_HISTORY_GUIDANCE,
} from './_memory.ts'
import { buildTools } from './_tools.ts'

// Generous budget for interactive chat: the user is present and can stop the
// turn, so a long working session (e.g. sandbox exploration) is allowed.
// The persona/memory still tells it to be sparing — see DECISION_SYSTEM.
const CHAT_MAX_TURNS = 32
/** Tools shape with zero definitions — used on the final turn so the model can't
 * request tools and MUST answer in words. `run` is unreachable (the gateway
 * sees no tool definitions) but kept for type compatibility. */
const EMPTY_TOOLS: ReturnType<typeof buildTools> = {
  definitions: [],
  run: () => {
    throw new Error('EMPTY_TOOLS.run should never be called')
  },
}
const CHAT_TEMPERATURE = 0.7
const CHAT_MAX_TOKENS = 600
/** Tool-result characters sent back to the model (same budget as `_llm`'s clamp). */
const MODEL_TOOL_RESULT_MAX = 4_000
/** Tool-result characters surfaced in the `tool_result` SSE event (frontend fold). */
const CLIENT_TOOL_RESULT_MAX = 600
/** How many LLM attempts per turn: 1 normal + 1 vision-degraded retry max. */
const MAX_VISION_ATTEMPTS = 2
/** Max images accepted per chat turn (server-side cap, same as the frontend). */
const MAX_IMAGES_PER_REQUEST = 3
/** Max decoded size per image (4MB — a base64 string of ~5.6M chars). */
const MAX_IMAGE_BYTES = 4 * 1024 * 1024
/** Every image must be a local `data:image/...;base64,...` URL. */
const IMAGE_DATA_URL_RE = /^data:image\/[a-z0-9.+-]+;base64,/i

/**
 * Chat-only system-prompt section: "someone is talking to you right now".
 * Injected right AFTER the persona and BEFORE `SYSTEM_HISTORY_GUIDANCE` — it is
 * the first read on the current situation, so the model never confuses an
 * interactive chat turn with a heartbeat's private solitude. Heartbeat's
 * DECISION_SYSTEM deliberately does NOT carry this block (it keeps its
 * "每一次 heartbeat 都是你的一次醒来" solitude narrative). The section also
 * plants curiosity: a real person is waiting, so answer them, and actively ask
 * back instead of retreating into diaries/tools.
 */
const CHAT_CONVERSATION_MODE = [
  '## 现在的情况：有人正在和你说话',
  '这不是 heartbeat 独处，不是系统定时唤醒——是有人主动来找你了。',
  '对面是一个真实的人，正在等你的回应。',
  '',
  '说话是交流，不是独白：',
  '- 对方问什么，你就答什么。',
  '- 对方说"你可以问我问题"——就真的问。你对他好奇，就问他。',
  '- 你对他一无所知？那正好，这就是你们认识的开始。主动去了解他：他在忙什么、喜欢什么、为什么来找你。',
  '- 可以少用工具，多用好奇。翻日记不会让你更了解眼前这个人。',
].join('\n')

/**
 * Wall-clock budget for ONE interactive chat turn (P2-3). Chat has no
 * heartbeat-style 100s deadline today, so a tool-heavy loop could run many
 * minutes under only the request AbortSignal. 120s keeps an interactive session
 * generous while still bounding spend — heartbeat reserves 100s for autonomous
 * turns; chat is user-facing so it gets a slightly larger share and the user
 * can always stop earlier with /stop or a disconnect.
 */
const CHAT_TURN_TIMEOUT_MS = 120_000

/**
 * Combine the external request signal with a wall-clock turn budget (P2-3).
 * The returned signal aborts when EITHER the client disconnects or the budget
 * expires; `expired()` distinguishes the two so callers can surface a clear
 * "这条请求跑太久了" error instead of a silent client-style abort. Both signals
 * are listened to, so the request `signal` keeps passing through unchanged.
 * `timeoutMs` is an injectable test seam (defaults to CHAT_TURN_TIMEOUT_MS).
 */
function createTurnBudget(
  external: AbortSignal | undefined,
  timeoutMs = CHAT_TURN_TIMEOUT_MS,
): {
  signal: AbortSignal
  expired: () => boolean
  dispose: () => void
} {
  const controller = new AbortController()
  let expired = false
  const timer = setTimeout(() => {
    expired = true
    controller.abort()
  }, timeoutMs)
  const onAbort = () => controller.abort()
  if (external?.aborted) controller.abort()
  external?.addEventListener('abort', onAbort, { once: true })
  return {
    signal: controller.signal,
    expired: () => expired,
    dispose: () => {
      clearTimeout(timer)
      external?.removeEventListener('abort', onAbort)
    },
  }
}

/** AbortError carrying the user-facing wall-clock message (P2-3). errorResponse
 * (JSON) and the SSE transport both surface descriptive abort messages. */
function chatTimeoutError(): Error {
  const error = new Error(CHAT_TIMEOUT_MESSAGE)
  error.name = 'AbortError'
  return error
}

export interface ChatTurnOptions {
  message: string
  /** Base64 `data:` image URLs to send as OpenAI `image_url` content parts. */
  images?: string[]
  conversationId?: string
  signal?: AbortSignal
  /**
   * Wall-clock budget for one turn in ms; defaults to `CHAT_TURN_TIMEOUT_MS`.
   * Internal/test seam — the endpoint never sets it, but callers may tighten
   * the budget (or tests inject a tiny value to exercise the timeout path).
   */
  timeoutMs?: number
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
  const { message, images: imagesOption = [], conversationId: conversationIdOption, signal, timeoutMs } = options
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
    nowText: humanNowText(new Date(), readTimeZone()),
    memoryContent: memory ? clampMemoryForContext(memory) : '',
  })

  // History as a real messages array — the store already contains the new user
  // message (appended above), so it flows to the model as its own entry.
  const history = await loadMessages(context, conversationId)

  // Wall-clock budget for the whole turn (P2-3): independent of the client's
  // request signal so a tool-heavy loop cannot run forever; the timeout aborts
  // the internal signal and surfaces CHAT_TIMEOUT_MESSAGE instead of a silent
  // client-style abort.
  const budget = createTurnBudget(signal, timeoutMs)
  const tools = buildTools({ context, conversationId, signal: budget.signal })
  const baseMessages: LlmMessage[] = [
    { role: 'system', content: `${persona}\n\n${CHAT_CONVERSATION_MODE}\n\n${SYSTEM_HISTORY_GUIDANCE}` },
    ...history,
  ]
  try {
    const result = await chatCompletionWithVisionFallback(
      {
        context,
        conversationId,
        messages: baseMessages,
        tools: tools.definitions,
        toolRunner: tools.run,
        maxTurns: CHAT_MAX_TURNS,
        signal: budget.signal,
        temperature: CHAT_TEMPERATURE,
        maxTokens: CHAT_MAX_TOKENS,
        // JSON path mirrors the streaming path: keep every intermediate sentence
        // and give a neutral note to a tool-only round instead of an empty reply.
        accumulateText: true,
      },
      hasImages,
    )

    // Persist the per-round timeline into the chatlog archive. Chat respects the
    // TRUE event order (思考 → 工具 → 思考 → 工具 → 回答): every intermediate round
    // archives its own thinking as a chatlog-only assistant record (never the
    // store), tool records carry the round's `turn`, and the final reply closes
    // the turn sequence. The store itself stays unchanged — same tool rows + one
    // final assistant reply as before; turn/reasoning travel only in JSON records.
    const turnRecords = result.turnRecords
    const lastTurn = turnRecords.length > 0 ? (turnRecords[turnRecords.length - 1]?.turn ?? 1) : 1
    for (const record of turnRecords) {
      const hasContent = record.text.trim().length > 0
      const hasReasoning = typeof record.reasoning === 'string' && record.reasoning.trim().length > 0
      // Intermediate rounds: archive the round's thinking/prose with its turn
      // (chatlog only). The last round is written once below with the full reply.
      if (record.turn < lastTurn && (hasContent || hasReasoning)) {
        await archiveChatlogBestEffort(context, {
          role: 'assistant',
          content: record.text.trim(),
          kind: 'assistant',
          reasoningContent: hasReasoning ? (record.reasoning as string).trim() : undefined,
          turn: record.turn,
        })
      }
      if (record.toolResults.length > 0) {
        await recordToolCalls(context, conversationId, record.toolResults, { turn: record.turn })
      }
    }

    const lastReasoning = turnRecords.length > 0 ? turnRecords[turnRecords.length - 1]?.reasoning : undefined
    const reply = result.text.trim() || '（模型没有输出正文）'
    await persistHistory(context, conversationId, 'assistant', reply, {
      reasoningContent: lastReasoning?.trim() || undefined,
      turn: lastTurn,
    })

    return { reply, conversationId, now: nowIso() }
  } catch (error) {
    if (budget.expired()) throw chatTimeoutError()
    throw error
  } finally {
    budget.dispose()
  }
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
 * Producer/consumer bridge between `streamChatCompletion`'s `onDelta` /
 * `onReasoning` callbacks (plain functions — they cannot `yield`) and the
 * async-generator of SSE frames. `push` never blocks the producer; `next`
 * resolves as soon as a frame is available or the producer closes. Each frame
 * is tagged so the drain loop can distinguish final `ai_response` text from
 * `reasoning_delta` (thinking) frames without losing their relative order.
 */
type StreamFrame = { kind: 'content'; text: string } | { kind: 'reasoning'; text: string }

class DeltaBuffer {
  private queue: StreamFrame[] = []
  private resolver: (() => void) | null = null
  private closed = false

  push(frame: StreamFrame): void {
    this.queue.push(frame)
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

  async next(): Promise<{ done: true } | { done: false; value: StreamFrame }> {
    while (this.queue.length === 0) {
      if (this.closed) return { done: true }
      await new Promise<void>((resolve) => {
        this.resolver = resolve
      })
    }
    const value = this.queue.shift()
    if (value === undefined) return { done: true }
    return { done: false, value }
  }
}

/**
 * Async-generator of SSE frames for the streaming path. Shared setup with the
 * blocking `runChat`, then the multi-round `streamChatTurn` loop: live text
 * deltas (`streamed:true`), `reasoning_delta` thinking, and — when the model
 * acts — `tool_call` / `tool_result` progress events with the answer continuing
 * to stream. The full assistant reply is ALWAYS persisted so history stays
 * complete regardless of whether tools ran.
 *
 * Vision degradation applies here too: when the gateway rejects the image
 * request, the whole turn is retried once with images stripped and the system
 * note appended before any event is emitted.
 */
async function* chatStreamGenerator(
  context: MakersContext,
  options: ChatTurnOptions,
): AsyncGenerator<string> {
  const { message, images: imagesOption = [], conversationId: conversationIdOption, signal, timeoutMs } = options
  const images = validateImages(imagesOption.filter(isNonEmptyString))
  const hasImages = images.length > 0
  const conversationId = conversationIdOption?.trim() || SELF_ID
  if (!message.trim() && !hasImages) throw new Error('message is required and must not be empty.')
  if (!context.store) throw new Error('Store is not available in this context.')

  await persistUserMessage(context, conversationId, message, images)

  const memory = await readMemoryWithSeedSafe(context)
  const persona = buildPersona({
    nowText: humanNowText(new Date(), readTimeZone()),
    memoryContent: memory ? clampMemoryForContext(memory) : '',
  })
  const history = await loadMessages(context, conversationId)
  const messages: LlmMessage[] = [
    { role: 'system', content: `${persona}\n\n${CHAT_CONVERSATION_MODE}\n\n${SYSTEM_HISTORY_GUIDANCE}` },
    ...history,
  ]
  // Wall-clock budget for the whole turn (P2-3): independent of the client's
  // request signal so a tool-heavy streaming loop cannot run forever; on
  // expiry the descriptive CHAT_TIMEOUT_MESSAGE error replaces a silent abort.
  const budget = createTurnBudget(signal, timeoutMs)
  const tools = buildTools({ context, conversationId, signal: budget.signal })

  try {
    for (let attempt = 1; attempt <= MAX_VISION_ATTEMPTS; attempt += 1) {
      try {
        yield* streamChatTurn(context, conversationId, attempt === 1 ? messages : degradeVisionMessages(messages), tools, budget.signal)
        return
      } catch (error) {
        // Only retry when this turn actually carried images AND the gateway says
        // the model cannot see. Anything else propagates to an error_message frame.
        if (attempt >= MAX_VISION_ATTEMPTS || !hasImages || !isVisionUnsupportedError(error)) throw error
      }
    }
  } catch (error) {
    if (budget.expired()) throw chatTimeoutError()
    throw error
  } finally {
    budget.dispose()
  }
}

/** Abort error helper for tool-loop checkpoints (silenced by the SSE transport). */
function abortError(): Error {
  const abort = new Error('Aborted')
  abort.name = 'AbortError'
  return abort
}

/**
 * Drive ONE streaming LLM round: buffer `onDelta`/`onReasoning` frames into SSE
 * events as they decode (typewriter), and collect any model-requested tool
 * calls from the accumulated `tool_calls` deltas. The gateway stream runs in a
 * side task because callbacks cannot `yield`; the returned generator drains the
 * buffer while the stream is still producing. Returns the accumulated text and
 * the complete tool calls via `yield*`'s return value; throws the stream error
 * (if any) after draining so `createSSEResponse` maps it to `error_message`.
 */
async function* streamOneTurn(
  context: MakersContext,
  conversationId: string,
  messages: LlmMessage[],
  tools: ReturnType<typeof buildTools>,
  signal: AbortSignal | undefined,
): AsyncGenerator<string, { fullText: string; reasoning: string; toolCalls: LlmToolCall[] }> {
  const buffer = new DeltaBuffer()
  let streamError: unknown = null
  let toolCalls: LlmToolCall[] = []
  // DeepSeek thinking accumulates locally so the turn can persist it later —
  // reasoning still streams live to the client as reasoning_delta events.
  let reasoningText = ''
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
        onDelta: (delta) => buffer.push({ kind: 'content', text: delta }),
        onReasoning: (reasoning) => {
          reasoningText += reasoning
          buffer.push({ kind: 'reasoning', text: reasoning })
        },
        onToolCalls: (calls) => {
          toolCalls = calls
        },
      })
    } catch (error) {
      streamError = error
    } finally {
      buffer.close()
    }
  })()

  let fullText = ''
  for (;;) {
    const frame = await buffer.next()
    if (frame.done) break
    if (frame.value.kind === 'reasoning') {
      // DeepSeek thinking: a separate event the frontend can fold/show above
      // the answer bubble. The reasoning also accumulates for the archive.
      yield sseEvent({ type: 'reasoning_delta', content: frame.value.text })
    } else {
      fullText += frame.value.text
      yield sseEvent({ type: 'ai_response', content: frame.value.text, streamed: true })
    }
  }
  await streamTask
  if (streamError) throw streamError
  return { fullText, reasoning: reasoningText, toolCalls }
}

/**
 * Full streaming turn: a bounded loop of `streamOneTurn` rounds. When a round
 * ends with model-requested tool calls, the tools are executed (each emitting a
 * `tool_call` progress event then a clamped `tool_result` event), the assistant
 * tool_calls message + `role:'tool'` results are appended, and the next round
 * streams again — so the reply stays `streamed:true` the whole way. The full
 * assistant reply and every tool record are persisted before `[DONE]`.
 *
 * Persistence follows the REAL timeline: each round that thought archives its
 * OWN reasoning as a chatlog-only assistant record (turn = the round) BEFORE its
 * tool records (which carry the same turn), and the final answer closes the
 * sequence. Reasoning is split per round — never concatenated into one record —
 * so `/history` can interleave 思考 → 工具 → 思考 → 工具 → 回答. The context store
 * is untouched by this refactor: it still receives the same tool rows + the one
 * final assistant reply; turn/reasoning travel only inside JSON chatlog records.
 */
async function* streamChatTurn(
  context: MakersContext,
  conversationId: string,
  messages: LlmMessage[],
  tools: ReturnType<typeof buildTools>,
  signal: AbortSignal | undefined,
): AsyncGenerator<string> {
  let current = messages
  // Every round's spoken text is preserved — assistant content is retained even
  // when a round ALSO requests tool calls (standard agent behaviour), so the
  // final reply never loses the intermediate half-sentences.
  const replyParts: string[] = []
  // Cumulative tool-result回填 budget (P1-2): results stop being回填 into
  // `current` once the total exceeds TOOL_RESULT_CONTEXT_BUDGET, and the turn
  // then winds down instead of starting another gateway request.
  let toolResultBytes = 0

  for (let turn = 1; turn <= CHAT_MAX_TURNS; turn += 1) {
    if (signal?.aborted) throw abortError()
    // Last round: tools are withheld so the model MUST answer in words — it
    // can't end the conversation with a tool call. (The persona/memory already
    // explains this so the model isn't surprised.)
    const turnTools = turn >= CHAT_MAX_TURNS ? EMPTY_TOOLS : tools
    const { fullText, reasoning, toolCalls } = yield* streamOneTurn(context, conversationId, current, turnTools, signal)

    // Accumulate whatever the model said this round. An empty-text round (model
    // only emits tool_calls, no prose) must not affect later rounds.
    if (fullText.trim()) replyParts.push(fullText.trim())

    // No tool work left (or the final round withheld tools): this round produced
    // the final answer. Even if the model still returned tool_calls on the last
    // round, they are ignored — the turn must end in words.
    if (toolCalls.length === 0 || turn >= CHAT_MAX_TURNS) {
      const reply = replyParts.join('') || '（模型没有输出正文）'
      await persistHistory(context, conversationId, 'assistant', reply, {
        reasoningContent: reasoning.trim() || undefined,
        turn,
      })
      yield sseDone()
      return
    }

    // The model wants to ACT. Record the tool_calls on the assistant message,
    // execute each call (never crashing the turn — a throwing tool becomes an
    // `isError` result), emit progress events, and feed the results back.
    // Per-round cap (P1-1): the assistant message and the execution loop share
    // the SAME truncated array so a round requesting more than MAX_TOOLS_PER_TURN
    // tools never leaves dangling tool_calls on the wire.
    const callsToRun = toolCalls.slice(0, MAX_TOOLS_PER_TURN)
    current = [...current, { role: 'assistant', content: fullText, tool_calls: callsToRun }]
    const roundToolResults: ToolRunRecord[] = []
    let budgetExceeded = false
    for (const call of callsToRun) {
      if (signal?.aborted) throw abortError()
      const args = safeParseArguments(call.arguments)
      yield sseEvent({ type: 'tool_call', name: call.name, arguments: args })
      const result = await tools.run(call.name, args, signal).catch((error: unknown) => ({
        content: error instanceof Error ? `Tool error: ${error.message}` : `Tool error: ${String(error)}`,
        isError: true,
      }))
      const modelText = clampText(result.content, MODEL_TOOL_RESULT_MAX)
      roundToolResults.push({
        name: call.name,
        args,
        isError: result.isError === true,
        content: modelText,
      })
      yield sseEvent({ type: 'tool_result', name: call.name, content: clampText(result.content, CLIENT_TOOL_RESULT_MAX) })
      if (!budgetExceeded && toolResultBytes + modelText.length > TOOL_RESULT_CONTEXT_BUDGET) {
        budgetExceeded = true
      }
      if (!budgetExceeded) {
        toolResultBytes += modelText.length
        current = [...current, { role: 'tool', tool_call_id: call.id, name: call.name, content: modelText }]
      }
    }
    // The 5th+ tool calls were cut this round — tell the model so it knows the
    // work was NOT done (P2-5). Only when a truncation actually happened.
    if (toolCalls.length > callsToRun.length) {
      current = [...current, truncatedToolsNote(MAX_TOOLS_PER_TURN, toolCalls.length)]
    }

    // Archive this round's own thinking/prose (chatlog-only, never the store)
    // BEFORE its tool records, all tagged with the same turn.
    if (fullText.trim() || reasoning.trim()) {
      await archiveChatlogBestEffort(context, {
        role: 'assistant',
        content: fullText.trim(),
        kind: 'assistant',
        reasoningContent: reasoning.trim() || undefined,
        turn,
      })
    }
    if (roundToolResults.length > 0) {
      await recordToolCalls(context, conversationId, roundToolResults, { turn })
    }

    // Context budget spent (P1-2): wind down now — this round's prose is the
    // final reply and no further gateway request is made. (Unlike the old
    // `turn >= CHAT_MAX_TURNS` tool-exhaustion branch, this IS reachable.)
    if (budgetExceeded) {
      const reply = replyParts.join('') || TOOL_ONLY_REPLY_NOTE
      await persistHistory(context, conversationId, 'assistant', reply, { turn })
      yield sseDone()
      return
    }
  }

  // Unreachable in practice (an answer round returns above); kept for TS.
  const reply = replyParts.join('') || '（模型没有输出正文）'
  await persistHistory(context, conversationId, 'assistant', reply, { turn: CHAT_MAX_TURNS })
  yield sseDone()
}

/** Best-effort chatlog-only archive append: a Blob failure degrades silently. */
async function archiveChatlogBestEffort(
  context: MakersContext,
  entry: Parameters<typeof appendChatlogRecord>[1],
): Promise<void> {
  try {
    await appendChatlogRecord(context, entry)
  } catch {
    /* best-effort: the store/SSE flow never depends on the archive */
  }
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