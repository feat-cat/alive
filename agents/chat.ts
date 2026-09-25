/**
 * POST /chat — minimal conversation endpoint (reserved for future Matrix).
 *
 * Flow: append user message → build a persona (wall clock + MEMORY.md "self")
 * → feed the model a STANDARD messages array: [system(persona), ...history]
 * where history comes straight from the store via `loadMessages` (including the
 * just-appended user message) → bounded chatCompletion with the full tool
 * registry (blob + diary + chatlog + workspace + search, same as heartbeat) →
 * persist any tool calls as `kind:'tool'` records → append assistant reply.
 *
 * Heartbeat and chat share the fixed SELF_ID conversation, so private thoughts
 * and user conversations live in ONE history stream. Plain JSON, no SSE.
 */
import {
  SELF_ID,
  asMakersContext,
  errorResponse,
  jsonOk,
  nowIso,
  requireAuth,
  resolveConversationId,
  type MakersContext,
} from './_shared.ts'
import { chatCompletion } from './_llm.ts'
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

export type ChatResult = {
  reply: string
  conversationId: string
  now: string
}

export async function runChat(
  context: MakersContext,
  options: { message: string; conversationId?: string; signal?: AbortSignal },
): Promise<ChatResult> {
  const conversationId = options.conversationId?.trim() || SELF_ID
  if (!options.message.trim()) throw new Error('message is required and must not be empty.')

  const store = context.store
  if (!store) throw new Error('Store is not available in this context.')

  await persistHistory(context, conversationId, 'user', options.message)

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
  const tools = buildTools({ context, conversationId, signal: options.signal })
  const result = await chatCompletion({
    context,
    conversationId,
    messages: [
      { role: 'system', content: `${persona}\n\n${SYSTEM_HISTORY_GUIDANCE}` },
      ...history,
    ],
    tools: tools.definitions,
    toolRunner: tools.run,
    maxTurns: CHAT_MAX_TURNS,
    signal: options.signal,
    temperature: 0.7,
    maxTokens: 600,
  })

  // Persist tool calls into the history before the final assistant reply.
  await recordToolCalls(context, conversationId, result.toolResults)

  const reply = result.text.trim() || '（没有回复）'
  await persistHistory(context, conversationId, 'assistant', reply)

  return { reply, conversationId, now: nowIso() }
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

export async function onRequest(context: any): Promise<Response> {
  const ctx = asMakersContext(context)
  const denied = requireAuth(ctx)
  if (denied) return denied
  const signal = ctx.request?.signal
  try {
    const message = messageValue(ctx)
    const conversationId = resolveConversationId(ctx, SELF_ID)
    const result = await runChat(ctx, { message, conversationId, signal })
    return jsonOk(result)
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
