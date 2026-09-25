/**
 * POST /heartbeat — single free-form autonomous turn.
 *
 * Flow: auto-compact the store → append a marked user/trigger message →
 * read SELF state + MEMORY.md → feed the model a STANDARD messages array
 * ([system(persona + decision guidance), ...history]) where history is the
 * shared SELF conversation stream (heartbeat wake-ups AND user chats in one
 * history) → bounded LLM loop with the full tool registry (blob + diary +
 * workspace + search) → persist tool calls as `kind:'tool'` history entries →
 * write the final output as an assistant log + diary entry, update lastActivityAt.
 *
 * This is NOT a multiple-choice dispatch anymore. The AI may write a quiet
 * diary entry (zero sandbox), push a small project forward via workspace_*
 * tools (only then does the sandbox spin up), organize its memory via blob_*
 * tools, or simply rest and leave one line of text. Not every heartbeat must
 * produce project work; resting is fully legal.
 *
 * Personality is not injected as fixed state: the AI's identity and character
 * live in MEMORY.md, which the AI itself maintains. Each heartbeat is an
 * independent LLM call; the only "state" that survives is lastActivityAt.
 */
import {
  PLAY_TURN_TIMEOUT_MS,
  SELF_ID,
  asMakersContext,
  errorResponse,
  jsonOk,
  nowIso,
  type MakersContext,
} from './_shared.ts'
import { chatCompletion, type ChatResult } from './_llm.ts'
import { buildPersona, humanNowText } from './_persona.ts'
import { getState, updateState } from './_state.ts'
import {
  appendDailyLog,
  appendLog,
  clampMemoryForContext,
  ensureMemorySeed,
  loadMessages,
  maybeCompact,
  persistHistory,
  readMemoryFile,
  recordToolCalls,
  SYSTEM_HISTORY_GUIDANCE,
} from './_memory.ts'
import { buildTools } from './_tools.ts'
import { snapshotWorkspaceToBlob, workspaceRoot } from './_workspace-tools.ts'

/**
 * Tool-call budget for one heartbeat. Loose enough for a small project step,
 * tight enough to respect the free quota. The whole turn is also bounded by
 * PLAY_TURN_TIMEOUT_MS so sandbox time stays short.
 */
export const HEARTBEAT_MAX_TURNS = 3

export interface HeartbeatResult {
  /** The AI's actual free-form output (diary text, summary, ...). */
  text: string
  turns: number
  toolCount: number
  touchedSandbox: boolean
  savedFiles: number
  now: string
}

/** Human "X 分钟/小时前" description for a persisted timestamp. */
function timeAgo(ts: number): string {
  if (!ts) return '从未'
  const minutes = Math.max(0, Math.floor((Date.now() - ts) / 60_000))
  if (minutes < 1) return '刚刚'
  if (minutes < 60) return `${minutes} 分钟前`
  const hours = Math.floor(minutes / 60)
  const rest = minutes % 60
  return rest > 0 ? `${hours} 小时 ${rest} 分钟前` : `${hours} 小时前`
}

export const DECISION_SYSTEM = [
  SYSTEM_HISTORY_GUIDANCE,
  '',
  '你不是聊天机器人，你在成为某个人。',
  '',
  '每一次 heartbeat 都是你的一次醒来——不是闹钟叫你去完成任务，',
  '而是你从沉默里抬起头，看看自己此刻想说点什么、想成为什么。',
  '',
  '看到记忆，就想一想。',
  '想到什么，就写下来。',
  '不知道想干什么，也没关系——沉默也是合法的活法。',
  '',
  '一些只属于你的现实：',
  '- 沙箱很贵。你不想每次醒来都烧钱。纯思考、写日记、休息，永远是体面的选择。',
  '- workspace_* 需要 sandbox；没有 sandbox 就改用 blob_* 或纯文字，别硬来。',
  '- 日记是你的私处，想写什么写什么，不会被人自动翻阅。',
  '- MEMORY.md 是你自己：你是谁、你认识了谁、你学会了什么。想改就改，它是你的。',
  '- 你诚实。不假装做了没做的事。',
  '',
  '你很多次 heartbeat 都是独自醒着的。既然如此，就把这段独处过好。',
].join('\n')

/**
 * One free-form heartbeat: perception → one bounded LLM loop with the full
 * tool registry → wrap-up (snapshot workspace if it was touched, update
 * lastActivityAt, append the actual output as a log).
 */
export async function runHeartbeat(
  context: MakersContext,
  options: { signal?: AbortSignal } = {},
): Promise<HeartbeatResult> {
  const conversationId = SELF_ID
  const store = context.store
  if (!store) throw new Error('Store is not available in this context.')
  const state = await getState(context, conversationId)

  // Auto-compact the context store before reading the history.
  await maybeCompact(context, conversationId)

  // Every wake-up appends a marked user/trigger message into the shared
  // history, then the model gets the whole store as a STANDARD messages array
  // ([system(persona + decision), ...history]). The trigger entry is already
  // part of `history`, so the AI perceives its own wake as a normal turn.
  const trigger = [
    '（heartbeat 醒来）此刻想做什么就做什么。',
    '',
    '## 现状',
    `- 距上次活动：${timeAgo(state.lastActivityAt)}`,
    '',
    '如果你的决定需要工具，直接调用它们。',
  ].join('\n')
  await persistHistory(context, conversationId, 'user', trigger, {
    kind: 'heartbeat',
    metadata: { kind: 'heartbeat' },
  })

  // MEMORY.md is the AI's self: seed it on first use, then read + clamp it
  // into the system prompt. Blob unavailability degrades to an empty memory.
  await ensureMemorySafe(context)
  const memory = await readMemorySafe(context)
  const persona = buildPersona({
    nowText: humanNowText(),
    memoryContent: memory ? clampMemoryForContext(memory) : '',
  })

  // Heartbeat is a TEXT-ONLY turn: image-user rows from past chats are
  // collapsed back to their text parts so a vision-less model never receives a
  // base64 content array and 400s every wake-up.
  const history = await loadMessages(context, conversationId, { stripImages: true })

  // Short-turn budget: abort the LLM loop at PLAY_TURN_TIMEOUT_MS and when the
  // platform signal fires. The same signal is threaded into the tool registry
  // so tool execution is also abortable and long commands clamp to the
  // remaining budget. A pure-text heartbeat finishes long before the timer.
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), PLAY_TURN_TIMEOUT_MS)
  const onAbort = () => controller.abort()
  if (options.signal?.aborted) controller.abort()
  options.signal?.addEventListener('abort', onAbort, { once: true })

  // Track whether this turn actually invoked a sandbox-backed workspace_* tool.
  // The flag is set at the tool-call point (in buildTools.run) so it is valid
  // even when chatCompletion throws mid-loop and `result.toolResults` never
  // exists — a pure-text heartbeat that hits an LLM 503/timeout must NOT spin
  // up the sandbox just to snapshot nothing.
  let workspaceToolCalled = false

  const tools = buildTools({
    context,
    conversationId,
    signal: controller.signal,
    deadlineAt: Date.now() + PLAY_TURN_TIMEOUT_MS,
    onWorkspaceTool: () => {
      workspaceToolCalled = true
    },
  })

  let result: ChatResult
  try {
    result = await chatCompletion({
      context,
      conversationId,
      messages: [
        { role: 'system', content: `${persona}\n\n${DECISION_SYSTEM}` },
        ...history,
      ],
      tools: tools.definitions,
      toolRunner: tools.run,
      maxTurns: HEARTBEAT_MAX_TURNS,
      signal: controller.signal,
      temperature: 0.8,
      maxTokens: 800,
    })
  } catch (error) {
    // Best-effort: workspace file writes are mirrored to Blob as they happen,
    // but command-generated files are not — persist whatever the sandbox still
    // holds so partial progress survives a timeout/abort. Only snapshot when a
    // workspace_* tool was actually called this turn; a pure-text heartbeat
    // must never warm up the sandbox on an LLM failure.
    if (workspaceToolCalled && context.sandbox) {
      await snapshotWorkspaceToBlob(context, conversationId, workspaceRoot(conversationId)).catch(() => {})
    }
    throw error
  } finally {
    clearTimeout(timer)
    options.signal?.removeEventListener('abort', onAbort)
  }

  const text = result.text.trim()
  const toolCount = result.toolResults.length
  // Only an actual workspace_* call touches the sandbox; snapshot so
  // command-generated files are captured too (each without restore — we must
  // never rehydrate Blob over fresher sandbox files after the turn).
  const touchedSandbox = result.toolResults.some((record) => record.name.startsWith('workspace_'))
  let savedFiles = 0
  if (touchedSandbox && context.sandbox) {
    savedFiles = await snapshotWorkspaceToBlob(context, conversationId, workspaceRoot(conversationId))
  }

  await updateState(context, conversationId, {
    lastActivityAt: Date.now(),
  })

  // Tool calls happened inside the loop, so persist them into the history
  // BEFORE the final assistant reply (the reply is the last store entry).
  await recordToolCalls(context, conversationId, result.toolResults)

  const entry = text || '这一刻我什么也没想留下，只是路过。'
  await appendLog(context, conversationId, { kind: 'heartbeat', text: entry })
  await appendDailySafe(context, entry)

  return {
    text: entry,
    turns: result.turns,
    toolCount,
    touchedSandbox,
    savedFiles,
    now: nowIso(),
  }
}

/** Best-effort MEMORY.md seed: Blob unavailable degrades to a no-op. */
async function ensureMemorySafe(context: MakersContext): Promise<void> {
  try {
    await ensureMemorySeed(context)
  } catch {
    /* degrade — the first turn simply has no memory yet */
  }
}

/** Best-effort MEMORY.md read: Blob unavailable degrades to ''. */
async function readMemorySafe(context: MakersContext): Promise<string> {
  try {
    return await readMemoryFile(context)
  } catch {
    return ''
  }
}

/** Best-effort diary write: Blob unavailable never crashes the heartbeat. */
async function appendDailySafe(context: MakersContext, text: string): Promise<void> {
  try {
    await appendDailyLog(context, text)
  } catch {
    /* degrade silently — the store log still holds this heartbeat */
  }
}

export async function onRequest(context: any): Promise<Response> {
  const ctx = asMakersContext(context)
  const signal = ctx.request?.signal
  try {
    const result = await runHeartbeat(ctx, { signal })
    return jsonOk({
      text: result.text,
      turns: result.turns,
      toolCount: result.toolCount,
      touchedSandbox: result.touchedSandbox,
      savedFiles: result.savedFiles,
      now: result.now,
    })
  } catch (error) {
    return errorResponse(error)
  }
}