/**
 * GET /history — full conversation archive reader.
 *
 * Reads the append-only Blob chatlog archive (`chatlog/YYYY-MM-DD.md`), which
 * compact NEVER touches: even after the context store folds the oldest 20% into
 * a summary message, the complete original history stays retrievable here.
 *
 * The store is deliberately NOT read — the archive is the source of truth for
 * history, so a missing store degrades to still serving whatever archive exists.
 *
 * Query params:
 *   conversation_id  conversation id (default SELF_ID=eo-self)
 *   days             how many recent calendar days to read (default 30, 1-90)
 *   keyword          optional case-insensitive search over the archive;
 *                    the keyword path returns matching archive LINE SNIPPETS
 *                    (not full parsed messages) and honors `limit` like the
 *                    plain read path
 *   include          "all" to include heartbeat triggers + compact summaries
 *                    (default hides kind=heartbeat / kind=summary)
 *   limit            max messages to return (default 200, 1-1000)
 *
 * Returns JSON: `{ ok: true, messages: [{ role, content, kind, ts }], ... }`.
 */
import {
  SELF_ID,
  asMakersContext,
  errorResponse,
  jsonOk,
  requireAuth,
  resolveConversationId,
  type MakersContext,
} from './_shared.ts'
import {
  loadChatlog,
  parseChatlogLine,
  searchChatlog,
  type ChatlogMessage,
  type ChatlogSearchHit,
} from './_memory.ts'

const HISTORY_DEFAULT_DAYS = 30
const HISTORY_MAX_DAYS = 90
const HISTORY_DEFAULT_LIMIT = 200
const HISTORY_MAX_LIMIT = 1_000

export async function onRequest(context: any): Promise<Response> {
  const ctx = asMakersContext(context)
  const denied = requireAuth(ctx)
  if (denied) return denied
  try {
    const conversationId = queryParam(ctx, 'conversation_id') || resolveConversationId(ctx, SELF_ID)
    const days = clampInt(queryParam(ctx, 'days'), HISTORY_DEFAULT_DAYS, 1, HISTORY_MAX_DAYS)
    const limit = clampInt(queryParam(ctx, 'limit'), HISTORY_DEFAULT_LIMIT, 1, HISTORY_MAX_LIMIT)
    const keyword = queryParam(ctx, 'keyword').trim()
    const includeAll = queryParam(ctx, 'include').trim() === 'all'

    let messages: ChatlogMessage[]
    if (keyword) {
      // Keyword search covers the WHOLE archive (including heartbeat/summary);
      // the user asked for a search, not a filtered reading. Each hit is a
      // matching archive LINE (snippet semantics), and `limit` applies to the
      // flattened snippet list exactly like the plain read path.
      const hits = await searchChatlog(ctx, conversationId, keyword, days)
      messages = flattenSearchHits(hits).slice(-limit)
    } else {
      // Full-history read: hide system-generated heartbeat wake prompts and
      // compact summaries by default (they are structural, not conversation);
      // real user messages, assistant replies and tool calls are kept. Pass
      // ?include=all to see everything, then clamp to the requested limit.
      const loaded = await loadChatlog(ctx, conversationId, days)
      messages = includeAll
        ? loaded
        : loaded.filter((message) => message.kind !== 'heartbeat' && message.kind !== 'summary')
      messages = messages.slice(-limit)
    }

    return jsonOk({
      messages,
      conversationId,
      days,
      count: messages.length,
    })
  } catch (error) {
    return errorResponse(error)
  }
}

/**
 * Flatten per-day search hits back into the uniform message shape. Hits are
 * archive LINE SNIPPETS, not full messages: a snippet that parses as a normal
 * chatlog row keeps its real kind/role/ts; one that does not (a continuation
 * line, a `…[truncated]` fragment) is emitted as `kind:'search'` with an empty
 * `ts`, so callers can tell a snippet fragment apart from a real message.
 */
function flattenSearchHits(hits: ChatlogSearchHit[]): ChatlogMessage[] {
  const messages: ChatlogMessage[] = []
  for (const hit of hits) {
    for (const snippet of hit.snippets) {
      messages.push(parseChatlogLine(snippet) ?? { role: 'assistant', content: snippet, kind: 'search', ts: '' })
    }
  }
  return messages
}

function clampInt(raw: string, fallback: number, min: number, max: number): number {
  const parsed = Number.parseInt(raw, 10)
  if (!Number.isFinite(parsed)) return fallback
  return Math.min(Math.max(parsed, min), max)
}

/** Read a query parameter out of `request.url` (GET endpoints have no body). */
function queryParam(context: MakersContext, key: string): string {
  const query = queryString(context)
  if (!query) return ''
  for (const pair of query.split('&')) {
    const eq = pair.indexOf('=')
    if (eq < 0) continue
    const name = pair.slice(0, eq).trim()
    if (name !== key) continue
    const value = pair.slice(eq + 1).trim()
    try {
      return decodeURIComponent(value.replace(/\+/g, ' '))
    } catch {
      return value
    }
  }
  return ''
}

function queryString(context: MakersContext): string {
  const url = context.request?.url
  if (!url) return ''
  const index = url.indexOf('?')
  return index >= 0 ? url.slice(index + 1) : ''
}