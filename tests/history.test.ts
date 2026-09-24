/**
 * Tests for the chatlog archive surface:
 *  - GET /history endpoint (agents/history.ts): default filtering of
 *    heartbeat/summary, include=all, keyword search, conversation_id default,
 *    degradation when the store is missing, error mapping.
 *  - chatlog_search / chatlog_read tool registration + behavior (buildTools).
 *
 * The chatlog archive reads from injected Blob only (never the store), so
 * `makeContext({})` (no store) is the normal case here.
 */
import { afterEach, describe, mock, test } from 'node:test'
import assert from 'node:assert/strict'
import { injectBlobStoreForTesting } from '../agents/_blob-tools.ts'
import { onRequest as historyOnRequest } from '../agents/history.ts'
import { SELF_ID, type Env } from '../agents/_shared.ts'
import { CHATLOG_DIR, dateKey, maybeCompact, persistHistory } from '../agents/_memory.ts'
import { buildTools } from '../agents/_tools.ts'
import {
  gatewayEnv,
  makeContext,
  makeMockBlobStore,
  makeMockStore,
  type MockBlobStore,
} from './_helpers.ts'

/** A realistic day's chatlog with all kinds present. */
function chatlogToday(): Record<string, string> {
  const today = dateKey(new Date())
  return {
    [`${CHATLOG_DIR}${today}.md`]: [
      '- [2026-09-11T10:00:00Z] user: 真实用户问题',
      '- [2026-09-11T10:01:00Z] assistant: 真实回复',
      '- [2026-09-11T10:02:00Z] tool: [调用工具 web_search] 参数={"query":"x"}',
      '- [2026-09-11T10:03:00Z] heartbeat: （heartbeat 醒来）此刻想做什么就做什么',
      '- [2026-09-11T10:04:00Z] summary: 旧的记录摘要',
    ].join('\n'),
  }
}

async function historyResponse(blob: MockBlobStore, url: string): Promise<{ status: number; body: any }> {
  injectBlobStoreForTesting(blob)
  const res = await historyOnRequest(makeContext({ url }))
  return { status: res.status, body: await res.json() as any }
}

afterEach(() => {
  mock.restoreAll()
  injectBlobStoreForTesting(null)
})

describe('GET /history', () => {
  test('default read filters out heartbeat triggers and compact summaries', async () => {
    const { status, body } = await historyResponse(makeMockBlobStore(chatlogToday()), `/history?conversation_id=${SELF_ID}`)

    assert.equal(status, 200)
    assert.equal(body.ok, true)
    assert.equal(body.conversationId, SELF_ID)
    const kinds = (body.messages as Array<{ kind: string }>).map((message) => message.kind)
    assert.ok(!kinds.includes('heartbeat'))
    assert.ok(!kinds.includes('summary'))
    assert.ok(kinds.includes('user'))
    assert.ok(kinds.includes('assistant'))
    assert.ok(kinds.includes('tool'))
  })

  test('include=all returns the complete archive including heartbeat and summary', async () => {
    const { status, body } = await historyResponse(makeMockBlobStore(chatlogToday()), '/history?include=all')

    assert.equal(status, 200)
    assert.equal(body.ok, true)
    const kinds = (body.messages as Array<{ kind: string }>).map((message) => message.kind)
    assert.ok(kinds.includes('heartbeat'))
    assert.ok(kinds.includes('summary'))
    assert.equal(body.messages.length, 5)
  })

  test('conversation_id defaults to eo-self and is returned in the payload', async () => {
    const { status, body } = await historyResponse(makeMockBlobStore(chatlogToday()), '/history')
    assert.equal(status, 200)
    assert.equal(body.conversationId, SELF_ID)
  })

  test('honors an explicit conversation_id from the query string', async () => {
    const { status, body } = await historyResponse(makeMockBlobStore(chatlogToday()), '/history?conversation_id=other-conv')
    assert.equal(status, 200)
    assert.equal(body.conversationId, 'other-conv')
  })

  test('keyword search returns matching archive lines (heartbeat/summary included)', async () => {
    const today = dateKey(new Date())
    const blob = makeMockBlobStore({
      [`${CHATLOG_DIR}${today}.md`]: [
        '- [2026-09-11T10:00:00Z] user: 目标词 出现在用户消息',
        '- [2026-09-11T10:01:00Z] assistant: 无关回复',
        '- [2026-09-11T10:02:00Z] summary: 目标词 出现在摘要',
      ].join('\n'),
    })

    const { status, body } = await historyResponse(blob, `/history?keyword=${encodeURIComponent('目标词')}`)

    assert.equal(status, 200)
    assert.equal(body.ok, true)
    const messages = body.messages as Array<{ content: string; kind: string }>
    assert.ok(messages.length >= 1)
    assert.ok(messages.some((message) => message.content.includes('目标词')))
  })

  test('keyword search returns an empty messages array when nothing matches', async () => {
    const { status, body } = await historyResponse(makeMockBlobStore(chatlogToday()), '/history?keyword=不存在的词')
    assert.equal(status, 200)
    assert.deepEqual(body.messages, [])
  })

  test('keyword search honors limit and slices the tail (P2-1)', async () => {
    const today = dateKey(new Date())
    const yesterday = dateKey(new Date(Date.now() - 86_400_000))
    // searchChatlog caps at 3 snippets per file, so spread the matches across
    // two days to produce a flattened list longer than the requested limit.
    const blob = makeMockBlobStore({
      [`${CHATLOG_DIR}${today}.md`]: [
        '- [2026-09-11T10:00:00Z] user: 目标词 今天一条',
        '- [2026-09-11T10:01:00Z] assistant: 目标词 今天两条',
      ].join('\n'),
      [`${CHATLOG_DIR}${yesterday}.md`]: [
        '- [2026-09-10T10:00:00Z] user: 目标词 昨天一条',
        '- [2026-09-10T10:01:00Z] assistant: 目标词 昨天两条',
      ].join('\n'),
    })

    const { status, body } = await historyResponse(blob, `/history?keyword=${encodeURIComponent('目标词')}&limit=2`)

    assert.equal(status, 200)
    assert.equal(body.ok, true)
    const messages = body.messages as Array<{ content: string }>
    assert.equal(messages.length, 2)
    assert.equal(messages[0]?.content, '目标词 昨天一条')
    assert.equal(messages[1]?.content, '目标词 昨天两条')
  })

  test('keyword snippets that do not parse are marked kind=search with empty ts (P2-1)', async () => {
    const today = dateKey(new Date())
    const blob = makeMockBlobStore({
      [`${CHATLOG_DIR}${today}.md`]: [
        '- [2026-09-11T10:00:00Z] user: 目标词 开头',
        '  续行内容 目标词 也在', // bare continuation line, not a full row
        '- [2026-09-11T10:01:00Z] assistant: 目标词 结束',
      ].join('\n'),
    })

    const { status, body } = await historyResponse(blob, `/history?keyword=${encodeURIComponent('目标词')}`)

    assert.equal(status, 200)
    const messages = body.messages as Array<{ kind: string; ts: string }>
    const fragment = messages.find((message) => message.kind === 'search')
    assert.ok(fragment, 'a non-row snippet must surface as kind=search')
    assert.equal(fragment.ts, '')
  })

  test('limit clamps the returned messages to the tail', async () => {
    const { status, body } = await historyResponse(makeMockBlobStore(chatlogToday()), '/history?limit=1&include=all')
    assert.equal(status, 200)
    assert.equal(body.messages.length, 1)
    assert.equal((body.messages[0] as { kind: string }).kind, 'summary')
  })

  test('reads from the archive even when the store is missing (no store degradation)', async () => {
    const { status, body } = await historyResponse(makeMockBlobStore(chatlogToday()), '/history')
    assert.equal(status, 200)
    assert.equal(body.ok, true)
    assert.ok(body.messages.length > 0)
  })

  test('returns an empty list when no archive exists yet', async () => {
    const { status, body } = await historyResponse(makeMockBlobStore(), '/history')
    assert.equal(status, 200)
    assert.deepEqual(body.messages, [])
  })
})

describe('compact + /history integration (P2-2)', () => {
  test('old messages stay queryable via /history after maybeCompact deletes them from the store', async () => {
    const store = makeMockStore()
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const context = makeContext({ store, env: gatewayEnv() as Env })

    // Write the "old" conversation through the REAL persistHistory path so each
    // message lands in BOTH the store and the append-only chatlog archive.
    for (let i = 1; i <= 5; i += 1) {
      await persistHistory(context, SELF_ID, 'user', `旧消息-${i}`)
    }
    // Bulk-fill the rest of the store to cross COMPACT_TRIGGER (0.6 × 10000).
    for (let i = 0; i < 5995; i += 1) {
      store.addMessage(SELF_ID, { role: 'assistant', content: `填充-${i}`, metadata: {} })
    }
    assert.equal(store.logs.get(SELF_ID)?.length, 6000)

    // Compact folds the oldest 20% (which includes 旧消息-1..5) into one summary
    // and DELETES those store rows.
    mock.method(globalThis, 'fetch', async () => new Response(
      JSON.stringify({ choices: [{ message: { content: '早期记录摘要。' } }] }),
      { status: 200, headers: { 'content-type': 'application/json' } },
    ))
    const result = await maybeCompact(context, SELF_ID)
    assert.equal(result.compacted, true)
    assert.equal(result.removedCount, 1200)

    // The old rows are GONE from the store…
    const remaining = store.logs.get(SELF_ID) ?? []
    assert.ok(!remaining.some((message) => message.content.startsWith('旧消息-')))

    // …but GET /history (reading the append-only archive) still serves them.
    const res = await historyOnRequest(makeContext({ url: '/history?include=all' }))
    assert.equal(res.status, 200)
    const body = await res.json() as { ok: boolean; messages: Array<{ content: string }> }
    assert.equal(body.ok, true)
    const contents = body.messages.map((message) => message.content)
    for (let i = 1; i <= 5; i += 1) {
      assert.ok(contents.includes(`旧消息-${i}`))
    }
    // The compact summary itself is archived too.
    assert.ok(contents.includes('早期记录摘要。'))
  })
})

describe('chatlog tools (registered in buildTools)', () => {
  function contextFor(blob: MockBlobStore): any {
    injectBlobStoreForTesting(blob)
    return makeContext({})
  }

  test('chatlog_search and chatlog_read are registered', () => {
    const tools = buildTools({ context: contextFor(makeMockBlobStore()), conversationId: SELF_ID })
    const names = tools.definitions.map((definition) => definition.name)
    assert.ok(names.includes('chatlog_search'))
    assert.ok(names.includes('chatlog_read'))
  })

  test('chatlog_search returns matching snippets', async () => {
    const tools = buildTools({ context: contextFor(makeMockBlobStore(chatlogToday())), conversationId: SELF_ID })
    const result = await tools.run('chatlog_search', { keyword: '真实用户' })
    assert.equal(result.isError, undefined)
    assert.match(result.content, /真实用户问题/)
  })

  test('chatlog_search reports no matches', async () => {
    const tools = buildTools({ context: contextFor(makeMockBlobStore(chatlogToday())), conversationId: SELF_ID })
    const result = await tools.run('chatlog_search', { keyword: '缺失词' })
    assert.match(result.content, /No chatlog matches/)
  })

  test('chatlog_search requires a keyword', async () => {
    const tools = buildTools({ context: contextFor(makeMockBlobStore()), conversationId: SELF_ID })
    const result = await tools.run('chatlog_search', {})
    assert.equal(result.isError, true)
    assert.match(result.content, /requires "keyword"/)
  })

  test('chatlog_read reads one day by YYYY-MM-DD', async () => {
    const today = dateKey(new Date())
    const tools = buildTools({ context: contextFor(makeMockBlobStore(chatlogToday())), conversationId: SELF_ID })
    const result = await tools.run('chatlog_read', { day: today })
    assert.equal(result.isError, undefined)
    assert.match(result.content, /真实用户问题/)
    assert.match(result.content, new RegExp(`## ${today}`))
  })

  test('chatlog_read rejects a malformed day', async () => {
    const tools = buildTools({ context: contextFor(makeMockBlobStore()), conversationId: SELF_ID })
    const result = await tools.run('chatlog_read', { day: 'not-a-date' })
    assert.equal(result.isError, true)
    assert.match(result.content, /must be/)
  })

  test('chatlog_read without a day returns the most recent N days', async () => {
    const tools = buildTools({ context: contextFor(makeMockBlobStore(chatlogToday())), conversationId: SELF_ID })
    const result = await tools.run('chatlog_read', {})
    assert.equal(result.isError, undefined)
    assert.match(result.content, /真实用户问题/)
  })

  test('chatlog_read can read the full chatlog key form', async () => {
    const today = dateKey(new Date())
    const tools = buildTools({ context: contextFor(makeMockBlobStore(chatlogToday())), conversationId: SELF_ID })
    const result = await tools.run('chatlog_read', { day: `${CHATLOG_DIR}${today}.md` })
    assert.equal(result.isError, undefined)
    assert.match(result.content, /真实用户问题/)
  })
})