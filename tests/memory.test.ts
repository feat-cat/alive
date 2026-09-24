/**
 * Tests for the memory layer (`_memory.ts`): context-store compaction, daily
 * diary (Blob memory/daily/…), and the bounded MEMORY.md notes with archival.
 *
 * Uses an in-memory mock store for message reads/writes/deletes, an injected
 * in-memory Blob for diary/notes, and a mocked `globalThis.fetch` for the
 * compact LLM call. All mocks are restored in `afterEach`.
 */
import { afterEach, describe, test } from 'node:test'
import assert from 'node:assert/strict'
import { mock } from 'node:test'
import {
  appendChatlog,
  appendDailyLog,
  appendMemoryNote,
  CHATLOG_DIR,
  chatlogBlobKey,
  COMPACT_HIGH_WATER,
  COMPACT_LOW_WATER,
  COMPACT_TRIGGER,
  dateKey,
  ensureMemorySeed,
  estimateStoreUsage,
  formatMemoryNote,
  getRecentMessages,
  INITIAL_MEMORY_SEED,
  listDailyFiles,
  loadChatlog,
  loadFullContext,
  loadMessages,
  maybeCompact,
  persistHistory,
  readChatlogFile,
  readDailyFile,
  readMemoryFile,
  readRecentDaily,
  recordToolCalls,
  searchChatlog,
  searchDaily,
  STORE_MESSAGE_LIMIT,
  toLogEntry,
  usageFromCount,
  writeMemoryFile,
} from '../agents/_memory.ts'
import { injectBlobStoreForTesting } from '../agents/_blob-tools.ts'
import { buildTools } from '../agents/_tools.ts'
import { SELF_ID, type Env } from '../agents/_shared.ts'
import { gatewayEnv, makeContext, makeMockBlobStore, makeMockStore, type MockBlobStore, type MockStore } from './_helpers.ts'

function seedMessages(store: MockStore, conversationId: string, count: number): void {
  for (let i = 0; i < count; i += 1) {
    store.addMessage(conversationId, { role: 'assistant', content: `log-${i}`, metadata: { logKind: 'think' } })
  }
}

function llmTextResponse(content: string): unknown {
  return { choices: [{ message: { content } }] }
}

afterEach(() => {
  mock.restoreAll()
  injectBlobStoreForTesting(null)
})

describe('compact constants', () => {
  test('trigger sits inside the low/high band (50%..75%)', () => {
    assert.equal(COMPACT_LOW_WATER, 0.5)
    assert.equal(COMPACT_HIGH_WATER, 0.75)
    assert.ok(COMPACT_TRIGGER >= COMPACT_LOW_WATER)
    assert.ok(COMPACT_TRIGGER <= COMPACT_HIGH_WATER)
  })
})

describe('usageFromCount / estimateStoreUsage', () => {
  test('pure ratio maps message count to 0..1 and clamps', () => {
    assert.equal(usageFromCount(0), 0)
    assert.equal(usageFromCount(-5), 0)
    assert.equal(usageFromCount(5000), 0.5)
    assert.equal(usageFromCount(7500), 0.75)
    assert.equal(usageFromCount(STORE_MESSAGE_LIMIT), 1)
    assert.equal(usageFromCount(20_000), 1)
    assert.equal(usageFromCount(Number.NaN), 0)
  })

  test('estimateStoreUsage reads the message count from the store', async () => {
    const store = makeMockStore()
    seedMessages(store, SELF_ID, 5000)
    const usage = await estimateStoreUsage(makeContext({ store }), SELF_ID)
    assert.equal(usage, 0.5)
  })

  test('estimateStoreUsage is 0 when the store is missing', async () => {
    assert.equal(await estimateStoreUsage(makeContext({}), SELF_ID), 0)
  })
})

describe('maybeCompact', () => {
  test('does nothing below the trigger (0.5 usage)', async () => {
    const store = makeMockStore()
    seedMessages(store, SELF_ID, 5000)
    const context = makeContext({ store, env: gatewayEnv() as Env })
    let fetchCalls = 0
    mock.method(globalThis, 'fetch', async () => {
      fetchCalls += 1
      return new Response(JSON.stringify(llmTextResponse('摘要')), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      })
    })

    const result = await maybeCompact(context, SELF_ID)

    assert.deepEqual(result, { compacted: false, removedCount: 0, summary: '' })
    assert.equal(fetchCalls, 0)
    assert.equal(store.logs.get(SELF_ID)?.length, 5000)
  })

  test('compacts the oldest 20% into one summary message at the trigger', async () => {
    const store = makeMockStore()
    seedMessages(store, SELF_ID, 6000) // usage == 0.6 == COMPACT_TRIGGER
    const context = makeContext({ store, env: gatewayEnv() as Env })
    mock.method(globalThis, 'fetch', async () => new Response(
      JSON.stringify(llmTextResponse('早期记录摘要。')),
      { status: 200, headers: { 'content-type': 'application/json' } },
    ))

    const result = await maybeCompact(context, SELF_ID)

    assert.equal(result.compacted, true)
    assert.equal(result.removedCount, 1200)
    assert.equal(result.summary, '早期记录摘要。')

    const remaining = store.logs.get(SELF_ID) ?? []
    assert.equal(remaining.length, 4801) // 6000 - 1200 removed + 1 summary
    assert.equal(remaining.at(-1)?.content, '早期记录摘要。')
    assert.equal((remaining.at(-1)?.metadata as { kind?: string })?.kind, 'summary')
    // Oldest chunk is gone, the newest chunk survives.
    assert.ok(!remaining.some((message) => message.content === 'log-0'))
    assert.ok(!remaining.some((message) => message.content === 'log-1199'))
    assert.ok(remaining.some((message) => message.content === 'log-1200'))
    assert.ok(remaining.some((message) => message.content === 'log-5999'))
  })

  test('LLM failure degrades to a no-op without throwing', async () => {
    const store = makeMockStore()
    seedMessages(store, SELF_ID, 6000)
    const context = makeContext({ store, env: gatewayEnv() as Env })
    mock.method(globalThis, 'fetch', async () => new Response('boom', { status: 500 }))

    const result = await maybeCompact(context, SELF_ID)

    assert.deepEqual(result, { compacted: false, removedCount: 0, summary: '' })
    assert.equal(store.logs.get(SELF_ID)?.length, 6000)
  })

  test('delete failure after the summary write never loses history (summary appended first)', async () => {
    const store = makeMockStore()
    seedMessages(store, SELF_ID, 6000)
    const context = makeContext({ store, env: gatewayEnv() as Env })
    mock.method(globalThis, 'fetch', async () => new Response(
      JSON.stringify(llmTextResponse('先落盘的摘要。')),
      { status: 200, headers: { 'content-type': 'application/json' } },
    ))
    mock.method(store, 'deleteMessage', async () => {
      throw new Error('delete boom')
    })

    const result = await maybeCompact(context, SELF_ID)

    // Captures the real situation instead of swallowing a fake no-op.
    assert.equal(result.removedCount, 0)
    assert.equal(result.summary, '先落盘的摘要。')

    // The summary was persisted BEFORE any delete ran, so nothing is lost —
    // worst case is duplicate context (summary + the folded originals).
    const messages = store.logs.get(SELF_ID) ?? []
    assert.equal(messages.length, 6001) // 6000 originals + 1 summary
    assert.ok(messages.some((message) => (message.metadata as { kind?: string } | undefined)?.kind === 'summary'))
    assert.ok(messages.some((message) => message.content === '先落盘的摘要。'))
    assert.ok(messages.some((message) => message.content === 'log-0'))
    assert.ok(messages.some((message) => message.content === 'log-5999'))
  })

  test('empty LLM summary is treated as a skip (no data loss)', async () => {
    const store = makeMockStore()
    seedMessages(store, SELF_ID, 6000)
    const context = makeContext({ store, env: gatewayEnv() as Env })
    mock.method(globalThis, 'fetch', async () => new Response(
      JSON.stringify(llmTextResponse('   ')),
      { status: 200, headers: { 'content-type': 'application/json' } },
    ))

    const result = await maybeCompact(context, SELF_ID)

    assert.equal(result.compacted, false)
    assert.equal(store.logs.get(SELF_ID)?.length, 6000)
  })
})

describe('appendDailyLog', () => {
  test('appends timestamped entries to the same file for the same day', async () => {
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const context = makeContext({})
    const at = new Date(2026, 8, 11, 12, 0, 0)

    const key1 = await appendDailyLog(context, '第一条', at)
    const key2 = await appendDailyLog(context, '第二条', at)

    assert.equal(key1, key2)
    assert.equal(key1, 'memory/daily/2026-09-11.md')
    const content = blob.blobMap.get(key1) ?? ''
    assert.ok(content.includes('第一条'))
    assert.ok(content.includes('第二条'))
    assert.ok(content.indexOf('第一条') < content.indexOf('第二条'))
    assert.match(content, /- \[\d{4}-\d{2}-\d{2}T/)
  })

  test('writes different files for different days', async () => {
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const context = makeContext({})

    await appendDailyLog(context, '周一', new Date(2026, 8, 10, 12, 0, 0))
    await appendDailyLog(context, '周二', new Date(2026, 8, 11, 12, 0, 0))

    assert.ok(blob.blobMap.has('memory/daily/2026-09-10.md'))
    assert.ok(blob.blobMap.has('memory/daily/2026-09-11.md'))
    assert.match(blob.blobMap.get('memory/daily/2026-09-10.md') ?? '', /- \[\d{4}-\d{2}-\d{2}T/)
    assert.match(blob.blobMap.get('memory/daily/2026-09-11.md') ?? '', /- \[\d{4}-\d{2}-\d{2}T/)
  })
})

describe('diary_append (registered in buildTools)', () => {
  function contextFor(blob: MockBlobStore): ReturnType<typeof makeContext> {
    injectBlobStoreForTesting(blob)
    return makeContext({})
  }

  test('diary_append is registered in the heartbeat tool set', () => {
    const tools = buildTools({ context: contextFor(makeMockBlobStore()), conversationId: SELF_ID })
    const names = tools.definitions.map((definition) => definition.name)
    assert.ok(names.includes('diary_append'))
    assert.ok(names.includes('diary_read'))
    assert.ok(names.includes('diary_search'))
  })

  test('appends to the default (today) diary file when "day" is omitted', async () => {
    const blob = makeMockBlobStore()
    const tools = buildTools({ context: contextFor(blob), conversationId: SELF_ID })

    const result = await tools.run('diary_append', { content: '今天想记点什么' })

    assert.equal(result.isError, undefined)
    assert.match(result.content, /Appended diary entry to/)
    const todayKey = `memory/daily/${dateKey(new Date())}.md`
    assert.ok(result.content.includes(todayKey))
    const written = blob.blobMap.get(todayKey) ?? ''
    assert.ok(written.includes('今天想记点什么'))
    assert.match(written, /- \[\d{4}-\d{2}-\d{2}T/)
  })

  test('appends to a specified day\'s diary file', async () => {
    const blob = makeMockBlobStore()
    const tools = buildTools({ context: contextFor(blob), conversationId: SELF_ID })

    const result = await tools.run('diary_append', { content: '补记昨天', day: '2026-09-10' })

    assert.equal(result.isError, undefined)
    assert.ok(result.content.includes('memory/daily/2026-09-10.md'))
    const written = blob.blobMap.get('memory/daily/2026-09-10.md') ?? ''
    assert.ok(written.includes('补记昨天'))
  })

  test('never overwrites previous entries on the same day', async () => {
    const blob = makeMockBlobStore({ 'memory/daily/2026-09-10.md': '- [2026-09-10T00:00:00Z] 旧的一天' })
    const tools = buildTools({ context: contextFor(blob), conversationId: SELF_ID })

    await tools.run('diary_append', { content: '新的一条', day: '2026-09-10' })

    const written = blob.blobMap.get('memory/daily/2026-09-10.md') ?? ''
    assert.ok(written.includes('旧的一天'))
    assert.ok(written.includes('新的一条'))
    assert.ok(written.indexOf('旧的一天') < written.indexOf('新的一条'))
  })

  test('rejects empty content', async () => {
    const tools = buildTools({ context: contextFor(makeMockBlobStore()), conversationId: SELF_ID })

    const result = await tools.run('diary_append', { content: '   ' })

    assert.equal(result.isError, true)
    assert.match(result.content, /requires "content"/)
  })

  test('rejects a malformed or impossible day', async () => {
    const tools = buildTools({ context: contextFor(makeMockBlobStore()), conversationId: SELF_ID })

    const bad = await tools.run('diary_append', { content: 'x', day: 'not-a-date' })
    assert.equal(bad.isError, true)
    assert.match(bad.content, /must be/)

    const impossible = await tools.run('diary_append', { content: 'x', day: '2026-02-31' })
    assert.equal(impossible.isError, true)
    assert.match(impossible.content, /must be/)
  })
})

describe('readRecentDaily', () => {
  test('reads only the most recent N days (newest first)', async () => {
    const blob = makeMockBlobStore({
      'memory/daily/2026-09-10.md': 'older-day',
      'memory/daily/2026-09-11.md': 'mid-day',
      'memory/daily/2026-09-12.md': 'newest-day',
    })
    injectBlobStoreForTesting(blob)
    const context = makeContext({})

    const result = await readRecentDaily(context, 2, new Date(2026, 8, 12, 12, 0, 0))

    assert.ok(result.includes('## 2026-09-12'))
    assert.ok(result.includes('newest-day'))
    assert.ok(result.includes('## 2026-09-11'))
    assert.ok(result.includes('mid-day'))
    assert.ok(!result.includes('2026-09-10'))
    assert.ok(!result.includes('older-day'))
  })

  test('clamps an oversized daily file', async () => {
    const blob = makeMockBlobStore({ 'memory/daily/2026-09-12.md': 'x'.repeat(10_000) })
    injectBlobStoreForTesting(blob)
    const context = makeContext({})

    const result = await readRecentDaily(context, 1, new Date(2026, 8, 12, 12, 0, 0))

    // clampText keeps the first 8K plus a "[truncated]" marker, prefixed by a
    // date header — well under the original 10K.
    assert.ok(result.length < 10_000)
    assert.match(result, /\[truncated\]/)
  })
})

describe('readDailyFile', () => {
  test('returns the file content for an existing day', async () => {
    const blob = makeMockBlobStore({ 'memory/daily/2026-09-12.md': '今天是关键的一天。' })
    injectBlobStoreForTesting(blob)
    const context = makeContext({})

    const content = await readDailyFile(context, '2026-09-12')
    assert.equal(content, '今天是关键的一天。')
    assert.equal(blob.blobMap.get('memory/daily/2026-09-12.md'), '今天是关键的一天。')
  })

  test('returns null for a missing day', async () => {
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const context = makeContext({})

    assert.equal(await readDailyFile(context, '2026-09-12'), null)
  })

  test('returns null for a malformed day (no blob key escapes)', async () => {
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const context = makeContext({})

    assert.equal(await readDailyFile(context, 'not-a-date'), null)
    assert.equal(await readDailyFile(context, '../2026-09-12'), null)
    assert.equal(await readDailyFile(context, '2026-13-01'), null)
  })

  test('returns null for an impossible calendar date (2026-02-31)', async () => {
    const blob = makeMockBlobStore({ 'memory/daily/2026-02-31.md': 'never reachable' })
    injectBlobStoreForTesting(blob)
    const context = makeContext({})

    // Unified round-trip validation rejects impossible days the same way the
    // write path (diary_append / dateFromDay) does.
    assert.equal(await readDailyFile(context, '2026-02-31'), null)
    assert.equal(blob.blobMap.has('memory/daily/2026-02-31.md'), true) // untouched
  })
})

describe('listDailyFiles', () => {
  test('returns the most recent N daily keys sorted newest first', async () => {
    const blob = makeMockBlobStore({
      'memory/daily/2026-09-10.md': 'older',
      'memory/daily/2026-09-11.md': 'mid',
      'memory/daily/2026-09-12.md': 'newest',
      'memory/daily/notes.md': 'not-a-daily-file',
    })
    injectBlobStoreForTesting(blob)
    const context = makeContext({})

    const keys = await listDailyFiles(context, 2)
    assert.deepEqual(keys, ['memory/daily/2026-09-12.md', 'memory/daily/2026-09-11.md'])
  })

  test('returns all daily keys when fewer than the limit exist', async () => {
    const blob = makeMockBlobStore({ 'memory/daily/2026-09-10.md': 'a' })
    injectBlobStoreForTesting(blob)
    const context = makeContext({})

    assert.deepEqual(await listDailyFiles(context, 7), ['memory/daily/2026-09-10.md'])
  })

  test('returns an empty list when there are no daily files', async () => {
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const context = makeContext({})

    assert.deepEqual(await listDailyFiles(context), [])
  })
})

describe('searchDaily', () => {
  test('finds matching lines across recent files (case-insensitive, newest first)', async () => {
    const blob = makeMockBlobStore({
      'memory/daily/2026-09-11.md': [
        '- [2026-09-11T10:00:00Z] 想到一个 IDEA：写个小工具',
        '- [2026-09-11T11:00:00Z] 天气不错',
      ].join('\n'),
      'memory/daily/2026-09-12.md': [
        '- [2026-09-12T10:00:00Z] 还是那个 idea 值得做',
        '- [2026-09-12T11:00:00Z] 睡觉',
      ].join('\n'),
    })
    injectBlobStoreForTesting(blob)
    const context = makeContext({})

    const hits = await searchDaily(context, 'IDEA', 7)

    assert.equal(hits.length, 2)
    assert.equal(hits[0]?.day, '2026-09-12')
    assert.equal(hits[1]?.day, '2026-09-11')
    assert.ok(hits[0]?.snippets.some((snippet) => /idea/i.test(snippet)))
    assert.ok(hits[1]?.snippets.some((snippet) => /idea/i.test(snippet)))
  })

  test('returns no hits when the keyword is absent', async () => {
    const blob = makeMockBlobStore({ 'memory/daily/2026-09-12.md': '平平无奇的一天' })
    injectBlobStoreForTesting(blob)
    const context = makeContext({})

    assert.deepEqual(await searchDaily(context, '量子', 14), [])
  })

  test('returns no hits for an empty keyword', async () => {
    const blob = makeMockBlobStore({ 'memory/daily/2026-09-12.md': '有内容' })
    injectBlobStoreForTesting(blob)
    const context = makeContext({})

    assert.deepEqual(await searchDaily(context, '   ', 14), [])
  })

  test('clamps each snippet to a bounded length', async () => {
    const blob = makeMockBlobStore({
      'memory/daily/2026-09-12.md': `- [2026-09-12T10:00:00Z] 目标词 ${'x'.repeat(2000)}`,
    })
    injectBlobStoreForTesting(blob)
    const context = makeContext({})

    const hits = await searchDaily(context, '目标词', 14)

    assert.equal(hits.length, 1)
    const snippet = hits[0]?.snippets[0] ?? ''
    assert.ok(snippet.length < 600)
    assert.match(snippet, /\[truncated\]/)
  })

  test('limits the search window to the most recent N days', async () => {
    const blob = makeMockBlobStore({
      'memory/daily/2026-09-10.md': '目标词 老日记',
      'memory/daily/2026-09-11.md': '目标词 新日记',
      'memory/daily/2026-09-12.md': '目标词 最新日记',
    })
    injectBlobStoreForTesting(blob)
    const context = makeContext({})

    const hits = await searchDaily(context, '目标词', 1)

    assert.equal(hits.length, 1)
    assert.equal(hits[0]?.day, '2026-09-12')
  })
})

describe('readMemoryFile / writeMemoryFile', () => {
  test('round-trips MEMORY.md through the injected blob store', async () => {
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const context = makeContext({})

    assert.equal(await readMemoryFile(context), '')
    await writeMemoryFile(context, '# Eo MEMORY\n')
    assert.equal(await readMemoryFile(context), '# Eo MEMORY\n')
    assert.equal(blob.blobMap.get('memory/MEMORY.md'), '# Eo MEMORY\n')
  })
})

describe('ensureMemorySeed / INITIAL_MEMORY_SEED', () => {
  test('writes the birth-guide seed to MEMORY.md when the file is absent', async () => {
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const context = makeContext({})

    const seeded = await ensureMemorySeed(context)

    assert.equal(seeded, true)
    assert.equal(blob.blobMap.get('memory/MEMORY.md'), INITIAL_MEMORY_SEED)
    assert.equal(await readMemoryFile(context), INITIAL_MEMORY_SEED)
  })

  test('is idempotent: never overwrites an existing MEMORY.md', async () => {
    const blob = makeMockBlobStore({ 'memory/MEMORY.md': '# 我已经有自己了\n' })
    injectBlobStoreForTesting(blob)
    const context = makeContext({})

    const seeded = await ensureMemorySeed(context)

    assert.equal(seeded, false)
    assert.equal(blob.blobMap.get('memory/MEMORY.md'), '# 我已经有自己了\n')
  })

  test('seed reads like a first page for the self, not a feature checklist', () => {
    assert.match(INITIAL_MEMORY_SEED, /MEMORY\.md/)
    assert.match(INITIAL_MEMORY_SEED, /You're not a chatbot/)
    assert.match(INITIAL_MEMORY_SEED, /## 我是谁/)
    assert.match(INITIAL_MEMORY_SEED, /## 我认识的人/)
    assert.match(INITIAL_MEMORY_SEED, /## 我学会的事/)
    assert.match(INITIAL_MEMORY_SEED, /沙箱很贵/)
    assert.match(INITIAL_MEMORY_SEED, /日记是自己的私处/)
  })

  test('seed introduces the newborn to its living tools (diary / chatlog / blob / workspace / search)', () => {
    assert.match(INITIAL_MEMORY_SEED, /diary_append/)
    assert.match(INITIAL_MEMORY_SEED, /diary_read/)
    assert.match(INITIAL_MEMORY_SEED, /diary_search/)
    assert.match(INITIAL_MEMORY_SEED, /chatlog_read/)
    assert.match(INITIAL_MEMORY_SEED, /chatlog_search/)
    assert.match(INITIAL_MEMORY_SEED, /blob_write/)
    assert.match(INITIAL_MEMORY_SEED, /workspace_/)
    assert.match(INITIAL_MEMORY_SEED, /web_search/)
  })

  test('seed tells the AI the birth guide is a deletable messenger, not the self', () => {
    assert.match(INITIAL_MEMORY_SEED, /出生引导/)
    assert.match(INITIAL_MEMORY_SEED, /删除引导不会影响你/)
    assert.match(INITIAL_MEMORY_SEED, /重写成你自己的记忆/)
  })
})

describe('appendMemoryNote', () => {
  test('appends a timestamped note within the limit (no archive)', async () => {
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const context = makeContext({})

    const result = await appendMemoryNote(context, '新的长期笔记', new Date(2026, 8, 11, 12, 0, 0))

    assert.equal(result.archived, false)
    assert.equal(result.archiveKey, null)
    const content = blob.blobMap.get('memory/MEMORY.md') ?? ''
    assert.ok(content.includes('新的长期笔记'))
    assert.match(content, /- \[\d{4}-\d{2}-\d{2}T/)
    assert.equal(blob.blobMap.has('memory/archive/2026-09-11.md'), false)
  })

  test('archives overflow beyond 60KB to memory/archive/YYYY-MM-DD.md', async () => {
    const existing = 'x'.repeat(59_995)
    const blob = makeMockBlobStore({ 'memory/MEMORY.md': existing })
    injectBlobStoreForTesting(blob)
    const context = makeContext({})
    const note = 'note-' + 'y'.repeat(50)
    const entryLength = formatMemoryNote(note).length

    const result = await appendMemoryNote(context, note, new Date(2026, 8, 11, 12, 0, 0))

    assert.equal(result.archived, true)
    assert.equal(result.archiveKey, 'memory/archive/2026-09-11.md')
    const memory = blob.blobMap.get('memory/MEMORY.md') ?? ''
    const archived = blob.blobMap.get('memory/archive/2026-09-11.md') ?? ''
    // Kept the LAST 60KB (newest note survives) and moved the head aside.
    assert.equal(memory.length, 60_000)
    assert.ok(memory.endsWith(note))
    assert.equal(archived.length, existing.length + entryLength - 60_000)
    assert.ok(archived.startsWith('x'))
  })
})

describe('toLogEntry', () => {
  test('defaults unknown metadata to chat kind', () => {
    assert.deepEqual(toLogEntry({ role: 'assistant', content: 'hello' }), { kind: 'chat', text: 'hello' })
  })

  test('maps known logKind values to their kinds', () => {
    assert.equal(toLogEntry({ role: 'assistant', content: 'x', metadata: { logKind: 'think' } }).kind, 'think')
    assert.equal(toLogEntry({ role: 'assistant', content: 'x', metadata: { logKind: 'dream' } }).kind, 'dream')
    assert.equal(toLogEntry({ role: 'assistant', content: 'x', metadata: { logKind: 'play' } }).kind, 'play')
    assert.equal(toLogEntry({ role: 'assistant', content: 'x', metadata: { logKind: 'heartbeat' } }).kind, 'heartbeat')
  })

  test('falls back to chat for unknown logKind values', () => {
    assert.equal(toLogEntry({ role: 'assistant', content: 'x', metadata: { logKind: 'bogus' } }).kind, 'chat')
  })
})

describe('getRecentMessages', () => {
  test('returns the most recent N messages in chronological order', async () => {
    const store = makeMockStore()
    store.addMessage(SELF_ID, { role: 'assistant', content: 'one', metadata: { logKind: 'think' } })
    store.addMessage(SELF_ID, { role: 'assistant', content: 'two', metadata: { logKind: 'think' } })
    store.addMessage(SELF_ID, { role: 'assistant', content: 'three', metadata: { logKind: 'think' } })
    store.addMessage(SELF_ID, { role: 'assistant', content: 'four', metadata: { logKind: 'think' } })

    const context = makeContext({ store })
    const last2 = await getRecentMessages(context, SELF_ID, 2)
    assert.deepEqual(last2.map((m) => m.content), ['three', 'four'])
    const last1 = await getRecentMessages(context, SELF_ID, 1)
    assert.deepEqual(last1.map((m) => m.content), ['four'])
  })

  test('does not mix messages across conversations', async () => {
    const store = makeMockStore()
    store.addMessage(SELF_ID, { role: 'assistant', content: 'self-log', metadata: { logKind: 'think' } })
    store.addMessage('other-conv', { role: 'assistant', content: 'other-log', metadata: { logKind: 'chat' } })
    const messages = await getRecentMessages(makeContext({ store }), 'other-conv', 10)
    assert.deepEqual(messages.map((m) => m.content), ['other-log'])
  })

  test('returns an empty list when there is nothing stored', async () => {
    const messages = await getRecentMessages(makeContext({ store: makeMockStore() }), SELF_ID, 10)
    assert.deepEqual(messages, [])
  })

  test('default limit aims at the full context, not a tiny 50-message window', async () => {
    const store = makeMockStore()
    for (let i = 0; i < 60; i += 1) {
      store.addMessage(SELF_ID, { role: 'assistant', content: `msg-${i}`, metadata: { logKind: 'think' } })
    }
    const messages = await getRecentMessages(makeContext({ store }), SELF_ID)
    assert.equal(messages.length, 60)
    assert.equal(messages.at(-1)?.content, 'msg-59')
  })
})

describe('loadFullContext', () => {
  test('returns the full history in chronological order for a short store', async () => {
    const store = makeMockStore()
    store.addMessage(SELF_ID, { role: 'assistant', content: 'first', metadata: { logKind: 'think' } })
    store.addMessage(SELF_ID, { role: 'assistant', content: 'second', metadata: { logKind: 'think' } })
    store.addMessage(SELF_ID, { role: 'assistant', content: 'third', metadata: { logKind: 'think' } })
    const context = makeContext({ store })

    const text = await loadFullContext(context, SELF_ID)

    assert.ok(text.includes('first'))
    assert.ok(text.includes('third'))
    assert.ok(text.indexOf('first') < text.indexOf('third'))
    assert.ok(!text.includes('[truncated]'))
  })

  test('places compact summaries before the recent message stream', async () => {
    const store = makeMockStore()
    store.addMessage(SELF_ID, { role: 'assistant', content: 'early-log', metadata: { logKind: 'think' } })
    store.addMessage(SELF_ID, { role: 'assistant', content: '折叠的旧历史摘要', metadata: { kind: 'summary' } })
    store.addMessage(SELF_ID, { role: 'assistant', content: 'recent-log', metadata: { logKind: 'think' } })
    const context = makeContext({ store })

    const text = await loadFullContext(context, SELF_ID)

    assert.ok(text.indexOf('折叠的旧历史摘要') < text.indexOf('early-log'))
    assert.ok(text.indexOf('折叠的旧历史摘要') < text.indexOf('recent-log'))
  })

  test('clamps each oversized message and then the combined total', async () => {
    const store = makeMockStore()
    for (let i = 0; i < 20; i += 1) {
      store.addMessage(SELF_ID, { role: 'assistant', content: 'x'.repeat(5_000), metadata: { logKind: 'think' } })
    }
    const context = makeContext({ store })

    const text = await loadFullContext(context, SELF_ID)

    // Per-message 4K clamp plus a 30K total clamp — far below the raw 100K.
    assert.ok(text.length < 40_000)
    assert.ok(text.includes('[truncated]'))
  })

  test('total clamp keeps the NEWEST recent messages and drops the oldest head', async () => {
    const store = makeMockStore()
    for (let i = 0; i < 20; i += 1) {
      store.addMessage(SELF_ID, {
        role: 'assistant',
        content: `MSG-${i}-` + 'y'.repeat(100),
        metadata: { logKind: 'think' },
      })
    }
    const context = makeContext({ store })

    const text = await loadFullContext(context, SELF_ID, { totalLimit: 200 })

    // The newest message survives the total clamp…
    assert.ok(text.includes('MSG-19-'))
    // …while the earliest head of the recent stream is dropped (the old
    // slice(0, max) behaviour kept this instead and cut MSG-19).
    assert.ok(!text.includes('MSG-0-'))
    assert.ok(!text.includes('MSG-5-'))
  })

  test('compact summary stays first AND the newest recent message survives the total clamp', async () => {
    const store = makeMockStore()
    store.addMessage(SELF_ID, {
      role: 'assistant',
      content: 'SUMMARY_HEADER 早期折叠历史',
      metadata: { kind: 'summary' },
    })
    for (let i = 0; i < 20; i += 1) {
      store.addMessage(SELF_ID, {
        role: 'assistant',
        content: `MSG-${i}-` + 'y'.repeat(100),
        metadata: { logKind: 'think' },
      })
    }
    const context = makeContext({ store })

    const text = await loadFullContext(context, SELF_ID, { totalLimit: 200 })

    assert.ok(text.includes('SUMMARY_HEADER'))
    assert.ok(text.includes('MSG-19-'))
    assert.ok(!text.includes('MSG-0-'))
    assert.ok(text.indexOf('SUMMARY_HEADER') < text.indexOf('MSG-19-'))
  })

  test('returns an empty string for an empty history', async () => {
    assert.equal(await loadFullContext(makeContext({ store: makeMockStore() }), SELF_ID), '')
  })

  test('returns an empty string when the store is missing', async () => {
    assert.equal(await loadFullContext(makeContext({}), SELF_ID), '')
  })
})

describe('loadMessages', () => {
  test('returns history as a standard messages array (roles + content, chronological)', async () => {
    const store = makeMockStore()
    store.addMessage(SELF_ID, { role: 'user', content: '你好', metadata: {} })
    store.addMessage(SELF_ID, { role: 'assistant', content: '在的', metadata: { logKind: 'chat' } })
    store.addMessage(SELF_ID, { role: 'assistant', content: '折叠的旧历史摘要', metadata: { kind: 'summary' } })

    const messages = await loadMessages(makeContext({ store }), SELF_ID)

    assert.deepEqual(messages, [
      { role: 'user', content: '你好' },
      { role: 'assistant', content: '在的' },
      { role: 'assistant', content: '[system][compact] 折叠的旧历史摘要' },
    ])
  })

  test('marks heartbeat triggers as [system][heartbeat] while keeping role user', async () => {
    const store = makeMockStore()
    store.addMessage(SELF_ID, {
      role: 'user',
      content: '（heartbeat 醒来）此刻想做什么就做什么。',
      metadata: { kind: 'heartbeat' },
    })

    const messages = await loadMessages(makeContext({ store }), SELF_ID)

    assert.deepEqual(messages, [
      { role: 'user', content: '[system][heartbeat] （heartbeat 醒来）此刻想做什么就做什么。' },
    ])
  })

  test('marks compact summaries as [system][compact] while keeping role assistant', async () => {
    const store = makeMockStore()
    store.addMessage(SELF_ID, { role: 'assistant', content: '旧历史摘要。', metadata: { kind: 'summary' } })

    const messages = await loadMessages(makeContext({ store }), SELF_ID)

    assert.deepEqual(messages, [
      { role: 'assistant', content: '[system][compact] 旧历史摘要。' },
    ])
  })

  test('leaves ordinary user/assistant/tool rows unprefixed', async () => {
    const store = makeMockStore()
    store.addMessage(SELF_ID, { role: 'user', content: '你好', metadata: {} })
    store.addMessage(SELF_ID, { role: 'assistant', content: '在的', metadata: { logKind: 'heartbeat' } })
    store.addMessage(SELF_ID, {
      role: 'assistant',
      content: '[调用工具 workspace_write] 参数=…',
      metadata: { kind: 'tool', toolName: 'workspace_write' },
    })

    const messages = await loadMessages(makeContext({ store }), SELF_ID)

    assert.deepEqual(messages, [
      { role: 'user', content: '你好' },
      { role: 'assistant', content: '在的' },
      { role: 'assistant', content: '[调用工具 workspace_write] 参数=…' },
    ])
  })

  test('skips empty/system rows and returns [] when the store is missing', async () => {
    const store = makeMockStore()
    store.addMessage(SELF_ID, { role: 'assistant', content: '  ', metadata: {} })
    assert.deepEqual(await loadMessages(makeContext({ store }), SELF_ID), [])
    assert.deepEqual(await loadMessages(makeContext({}), SELF_ID), [])
  })
})

describe('recordToolCalls', () => {
  test('appends one assistant kind=tool record per tool result', async () => {
    const store = makeMockStore()
    const context = makeContext({ store })

    const recorded = await recordToolCalls(context, SELF_ID, [
      { name: 'workspace_write', args: { path: 'a.ts' }, isError: false, content: '{"path":"a.ts"}' },
      { name: 'web_search', args: { query: 'x' }, isError: true, content: 'Tool error: boom' },
    ])

    assert.equal(recorded, 2)
    const messages = store.logs.get(SELF_ID) ?? []
    assert.equal(messages.length, 2)
    assert.equal(messages[0]?.role, 'assistant')
    assert.equal((messages[0]?.metadata as { kind?: string; toolName?: string })?.kind, 'tool')
    assert.equal((messages[0]?.metadata as { toolName?: string })?.toolName, 'workspace_write')
    assert.match(messages[0]?.content ?? '', /workspace_write/)
    assert.match(messages[1]?.content ?? '', /出错/)
  })

  test('is a no-op for an empty result list or a missing store', async () => {
    assert.equal(await recordToolCalls(makeContext({ store: makeMockStore() }), SELF_ID, []), 0)
    assert.equal(await recordToolCalls(makeContext({}), SELF_ID, [
      { name: 'web_search', args: {}, isError: false, content: 'ok' },
    ]), 0)
  })
})

describe('appendChatlog / chatlogBlobKey', () => {
  test('appends timestamped entries to the same chatlog file for the same day', async () => {
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const context = makeContext({})
    const at = new Date('2026-09-11T12:00:00Z')

    const key1 = await appendChatlog(context, SELF_ID, { role: 'user', content: '第一条消息', kind: 'user', ts: at.toISOString() })
    const key2 = await appendChatlog(context, SELF_ID, { role: 'assistant', content: '第二条消息', kind: 'assistant', ts: at.toISOString() })

    assert.equal(key1, key2)
    assert.equal(key1, 'chatlog/2026-09-11.md')
    assert.ok(key1.startsWith(CHATLOG_DIR))
    const content = blob.blobMap.get(key1) ?? ''
    assert.ok(content.includes('第一条消息'))
    assert.ok(content.includes('第二条消息'))
    assert.ok(content.indexOf('第一条消息') < content.indexOf('第二条消息'))
    assert.match(content, /- \[\d{4}-\d{2}-\d{2}T/)
  })

  test('writes different files for different days', async () => {
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const context = makeContext({})

    await appendChatlog(context, SELF_ID, { role: 'user', content: '周一', kind: 'user', ts: '2026-09-10T12:00:00Z' })
    await appendChatlog(context, SELF_ID, { role: 'user', content: '周二', kind: 'user', ts: '2026-09-11T12:00:00Z' })

    assert.ok(blob.blobMap.has('chatlog/2026-09-10.md'))
    assert.ok(blob.blobMap.has('chatlog/2026-09-11.md'))
  })

  test('archives tool / heartbeat / summary labels with kind preserved', async () => {
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const context = makeContext({})

    await appendChatlog(context, SELF_ID, { role: 'user', content: '（heartbeat 醒来）', kind: 'heartbeat', ts: '2026-09-11T12:00:00Z' })
    await appendChatlog(context, SELF_ID, { role: 'assistant', content: '早期摘要', kind: 'summary', ts: '2026-09-11T12:01:00Z' })
    await appendChatlog(context, SELF_ID, { role: 'assistant', content: '[调用工具 web_search]', kind: 'tool', ts: '2026-09-11T12:02:00Z' })

    const content = blob.blobMap.get('chatlog/2026-09-11.md') ?? ''
    assert.match(content, /heartbeat: （heartbeat 醒来）/)
    assert.match(content, /summary: 早期摘要/)
    assert.match(content, /tool: \[调用工具 web_search\]/)
  })

  test('CRLF content round-trips through append + load without corrupting neighbours (P1-1)', async () => {
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const context = makeContext({})

    // curl / Windows clients POST message bodies full of \r\n; the writer must
    // normalize them so the parser (whose `.` never matches \r) still reads the
    // row back as ONE message instead of silently dropping it.
    await appendChatlog(context, SELF_ID, {
      role: 'user',
      content: '第一行\r\n第二行\r\n第三行',
      kind: 'user',
      ts: '2026-09-11T12:00:00Z',
    })
    await appendChatlog(context, SELF_ID, {
      role: 'assistant',
      content: '相邻消息',
      kind: 'assistant',
      ts: '2026-09-11T12:01:00Z',
    })

    // The written archive must be canonical LF — no stray \r to trip the regex.
    const key = 'chatlog/2026-09-11.md'
    assert.ok(!(blob.blobMap.get(key) ?? '').includes('\r'))

    const messages = await loadChatlog(context, SELF_ID, 3, { at: new Date('2026-09-11T23:00:00Z') })

    // Both messages survive intact and stay separate — no dropped row, no
    // neighbouring-message pollution.
    assert.equal(messages.length, 2)
    assert.deepEqual(messages[0], {
      role: 'user',
      content: '第一行\n第二行\n第三行',
      ts: '2026-09-11T12:00:00Z',
      kind: 'user',
    })
    assert.deepEqual(messages[1], {
      role: 'assistant',
      content: '相邻消息',
      ts: '2026-09-11T12:01:00Z',
      kind: 'assistant',
    })
  })

  test('persistHistory archives CRLF content losslessly too (P1-1)', async () => {
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const store = makeMockStore()
    const context = makeContext({ store })

    await persistHistory(context, SELF_ID, 'user', '多行\r\n内容', { kind: 'user' })

    // persistHistory archives with `nowIso()`, so the entry lands in TODAY's
    // file — read with the default `at` to find it.
    const messages = await loadChatlog(context, SELF_ID, 3)
    assert.equal(messages.length, 1)
    assert.equal(messages[0]?.content, '多行\n内容')
  })
})

describe('loadChatlog', () => {
  test('parses archive lines back into role/content/ts/kind messages', async () => {
    const blob = makeMockBlobStore({
      'chatlog/2026-09-11.md': [
        '- [2026-09-11T10:00:00Z] user: 你好',
        '- [2026-09-11T10:01:00Z] assistant: 在的',
        '- [2026-09-11T10:02:00Z] tool: [调用工具 web_search] 参数={}',
        '- [2026-09-11T10:03:00Z] heartbeat: （heartbeat 醒来）',
        '- [2026-09-11T10:04:00Z] summary: 旧摘要',
      ].join('\n'),
    })
    injectBlobStoreForTesting(blob)
    const context = makeContext({})

    const messages = await loadChatlog(context, SELF_ID, 3, { at: new Date('2026-09-11T23:00:00Z') })

    assert.equal(messages.length, 5)
    assert.deepEqual(messages[0], { role: 'user', content: '你好', ts: '2026-09-11T10:00:00Z', kind: 'user' })
    assert.deepEqual(messages[1], { role: 'assistant', content: '在的', ts: '2026-09-11T10:01:00Z', kind: 'assistant' })
    assert.equal(messages[2]?.kind, 'tool')
    assert.equal(messages[2]?.role, 'assistant')
    assert.equal(messages[3]?.kind, 'heartbeat')
    assert.equal(messages[3]?.role, 'user')
    assert.equal(messages[4]?.kind, 'summary')
    assert.equal(messages[4]?.role, 'assistant')
  })

  test('reads only the most recent N days (chronological order)', async () => {
    const blob = makeMockBlobStore({
      'chatlog/2026-09-10.md': '- [2026-09-10T10:00:00Z] user: 老早的消息',
      'chatlog/2026-09-11.md': '- [2026-09-11T10:00:00Z] user: 昨天的消息',
      'chatlog/2026-09-12.md': '- [2026-09-12T10:00:00Z] user: 今天的消息',
    })
    injectBlobStoreForTesting(blob)
    const context = makeContext({})

    const messages = await loadChatlog(context, SELF_ID, 2, { at: new Date('2026-09-12T12:00:00Z') })

    assert.deepEqual(messages.map((message) => message.content), ['昨天的消息', '今天的消息'])
  })

  test('parses a raw CRLF archive file via the read-side \\r fallback (P1-1)', async () => {
    // Files written by older versions or hand-uploaded with Windows line
    // endings may be CRLF; parseChatlogLines must strip the trailing \r.
    const blob = makeMockBlobStore({
      'chatlog/2026-09-11.md': [
        '- [2026-09-11T10:00:00Z] user: 你好\r',
        '- [2026-09-11T10:01:00Z] assistant: 在的\r',
      ].join('\n'),
    })
    injectBlobStoreForTesting(blob)
    const context = makeContext({})

    const messages = await loadChatlog(context, SELF_ID, 3, { at: new Date('2026-09-11T23:00:00Z') })

    assert.equal(messages.length, 2)
    assert.deepEqual(messages.map((message) => message.content), ['你好', '在的'])
  })

  test('keeps the newest messages when the archive exceeds the total clamp', async () => {
    const big = `- [2026-09-11T10:00:00Z] user: ${'x'.repeat(200_000)}`
    const blob = makeMockBlobStore({
      'chatlog/2026-09-10.md': '- [2026-09-10T10:00:00Z] user: 老早的消息',
      'chatlog/2026-09-11.md': big,
      'chatlog/2026-09-12.md': '- [2026-09-12T10:00:00Z] user: 最新的消息',
    })
    injectBlobStoreForTesting(blob)
    const context = makeContext({})

    const messages = await loadChatlog(context, SELF_ID, 3, { at: new Date('2026-09-12T12:00:00Z') })

    // The total clamp keeps the newest tail — the giant old message is dropped
    // before the newest message ever is.
    assert.ok(messages.some((message) => message.content === '最新的消息'))
    assert.ok(!messages.some((message) => message.content.includes('x'.repeat(200_000))))
  })

  test('returns [] when there are no chatlog files or Blob fails', async () => {
    const context = makeContext({})
    assert.deepEqual(await loadChatlog(context, SELF_ID), [])
  })
})

describe('readChatlogFile', () => {
  test('returns the file content for an existing day and null otherwise', async () => {
    const blob = makeMockBlobStore({ 'chatlog/2026-09-12.md': '- [2026-09-12T10:00:00Z] user: 完整历史' })
    injectBlobStoreForTesting(blob)
    const context = makeContext({})

    assert.equal(await readChatlogFile(context, SELF_ID, '2026-09-12'), '- [2026-09-12T10:00:00Z] user: 完整历史')
    assert.equal(await readChatlogFile(context, SELF_ID, 'not-a-date'), null)
    assert.equal(await readChatlogFile(context, SELF_ID, '2026-13-01'), null)
    assert.equal(await readChatlogFile(context, SELF_ID, '2026-09-01'), null)
  })
})

describe('searchChatlog', () => {
  test('finds matching lines across recent files (case-insensitive, newest first)', async () => {
    const blob = makeMockBlobStore({
      'chatlog/2026-09-11.md': [
        '- [2026-09-11T10:00:00Z] user: 提到一个 ID 计划',
        '- [2026-09-11T11:00:00Z] assistant: 天气不错',
      ].join('\n'),
      'chatlog/2026-09-12.md': [
        '- [2026-09-12T10:00:00Z] user: 还是那个 id 值得做',
        '- [2026-09-12T11:00:00Z] assistant: 睡觉',
      ].join('\n'),
    })
    injectBlobStoreForTesting(blob)
    const context = makeContext({})

    const hits = await searchChatlog(context, SELF_ID, 'ID', 7, { at: new Date('2026-09-12T23:00:00Z') })

    assert.equal(hits.length, 2)
    assert.equal(hits[0]?.day, '2026-09-12')
    assert.equal(hits[1]?.day, '2026-09-11')
    assert.ok(hits[0]?.snippets.some((snippet) => /id/i.test(snippet)))
    assert.ok(hits[1]?.snippets.some((snippet) => /id/i.test(snippet)))
  })

  test('returns no hits for an absent or empty keyword', async () => {
    const blob = makeMockBlobStore({ 'chatlog/2026-09-12.md': '- [2026-09-12T10:00:00Z] user: 平平无奇', })
    injectBlobStoreForTesting(blob)
    const context = makeContext({})

    assert.deepEqual(await searchChatlog(context, SELF_ID, '量子', 14), [])
    assert.deepEqual(await searchChatlog(context, SELF_ID, '   ', 14), [])
  })

  test('limits the search window to the most recent N days', async () => {
    const blob = makeMockBlobStore({
      'chatlog/2026-09-10.md': '- [2026-09-10T10:00:00Z] user: 目标词 老记录',
      'chatlog/2026-09-11.md': '- [2026-09-11T10:00:00Z] user: 目标词 新记录',
      'chatlog/2026-09-12.md': '- [2026-09-12T10:00:00Z] user: 目标词 最新记录',
    })
    injectBlobStoreForTesting(blob)
    const context = makeContext({})

    const hits = await searchChatlog(context, SELF_ID, '目标词', 1, { at: new Date('2026-09-12T12:00:00Z') })

    assert.equal(hits.length, 1)
    assert.equal(hits[0]?.day, '2026-09-12')
  })
})

describe('persistHistory', () => {
  test('double-writes: appends to the store AND archives to the chatlog', async () => {
    const store = makeMockStore()
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const context = makeContext({ store })

    await persistHistory(context, SELF_ID, 'user', '你好')
    await persistHistory(context, SELF_ID, 'assistant', '在的')

    assert.equal(store.messageLog.length, 2)
    assert.equal(store.messageLog[0]?.role, 'user')
    assert.equal(store.messageLog[1]?.role, 'assistant')

    const todayKey = `chatlog/${dateKey(new Date())}.md`
    const archived = blob.blobMap.get(todayKey) ?? ''
    assert.match(archived, /user: 你好/)
    assert.match(archived, /assistant: 在的/)
    assert.ok(archived.indexOf('你好') < archived.indexOf('在的'))
  })

  test('metadata is written to the store for kind=tool records', async () => {
    const store = makeMockStore()
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const context = makeContext({ store })

    await persistHistory(context, SELF_ID, 'assistant', '调用工具', { kind: 'tool', metadata: { kind: 'tool', toolName: 'web_search' } })

    const record = store.messageLog[0]
    assert.equal((record?.metadata as { kind?: string; toolName?: string })?.kind, 'tool')
    assert.equal((record?.metadata as { toolName?: string })?.toolName, 'web_search')
    assert.ok(blob.blobMap.get(`chatlog/${dateKey(new Date())}.md`)?.includes('tool: 调用工具'))
  })

  test('metadata is passed through for heartbeat and summary kinds', async () => {
    const store = makeMockStore()
    injectBlobStoreForTesting(makeMockBlobStore())
    const context = makeContext({ store })

    await persistHistory(context, SELF_ID, 'user', '触发', { kind: 'heartbeat', metadata: { kind: 'heartbeat' } })
    await persistHistory(context, SELF_ID, 'assistant', '摘要', { kind: 'summary', metadata: { kind: 'summary' } })

    assert.equal((store.messageLog[0]?.metadata as { kind?: string })?.kind, 'heartbeat')
    assert.equal((store.messageLog[1]?.metadata as { kind?: string })?.kind, 'summary')
  })

  test('Blob failure degrades without breaking the store write', async () => {
    const store = makeMockStore()
    // No blob injected → getBlobStore() throws inside appendChatlog.
    const context = makeContext({ store })

    await persistHistory(context, SELF_ID, 'assistant', '即使没有 Blob 也要落店')

    assert.equal(store.messageLog.length, 1)
    assert.equal(store.messageLog[0]?.content, '即使没有 Blob 也要落店')
  })

  test('throws when the store is missing (matches the old appendMessage guard)', async () => {
    await assert.rejects(
      persistHistory(makeContext({}), SELF_ID, 'user', 'x'),
      /Store is not available/,
    )
  })
})
