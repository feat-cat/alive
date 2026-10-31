/**
 * Tests for the three new chat/heartbeat features:
 *  ① user-message resend/correction merge (`detectMessageMerge` + `levenshtein`),
 *  ② interrupt message queue (turn-active flag + drain into the next LLM call),
 *  ③ decline_response (the model's legal "no text" exit).
 *
 * The AI Gateway fetch is mocked (blocking JSON bodies and SSE streams); Blob
 * is injected with an in-memory store so the queue / turn-active / chatlog keys
 * are exercised against the real helper code.
 */
import { beforeEach, afterEach, describe, test } from 'node:test'
import assert from 'node:assert/strict'
import { mock } from 'node:test'
import { injectBlobStoreForTesting } from '../agents/_blob-tools.ts'
import { onRequest as chatOnRequest, runChat, detectMessageMerge, QUEUED_MESSAGE } from '../agents/chat.ts'
import { runHeartbeat } from '../agents/heartbeat.ts'
import { setLlmRetryForTesting } from '../agents/_llm.ts'
import {
  activeTurnBlobKey,
  dateKey,
  queueBlobKey,
  setActiveTurn,
  getActiveTurn,
  clearActiveTurn,
  drainQueue,
} from '../agents/_memory.ts'
import { levenshtein, SELF_ID, type Env } from '../agents/_shared.ts'
import { SELF_STATE_KEY, type AgentState } from '../agents/_state.ts'
import {
  gatewayEnv,
  makeContext,
  makeMockBlobStore,
  makeMockStore,
  type MockBlobStore,
  type MockStore,
} from './_helpers.ts'

/** Instant retries: retries still happen but never sleep. */
const INSTANT_RETRY = { maxRetries: 3, backoffMs: [0, 0, 0] as const }

beforeEach(() => {
  setLlmRetryForTesting(INSTANT_RETRY)
})

afterEach(() => {
  mock.restoreAll()
  injectBlobStoreForTesting(null)
  setLlmRetryForTesting(null)
})

function llmTextResponse(content: string): unknown {
  return { choices: [{ message: { content } }] }
}

function llmToolCallResponse(name: string, args: Record<string, unknown>): unknown {
  return {
    choices: [
      {
        message: {
          content: '',
          tool_calls: [
            { id: 'call_1', type: 'function', function: { name, arguments: JSON.stringify(args) } },
          ],
        },
      },
    ],
  }
}

/** Mock the blocking AI gateway and expose how many LLM fetches happened. */
function mockGateway(...payloads: unknown[]): { calls: number } {
  let index = 0
  const counter = { calls: 0 }
  mock.method(globalThis, 'fetch', async () => {
    counter.calls += 1
    const payload = payloads[Math.min(index, payloads.length - 1)]
    index += 1
    return new Response(JSON.stringify(payload), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    })
  })
  return counter
}

/** Parse today's chatlog JSONL into records. */
function chatlogRecords(blob: MockBlobStore): Array<Record<string, unknown>> {
  const raw = blob.blobMap.get(`chatlog/${dateKey(new Date())}.jsonl`) ?? ''
  return raw
    .split('\n')
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line) as Record<string, unknown>)
}

function userRows(store: MockStore): MockStore['messageLog'] {
  return store.messageLog.filter((m) => m.role === 'user' && (m.metadata as { kind?: string } | undefined)?.kind !== 'heartbeat')
}

function defaultSelfState(): AgentState {
  return { lastActivityAt: 0, created: 1 }
}

/* ------------------------------------------------------------------ */
/* levenshtein (shared)                                                */
/* ------------------------------------------------------------------ */

describe('levenshtein', () => {
  test('counts single-character substitutions', () => {
    assert.equal(levenshtein('横', '很'), 1)
    assert.equal(levenshtein('abc', 'abd'), 1)
    assert.equal(levenshtein('今天天气横好', '今天天气很好'), 1)
  })

  test('handles empty strings and insertions', () => {
    assert.equal(levenshtein('', ''), 0)
    assert.equal(levenshtein('abc', ''), 3)
    assert.equal(levenshtein('', 'abc'), 3)
    assert.equal(levenshtein('a', 'ab'), 1)
  })

  test('classic kitten/sitting distance', () => {
    assert.equal(levenshtein('kitten', 'sitting'), 3)
  })

  test('is symmetric', () => {
    assert.equal(levenshtein('今天天气横好', '今天天气很好'), levenshtein('今天天气很好', '今天天气横好'))
  })
})

/* ------------------------------------------------------------------ */
/* detectMessageMerge (pure)                                           */
/* ------------------------------------------------------------------ */

describe('detectMessageMerge', () => {
  const now = Date.now()

  test('Case 1: a similar resend within the window is a replace', () => {
    assert.equal(detectMessageMerge('今天天气横好', '今天天气很好', now, now + 3000), 'replace')
  })

  test('Case 2: a short correction fragment is a fragment', () => {
    assert.equal(detectMessageMerge('今天天气横好', '很', now, now + 2000), 'fragment')
  })

  test('Case 3: outside the window is a new message', () => {
    assert.equal(detectMessageMerge('今天天气横好', '今天天气很好', now, now + 5 * 60 * 1000), null)
  })

  test('two unrelated same-length messages never merge', () => {
    assert.equal(detectMessageMerge('早先的问题', '现在是什么', now, now + 1000), null)
  })

  test('empty text never merges', () => {
    assert.equal(detectMessageMerge('', 'x', now, now + 1000), null)
    assert.equal(detectMessageMerge('x', '', now, now + 1000), null)
  })

  test('injectable window controls the merge decision', () => {
    assert.equal(detectMessageMerge('abc', 'abd', now, now + 1000, 500), null)
    assert.equal(detectMessageMerge('abc', 'abd', now, now + 1000, 2000), 'replace')
  })
})

/* ------------------------------------------------------------------ */
/* Feature ①: endpoint-level merge                                     */
/* ------------------------------------------------------------------ */

describe('POST /chat message merge', () => {
  test('Case 1: replace — B supersedes A, AI answers once', async () => {
    const store = makeMockStore()
    store.addMessage(SELF_ID, { role: 'user', content: '今天天气横好' })
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const gateway = mockGateway(llmTextResponse('好的，天气不错。'))

    const result = await runChat(makeContext({ store, env: gatewayEnv() as Env }), { message: '今天天气很好' })

    assert.equal(result.reply, '好的，天气不错。')
    assert.equal(gateway.calls, 1)
    // Store holds exactly ONE user row — B (A was superseded).
    const users = userRows(store)
    assert.equal(users.length, 1)
    assert.equal(users[0]?.content, '今天天气很好')
    // chatlog: A (superseded) + B (merged).
    const records = chatlogRecords(blob)
    assert.ok(records.some((r) => r.content === '今天天气横好' && (r.metadata as Record<string, unknown>)?.superseded === true))
    assert.ok(records.some((r) => r.content === '今天天气很好' && (r.metadata as Record<string, unknown>)?.merged === true))
  })

  test('Case 2: fragment — B is archived only, AI answers A once', async () => {
    const store = makeMockStore()
    store.addMessage(SELF_ID, { role: 'user', content: '今天天气横好' })
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const gateway = mockGateway(llmTextResponse('好。'))

    const result = await runChat(makeContext({ store, env: gatewayEnv() as Env }), { message: '很' })

    assert.equal(result.reply, '好。')
    assert.equal(gateway.calls, 1)
    // Store keeps A only — no new user row for the fragment.
    const users = userRows(store)
    assert.equal(users.length, 1)
    assert.equal(users[0]?.content, '今天天气横好')
    // chatlog has the fragment with mergeFragment:true.
    const records = chatlogRecords(blob)
    assert.ok(records.some((r) => r.content === '很' && (r.metadata as Record<string, unknown>)?.mergeFragment === true))
  })

  test('Case 3: a message outside the window starts its own turn', async () => {
    const store = makeMockStore()
    store.addMessage(SELF_ID, { role: 'user', content: '今天天气横好' })
    // Age A beyond the 30s window.
    const a = store.messageLog[0]
    if (a) a.createdAt = new Date(Date.now() - 5 * 60 * 1000).toISOString()
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const gateway = mockGateway(llmTextResponse('收到。'))

    const result = await runChat(makeContext({ store, env: gatewayEnv() as Env }), { message: '今天天气很好' })

    assert.equal(result.reply, '收到。')
    assert.equal(gateway.calls, 1)
    // Store now has BOTH A and B as independent user rows.
    const users = userRows(store)
    assert.equal(users.length, 2)
    assert.equal(users[0]?.content, '今天天气横好')
    assert.equal(users[1]?.content, '今天天气很好')
  })
})

/* ------------------------------------------------------------------ */
/* Feature ②: interrupt queue                                          */
/* ------------------------------------------------------------------ */

describe('interrupt message queue', () => {
  test('a message arriving mid-turn is queued, no new turn starts', async () => {
    const store = makeMockStore()
    const blob = makeMockBlobStore()
    const now = Date.now()
    blob.blobMap.set(activeTurnBlobKey(SELF_ID), JSON.stringify({ startedAt: now, expiresAt: now + 120_000 }))
    injectBlobStoreForTesting(blob)
    const gateway = mockGateway(llmTextResponse('不应出现'))

    const result = await runChat(makeContext({ store, env: gatewayEnv() as Env }), { message: '插嘴' })

    assert.equal(result.queued, true)
    assert.equal(result.message, QUEUED_MESSAGE)
    assert.equal(gateway.calls, 0)
    // The queue blob holds the message.
    const raw = blob.blobMap.get(queueBlobKey(SELF_ID))
    assert.ok(raw)
    const items = JSON.parse(raw) as Array<{ text: string }>
    assert.equal(items[0]?.text, '插嘴')
    // The store is untouched (the message is not a user row yet).
    assert.equal(store.messageLog.length, 0)
  })

  test('SSE path emits a queued event instead of an error', async () => {
    const store = makeMockStore()
    const blob = makeMockBlobStore()
    const now = Date.now()
    blob.blobMap.set(activeTurnBlobKey(SELF_ID), JSON.stringify({ startedAt: now, expiresAt: now + 120_000 }))
    injectBlobStoreForTesting(blob)

    const res = await chatOnRequest(makeContext({ store, env: gatewayEnv() as Env, body: { message: '插嘴' } }))
    assert.equal(res.status, 200)
    const text = await res.text()
    assert.ok(text.includes('"type":"queued"'))
    assert.ok(text.includes(QUEUED_MESSAGE))
  })

  test('a message with no active turn starts a normal turn', async () => {
    const store = makeMockStore()
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const gateway = mockGateway(llmTextResponse('你好'))

    const result = await runChat(makeContext({ store, env: gatewayEnv() as Env }), { message: '正常消息' })

    assert.equal(result.queued, undefined)
    assert.equal(result.reply, '你好')
    assert.equal(gateway.calls, 1)
    const users = userRows(store)
    assert.equal(users.length, 1)
    assert.equal(users[0]?.content, '正常消息')
  })

  test('a merge fragment is not queued even when a turn is active', async () => {
    const store = makeMockStore()
    store.addMessage(SELF_ID, { role: 'user', content: '今天天气横好' })
    const blob = makeMockBlobStore()
    const now = Date.now()
    blob.blobMap.set(activeTurnBlobKey(SELF_ID), JSON.stringify({ startedAt: now, expiresAt: now + 120_000 }))
    injectBlobStoreForTesting(blob)
    const gateway = mockGateway(llmTextResponse('好。'))

    const result = await runChat(makeContext({ store, env: gatewayEnv() as Env }), { message: '很' })

    assert.equal(result.queued, undefined)
    assert.equal(result.reply, '好。')
    assert.equal(gateway.calls, 1)
    // No new user row (fragment archived only), queue untouched.
    assert.equal(userRows(store).length, 1)
    assert.equal(blob.blobMap.has(queueBlobKey(SELF_ID)), false)
  })

  test('drainQueue folds queued messages into the running LLM request and persists them', async () => {
    const store = makeMockStore()
    const blob = makeMockBlobStore()
    blob.blobMap.set(queueBlobKey(SELF_ID), JSON.stringify([{ text: '插嘴B', ts: new Date().toISOString() }]))
    injectBlobStoreForTesting(blob)
    const bodies: Array<Record<string, unknown>> = []
    mock.method(globalThis, 'fetch', async (_input: unknown, init?: RequestInit) => {
      if (init?.body) bodies.push(JSON.parse(String(init.body)) as Record<string, unknown>)
      return new Response(JSON.stringify(llmTextResponse('收到')), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      })
    })

    const result = await runChat(makeContext({ store, env: gatewayEnv() as Env }), { message: '新问题' })

    assert.equal(result.reply, '收到')
    const messages = (bodies[0] as { messages: Array<{ role: string; content: string }> }).messages
    assert.ok(messages.some((m) => m.role === 'user' && m.content === '新问题'))
    assert.ok(messages.some((m) => m.role === 'user' && m.content === '插嘴B'))
    // The store gained both user rows; the queue was consumed.
    const users = userRows(store)
    assert.ok(users.some((m) => m.content === '新问题'))
    assert.ok(users.some((m) => m.content === '插嘴B'))
    assert.equal(blob.blobMap.has(queueBlobKey(SELF_ID)), false)
  })

  test('drainQueue returns [] when the queue is empty', async () => {
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const drained = await drainQueue(makeContext({ store: makeMockStore(), env: gatewayEnv() as Env }), SELF_ID)
    assert.deepEqual(drained, [])
  })

  test('active-turn helpers write / read / clear the marker', async () => {
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const context = makeContext({ store: makeMockStore(), env: gatewayEnv() as Env })
    assert.equal(await getActiveTurn(context, SELF_ID), null)
    await setActiveTurn(context, SELF_ID, { startedAt: Date.now(), expiresAt: Date.now() + 120_000 })
    const active = await getActiveTurn(context, SELF_ID)
    assert.ok(active)
    assert.ok(active.expiresAt > active.startedAt)
    await clearActiveTurn(context, SELF_ID)
    assert.equal(await getActiveTurn(context, SELF_ID), null)
  })

  test('an expired active turn reads as null', async () => {
    const blob = makeMockBlobStore()
    const now = Date.now()
    blob.blobMap.set(activeTurnBlobKey(SELF_ID), JSON.stringify({ startedAt: now - 200_000, expiresAt: now - 100_000 }))
    injectBlobStoreForTesting(blob)
    const active = await getActiveTurn(makeContext({ store: makeMockStore(), env: gatewayEnv() as Env }), SELF_ID)
    assert.equal(active, null)
  })
})

/* ------------------------------------------------------------------ */
/* Feature ③: decline_response                                         */
/* ------------------------------------------------------------------ */

describe('decline_response', () => {
  test('JSON path: ends the turn with no assistant text, records a declined row', async () => {
    const store = makeMockStore()
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const gateway = mockGateway(
      llmToolCallResponse('decline_response', { reason: '超出能力' }),
      llmTextResponse('不应出现'),
    )

    const result = await runChat(makeContext({ store, env: gatewayEnv() as Env }), { message: '敏感问题' })

    assert.equal(result.declined, true)
    assert.equal(result.replied, false)
    assert.equal(result.reply, '')
    // Only the decline round ran — no second LLM request.
    assert.equal(gateway.calls, 1)
    const declined = store.messageLog.filter((m) => (m.metadata as { kind?: string } | undefined)?.kind === 'declined')
    assert.equal(declined.length, 1)
    assert.equal(declined[0]?.content, '（选择不回应：超出能力）')
    assert.ok(chatlogRecords(blob).some((r) => r.kind === 'declined'))
  })

  test('decline without a reason uses the plain marker', async () => {
    const store = makeMockStore()
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    mockGateway(llmToolCallResponse('decline_response', {}))

    const result = await runChat(makeContext({ store, env: gatewayEnv() as Env }), { message: 'x' })

    assert.equal(result.declined, true)
    const declined = store.messageLog.filter((m) => (m.metadata as { kind?: string } | undefined)?.kind === 'declined')
    assert.equal(declined[0]?.content, '（选择不回应）')
  })

  test('SSE path emits a declined event instead of an error', async () => {
    const store = makeMockStore()
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const encoder = new TextEncoder()
    const toolArgs = JSON.stringify({ reason: '敏感' })
    mock.method(globalThis, 'fetch', async () => {
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(
            encoder.encode(
              `data: ${JSON.stringify({ choices: [{ delta: { tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: 'decline_response', arguments: toolArgs } }] } }] })}\n\n`,
            ),
          )
          controller.enqueue(encoder.encode('data: [DONE]\n\n'))
          controller.close()
        },
      })
      return new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } })
    })

    const res = await chatOnRequest(makeContext({ store, env: gatewayEnv() as Env, body: { message: 'x' } }))
    assert.equal(res.status, 200)
    const text = await res.text()
    assert.ok(text.includes('"type":"declined"'))
    assert.ok(text.includes('（选择不回应：敏感）'))
    const declined = store.messageLog.filter((m) => (m.metadata as { kind?: string } | undefined)?.kind === 'declined')
    assert.equal(declined.length, 1)
  })

  test('heartbeat: decline ends silently, records a declined row, writes no diary', async () => {
    const store = makeMockStore({ [SELF_ID]: { [SELF_STATE_KEY]: defaultSelfState() } })
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const gateway = mockGateway(
      llmToolCallResponse('decline_response', {}),
      llmTextResponse('不应出现'),
    )

    const result = await runHeartbeat(makeContext({ store, env: gatewayEnv() as Env }))

    assert.equal(result.text, '')
    assert.equal(result.turns, 1)
    assert.equal(gateway.calls, 1)
    const declined = store.messageLog.filter((m) => (m.metadata as { kind?: string } | undefined)?.kind === 'declined')
    assert.equal(declined.length, 1)
    assert.equal(declined[0]?.content, '（选择不回应）')
    // No heartbeat log (the declined row is the only assistant row).
    const logs = store.messageLog.filter((m) => (m.metadata as { logKind?: string } | undefined)?.logKind === 'heartbeat')
    assert.equal(logs.length, 0)
  })
})
