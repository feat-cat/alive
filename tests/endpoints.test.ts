/**
 * End-to-end endpoint tests for chat/stop (the remaining routable endpoints;
 * heartbeat has its own test file and think/dream/play endpoints were removed
 * with the free-form heartbeat redesign).
 *
 * A mock `MakersContext` (store + env) drives each handler while
 * `globalThis.fetch` is mocked to simulate the AI Gateway. All mocks are
 * restored in `afterEach`.
 */
import { afterEach, describe, test } from 'node:test'
import assert from 'node:assert/strict'
import { mock } from 'node:test'
import { onRequest as chatOnRequest, runChat } from '../agents/chat.ts'
import { onRequest as stopOnRequest, runStop } from '../agents/stop.ts'
import { SELF_ID, type Env } from '../agents/_shared.ts'
import { gatewayEnv, makeContext, makeMockStore } from './_helpers.ts'

function llmTextResponse(content: string): unknown {
  return { choices: [{ message: { content } }] }
}

/** Mock the AI gateway and expose how many LLM fetches actually happened. */
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

afterEach(() => {
  mock.restoreAll()
})

describe('POST /chat', () => {
  test('main path: appends user + assistant messages and returns a reply', async () => {
    const store = makeMockStore()
    mockGateway(llmTextResponse('你好呀，有什么可以帮你？'))

    const result = await runChat(makeContext({ store, env: gatewayEnv() as Env }), {
      message: '你好',
    })

    assert.equal(result.reply, '你好呀，有什么可以帮你？')
    assert.equal(result.conversationId, SELF_ID)
    assert.equal(store.messageLog.at(-2)?.role, 'user')
    assert.equal(store.messageLog.at(-1)?.role, 'assistant')
  })

  test('messages passed to the LLM are a standard array with independent history entries', async () => {
    const store = makeMockStore()
    store.addMessage(SELF_ID, { role: 'assistant', content: '早先的回复', metadata: { logKind: 'heartbeat' } })
    store.addMessage(SELF_ID, { role: 'user', content: '早先的提问', metadata: {} })
    const bodies: Array<Record<string, unknown>> = []
    mock.method(globalThis, 'fetch', async (_input: unknown, init?: RequestInit) => {
      if (init?.body) bodies.push(JSON.parse(String(init.body)) as Record<string, unknown>)
      return new Response(JSON.stringify(llmTextResponse('好的。')), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      })
    })

    const res = await chatOnRequest(makeContext({
      store,
      env: gatewayEnv() as Env,
      body: { message: '今天做什么？' },
    }))

    assert.equal(res.status, 200)
    assert.ok(bodies.length >= 1)
    const messages = (bodies[0] as { messages: Array<{ role: string; content: string }> }).messages
    assert.ok(Array.isArray(messages))
    // Independent role entries — NOT a single clamped user text block.
    assert.ok(messages.some((message) => message.role === 'system'))
    assert.ok(messages.some((message) => message.role === 'assistant' && message.content === '早先的回复'))
    assert.ok(messages.some((message) => message.role === 'user' && message.content === '早先的提问'))
    assert.ok(messages.some((message) => message.role === 'user' && message.content === '今天做什么？'))
    // Exactly one system message, always at index 0 (heartbeat/compact history
    // never injects another system row).
    const systemMessages = messages.filter((message) => message.role === 'system')
    assert.equal(systemMessages.length, 1)
    assert.equal(messages[0]?.role, 'system')
    const system = messages[0]?.content ?? ''
    assert.match(system, /\[system\]\[heartbeat\]/)
    assert.match(system, /\[system\]\[compact\]/)
    assert.match(system, /不是用户说的/)
    // The store gained the new user message + the assistant reply.
    assert.equal(store.messageLog.at(-1)?.role, 'assistant')
  })

  test('system prompt tells the model only web_search is available in chat', async () => {
    const store = makeMockStore()
    const bodies: Array<Record<string, unknown>> = []
    mock.method(globalThis, 'fetch', async (_input: unknown, init?: RequestInit) => {
      if (init?.body) bodies.push(JSON.parse(String(init.body)) as Record<string, unknown>)
      return new Response(JSON.stringify(llmTextResponse('好的。')), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      })
    })

    const res = await chatOnRequest(makeContext({
      store,
      env: gatewayEnv() as Env,
      body: { message: '你好' },
    }))

    assert.equal(res.status, 200)
    assert.ok(bodies.length >= 1)
    const messages = (bodies[0] as { messages: Array<{ role: string; content: string }> }).messages
    const system = messages.find((message) => message.role === 'system')?.content ?? ''
    // The AI must never call diary_*/chatlog_*/blob_*/workspace_* here just
    // because the MEMORY.md seed mentions them.
    assert.match(system, /只有 web_search/)
    assert.match(system, /不在此会话提供/)
  })

  test('rejects an empty message', async () => {
    const store = makeMockStore()
    await assert.rejects(
      runChat(makeContext({ store }), { message: '   ' }),
      /message is required/,
    )
  })

  test('onRequest reads message from the request body', async () => {
    const store = makeMockStore()
    mockGateway(llmTextResponse('收到。'))
    const context = makeContext({ store, env: gatewayEnv() as Env, body: { message: '测试' } })
    const res = await chatOnRequest(context)
    assert.equal(res.status, 200)
    const body = (await res.json()) as { ok: boolean; reply: string }
    assert.equal(body.ok, true)
    assert.equal(body.reply, '收到。')
  })
})

describe('POST /stop', () => {
  test('reports aborted when the platform util confirms it', async () => {
    const context = makeContext({ body: { conversation_id: 'eo-self' }, abortActiveRun: async () => ({ aborted: true }) })
    const res = await stopOnRequest(context)
    assert.equal(res.status, 200)
    const body = (await res.json()) as { ok: boolean; aborted: boolean }
    assert.equal(body.ok, true)
    assert.equal(body.aborted, true)
  })

  test('returns 400 when conversation_id is missing', async () => {
    const res = await stopOnRequest(makeContext({ body: {} }))
    assert.equal(res.status, 400)
    const body = (await res.json()) as { ok: boolean; error: string }
    assert.match(body.error, /conversation_id is required/)
  })

  test('runStop without utils reports not aborted', async () => {
    const result = await runStop(makeContext({}), 'eo-self')
    assert.equal(result.aborted, false)
  })
})