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
import { CHAT_TIMEOUT_MESSAGE, errorResponse, SELF_ID, type Env } from '../agents/_shared.ts'
import { injectBlobStoreForTesting } from '../agents/_blob-tools.ts'
import { dateKey } from '../agents/_memory.ts'
import { gatewayEnv, makeContext, makeMockBlobStore, makeMockStore } from './_helpers.ts'

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
  injectBlobStoreForTesting(null)
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

    // stream:false forces the JSON path so the request body can be inspected
    // directly (the streaming path lazily fetches only when the response body
    // is drained).
    const res = await chatOnRequest(makeContext({
      store,
      env: gatewayEnv() as Env,
      body: { message: '今天做什么？', stream: false },
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

  test('P0 regression: the current user message ALWAYS reaches the gateway body, even past the store read cap', async () => {
    // The real platform store caps `getMessages({ limit })` at 100 and THROWS
    // `MemoryValidationError` for a larger value. loadMessages used to pass
    // `limit: STORE_MESSAGE_LIMIT (10000)`, swallow the throw into `[]`, and
    // feed the model a system-only array — the user message never entered the
    // model context and it hallucinated an input. Now loadMessages reads the
    // NEWEST page inside the platform limit, so the just-persisted user message
    // (always the newest row) is guaranteed to be in the gateway request.
    const store = makeMockStore()
    for (let i = 0; i < 150; i += 1) {
      store.addMessage(SELF_ID, { role: 'assistant', content: `old-${i}`, metadata: { logKind: 'heartbeat' } })
    }
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
      body: { message: '今天做什么？', stream: false },
    }))

    assert.equal(res.status, 200)
    assert.ok(bodies.length >= 1)
    const messages = (bodies[0] as { messages: Array<{ role: string; content: string }> }).messages
    const freshUser = messages.filter((message) => message.role === 'user' && message.content === '今天做什么？')
    assert.equal(freshUser.length, 1, 'the just-sent user message must be fed to the model')
    // The model sees the newest history within the platform read cap, never the
    // full 150-message dump and never an empty array.
    assert.ok(messages.some((message) => message.role === 'assistant' && message.content === 'old-149'))
    assert.ok(!messages.some((message) => message.role === 'assistant' && message.content === 'old-0'))
  })

  test('chat system message leads with the conversation-mode section, not heartbeat solitude', async () => {
    const store = makeMockStore()
    const bodies: Array<Record<string, unknown>> = []
    mock.method(globalThis, 'fetch', async (_input: unknown, init?: RequestInit) => {
      if (init?.body) bodies.push(JSON.parse(String(init.body)) as Record<string, unknown>)
      return new Response(JSON.stringify(llmTextResponse('好啊，你最近在忙什么？')), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      })
    })

    const res = await chatOnRequest(makeContext({
      store,
      env: gatewayEnv() as Env,
      body: { message: '你可以问我一些想了解的问题', stream: false },
    }))

    assert.equal(res.status, 200)
    assert.ok(bodies.length >= 1)
    const messages = (bodies[0] as { messages: Array<{ role: string; content: string }> }).messages
    const system = messages[0]?.content ?? ''
    // A chat turn is explicitly framed as "someone is talking to you" — the
    // exact antidote to the heartbeat solitude narrative.
    assert.match(system, /有人正在和你说话/)
    // The conversation-mode section is an attitude demonstration (OpenClaw
    // SOUL.md spirit), not a fill-in-the-blank question template: have an
    // opinion, skip the pleasantries, stay terse, call out nonsense.
    assert.match(system, /有观点/)
    assert.match(system, /别客套/)
    assert.match(system, /万能问题/)
    assert.match(system, /敢说破/)
    assert.match(system, /少用工具，多用好奇/)
    // Ordering: wall-clock persona first, then the conversation-mode section,
    // then the [system]-marker history guidance.
    assert.ok(system.indexOf('现在是') < system.indexOf('有人正在和你说话'))
    assert.ok(system.indexOf('有人正在和你说话') < system.indexOf('[system]'))
  })

  test('chat registers the full tool set (blob + workspace + diary + chatlog + web_search)', async () => {
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
      body: { message: '你好', stream: false },
    }))

    assert.equal(res.status, 200)
    assert.ok(bodies.length >= 1)
    // The full heartbeat registry is sent, not just web_search — chat can both
    // talk and act (blob/diary/chatlog/workspace/search).
    const tools = (bodies[0] as { tools?: Array<{ function: { name: string } }> }).tools ?? []
    const names = tools.map((tool) => tool.function.name)
    assert.ok(names.includes('web_search'))
    assert.ok(names.includes('blob_read'))
    assert.ok(names.includes('blob_write'))
    assert.ok(names.includes('workspace_list'))
    assert.ok(names.includes('workspace_write'))
    assert.ok(names.includes('diary_append'))
    assert.ok(names.includes('diary_read'))
    assert.ok(names.includes('chatlog_search'))
    assert.ok(names.includes('chatlog_read'))
    assert.ok(names.length >= 10)
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
    const context = makeContext({ store, env: gatewayEnv() as Env, body: { message: '测试', stream: false } })
    const res = await chatOnRequest(context)
    assert.equal(res.status, 200)
    assert.match(res.headers.get('content-type') ?? '', /application\/json/)
    const body = (await res.json()) as { ok: boolean; reply: string }
    assert.equal(body.ok, true)
    assert.equal(body.reply, '收到。')
    // History is still written by the JSON path.
    assert.equal(store.messageLog.at(-1)?.role, 'assistant')
    assert.equal(store.messageLog.at(-1)?.content, '收到。')
  })

  test('streaming path: text deltas arrive as SSE ai_response events and history persists the full text', async () => {
    const store = makeMockStore()
    const encoder = new TextEncoder()
    const payloads = [
      { choices: [{ delta: { content: '你' } }] },
      { choices: [{ delta: { content: '好' } }] },
      { choices: [{ delta: { content: '呀' } }] },
    ]
    mock.method(globalThis, 'fetch', async () => {
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          for (const payload of payloads) {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(payload)}\n\n`))
          }
          controller.enqueue(encoder.encode('data: [DONE]\n\n'))
          controller.close()
        },
      })
      return new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } })
    })

    const res = await chatOnRequest(makeContext({ store, env: gatewayEnv() as Env, body: { message: '你好' } }))
    assert.equal(res.status, 200)
    assert.match(res.headers.get('content-type') ?? '', /^text\/event-stream/)
    const bodyText = await res.text()
    assert.ok(bodyText.includes('data: [DONE]'))
    assert.ok(bodyText.includes('"type":"ai_response"'))
    assert.ok(bodyText.includes('"content":"你"'))
    assert.ok(bodyText.includes('"content":"好"'))
    assert.ok(bodyText.includes('"content":"呀"'))
    // A pure content stream must not fabricate reasoning_delta events.
    assert.ok(!bodyText.includes('"type":"reasoning_delta"'))
    // The client saw per-token deltas; history stored the accumulated reply.
    assert.equal(store.messageLog.at(-1)?.role, 'assistant')
    assert.equal(store.messageLog.at(-1)?.content, '你好呀')
  })

  test('streaming path: reasoning deltas arrive as reasoning_delta SSE events and are archived into the chatlog JSON record', async () => {
    const store = makeMockStore()
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const encoder = new TextEncoder()
    const payloads = [
      { choices: [{ delta: { reasoning_content: '先' } }] },
      { choices: [{ delta: { reasoning_content: '思考' } }] },
      { choices: [{ delta: { content: '你好' } }] },
    ]
    mock.method(globalThis, 'fetch', async () => {
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          for (const payload of payloads) {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(payload)}\n\n`))
          }
          controller.enqueue(encoder.encode('data: [DONE]\n\n'))
          controller.close()
        },
      })
      return new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } })
    })

    const res = await chatOnRequest(makeContext({ store, env: gatewayEnv() as Env, body: { message: '你好' } }))
    assert.equal(res.status, 200)
    assert.match(res.headers.get('content-type') ?? '', /^text\/event-stream/)
    const bodyText = await res.text()
    assert.ok(bodyText.includes('"type":"reasoning_delta"'))
    assert.ok(bodyText.includes('"content":"先"'))
    assert.ok(bodyText.includes('"content":"思考"'))
    assert.ok(bodyText.includes('"type":"ai_response"'))
    assert.ok(bodyText.includes('data: [DONE]'))
    // The store row stays concise — content only, no reasoning field.
    assert.equal(store.messageLog.at(-1)?.role, 'assistant')
    assert.equal(store.messageLog.at(-1)?.content, '你好')
    assert.equal(store.messageLog.at(-1)?.metadata?.reasoningContent, undefined)
    // The JSON chatlog record carries the full accumulated thinking.
    const assistantRecord = (blob.blobMap.get(`chatlog/${dateKey(new Date())}.jsonl`) ?? '')
      .split('\n')
      .filter((line) => line.trim())
      .map((line) => JSON.parse(line) as { role: string; kind: string; content: string; reasoningContent?: string })
      .find((record) => record.kind === 'assistant')
    assert.ok(assistantRecord, 'assistant record is archived')
    assert.equal(assistantRecord?.content, '你好')
    assert.equal(assistantRecord?.reasoningContent, '先思考')
  })

  test('streaming path: reasoning is archived per-round with turn, not concatenated into one record', async () => {
    const store = makeMockStore()
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const encoder = new TextEncoder()
    const toolArgs = JSON.stringify({ query: 'x' })
    let fetchCalls = 0
    mock.method(globalThis, 'fetch', async () => {
      fetchCalls += 1
      const stream = (chunks: unknown[]) => {
        const body = new ReadableStream<Uint8Array>({
          start(controller) {
            for (const chunk of chunks) controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`))
            controller.enqueue(encoder.encode('data: [DONE]\n\n'))
            controller.close()
          },
        })
        return new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } })
      }
      if (fetchCalls === 1) {
        return stream([
          { choices: [{ delta: { reasoning_content: '第一轮思考' } }] },
          { choices: [{ delta: { content: '好的，' } }] },
          { choices: [{ delta: { tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: 'web_search', arguments: toolArgs } }] } }] },
        ])
      }
      return stream([
        { choices: [{ delta: { reasoning_content: '第二轮思考' } }] },
        { choices: [{ delta: { content: '查完了。' } }] },
      ])
    })

    const res = await chatOnRequest(makeContext({ store, env: gatewayEnv() as Env, body: { message: '查一下' } }))
    assert.equal(res.status, 200)
    await res.text()
    assert.equal(fetchCalls, 2)

    const assistantRecords = (blob.blobMap.get(`chatlog/${dateKey(new Date())}.jsonl`) ?? '')
      .split('\n')
      .filter((line) => line.trim())
      .map((line) => JSON.parse(line) as { kind?: string; content: string; reasoningContent?: string; turn?: number })
      .filter((record) => record.kind === 'assistant' || record.reasoningContent)
    assert.equal(assistantRecords.length, 2, 'each round archives its own assistant record')
    assert.deepEqual(assistantRecords.map((record) => ({ content: record.content, reasoningContent: record.reasoningContent, turn: record.turn })), [
      { content: '好的，', reasoningContent: '第一轮思考', turn: 1 },
      { content: '好的，查完了。', reasoningContent: '第二轮思考', turn: 2 },
    ])
    // The tool record also carries the round it ran in.
    const toolRecord = (blob.blobMap.get(`chatlog/${dateKey(new Date())}.jsonl`) ?? '')
      .split('\n')
      .filter((line) => line.trim())
      .map((line) => JSON.parse(line) as { kind?: string; turn?: number })
      .find((record) => record.kind === 'tool')
    assert.equal(toolRecord?.turn, 1)
    // The store still grows only the real tool rows + the one final reply.
    const toolRows = store.messageLog.filter((row) => row.role === 'assistant' && row.metadata?.kind === 'tool')
    assert.equal(toolRows.length, 1)
    assert.equal(store.messageLog.at(-1)?.role, 'assistant')
    assert.equal(store.messageLog.at(-1)?.content, '好的，查完了。')
  })

  test('JSON path (stream:false): assistant reasoning from the gateway is archived into the chatlog JSON record', async () => {
    const store = makeMockStore()
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    mock.method(globalThis, 'fetch', async () =>
      new Response(
        JSON.stringify({ choices: [{ message: { content: '非流式回复', reasoning_content: '非流式思考' } }] }),
        { status: 200 },
      ))

    const res = await chatOnRequest(makeContext({
      store,
      env: gatewayEnv() as Env,
      body: { message: '你好', stream: false },
    }))
    assert.equal(res.status, 200)
    const body = (await res.json()) as { ok: boolean; reply: string }
    assert.equal(body.reply, '非流式回复')
    // The store row stays concise; the JSON chatlog record carries the thinking.
    assert.equal(store.messageLog.at(-1)?.content, '非流式回复')
    assert.equal(store.messageLog.at(-1)?.metadata?.reasoningContent, undefined)
    const assistantRecord = (blob.blobMap.get(`chatlog/${dateKey(new Date())}.jsonl`) ?? '')
      .split('\n')
      .filter((line) => line.trim())
      .map((line) => JSON.parse(line) as { kind: string; content: string; reasoningContent?: string })
      .find((record) => record.kind === 'assistant')
    assert.equal(assistantRecord?.content, '非流式回复')
    assert.equal(assistantRecord?.reasoningContent, '非流式思考')
  })

  test('JSON path (stream:false): per-round reasoning is archived with turn, not concatenated', async () => {
    const store = makeMockStore()
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const toolArgs = JSON.stringify({ query: 'x' })
    let fetchCalls = 0
    mock.method(globalThis, 'fetch', async () => {
      fetchCalls += 1
      if (fetchCalls === 1) {
        return new Response(
          JSON.stringify({
            choices: [{
              message: {
                content: '我查一下。',
                reasoning_content: '第一轮思考',
                tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'web_search', arguments: toolArgs } }],
              },
            }],
          }),
          { status: 200 },
        )
      }
      return new Response(
        JSON.stringify({ choices: [{ message: { content: '结果是 7。', reasoning_content: '第二轮思考' } }] }),
        { status: 200 },
      )
    })

    const res = await chatOnRequest(makeContext({
      store,
      env: gatewayEnv() as Env,
      body: { message: '帮我算一下', stream: false },
    }))
    assert.equal(res.status, 200)
    const body = (await res.json()) as { ok: boolean; reply: string }
    assert.equal(body.reply, '我查一下。结果是 7。')
    assert.equal(fetchCalls, 2)

    const chatlog = (blob.blobMap.get(`chatlog/${dateKey(new Date())}.jsonl`) ?? '')
      .split('\n')
      .filter((line) => line.trim())
      .map((line) => JSON.parse(line) as { kind?: string; content: string; reasoningContent?: string; turn?: number })
    // Two assistant thinking records: round 1's own thinking + round 2's own
    // thinking on the final answer — never concatenated into one.
    const thinking = chatlog.filter((record) => record.kind === 'assistant' && record.reasoningContent)
    assert.deepEqual(thinking.map((record) => ({ content: record.content, reasoningContent: record.reasoningContent, turn: record.turn })), [
      { content: '我查一下。', reasoningContent: '第一轮思考', turn: 1 },
      { content: '我查一下。结果是 7。', reasoningContent: '第二轮思考', turn: 2 },
    ])
    const tool = chatlog.find((record) => record.kind === 'tool')
    assert.equal(tool?.turn, 1)
    // The store still contains only the real tool row + the one final reply.
    const toolRows = store.messageLog.filter((row) => row.role === 'assistant' && row.metadata?.kind === 'tool')
    assert.equal(toolRows.length, 1)
    assert.equal(store.messageLog.at(-1)?.content, '我查一下。结果是 7。')
    assert.equal(store.messageLog.at(-1)?.metadata?.reasoningContent, undefined)
  })

  test('tool branch: tool_call/tool_result events stream in and the answer continues ai_response streamed:true', async () => {
    const store = makeMockStore()
    const encoder = new TextEncoder()
    // The gateway sends `arguments` as incremental chunks; slice the full JSON
    // string so no fragile literal with a closing brace trips the parser.
    const webArgs = JSON.stringify({ query: 'x' })
    const webPart1 = webArgs.slice(0, 10)
    const webPart2 = webArgs.slice(10)
    let fetchCalls = 0
    mock.method(globalThis, 'fetch', async () => {
      fetchCalls += 1
      if (fetchCalls === 1) {
        // Round 1: the model streams tool_calls deltas (arguments split across
        // chunks like a real gateway) then ends the stream.
        const body = new ReadableStream<Uint8Array>({
          start(controller) {
            const tool1 = { choices: [{ delta: { tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: 'web_search', arguments: webPart1 } }] } }] }
            const tool2 = { choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: webPart2 } }] } }] }
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(tool1)}\n\n`))
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(tool2)}\n\n`))
            controller.enqueue(encoder.encode('data: [DONE]\n\n'))
            controller.close()
          },
        })
        return new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } })
      }
      // Round 2: the continuation turn streams the final answer — still
      // typewriter (`streamed:true`), NOT a one-shot degraded reply.
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({ choices: [{ delta: { content: '我查完' } }] })}\n\n`))
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({ choices: [{ delta: { content: '了，答案是 11。' } }] })}\n\n`))
          controller.enqueue(encoder.encode('data: [DONE]\n\n'))
          controller.close()
        },
      })
      return new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } })
    })

    const res = await chatOnRequest(makeContext({ store, env: gatewayEnv() as Env, body: { message: '帮我查一下' } }))
    assert.equal(res.status, 200)
    const bodyText = await res.text()
    // Two streaming rounds happened, each with ai_response deltas.
    assert.ok(bodyText.includes('"type":"ai_response"'))
    assert.ok(bodyText.includes('"streamed":true'))
    assert.ok(bodyText.includes('"content":"我查完"'))
    assert.ok(bodyText.includes('"content":"了，答案是 11。'))
    // Tool progress is inserted as its own events between the streaming rounds.
    assert.ok(bodyText.includes('"type":"tool_call"'))
    assert.ok(bodyText.includes('"type":"tool_result"'))
    assert.ok(bodyText.includes('"name":"web_search"'))
    assert.ok(bodyText.includes('data: [DONE]'))
    assert.equal(fetchCalls, 2)
    // The tool record is persisted BEFORE the final assistant reply; history
    // stores the accumulated full answer.
    const toolRows = store.messageLog.filter((row) => row.role === 'assistant' && row.metadata?.kind === 'tool')
    assert.equal(toolRows.length, 1)
    assert.match(toolRows[0]?.content ?? '', /web_search/)
    assert.equal(store.messageLog.at(-1)?.role, 'assistant')
    assert.equal(store.messageLog.at(-1)?.content, '我查完了，答案是 11。')
  })

  test('tool rounds end in a forced-text final round: reply keeps intermediate prose and every tool round is recorded', async () => {
    const store = makeMockStore()
    const encoder = new TextEncoder()
    const toolArgs = JSON.stringify({ query: 'x' })
    const toolCallDelta = {
      choices: [{
        delta: {
          tool_calls: [{ index: 0, id: 'call', type: 'function', function: { name: 'web_search', arguments: toolArgs } }],
        },
      }],
    }
    const stream = (chunks: unknown[]) => {
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          for (const chunk of chunks) controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`))
          controller.enqueue(encoder.encode('data: [DONE]\n\n'))
          controller.close()
        },
      })
      return new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } })
    }
    let fetchCalls = 0
    mock.method(globalThis, 'fetch', async () => {
      fetchCalls += 1
      // Round 1: half a sentence + a tool call. Round 2: half a sentence + a
      // tool call. Round 3: the model answers in WORDS — the final round has no
      // tools, so even a stub asking for one would be ignored.
      if (fetchCalls === 1) {
        return stream([
          { choices: [{ delta: { content: '好的，' } }] },
          toolCallDelta,
        ])
      }
      if (fetchCalls === 2) {
        return stream([
          { choices: [{ delta: { content: '我先查一下。' } }] },
          toolCallDelta,
        ])
      }
      return stream([
        { choices: [{ delta: { content: '查完了，答案是 11。' } }] },
      ])
    })

    const res = await chatOnRequest(makeContext({ store, env: gatewayEnv() as Env, body: { message: '查一下' } }))
    assert.equal(res.status, 200)
    const bodyText = await res.text()
    assert.equal(fetchCalls, 3) // two tool rounds + one forced-text final round
    // Intermediate round text is streamed to the client (typewriter).
    assert.ok(bodyText.includes('"content":"好的，"'))
    assert.ok(bodyText.includes('"content":"我先查一下。"'))
    assert.ok(bodyText.includes('"content":"查完了，答案是 11。"'))
    // The final round ended in words — no "（没有回复）" and no neutral note.
    assert.ok(!bodyText.includes('（没有回复）'))
    assert.ok(!bodyText.includes('（这一轮以工具调用结束，没有生成正文）'))
    assert.ok(bodyText.includes('data: [DONE]'))
    // History keeps round 1 + round 2 intermediate text plus round 3's answer;
    // only rounds 1-2 recorded a tool (the final round had none).
    const toolRows = store.messageLog.filter((row) => row.role === 'assistant' && row.metadata?.kind === 'tool')
    assert.equal(toolRows.length, 2)
    assert.equal(store.messageLog.at(-1)?.role, 'assistant')
    assert.equal(store.messageLog.at(-1)?.content, '好的，我先查一下。查完了，答案是 11。')
  })

  test('tool-only rounds with zero prose end in a forced-text final round: pure-text answer, no neutral note', async () => {
    const store = makeMockStore()
    const encoder = new TextEncoder()
    const toolArgs = JSON.stringify({ query: 'x' })
    const toolCallDelta = {
      choices: [{ delta: { tool_calls: [{ index: 0, id: 'call', type: 'function', function: { name: 'web_search', arguments: toolArgs } }] } }],
    }
    const stream = (chunks: unknown[]) => {
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          for (const chunk of chunks) controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`))
          controller.enqueue(encoder.encode('data: [DONE]\n\n'))
          controller.close()
        },
      })
      return new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } })
    }
    let fetchCalls = 0
    mock.method(globalThis, 'fetch', async () => {
      fetchCalls += 1
      // Rounds 1-2: pure tool calls with zero prose. Round 3: the final round
      // withholds tools, so the model answers in words — no neutral note.
      if (fetchCalls <= 2) return stream([toolCallDelta])
      return stream([
        { choices: [{ delta: { content: '查完了。' } }] },
      ])
    })

    const res = await chatOnRequest(makeContext({ store, env: gatewayEnv() as Env, body: { message: '查一下' } }))
    assert.equal(res.status, 200)
    const bodyText = await res.text()
    assert.equal(fetchCalls, 3)
    assert.ok(bodyText.includes('data: [DONE]'))
    assert.ok(bodyText.includes('"content":"查完了。"'))
    // The forced-text final round replaces the old neutral note: the reply is
    // the model's actual words.
    assert.ok(!bodyText.includes('（没有回复）'))
    assert.ok(!bodyText.includes('（这一轮以工具调用结束，没有生成正文）'))
    assert.equal(store.messageLog.at(-1)?.role, 'assistant')
    assert.equal(store.messageLog.at(-1)?.content, '查完了。')
    const toolRows = store.messageLog.filter((row) => row.role === 'assistant' && row.metadata?.kind === 'tool')
    assert.equal(toolRows.length, 2)
  })

  test('mixed turn: round 1 tool_calls + half-sentence, round 2 no tools -> reply concatenates both rounds (middle text kept)', async () => {
    const store = makeMockStore()
    const encoder = new TextEncoder()
    const toolArgs = JSON.stringify({ query: 'x' })
    const stream = (chunks: unknown[]) => {
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          for (const chunk of chunks) controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`))
          controller.enqueue(encoder.encode('data: [DONE]\n\n'))
          controller.close()
        },
      })
      return new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } })
    }
    let fetchCalls = 0
    mock.method(globalThis, 'fetch', async () => {
      fetchCalls += 1
      if (fetchCalls === 1) {
        // Round 1: the model says half a sentence AND asks for a tool.
        return stream([
          { choices: [{ delta: { content: '好的，' } }] },
          { choices: [{ delta: { tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: 'web_search', arguments: toolArgs } }] } }] },
        ])
      }
      // Round 2: the final answer, no more tools — the turn ends naturally.
      return stream([
        { choices: [{ delta: { content: '查完了，答案是 42。' } }] },
      ])
    })

    const res = await chatOnRequest(makeContext({ store, env: gatewayEnv() as Env, body: { message: '查一下' } }))
    assert.equal(res.status, 200)
    const bodyText = await res.text()
    assert.equal(fetchCalls, 2)
    assert.ok(bodyText.includes('"content":"好的，"'))
    assert.ok(bodyText.includes('"content":"查完了，答案是 42。"'))
    assert.ok(bodyText.includes('"type":"tool_call"'))
    assert.ok(bodyText.includes('data: [DONE]'))
    // Reply = round 1 + round 2 text concatenated; the half-sentence is NOT lost.
    const toolRows = store.messageLog.filter((row) => row.role === 'assistant' && row.metadata?.kind === 'tool')
    assert.equal(toolRows.length, 1)
    assert.equal(store.messageLog.at(-1)?.role, 'assistant')
    assert.equal(store.messageLog.at(-1)?.content, '好的，查完了，答案是 42。')
  })

  test('JSON path (stream:false): tool rounds preserve intermediate half-sentences in the reply', async () => {
    const store = makeMockStore()
    const toolArgs = JSON.stringify({ query: 'x' })
    let fetchCalls = 0
    mock.method(globalThis, 'fetch', async () => {
      fetchCalls += 1
      if (fetchCalls === 1) {
        return new Response(
          JSON.stringify({
            choices: [{ message: { content: '好的，', tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'web_search', arguments: toolArgs } }] } }],
          }),
          { status: 200 },
        )
      }
      if (fetchCalls === 2) {
        return new Response(
          JSON.stringify({
            choices: [{ message: { content: '我先查一下。', tool_calls: [{ id: 'call_2', type: 'function', function: { name: 'web_search', arguments: toolArgs } }] } }],
          }),
          { status: 200 },
        )
      }
      return new Response(JSON.stringify({ choices: [{ message: { content: '查完了，答案是 42。' } }] }), { status: 200 })
    })

    const res = await chatOnRequest(makeContext({
      store,
      env: gatewayEnv() as Env,
      body: { message: '查一下', stream: false },
    }))
    assert.equal(res.status, 200)
    const body = (await res.json()) as { ok: boolean; reply: string }
    assert.equal(fetchCalls, 3)
    // Round 1 + round 2 prose survive the tool rounds — no more last-turn loss.
    assert.equal(body.reply, '好的，我先查一下。查完了，答案是 42。')
    assert.equal(store.messageLog.at(-1)?.role, 'assistant')
    assert.equal(store.messageLog.at(-1)?.content, '好的，我先查一下。查完了，答案是 42。')
    const toolRows = store.messageLog.filter((row) => row.role === 'assistant' && row.metadata?.kind === 'tool')
    assert.equal(toolRows.length, 2)
  })

  test('JSON path (stream:false): tool-only rounds end in a forced-text final round (no neutral note)', async () => {
    const store = makeMockStore()
    const toolArgs = JSON.stringify({ query: 'x' })
    let fetchCalls = 0
    mock.method(globalThis, 'fetch', async () => {
      fetchCalls += 1
      // Rounds 1-2: pure tool calls with zero prose. Round 3: the final round
      // withholds tools, so the model answers in words — the reply is the text.
      if (fetchCalls <= 2) {
        return new Response(
          JSON.stringify({
            choices: [{ message: { content: '', tool_calls: [{ id: `call_${fetchCalls}`, type: 'function', function: { name: 'web_search', arguments: toolArgs } }] } }],
          }),
          { status: 200 },
        )
      }
      return new Response(
        JSON.stringify({ choices: [{ message: { content: '查完了，答案是 42。' } }] }),
        { status: 200 },
      )
    })

    const res = await chatOnRequest(makeContext({
      store,
      env: gatewayEnv() as Env,
      body: { message: '查一下', stream: false },
    }))
    assert.equal(res.status, 200)
    const body = (await res.json()) as { ok: boolean; reply: string }
    assert.equal(body.reply, '查完了，答案是 42。')
    assert.equal(store.messageLog.at(-1)?.role, 'assistant')
    assert.equal(store.messageLog.at(-1)?.content, '查完了，答案是 42。')
    // Rounds 1-2 recorded their tools; the forced-text final round had none.
    const toolRows = store.messageLog.filter((row) => row.role === 'assistant' && row.metadata?.kind === 'tool')
    assert.equal(toolRows.length, 2)
  })

  test('streaming gateway failure surfaces as an SSE error_message event', async () => {
    const store = makeMockStore()
    mock.method(globalThis, 'fetch', async () => new Response('boom', { status: 503 }))

    const res = await chatOnRequest(makeContext({ store, env: gatewayEnv() as Env, body: { message: '你好' } }))
    assert.equal(res.status, 200) // SSE transports errors inside the event stream
    assert.match(res.headers.get('content-type') ?? '', /^text\/event-stream/)
    const bodyText = await res.text()
    assert.ok(bodyText.includes('"type":"error_message"'))
    assert.ok(bodyText.includes('AI gateway HTTP 503'))
  })

  test('image request sends multimodal content parts to the gateway and persists a marked row', async () => {
    const store = makeMockStore()
    const bodies: Array<Record<string, unknown>> = []
    mock.method(globalThis, 'fetch', async (_input: unknown, init?: RequestInit) => {
      if (init?.body) bodies.push(JSON.parse(String(init.body)) as Record<string, unknown>)
      return new Response(JSON.stringify(llmTextResponse('我看到你发来的图片了。')), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      })
    })

    const res = await chatOnRequest(makeContext({
      store,
      env: gatewayEnv() as Env,
      body: { message: '看看这张图', images: ['data:image/png;base64,AAA'], stream: false },
    }))
    assert.equal(res.status, 200)
    const env = (await res.json()) as { ok: boolean; reply: string }
    assert.equal(env.reply, '我看到你发来的图片了。')

    const messages = (bodies[0] as { messages: Array<{ role: string; content: unknown }> }).messages
    const last = messages.at(-1)
    assert.equal(last?.role, 'user')
    const parts = last?.content as Array<{ type: string; text?: string; image_url?: { url: string } }>
    assert.ok(Array.isArray(parts))
    assert.equal(parts[0]?.type, 'text')
    assert.equal(parts[0]?.text, '看看这张图')
    assert.equal(parts[1]?.type, 'image_url')
    assert.equal(parts[1]?.image_url?.url, 'data:image/png;base64,AAA')

    // History stored the array as a JSON string tagged image-user.
    const userRow = store.messageLog.find((row) => row.role === 'user' && row.metadata?.kind === 'image-user')
    assert.ok(userRow, 'image user row is persisted')
    assert.equal(typeof userRow?.content, 'string')
    const stored = JSON.parse(userRow?.content ?? '') as Array<{ type: string }>
    assert.equal(stored[1]?.type, 'image_url')
  })

  test('image-enabled turn accepts an image-only request (no text)', async () => {
    const store = makeMockStore()
    mock.method(globalThis, 'fetch', async () =>
      new Response(JSON.stringify(llmTextResponse('我收到了图片。')), { status: 200 }))
    const res = await chatOnRequest(makeContext({
      store,
      env: gatewayEnv() as Env,
      body: { message: '', images: ['data:image/png;base64,AAA'], stream: false },
    }))
    assert.equal(res.status, 200)
    const body = (await res.json()) as { ok: boolean; reply: string }
    assert.equal(body.reply, '我收到了图片。')
  })

  test('rejects a remote http(s) image URL (P2-2: SSRF / cost transfer)', async () => {
    const store = makeMockStore()
    let calls = 0
    mock.method(globalThis, 'fetch', async () => {
      calls += 1
      return new Response(JSON.stringify(llmTextResponse('不应走到网关。')), { status: 200 })
    })

    await assert.rejects(
      runChat(makeContext({ store, env: gatewayEnv() as Env }), {
        message: '看图',
        images: ['https://example.com/pic.png'],
      }),
      /data:image/,
    )
    assert.equal(calls, 0)
  })

  test('rejects more than 3 images (P2-2)', async () => {
    const store = makeMockStore()
    let calls = 0
    mock.method(globalThis, 'fetch', async () => {
      calls += 1
      return new Response(JSON.stringify(llmTextResponse('x')), { status: 200 })
    })

    await assert.rejects(
      runChat(makeContext({ store, env: gatewayEnv() as Env }), {
        message: '看图',
        images: Array.from({ length: 4 }, (_, i) => `data:image/png;base64,AAAA${i}`),
      }),
      /最多发送 3 张/,
    )
    assert.equal(calls, 0)
  })

  test('rejects an oversized base64 image over 4MB (P2-2)', async () => {
    const store = makeMockStore()
    let calls = 0
    mock.method(globalThis, 'fetch', async () => {
      calls += 1
      return new Response(JSON.stringify(llmTextResponse('x')), { status: 200 })
    })
    // ~4.2MB decoded: 5.6M base64 chars → floor(5.6M*3/4) = 4.2M bytes > 4MB.
    const big = `data:image/png;base64,${'A'.repeat(5_600_000)}`

    await assert.rejects(
      runChat(makeContext({ store, env: gatewayEnv() as Env }), {
        message: '看图',
        images: [big],
      }),
      /4MB/,
    )
    assert.equal(calls, 0)
  })

  test('vision-unsupported 400 retries WITHOUT images and appends the degraded system note', async () => {
    const store = makeMockStore()
    const bodies: Array<Record<string, unknown>> = []
    let calls = 0
    mock.method(globalThis, 'fetch', async (_input: unknown, init?: RequestInit) => {
      calls += 1
      if (init?.body) bodies.push(JSON.parse(String(init.body)) as Record<string, unknown>)
      if (calls === 1) {
        return new Response('{"error":"image_url is not supported by this model"}', { status: 400 })
      }
      return new Response(JSON.stringify(llmTextResponse('抱歉我无法直接看图，你把图片内容告诉我好吗？')), { status: 200 })
    })

    const res = await chatOnRequest(makeContext({
      store,
      env: gatewayEnv() as Env,
      body: { message: '看图', images: ['data:image/png;base64,AAA'], stream: false },
    }))
    assert.equal(res.status, 200)
    const body = (await res.json()) as { ok: boolean; reply: string }
    assert.equal(body.reply, '抱歉我无法直接看图，你把图片内容告诉我好吗？')
    assert.equal(calls, 2)

    // First attempt carries the image parts.
    const first = (bodies[0]?.messages as Array<{ role: string; content: unknown }>).at(-1)
    assert.ok(Array.isArray(first?.content))

    // Retry: no image_url anywhere, system prompt mentions the degraded vision.
    const second = bodies[1]?.messages as Array<{ role: string; content: unknown }>
    assert.match(String(second[0]?.content), /当前模型不支持视觉输入/)
    assert.ok(!JSON.stringify(second).includes('image_url'))
    const degradedUser = second.at(-1)
    assert.equal(typeof degradedUser?.content, 'string')
  })

  test('vision-unsupported 400 retries inside the SSE streaming path', async () => {
    const store = makeMockStore()
    let calls = 0
    mock.method(globalThis, 'fetch', async () => {
      calls += 1
      if (calls === 1) {
        return new Response('{"error":"this model does not support vision input"}', { status: 400 })
      }
      const encoder = new TextEncoder()
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({ choices: [{ delta: { content: '我看' } }] })}\n\n`))
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({ choices: [{ delta: { content: '不了图' } }] })}\n\n`))
          controller.enqueue(encoder.encode('data: [DONE]\n\n'))
          controller.close()
        },
      })
      return new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } })
    })

    const res = await chatOnRequest(makeContext({
      store,
      env: gatewayEnv() as Env,
      body: { message: '看图', images: ['data:image/png;base64,AAA'] },
    }))
    assert.match(res.headers.get('content-type') ?? '', /^text\/event-stream/)
    // The SSE generator runs lazily while the body is drained — fetch counters
    // and persistence are only meaningful AFTER the stream fully resolves.
    const bodyText = await res.text()
    assert.equal(calls, 2)
    assert.ok(bodyText.includes('data: [DONE]'))
    // Deltas arrive as separate SSE events; the client assembles them itself.
    assert.ok(bodyText.includes('"content":"我看"'))
    assert.ok(bodyText.includes('"content":"不了图"'))
    // The degraded reply persists the ACCUMULATED full text.
    assert.equal(store.messageLog.at(-1)?.role, 'assistant')
    assert.equal(store.messageLog.at(-1)?.content, '我看不了图')
  })

  test('a non-image 400 is NOT retried and surfaces as a JSON error', async () => {
    const store = makeMockStore()
    let calls = 0
    mock.method(globalThis, 'fetch', async () => {
      calls += 1
      return new Response('{"error":"messages[1]: missing field name"}', { status: 400 })
    })

    const res = await chatOnRequest(makeContext({
      store,
      env: gatewayEnv() as Env,
      body: { message: '看图', images: ['data:image/png;base64,AAA'], stream: false },
    }))
    assert.equal(calls, 1)
    assert.equal(res.status, 500) // errorResponse maps the gateway 400 to a stable 500
    const body = (await res.json()) as { ok: boolean; error: string }
    assert.match(body.error, /AI gateway HTTP 400/)
  })

  test('P1-2: the streaming turn winds down once the cumulative tool-result budget is spent', async () => {
    const store = makeMockStore()
    const blob = makeMockBlobStore({ 'memory/big.txt': 'x'.repeat(4000) })
    injectBlobStoreForTesting(blob)
    const encoder = new TextEncoder()
    const toolArgs = JSON.stringify({ key: 'memory/big.txt' })
    const fourToolDelta = {
      choices: [{
        delta: {
          tool_calls: Array.from({ length: 4 }, (_, i) => ({
            index: i,
            id: `call_${i + 1}`,
            type: 'function',
            function: { name: 'blob_read', arguments: toolArgs },
          })),
        },
      }],
    }
    const stream = (chunks: unknown[]) => {
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          for (const chunk of chunks) controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`))
          controller.enqueue(encoder.encode('data: [DONE]\n\n'))
          controller.close()
        },
      })
      return new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } })
    }
    let fetchCalls = 0
    mock.method(globalThis, 'fetch', async () => {
      fetchCalls += 1
      // Rounds 1-2: 4×4000-char blob reads each (32K total). Round 3: one more
      // read would push past the budget, so the turn winds down immediately.
      if (fetchCalls <= 2) return stream([fourToolDelta])
      if (fetchCalls === 3) {
        return stream([
          { choices: [{ delta: { content: '到这里预算不够了，我先停手。' } }] },
          { choices: [{ delta: { tool_calls: [{ index: 0, id: 'call_9', type: 'function', function: { name: 'blob_read', arguments: toolArgs } }] } }] },
        ])
      }
      return stream([{ choices: [{ delta: { content: '不该到达的第 4 轮。' } }] }])
    })

    const res = await chatOnRequest(makeContext({ store, env: gatewayEnv() as Env, body: { message: '查一下' } }))
    assert.equal(res.status, 200)
    const bodyText = await res.text()
    // The wind-down happens on round 3 — no round 4 fetch is ever made.
    assert.equal(fetchCalls, 3)
    // 4 + 4 + 1 tools executed before the wind-down.
    assert.equal((bodyText.match(/"type":"tool_result"/g) ?? []).length, 9)
    assert.equal(store.messageLog.at(-1)?.role, 'assistant')
    assert.equal(store.messageLog.at(-1)?.content, '到这里预算不够了，我先停手。')
  })

  test('P2-6: the streaming final round ignores a stubborn tool_call and ends in words (exactly CHAT_MAX_TURNS rounds)', async () => {
    const store = makeMockStore()
    const encoder = new TextEncoder()
    const toolArgs = JSON.stringify({ query: 'x' })
    const toolCallDelta = {
      choices: [{ delta: { tool_calls: [{ index: 0, id: 'call', type: 'function', function: { name: 'web_search', arguments: toolArgs } }] } }],
    }
    const stream = (chunks: unknown[]) => {
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          for (const chunk of chunks) controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`))
          controller.enqueue(encoder.encode('data: [DONE]\n\n'))
          controller.close()
        },
      })
      return new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } })
    }
    let fetchCalls = 0
    const bodies: Array<Record<string, unknown>> = []
    mock.method(globalThis, 'fetch', async (_input: unknown, init?: RequestInit) => {
      fetchCalls += 1
      if (init?.body) bodies.push(JSON.parse(String(init.body)) as Record<string, unknown>)
      if (fetchCalls < 32) return stream([toolCallDelta])
      // Final round: tools are withheld, but the stub STILL emits a tool_call
      // delta plus words — the turn must end in words anyway.
      return stream([
        toolCallDelta,
        { choices: [{ delta: { content: '最后的话。' } }] },
      ])
    })

    const res = await chatOnRequest(makeContext({ store, env: gatewayEnv() as Env, body: { message: '查一下' } }))
    assert.equal(res.status, 200)
    const bodyText = await res.text()
    assert.equal(fetchCalls, 32) // terminated exactly at CHAT_MAX_TURNS, no infinite loop
    assert.ok(bodyText.includes('"content":"最后的话。"'))
    assert.ok(bodyText.includes('data: [DONE]'))
    assert.equal(store.messageLog.at(-1)?.role, 'assistant')
    assert.equal(store.messageLog.at(-1)?.content, '最后的话。')
    // The final round withholds tools: its request body carries no tools key.
    assert.ok(bodies.length >= 32)
    assert.ok(Array.isArray(bodies[0]?.tools), 'a normal round carries the tool definitions')
    assert.ok(!('tools' in (bodies[31] ?? {})), 'the final round must not send tools (forced text)')
  })

  test('P2-6/P2-5: streaming 5→4 truncation keeps the wire consistent and appends the truncation note', async () => {
    const store = makeMockStore()
    const encoder = new TextEncoder()
    const fiveToolDelta = {
      choices: [{
        delta: {
          tool_calls: Array.from({ length: 5 }, (_, i) => ({
            index: i,
            id: `call_${i + 1}`,
            type: 'function',
            function: { name: 'web_search', arguments: JSON.stringify({ query: `q${i}` }) },
          })),
        },
      }],
    }
    const stream = (chunks: unknown[]) => {
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          for (const chunk of chunks) controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`))
          controller.enqueue(encoder.encode('data: [DONE]\n\n'))
          controller.close()
        },
      })
      return new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } })
    }
    let fetchCalls = 0
    const bodies: Array<Record<string, unknown>> = []
    mock.method(globalThis, 'fetch', async (_input: unknown, init?: RequestInit) => {
      fetchCalls += 1
      if (init?.body) bodies.push(JSON.parse(String(init.body)) as Record<string, unknown>)
      if (fetchCalls === 1) return stream([fiveToolDelta])
      return stream([{ choices: [{ delta: { content: 'done' } }] }])
    })

    const res = await chatOnRequest(makeContext({ store, env: gatewayEnv() as Env, body: { message: '查一下' } }))
    assert.equal(res.status, 200)
    const bodyText = await res.text()
    assert.equal(fetchCalls, 2)
    // Only 4 of the 5 requested tools executed.
    assert.equal((bodyText.match(/"type":"tool_result"/g) ?? []).length, 4)
    assert.equal((bodyText.match(/"type":"tool_call"/g) ?? []).length, 4)

    // The final-round request carries the truncated assistant tool_calls (P1-1)
    // plus the truncation note (P2-5).
    assert.ok(bodies.length >= 2)
    const sent = bodies[1]?.messages as Array<Record<string, unknown>>
    const assistant = sent.find((message) => message?.role === 'assistant')
    assert.ok(assistant, 'final-round body carries the assistant tool_calls message')
    const toolCalls = assistant?.tool_calls as Array<{ id: string }>
    assert.equal(toolCalls.length, 4)
    const toolResults = sent.filter((message) => message?.role === 'tool')
    assert.equal(toolResults.length, 4)
    for (const call of toolCalls) {
      assert.ok(
        toolResults.some((message) => message?.tool_call_id === call.id),
        `tool_call ${call.id} has a matching role:'tool' result`,
      )
    }
    assert.ok(
      sent.some((message) => message?.role === 'system' && String(message?.content).includes('本轮工具数达到上限 4')),
      'the truncation note is appended for the model',
    )
  })

  test('P2-6: an empty final-round reply falls back to the descriptive（模型没有输出正文）', async () => {
    const store = makeMockStore()
    const encoder = new TextEncoder()
    mock.method(globalThis, 'fetch', async () => {
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(encoder.encode('data: [DONE]\n\n'))
          controller.close()
        },
      })
      return new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } })
    })

    const res = await chatOnRequest(makeContext({ store, env: gatewayEnv() as Env, body: { message: '你好' } }))
    assert.equal(res.status, 200)
    await res.text()
    // No ai_response deltas were emitted, yet the persisted reply is informative.
    assert.equal(store.messageLog.at(-1)?.role, 'assistant')
    assert.equal(store.messageLog.at(-1)?.content, '（模型没有输出正文）')
  })

  test('P2-6: JSON path (stream:false) maps an empty model reply to the descriptive fallback', async () => {
    const store = makeMockStore()
    mock.method(globalThis, 'fetch', async () =>
      new Response(JSON.stringify(llmTextResponse('')), { status: 200 }))

    const res = await chatOnRequest(makeContext({
      store,
      env: gatewayEnv() as Env,
      body: { message: '你好', stream: false },
    }))
    assert.equal(res.status, 200)
    const body = (await res.json()) as { ok: boolean; reply: string }
    assert.equal(body.reply, '（模型没有输出正文）')
    assert.equal(store.messageLog.at(-1)?.role, 'assistant')
    assert.equal(store.messageLog.at(-1)?.content, '（模型没有输出正文）')
  })

  test('P2-3: JSON path rejects with the descriptive wall-clock timeout message', async () => {
    const store = makeMockStore()
    mock.method(globalThis, 'fetch', async (_input: unknown, init?: RequestInit) => {
      // Hang until the turn budget's abort fires (never resolves on its own).
      return new Promise((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => reject(new Error('aborted by budget')))
      })
    })

    await assert.rejects(
      runChat(makeContext({ store, env: gatewayEnv() as Env }), { message: 'hi', timeoutMs: 5 }),
      (error: unknown) =>
        error instanceof Error && error.name === 'AbortError' && error.message.includes('跑太久了'),
    )
  })

  test('P2-3: errorResponse maps the descriptive chat timeout AbortError to a clear 499 body', async () => {
    const error = new Error(CHAT_TIMEOUT_MESSAGE)
    error.name = 'AbortError'
    const res = errorResponse(error)
    assert.equal(res.status, 499)
    const body = (await res.json()) as { ok: boolean; error: string }
    assert.equal(body.error, CHAT_TIMEOUT_MESSAGE)
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