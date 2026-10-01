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

  test('streaming path: reasoning survives tool rounds by concatenating all rounds into the assistant record', async () => {
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
    const assistantRecord = (blob.blobMap.get(`chatlog/${dateKey(new Date())}.jsonl`) ?? '')
      .split('\n')
      .filter((line) => line.trim())
      .map((line) => JSON.parse(line) as { kind: string; content: string; reasoningContent?: string })
      .find((record) => record.kind === 'assistant')
    assert.equal(assistantRecord?.content, '好的，查完了。')
    assert.equal(assistantRecord?.reasoningContent, '第一轮思考第二轮思考')
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

  test('tool-only rounds that exhaust CHAT_MAX_TURNS still reply with a body (not "（没有回复）") and keep intermediate text', async () => {
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
      // tool call. Round 3: ANOTHER tool call with NO prose — the budget runs
      // outs here, so every round requested a tool.
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
      return stream([toolCallDelta])
    })

    const res = await chatOnRequest(makeContext({ store, env: gatewayEnv() as Env, body: { message: '查一下' } }))
    assert.equal(res.status, 200)
    const bodyText = await res.text()
    assert.equal(fetchCalls, 3) // all three rounds streamed through the gateway
    // Intermediate round text is streamed to the client (typewriter).
    assert.ok(bodyText.includes('"content":"好的，"'))
    assert.ok(bodyText.includes('"content":"我先查一下。"'))
    // The reply is NOT the misleading "（没有回复）" after a tool-only run — the
    // accumulated intermediate words form the persisted reply.
    assert.ok(!bodyText.includes('（没有回复）'))
    assert.ok(!bodyText.includes('（这一轮以工具调用结束，没有生成正文）'))
    assert.ok(bodyText.includes('data: [DONE]'))
    // History keeps round 1 + round 2 intermediate text; every round's tool
    // call was recorded BEFORE the final reply.
    const toolRows = store.messageLog.filter((row) => row.role === 'assistant' && row.metadata?.kind === 'tool')
    assert.equal(toolRows.length, 3)
    assert.equal(store.messageLog.at(-1)?.role, 'assistant')
    assert.equal(store.messageLog.at(-1)?.content, '好的，我先查一下。')
  })

  test('tool-only rounds with zero prose fall back to a neutral note when the budget is exhausted', async () => {
    const store = makeMockStore()
    const encoder = new TextEncoder()
    const toolArgs = JSON.stringify({ query: 'x' })
    const toolCallDelta = {
      choices: [{ delta: { tool_calls: [{ index: 0, id: 'call', type: 'function', function: { name: 'web_search', arguments: toolArgs } }] } }],
    }
    mock.method(globalThis, 'fetch', async () => {
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(toolCallDelta)}\n\n`))
          controller.enqueue(encoder.encode('data: [DONE]\n\n'))
          controller.close()
        },
      })
      return new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } })
    })

    const res = await chatOnRequest(makeContext({ store, env: gatewayEnv() as Env, body: { message: '查一下' } }))
    assert.equal(res.status, 200)
    const bodyText = await res.text()
    assert.ok(bodyText.includes('data: [DONE]'))
    // No misleading "（没有回复）" — the neutral note explains the tool-only run,
    // and the tool records were persisted before it.
    assert.ok(!bodyText.includes('（没有回复）'))
    assert.equal(store.messageLog.at(-1)?.role, 'assistant')
    assert.equal(store.messageLog.at(-1)?.content, '（这一轮以工具调用结束，没有生成正文）')
    const toolRows = store.messageLog.filter((row) => row.role === 'assistant' && row.metadata?.kind === 'tool')
    assert.equal(toolRows.length, 3)
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

  test('JSON path (stream:false): a tool-only budget exhaustion returns the neutral note, not "（没有回复）"', async () => {
    const store = makeMockStore()
    const toolArgs = JSON.stringify({ query: 'x' })
    mock.method(globalThis, 'fetch', async () =>
      new Response(
        JSON.stringify({
          choices: [{ message: { content: '', tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'web_search', arguments: toolArgs } }] } }],
        }),
        { status: 200 },
      ))

    const res = await chatOnRequest(makeContext({
      store,
      env: gatewayEnv() as Env,
      body: { message: '查一下', stream: false },
    }))
    assert.equal(res.status, 200)
    const body = (await res.json()) as { ok: boolean; reply: string }
    assert.equal(body.reply, '（这一轮以工具调用结束，没有生成正文）')
    assert.equal(store.messageLog.at(-1)?.role, 'assistant')
    assert.equal(store.messageLog.at(-1)?.content, '（这一轮以工具调用结束，没有生成正文）')
    const toolRows = store.messageLog.filter((row) => row.role === 'assistant' && row.metadata?.kind === 'tool')
    assert.equal(toolRows.length, 3)
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