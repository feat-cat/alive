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
    // The client saw per-token deltas; history stored the accumulated reply.
    assert.equal(store.messageLog.at(-1)?.role, 'assistant')
    assert.equal(store.messageLog.at(-1)?.content, '你好呀')
  })

  test('tool branch: aborts the stream and sends the non-streaming tool-loop reply as one SSE event', async () => {
    const store = makeMockStore()
    let fetchCalls = 0
    mock.method(globalThis, 'fetch', async () => {
      fetchCalls += 1
      if (fetchCalls === 1) {
        const encoder = new TextEncoder()
        const delta = {
          choices: [{
            delta: {
              tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'web_search', arguments: '{"query":"x"}' } }],
            },
          }],
        }
        const body = new ReadableStream<Uint8Array>({
          start(controller) {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(delta)}\n\n`))
            controller.enqueue(encoder.encode('data: [DONE]\n\n'))
            controller.close()
          },
        })
        return new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } })
      }
      // Non-streaming fallback answers with plain text (tools already consumed).
      return new Response(JSON.stringify({ choices: [{ message: { content: '我查完了，答案是 11。' } }] }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      })
    })

    const res = await chatOnRequest(makeContext({ store, env: gatewayEnv() as Env, body: { message: '帮我查一下' } }))
    assert.equal(res.status, 200)
    const bodyText = await res.text()
    assert.ok(bodyText.includes('data: [DONE]'))
    assert.ok(bodyText.includes('我查完了，答案是 11。'))
    assert.equal(fetchCalls, 2)
    assert.equal(store.messageLog.at(-1)?.role, 'assistant')
    assert.equal(store.messageLog.at(-1)?.content, '我查完了，答案是 11。')
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