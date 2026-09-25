/**
 * Tests for the bounded LLM tool loop (`_llm.ts`): abort checkpoints that
 * cover tool execution (P2-1), final-turn tool calls not being dropped (P2-2),
 * and `arguments` arriving as either a JSON string or an object (P2-9).
 */
import { afterEach, describe, test } from 'node:test'
import assert from 'node:assert/strict'
import { mock } from 'node:test'
import { chatCompletion, withProviderMessageName, type LlmMessage } from '../agents/_llm.ts'
import { gatewayEnv, makeContext } from './_helpers.ts'

afterEach(() => {
  mock.restoreAll()
})

describe('withProviderMessageName (DeepSeek strict serde: messages[1] missing field name)', () => {
  test('adds a role-derived name to user/assistant messages; system and tool pass through', () => {
    const messages: LlmMessage[] = [
      { role: 'system', content: 'system' },
      { role: 'user', content: '[system][heartbeat] （heartbeat 醒来）此刻想做什么就做什么。' },
      { role: 'assistant', content: 'assistant reply' },
      { role: 'tool', tool_call_id: 'call_1', name: 'workspace_write', content: 'ok' },
    ]
    const normalized = withProviderMessageName(messages)
    assert.deepEqual(normalized, [
      { role: 'system', content: 'system' },
      { role: 'user', content: '[system][heartbeat] （heartbeat 醒来）此刻想做什么就做什么。', name: 'user' },
      { role: 'assistant', content: 'assistant reply', name: 'assistant' },
      { role: 'tool', tool_call_id: 'call_1', name: 'workspace_write', content: 'ok' },
    ])
    // Original messages are not mutated.
    assert.equal(Object.hasOwn(messages[1] ?? {}, 'name'), false)
  })

  test('keeps an explicit name and does not duplicate the field', () => {
    const messages: LlmMessage[] = [
      { role: 'user', content: 'hi', name: 'persona' },
      { role: 'assistant', content: 'hey', name: 'assistant' },
    ]
    const normalized = withProviderMessageName(messages)
    assert.equal(normalized[0]?.name, 'persona')
    assert.equal(normalized[1]?.name, 'assistant')
  })

  test('wraps assistant tool_calls into OpenAI standard shape alongside the role-derived name', () => {
    const messages: LlmMessage[] = [
      { role: 'assistant', content: '', tool_calls: [{ id: 'call_1', name: 'echo', arguments: { n: 1 } }] },
    ]
    const normalized = withProviderMessageName(messages)
    assert.deepEqual(normalized, [
      {
        role: 'assistant',
        content: '',
        name: 'assistant',
        tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'echo', arguments: '{"n":1}' } }],
      },
    ])
  })

  test('every upstream message has a name for non-system roles (DeepSeek V4 requirement)', async () => {
    // First heartbeat: system message + the just-persisted wake trigger only.
    // loadMessages restores the trigger as { role:'user', content:'[system][heartbeat] ...' }
    // with NO name field — the exact shape the deployed DeepSeek gateway rejected.
    const messages: LlmMessage[] = [
      { role: 'system', content: 'persona + DECISION_SYSTEM' },
      { role: 'user', content: '[system][heartbeat] （heartbeat 醒来）此刻想做什么就做什么。' },
    ]
    const bodies: Array<Record<string, unknown>> = []
    mock.method(globalThis, 'fetch', async (_input: unknown, init?: RequestInit) => {
      if (init?.body) bodies.push(JSON.parse(String(init.body)) as Record<string, unknown>)
      return new Response(JSON.stringify({ choices: [{ message: { content: 'ok' } }] }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      })
    })

    await chatCompletion({
      context,
      conversationId: 'eo-test',
      messages,
      maxTurns: 1,
    })

    assert.ok(bodies.length >= 1)
    const sent = bodies[0]?.messages as Array<Record<string, unknown>>
    assert.equal(sent[0]?.role, 'system')
    assert.equal(sent[1]?.role, 'user')
    // DeepSeek V4's serde requires `name` on the non-system trigger message.
    assert.equal(sent[1]?.name, 'user')
    assert.equal(typeof sent[1]?.content, 'string')
    assert.equal(sent[1]?.tool_calls, undefined)
    assert.equal(sent[1]?.tool_call_id, undefined)
  })
})

function toolCallsResponse(...calls: Array<{ name: string; arguments: unknown }>): unknown {
  return {
    choices: [
      {
        message: {
          content: '',
          tool_calls: calls.map((call, index) => ({
            id: `call_${index + 1}`,
            type: 'function',
            function: { name: call.name, arguments: call.arguments },
          })),
        },
      },
    ],
  }
}

const context = makeContext({ env: gatewayEnv() })
const tools = [{ name: 'echo', description: 'echo args back', parameters: {} }]

describe('chatCompletion tool abort checkpoints (P2-1)', () => {
  test('throws AbortError before running tools when the signal is already aborted', async () => {
    mock.method(globalThis, 'fetch', async () =>
      new Response(JSON.stringify(toolCallsResponse({ name: 'echo', arguments: '{}' })), { status: 200 }))
    let toolCalls = 0
    const controller = new AbortController()
    controller.abort()

    await assert.rejects(
      chatCompletion({
        context,
        conversationId: 'eo-test',
        messages: [{ role: 'user', content: 'go' }],
        tools,
        toolRunner: async () => {
          toolCalls += 1
          return { content: 'ok' }
        },
        signal: controller.signal,
        maxTurns: 5,
      }),
      (error: unknown) => error instanceof Error && error.name === 'AbortError',
    )
    assert.equal(toolCalls, 0)
  })

  test('throws AbortError between tool calls when the signal fires mid-batch', async () => {
    mock.method(globalThis, 'fetch', async () =>
      new Response(
        JSON.stringify(
          toolCallsResponse(
            { name: 'echo', arguments: '{}' },
            { name: 'echo', arguments: '{}' },
          ),
        ),
        { status: 200 },
      ))
    const controller = new AbortController()
    const calledNames: string[] = []

    await assert.rejects(
      chatCompletion({
        context,
        conversationId: 'eo-test',
        messages: [{ role: 'user', content: 'go' }],
        tools,
        toolRunner: async (name) => {
          calledNames.push(name)
          if (calledNames.length === 1) controller.abort()
          return { content: 'ok' }
        },
        signal: controller.signal,
        maxTurns: 5,
      }),
      (error: unknown) => error instanceof Error && error.name === 'AbortError',
    )
    // The first tool ran, the second was stopped by the pre-call checkpoint.
    assert.deepEqual(calledNames, ['echo'])
  })
})

describe('chatCompletion final turn (P2-2)', () => {
  test('executes tool calls on the last turn instead of dropping them', async () => {
    let fetchCalls = 0
    mock.method(globalThis, 'fetch', async () => {
      fetchCalls += 1
      return new Response(JSON.stringify(toolCallsResponse({ name: 'echo', arguments: '{}' })), { status: 200 })
    })
    const called: string[] = []

    const result = await chatCompletion({
      context,
      conversationId: 'eo-test',
      messages: [{ role: 'user', content: 'go' }],
      tools,
      toolRunner: async (name) => {
        called.push(name)
        return { content: 'worked' }
      },
      maxTurns: 1,
    })

    // The work happened on the final (and only) turn and was recorded.
    assert.equal(fetchCalls, 1)
    assert.deepEqual(called, ['echo'])
    assert.equal(result.turns, 1)
    assert.equal(result.toolResults.length, 1)
    assert.equal(result.toolResults[0]?.name, 'echo')
    assert.equal(result.toolResults[0]?.content, 'worked')
    // The turn ended on tool calls, so there is no assistant text yet.
    assert.equal(result.text, '')
  })
})

describe('chatCompletion OpenAI tool schema (P2-11)', () => {
  test('sends tools with parameters wrapped as a full JSON Schema object ({ type: "object", properties, required })', async () => {
    const bodies: Array<Record<string, unknown>> = []
    mock.method(globalThis, 'fetch', async (_input: unknown, init?: RequestInit) => {
      if (init?.body) bodies.push(JSON.parse(String(init.body)) as Record<string, unknown>)
      return new Response(JSON.stringify(toolCallsResponse({ name: 'echo', arguments: '{}' })), { status: 200 })
    })

    await chatCompletion({
      context,
      conversationId: 'eo-test',
      messages: [{ role: 'user', content: 'go' }],
      tools: [
        {
          name: 'echo',
          description: 'echo args back',
          parameters: {
            path: { type: 'string', description: 'file path' },
            n: { type: 'number', description: 'count' },
          },
        },
      ],
      toolRunner: async () => ({ content: 'ok' }),
      maxTurns: 1,
    })

    assert.ok(bodies.length >= 1)
    // The real AI Gateway 400s both on the flat { name, description,
    // parameters } shape (`tools[0].type is invalid or missing`) and on the
    // bare property map (schema must be a JSON Schema of type object, got
    // 'type': null), so assert the exact OpenAI-compatible wrapper + full JSON
    // Schema parameters that `singleCall` sends.
    assert.deepEqual(bodies[0]?.tools, [
      {
        type: 'function',
        function: {
          name: 'echo',
          description: 'echo args back',
          parameters: {
            type: 'object',
            properties: {
              path: { type: 'string', description: 'file path' },
              n: { type: 'number', description: 'count' },
            },
            required: ['path', 'n'],
          },
        },
      },
    ])
  })

  test('wraps empty parameters as { type: "object", properties: {}, required: [] }', async () => {
    const bodies: Array<Record<string, unknown>> = []
    mock.method(globalThis, 'fetch', async (_input: unknown, init?: RequestInit) => {
      if (init?.body) bodies.push(JSON.parse(String(init.body)) as Record<string, unknown>)
      return new Response(JSON.stringify(toolCallsResponse({ name: 'echo', arguments: '{}' })), { status: 200 })
    })

    await chatCompletion({
      context,
      conversationId: 'eo-test',
      messages: [{ role: 'user', content: 'go' }],
      tools,
      toolRunner: async () => ({ content: 'ok' }),
      maxTurns: 1,
    })

    assert.ok(bodies.length >= 1)
    const sent = (bodies[0]?.tools as Array<{ function: { parameters: Record<string, unknown> } }>)?.[0]
    assert.deepEqual(sent?.function.parameters, { type: 'object', properties: {}, required: [] })
  })

  test('omits the tools key when no tools are provided', async () => {
    const bodies: Array<Record<string, unknown>> = []
    mock.method(globalThis, 'fetch', async (_input: unknown, init?: RequestInit) => {
      if (init?.body) bodies.push(JSON.parse(String(init.body)) as Record<string, unknown>)
      return new Response(JSON.stringify(toolCallsResponse()), { status: 200 })
    })

    await chatCompletion({
      context,
      conversationId: 'eo-test',
      messages: [{ role: 'user', content: 'go' }],
      maxTurns: 1,
    })

    assert.ok(bodies.length >= 1)
    assert.ok(!('tools' in (bodies[0] ?? {})), 'body.tools must be absent when no tools are configured')
  })
})

describe('chatCompletion assistant tool_calls wrapping (bug #3)', () => {
  test('sends assistant tool_calls as OpenAI standard { id, type, function } with string arguments', async () => {
    const bodies: Array<Record<string, unknown>> = []
    let fetchCalls = 0
    mock.method(globalThis, 'fetch', async (_input: unknown, init?: RequestInit) => {
      fetchCalls += 1
      if (init?.body) bodies.push(JSON.parse(String(init.body)) as Record<string, unknown>)
      if (fetchCalls === 1) {
        return new Response(
          JSON.stringify(toolCallsResponse({ name: 'echo', arguments: JSON.stringify({ path: 'a.txt' }) })),
          { status: 200 },
        )
      }
      return new Response(JSON.stringify({ choices: [{ message: { content: 'done' } }] }), { status: 200 })
    })

    await chatCompletion({
      context,
      conversationId: 'eo-test',
      messages: [{ role: 'user', content: 'go' }],
      tools,
      toolRunner: async () => ({ content: 'ok' }),
      maxTurns: 2,
    })

    assert.ok(bodies.length >= 2)
    const sent = bodies[1]?.messages as Array<Record<string, unknown>>
    const assistant = sent.find((message) => message?.role === 'assistant')
    assert.ok(assistant, 'second request carries the assistant tool_calls message')
    // The flat LlmToolCall must be wrapped so the gateway serde can read
    // `function.name`; otherwise the live 400 is `messages[i]: missing field name`.
    assert.deepEqual(assistant.tool_calls, [
      { id: 'call_1', type: 'function', function: { name: 'echo', arguments: JSON.stringify({ path: 'a.txt' }) } },
    ])
    // Name normalization still applies to the assistant message.
    assert.equal(assistant.name, 'assistant')
  })

  test('stringifies object arguments into the wrapped tool_call', async () => {
    const bodies: Array<Record<string, unknown>> = []
    let fetchCalls = 0
    mock.method(globalThis, 'fetch', async (_input: unknown, init?: RequestInit) => {
      fetchCalls += 1
      if (init?.body) bodies.push(JSON.parse(String(init.body)) as Record<string, unknown>)
      if (fetchCalls === 1) {
        return new Response(
          JSON.stringify(toolCallsResponse({ name: 'echo', arguments: { path: 'b.txt', n: 2 } })),
          { status: 200 },
        )
      }
      return new Response(JSON.stringify({ choices: [{ message: { content: 'done' } }] }), { status: 200 })
    })

    await chatCompletion({
      context,
      conversationId: 'eo-test',
      messages: [{ role: 'user', content: 'go' }],
      tools,
      toolRunner: async () => ({ content: 'ok' }),
      maxTurns: 2,
    })

    assert.ok(bodies.length >= 2)
    const sent = bodies[1]?.messages as Array<Record<string, unknown>>
    const assistant = sent.find((message) => message?.role === 'assistant')
    assert.ok(assistant, 'second request carries the assistant tool_calls message')
    assert.deepEqual(assistant.tool_calls, [
      { id: 'call_1', type: 'function', function: { name: 'echo', arguments: '{"path":"b.txt","n":2}' } },
    ])
  })

  test('omits the tool_calls field entirely when a message has none', async () => {
    const bodies: Array<Record<string, unknown>> = []
    mock.method(globalThis, 'fetch', async (_input: unknown, init?: RequestInit) => {
      if (init?.body) bodies.push(JSON.parse(String(init.body)) as Record<string, unknown>)
      return new Response(JSON.stringify({ choices: [{ message: { content: 'plain reply' } }] }), { status: 200 })
    })

    await chatCompletion({
      context,
      conversationId: 'eo-test',
      messages: [{ role: 'user', content: 'hi' }],
      maxTurns: 1,
    })

    assert.ok(bodies.length >= 1)
    const sent = bodies[0]?.messages as Array<Record<string, unknown>>
    for (const message of sent) {
      assert.equal(message?.tool_calls, undefined)
      assert.equal(message?.tool_call_id, undefined)
    }
  })
})

describe('chatCompletion tool arguments (P2-9)', () => {
  test('accepts arguments sent as a JSON string', async () => {
    mock.method(globalThis, 'fetch', async () =>
      new Response(
        JSON.stringify(toolCallsResponse({ name: 'echo', arguments: JSON.stringify({ path: 'a.txt' }) })),
        { status: 200 },
      ))
    let received: Record<string, unknown> | undefined

    const result = await chatCompletion({
      context,
      conversationId: 'eo-test',
      messages: [{ role: 'user', content: 'go' }],
      tools,
      toolRunner: async (_name, args) => {
        received = args
        return { content: 'ok' }
      },
      maxTurns: 2,
    })

    assert.deepEqual(received, { path: 'a.txt' })
    assert.deepEqual(result.toolResults[0]?.args, { path: 'a.txt' })
  })

  test('accepts arguments sent directly as an object', async () => {
    mock.method(globalThis, 'fetch', async () =>
      new Response(
        JSON.stringify(toolCallsResponse({ name: 'echo', arguments: { path: 'b.txt', n: 2 } })),
        { status: 200 },
      ))
    let received: Record<string, unknown> | undefined

    const result = await chatCompletion({
      context,
      conversationId: 'eo-test',
      messages: [{ role: 'user', content: 'go' }],
      tools,
      toolRunner: async (_name, args) => {
        received = args
        return { content: 'ok' }
      },
      maxTurns: 2,
    })

    assert.deepEqual(received, { path: 'b.txt', n: 2 })
    assert.deepEqual(result.toolResults[0]?.args, { path: 'b.txt', n: 2 })
  })
})
