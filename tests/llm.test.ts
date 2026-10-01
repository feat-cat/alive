/**
 * Tests for the bounded LLM tool loop (`_llm.ts`): abort checkpoints that
 * cover tool execution (P2-1), final-turn tool calls not being dropped (P2-2),
 * and `arguments` arriving as either a JSON string or an object (P2-9).
 */
import { afterEach, describe, test } from 'node:test'
import assert from 'node:assert/strict'
import { mock } from 'node:test'
import { buildChatBody, chatCompletion, degradeVisionMessages, isVisionUnsupportedError, parseGatewayModel, requireGatewayEnv, streamChatCompletion, stripImageContent, TOOL_ONLY_REPLY_NOTE, withProviderMessageName, type LlmMessage, type LlmToolCall } from '../agents/_llm.ts'
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

describe('chatCompletion final turn (forced text)', () => {
  test('final round withholds tools: a stub still returning tool_calls is ignored and the turn ends in words', async () => {
    let fetchCalls = 0
    mock.method(globalThis, 'fetch', async () => {
      fetchCalls += 1
      // The model (or a stub) still asks for a tool on the final round — the
      // loop withholds tools there, so the request MUST be ignored.
      return new Response(
        JSON.stringify({
          choices: [{
            message: {
              content: '最后的话。',
              tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'echo', arguments: '{}' } }],
            },
          }],
        }),
        { status: 200 },
      )
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

    // fetchCalls hit the budget; no tool ran; the turn ended with the model's
    // words instead of a tool call.
    assert.equal(fetchCalls, 1)
    assert.deepEqual(called, [])
    assert.equal(result.turns, 1)
    assert.equal(result.toolResults.length, 0)
    assert.equal(result.text, '最后的话。')
  })
})

describe('chatCompletion accumulated text (JSON path symmetry, P2-2)', () => {
  test('accumulateText preserves intermediate half-sentences across tool rounds', async () => {
    const toolArgs = JSON.stringify({ path: 'a.txt' })
    let fetchCalls = 0
    mock.method(globalThis, 'fetch', async () => {
      fetchCalls += 1
      if (fetchCalls === 1) {
        return new Response(
          JSON.stringify({
            choices: [{ message: { content: '好的，', tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'echo', arguments: toolArgs } }] } }],
          }),
          { status: 200 },
        )
      }
      if (fetchCalls === 2) {
        return new Response(
          JSON.stringify({
            choices: [{ message: { content: '我先查一下。', tool_calls: [{ id: 'call_2', type: 'function', function: { name: 'echo', arguments: toolArgs } }] } }],
          }),
          { status: 200 },
        )
      }
      return new Response(JSON.stringify({ choices: [{ message: { content: '查完了，答案是 42。' } }] }), { status: 200 })
    })

    const result = await chatCompletion({
      context,
      conversationId: 'eo-test',
      messages: [{ role: 'user', content: 'go' }],
      tools,
      toolRunner: async () => ({ content: 'ok' }),
      maxTurns: 3,
      accumulateText: true,
    })

    assert.equal(fetchCalls, 3)
    // Every round's prose survives the tool rounds (same semantics as streaming).
    assert.equal(result.text, '好的，我先查一下。查完了，答案是 42。')
  })

  test('last round is forced to words: tool-only rounds 1-2 then a pure-text final round (no neutral note)', async () => {
    const toolArgs = JSON.stringify({ path: 'a.txt' })
    let fetchCalls = 0
    mock.method(globalThis, 'fetch', async () => {
      fetchCalls += 1
      if (fetchCalls === 1 || fetchCalls === 2) {
        return new Response(
          JSON.stringify({
            choices: [{ message: { content: '', tool_calls: [{ id: `call_${fetchCalls}`, type: 'function', function: { name: 'echo', arguments: toolArgs } }] } }],
          }),
          { status: 200 },
        )
      }
      // The final round is forced to text — the model answers in words and the
      // conversation closes normally, never with the tool-only neutral note.
      return new Response(
        JSON.stringify({ choices: [{ message: { content: '查完了，答案是 42。' } }] }),
        { status: 200 },
      )
    })

    const result = await chatCompletion({
      context,
      conversationId: 'eo-test',
      messages: [{ role: 'user', content: 'go' }],
      tools,
      toolRunner: async () => ({ content: 'ok' }),
      maxTurns: 3,
      accumulateText: true,
    })

    // Rounds 1-2 ran their tool; round 3 was the budgeted text answer.
    assert.equal(fetchCalls, 3)
    assert.equal(result.turns, 3)
    assert.equal(result.toolResults.length, 2)
    assert.equal(result.text, '查完了，答案是 42。')
    // A normal close must never fall back to the neutral note.
    assert.notEqual(result.text, TOOL_ONLY_REPLY_NOTE)
  })

  test('without accumulateText the last-round tool request is ignored (heartbeat/compact regression)', async () => {
    const toolArgs = JSON.stringify({ path: 'a.txt' })
    let fetchCalls = 0
    mock.method(globalThis, 'fetch', async () => {
      fetchCalls += 1
      if (fetchCalls === 1) {
        return new Response(
          JSON.stringify({
            choices: [{ message: { content: '好的，', tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'echo', arguments: toolArgs } }] } }],
          }),
          { status: 200 },
        )
      }
      return new Response(
        JSON.stringify({
          choices: [{ message: { content: '', tool_calls: [{ id: 'call_2', type: 'function', function: { name: 'echo', arguments: toolArgs } }] } }],
        }),
        { status: 200 },
      )
    })

    const result = await chatCompletion({
      context,
      conversationId: 'eo-test',
      messages: [{ role: 'user', content: 'go' }],
      tools,
      toolRunner: async () => ({ content: 'ok' }),
      maxTurns: 2,
    })

    // Non-accumulate keeps the heartbeat/compact contract: only the LAST round's
    // assistant content is returned ('' here); the intermediate '好的，' is
    // dropped. New behaviour: the last round withholds tools, so round 2's
    // requested tool is IGNORED — only round 1's tool ran.
    assert.equal(result.text, '')
    assert.equal(result.turns, 2)
    assert.equal(result.toolResults.length, 1)
    assert.equal(result.toolResults[0]?.name, 'echo')
  })
})

describe('chatCompletion OpenAI tool schema (P2-11)', () => {
  test('sends tools with parameters wrapped as a full JSON Schema object ({ type: "object", properties, required })', async () => {
    const bodies: Array<Record<string, unknown>> = []
    let fetchCalls = 0
    mock.method(globalThis, 'fetch', async (_input: unknown, init?: RequestInit) => {
      fetchCalls += 1
      if (init?.body) bodies.push(JSON.parse(String(init.body)) as Record<string, unknown>)
      // Round 1 (non-final) carries the tools so the wrapped schema can be
      // inspected; round 2 is the forced-text close.
      if (fetchCalls === 1) {
        return new Response(JSON.stringify(toolCallsResponse({ name: 'echo', arguments: '{}' })), { status: 200 })
      }
      return new Response(JSON.stringify({ choices: [{ message: { content: 'done' } }] }), { status: 200 })
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
      maxTurns: 2,
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
    let fetchCalls = 0
    mock.method(globalThis, 'fetch', async (_input: unknown, init?: RequestInit) => {
      fetchCalls += 1
      if (init?.body) bodies.push(JSON.parse(String(init.body)) as Record<string, unknown>)
      if (fetchCalls === 1) {
        return new Response(JSON.stringify(toolCallsResponse({ name: 'echo', arguments: '{}' })), { status: 200 })
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

describe('chatCompletion per-round tool cap (MAX_TOOLS_PER_TURN)', () => {
  test('a round requesting 5 tools executes only the first 4; the 5th is truncated', async () => {
    let fetchCalls = 0
    const fiveCalls = Array.from({ length: 5 }, (_, i) => ({ name: 'echo', arguments: JSON.stringify({ n: i }) }))
    mock.method(globalThis, 'fetch', async () => {
      fetchCalls += 1
      if (fetchCalls === 1) {
        return new Response(JSON.stringify(toolCallsResponse(...fiveCalls)), { status: 200 })
      }
      return new Response(JSON.stringify({ choices: [{ message: { content: 'done' } }] }), { status: 200 })
    })
    const called: string[] = []

    const result = await chatCompletion({
      context,
      conversationId: 'eo-test',
      messages: [{ role: 'user', content: 'go' }],
      tools,
      toolRunner: async (name) => {
        called.push(name)
        return { content: 'ok' }
      },
      maxTurns: 2,
    })

    // Round 1 (non-final) requested 5 tools but only 4 ran; the 5th was cut and
    // never recorded. Round 2 is the forced-text close.
    assert.equal(fetchCalls, 2)
    assert.equal(called.length, 4)
    assert.equal(result.toolResults.length, 4)
    assert.deepEqual(result.toolResults.map((run) => run.name), ['echo', 'echo', 'echo', 'echo'])
    assert.equal(result.toolResults[4], undefined)
    assert.equal(result.text, 'done')
  })

  test('P1-1: the assistant tool_calls and the execution loop share the truncated array — no dangling ids', async () => {
    let fetchCalls = 0
    const fiveCalls = Array.from({ length: 5 }, (_, i) => ({ name: 'echo', arguments: JSON.stringify({ n: i }) }))
    const bodies: Array<Record<string, unknown>> = []
    mock.method(globalThis, 'fetch', async (_input: unknown, init?: RequestInit) => {
      fetchCalls += 1
      if (init?.body) bodies.push(JSON.parse(String(init.body)) as Record<string, unknown>)
      if (fetchCalls === 1) {
        return new Response(JSON.stringify(toolCallsResponse(...fiveCalls)), { status: 200 })
      }
      return new Response(JSON.stringify({ choices: [{ message: { content: 'done' } }] }), { status: 200 })
    })
    const called: string[] = []

    await chatCompletion({
      context,
      conversationId: 'eo-test',
      messages: [{ role: 'user', content: 'go' }],
      tools,
      toolRunner: async (name) => {
        called.push(name)
        return { content: 'ok' }
      },
      maxTurns: 2,
    })

    assert.equal(called.length, 4)
    assert.ok(bodies.length >= 2)
    const sent = bodies[1]?.messages as Array<Record<string, unknown>>
    const assistant = sent.find((message) => message?.role === 'assistant')
    assert.ok(assistant, 'round 2 body carries the assistant tool_calls message')
    const toolCalls = assistant?.tool_calls as Array<{ id: string; function: { name: string; arguments: string } }>
    // Only the first 4 calls are declared — call_5 is NOT on the wire.
    assert.equal(toolCalls.length, 4)
    const toolResults = sent.filter((message) => message?.role === 'tool')
    assert.equal(toolResults.length, 4)
    // Every declared id has a matching role:'tool' result — no dangling tool_call.
    for (const call of toolCalls) {
      assert.ok(
        toolResults.some((message) => message?.tool_call_id === call.id),
        `tool_call ${call.id} has a matching role:'tool' result`,
      )
    }
    assert.equal(called[4], undefined)
  })

  test('P2-5: a truncation note is appended only when tool_calls actually exceed MAX_TOOLS_PER_TURN', async () => {
    let fetchCalls = 0
    const fiveCalls = Array.from({ length: 5 }, (_, i) => ({ name: 'echo', arguments: JSON.stringify({ n: i }) }))
    const bodies: Array<Record<string, unknown>> = []
    mock.method(globalThis, 'fetch', async (_input: unknown, init?: RequestInit) => {
      fetchCalls += 1
      if (init?.body) bodies.push(JSON.parse(String(init.body)) as Record<string, unknown>)
      if (fetchCalls === 1) {
        return new Response(JSON.stringify(toolCallsResponse(...fiveCalls)), { status: 200 })
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
    const note = sent.find((message) => message?.role === 'system' && String(message?.content).includes('本轮工具数达到上限 4'))
    assert.ok(note, 'the truncation note tells the model that 4+ calls were cut')

    // A normal 1-4 tool round appends NO note.
    fetchCalls = 0
    const bodies2: Array<Record<string, unknown>> = []
    mock.method(globalThis, 'fetch', async (_input: unknown, init?: RequestInit) => {
      fetchCalls += 1
      if (init?.body) bodies2.push(JSON.parse(String(init.body)) as Record<string, unknown>)
      if (fetchCalls === 1) {
        return new Response(JSON.stringify(toolCallsResponse({ name: 'echo', arguments: '{}' })), { status: 200 })
      }
      return new Response(JSON.stringify({ choices: [{ message: { content: 'ok' } }] }), { status: 200 })
    })
    await chatCompletion({
      context,
      conversationId: 'eo-test',
      messages: [{ role: 'user', content: 'go' }],
      tools,
      toolRunner: async () => ({ content: 'ok' }),
      maxTurns: 2,
    })
    const allMessages = JSON.stringify(bodies2[1]?.messages ?? [])
    assert.ok(!allMessages.includes('本轮工具数达到上限'), 'a normal single-tool round must not append the truncation note')
  })
})

describe('chatCompletion cumulative tool-result context budget (P1-2)', () => {
  test('stop回填 results and wind down once the cumulative budget is spent — no further gateway request', async () => {
    const big = 'x'.repeat(4000)
    let fetchCalls = 0
    mock.method(globalThis, 'fetch', async () => {
      fetchCalls += 1
      // Rounds 1-2: 4 tool calls each (4×4000 = 16K per round, 32K total).
      // Round 3: a single tool call would push past the 32K budget.
      if (fetchCalls <= 2) {
        const calls = Array.from({ length: 4 }, (_, i) => ({ id: `call_${fetchCalls}_${i}`, type: 'function', function: { name: 'echo', arguments: JSON.stringify({ n: i }) } }))
        return new Response(JSON.stringify({ choices: [{ message: { content: '', tool_calls: calls } }] }), { status: 200 })
      }
      if (fetchCalls === 3) {
        return new Response(JSON.stringify({ choices: [{ message: { content: '到这里预算不够了，我先停手。', tool_calls: [{ id: 'call_3_0', type: 'function', function: { name: 'echo', arguments: '{}' } }] } }] }), { status: 200 })
      }
      return new Response(JSON.stringify({ choices: [{ message: { content: '不该到达的第 4 轮。' } }] }), { status: 200 })
    })

    const result = await chatCompletion({
      context,
      conversationId: 'eo-test',
      messages: [{ role: 'user', content: 'go' }],
      tools,
      toolRunner: async () => ({ content: big }),
      maxTurns: 10,
      accumulateText: true,
    })

    // Round 3's tool run pushed the cumulative byte count over budget, so the
    // turn winds down immediately — no round 4, no further gateway request.
    assert.equal(fetchCalls, 3)
    assert.ok(result.text.includes('到这里预算不够了，我先停手。'))
    // 4 + 4 + 1 tools executed before the wind-down.
    assert.equal(result.toolResults.length, 9)
  })

  test('P2-6: a maxTurns all-tool stub terminates exactly at maxTurns with a final text reply', async () => {
    let fetchCalls = 0
    mock.method(globalThis, 'fetch', async () => {
      fetchCalls += 1
      return new Response(
        JSON.stringify({
          choices: [{
            message: {
              content: fetchCalls < 8 ? '' : '收尾的话。',
              tool_calls: [{ id: `call_${fetchCalls}`, type: 'function', function: { name: 'echo', arguments: '{}' } }],
            },
          }],
        }),
        { status: 200 },
      )
    })

    const result = await chatCompletion({
      context,
      conversationId: 'eo-test',
      messages: [{ role: 'user', content: 'go' }],
      tools,
      toolRunner: async () => ({ content: 'ok' }),
      maxTurns: 8,
    })

    // The final round withheld tools; the stub's tool_calls was ignored and the
    // turn ended in words.
    assert.equal(fetchCalls, 8)
    assert.equal(result.turns, 8)
    assert.equal(result.text, '收尾的话。')
    assert.equal(result.toolResults.length, 7)
  })

  test('P2-6: the final round request body carries no tools key', async () => {
    let fetchCalls = 0
    const bodies: Array<Record<string, unknown>> = []
    mock.method(globalThis, 'fetch', async (_input: unknown, init?: RequestInit) => {
      fetchCalls += 1
      if (init?.body) bodies.push(JSON.parse(String(init.body)) as Record<string, unknown>)
      if (fetchCalls === 1) {
        return new Response(JSON.stringify(toolCallsResponse({ name: 'echo', arguments: '{}' })), { status: 200 })
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
    assert.ok(Array.isArray(bodies[0]?.tools), 'round 1 carries the tool definitions')
    assert.ok(!('tools' in (bodies[1] ?? {})), 'the final round must not send tools (forced text)')
  })
})

/* ------------------------------------------------------------------ */
/* streamChatCompletion (SSE)                                          */
/* ------------------------------------------------------------------ */

function sseData(payload: unknown): string {
  return `data: ${JSON.stringify(payload)}\n\n`
}

/** Mock AI Gateway SSE stream: whole frames as one response body. */
function sseStreamResponse(chunks: string[]): Response {
  const encoder = new TextEncoder()
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk))
      controller.close()
    },
  })
  return new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } })
}

describe('streamChatCompletion (SSE)', () => {
  test('calls onDelta with each content delta, accumulates full text, and stops at [DONE]', async () => {
    const deltas: string[] = []
    mock.method(globalThis, 'fetch', async () =>
      sseStreamResponse([
        sseData({ choices: [{ delta: { content: '你' } }] }),
        sseData({ choices: [{ delta: { content: '好' } }] }),
        sseData({ choices: [{ delta: { content: '呀' } }] }),
        'data: [DONE]\n\n',
      ]))

    const result = await streamChatCompletion({
      context,
      conversationId: 'eo-test',
      messages: [{ role: 'user', content: '你好' }],
      onDelta: (delta) => deltas.push(delta),
    })

    assert.deepEqual(deltas, ['你', '好', '呀'])
    assert.equal(result.text, '你好呀')
  })

  test('handles delta frames split across arbitrary byte boundaries and a stream that simply ends', async () => {
    const raw = sseData({ choices: [{ delta: { content: 'hello' } }] })
      + sseData({ choices: [{ delta: { content: ' world.' } }] })
      + 'data: [DONE]\n\n'
    const bytes = new TextEncoder().encode(raw)
    const slices: Uint8Array[] = []
    for (let i = 0; i < bytes.length; i += 7) slices.push(bytes.slice(i, i + 7))
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        for (const slice of slices) controller.enqueue(slice)
        controller.close()
      },
    })
    mock.method(globalThis, 'fetch', async () => new Response(body, { status: 200 }))
    const deltas: string[] = []
    const result = await streamChatCompletion({
      context,
      conversationId: 'eo-test',
      messages: [{ role: 'user', content: 'hi' }],
      onDelta: (delta) => deltas.push(delta),
    })
    assert.deepEqual(deltas, ['hello', ' world.'])
    assert.equal(result.text, 'hello world.')
  })

  test('request body carries stream:true plus strict-serif normalization and the tool wrapper', async () => {
    const bodies: Array<Record<string, unknown>> = []
    mock.method(globalThis, 'fetch', async (_input: unknown, init?: RequestInit) => {
      if (init?.body) bodies.push(JSON.parse(String(init.body)) as Record<string, unknown>)
      return sseStreamResponse([sseData({ choices: [{ delta: { content: 'ok' } }] }), 'data: [DONE]\n\n'])
    })

    await streamChatCompletion({
      context,
      conversationId: 'eo-test',
      messages: [{ role: 'user', content: 'go' }],
      tools,
      onDelta: () => {},
    })

    assert.ok(bodies.length >= 1)
    assert.equal(bodies[0]?.stream, true)
    const sent = (bodies[0]?.messages as Array<Record<string, unknown>>)[0]
    assert.equal(sent?.role, 'user')
    assert.equal(sent?.name, 'user') // DeepSeek strict serde name normalization
    assert.ok(Array.isArray(bodies[0]?.tools))
    assert.equal(
      ((bodies[0]?.tools as Array<{ function: { name: string } }>)[0])?.function.name,
      'echo',
    )
  })

  test('calls onReasoning for reasoning_content deltas; content and reasoning accumulate independently', async () => {
    const deltas: string[] = []
    const reasoning: string[] = []
    mock.method(globalThis, 'fetch', async () =>
      sseStreamResponse([
        sseData({ choices: [{ delta: { reasoning_content: '让我想' } }] }),
        sseData({ choices: [{ delta: { reasoning_content: '想一下' } }] }),
        sseData({ choices: [{ delta: { content: '答案' } }] }),
        sseData({ choices: [{ delta: { reasoning_content: '（补充思考）' } }] }),
        sseData({ choices: [{ delta: { content: '在此' } }] }),
        'data: [DONE]\n\n',
      ]))

    const result = await streamChatCompletion({
      context,
      conversationId: 'eo-test',
      messages: [{ role: 'user', content: '你好' }],
      onDelta: (delta) => deltas.push(delta),
      onReasoning: (text) => reasoning.push(text),
    })

    assert.deepEqual(reasoning, ['让我想', '想一下', '（补充思考）'])
    assert.deepEqual(deltas, ['答案', '在此'])
    assert.equal(result.text, '答案在此')
  })

  test('does not call onReasoning when the stream carries no reasoning_content', async () => {
    const reasoning: string[] = []
    mock.method(globalThis, 'fetch', async () =>
      sseStreamResponse([
        sseData({ choices: [{ delta: { content: '好' } }] }),
        'data: [DONE]\n\n',
      ]))

    await streamChatCompletion({
      context,
      conversationId: 'eo-test',
      messages: [{ role: 'user', content: 'hi' }],
      onDelta: () => {},
      onReasoning: (text) => reasoning.push(text),
    })

    assert.deepEqual(reasoning, [])
  })

  test('accumulates tool_call arguments across deltas and calls onToolCalls once', async () => {
    const calls: LlmToolCall[][] = []
    const fullArgs = JSON.stringify({ path: 'a.txt', n: 2 })
    // The gateway sends `arguments` as incremental chunks — slice the full JSON
    // string into pieces so no literal with a closing brace trips the parser.
    const part1 = fullArgs.slice(0, 10)
    const part2 = fullArgs.slice(10, 16)
    const part3 = fullArgs.slice(16)
    mock.method(globalThis, 'fetch', async () =>
      sseStreamResponse([
        sseData({ choices: [{ delta: { tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: 'echo', arguments: part1 } }] } }] }),
        sseData({ choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: part2 } }] } }] }),
        sseData({ choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: part3 } }] } }] }),
        'data: [DONE]\n\n',
      ]))

    const result = await streamChatCompletion({
      context,
      conversationId: 'eo-test',
      messages: [{ role: 'user', content: 'go' }],
      tools,
      onDelta: () => {},
      onToolCalls: (toolCalls) => calls.push(toolCalls),
    })

    assert.equal(calls.length, 1)
    assert.deepEqual(calls[0], [{ id: 'call_1', name: 'echo', arguments: fullArgs }])
    assert.deepEqual(result.toolCalls, calls[0])
    assert.equal(result.text, '')
  })

  test('accumulates multiple tool_call indices independently and sorts by index', async () => {
    const calls: LlmToolCall[][] = []
    const argsA = JSON.stringify({ a: 1 })
    const argsB = JSON.stringify({ b: 2 })
    mock.method(globalThis, 'fetch', async () =>
      sseStreamResponse([
        sseData({
          choices: [{
            delta: {
              tool_calls: [
                { index: 0, id: 'call_1', type: 'function', function: { name: 'echo', arguments: argsA.slice(0, 4) } },
                { index: 1, id: 'call_2', type: 'function', function: { name: 'echo', arguments: argsB.slice(0, 4) } },
              ],
            },
          }],
        }),
        sseData({
          choices: [{
            delta: {
              tool_calls: [
                { index: 1, function: { arguments: argsB.slice(4) } },
                { index: 0, function: { arguments: argsA.slice(4) } },
              ],
            },
          }],
        }),
        'data: [DONE]\n\n',
      ]))

    const result = await streamChatCompletion({
      context,
      conversationId: 'eo-test',
      messages: [{ role: 'user', content: 'go' }],
      tools,
      onDelta: () => {},
      onToolCalls: (toolCalls) => calls.push(toolCalls),
    })

    assert.equal(calls.length, 1)
    assert.deepEqual(calls[0], [
      { id: 'call_1', name: 'echo', arguments: argsA },
      { id: 'call_2', name: 'echo', arguments: argsB },
    ])
    assert.deepEqual(result.toolCalls, calls[0])
  })

  test('no tool_calls deltas means onToolCalls is never called and result.toolCalls is empty', async () => {
    const calls: LlmToolCall[][] = []
    mock.method(globalThis, 'fetch', async () =>
      sseStreamResponse([
        sseData({ choices: [{ delta: { content: '好' } }] }),
        'data: [DONE]\n\n',
      ]))

    const result = await streamChatCompletion({
      context,
      conversationId: 'eo-test',
      messages: [{ role: 'user', content: 'hi' }],
      tools,
      onDelta: () => {},
      onToolCalls: (toolCalls) => calls.push(toolCalls),
    })

    assert.equal(calls.length, 0)
    assert.deepEqual(result.toolCalls, [])
    assert.equal(result.text, '好')
  })

  test('throws a descriptive error on a non-2xx response', async () => {
    mock.method(globalThis, 'fetch', async () => new Response('boom', { status: 503 }))

    await assert.rejects(
      streamChatCompletion({
        context,
        conversationId: 'eo-test',
        messages: [{ role: 'user', content: 'go' }],
        onDelta: () => {},
      }),
      /AI gateway HTTP 503/,
    )
  })
})

describe('multimodal content (image parts)', () => {
  const imageUrl = 'data:image/png;base64,AAAB'
  const userImageMessage: LlmMessage = {
    role: 'user',
    content: [
      { type: 'text', text: '看看这张图' },
      { type: 'image_url', image_url: { url: imageUrl } },
    ],
  }

  test('normalizer passes content arrays through verbatim and still adds a role-derived name', () => {
    const normalized = withProviderMessageName([userImageMessage])
    assert.equal(normalized.length, 1)
    const out = normalized[0]
    assert.equal(out?.role, 'user')
    assert.equal(out?.name, 'user')
    // The array must not be flattened or stringified.
    assert.deepEqual(out?.content, [
      { type: 'text', text: '看看这张图' },
      { type: 'image_url', image_url: { url: imageUrl } },
    ])
  })

  test('chatCompletion sends content arrays as OpenAI image_url parts', async () => {
    const bodies: Array<Record<string, unknown>> = []
    mock.method(globalThis, 'fetch', async (_input: unknown, init?: RequestInit) => {
      if (init?.body) bodies.push(JSON.parse(String(init.body)) as Record<string, unknown>)
      return new Response(JSON.stringify({ choices: [{ message: { content: '收到图片' } }] }), { status: 200 })
    })

    await chatCompletion({
      context,
      conversationId: 'eo-test',
      messages: [userImageMessage],
      maxTurns: 1,
    })

    assert.ok(bodies.length >= 1)
    const sent = (bodies[0]?.messages as Array<{ role: string; content: unknown }>)[0]
    assert.equal(sent?.role, 'user')
    assert.deepEqual(sent?.content, [
      { type: 'text', text: '看看这张图' },
      { type: 'image_url', image_url: { url: imageUrl } },
    ])
  })

  test('buildChatBody keeps string content untouched for plain messages', () => {
    const gateway = { apiKey: 'k', baseUrl: 'https://gateway.test', model: 'm' }
    const body = buildChatBody(
      gateway,
      [{ role: 'user', content: 'hi' }],
      undefined,
      0.6,
      undefined,
      false,
    )
    const sent = (body.messages as Array<{ role: string; content: unknown }>)[0]
    assert.equal(sent?.content, 'hi')
  })
})

describe('model reasoning-effort suffix (:none|:low|:medium|:high|:max)', () => {
  test('parseGatewayModel strips a known suffix and returns the matching effort', () => {
    assert.deepEqual(parseGatewayModel('@makers/deepseek-v4-flash:none'), {
      modelName: '@makers/deepseek-v4-flash',
      reasoningEffort: 'none',
    })
    assert.deepEqual(parseGatewayModel('@makers/deepseek-v4-flash:low'), {
      modelName: '@makers/deepseek-v4-flash',
      reasoningEffort: 'low',
    })
    assert.deepEqual(parseGatewayModel('@makers/deepseek-v4-flash:medium'), {
      modelName: '@makers/deepseek-v4-flash',
      reasoningEffort: 'medium',
    })
    assert.deepEqual(parseGatewayModel('@makers/deepseek-v4-flash:high'), {
      modelName: '@makers/deepseek-v4-flash',
      reasoningEffort: 'high',
    })
    assert.deepEqual(parseGatewayModel('@makers/deepseek-v4-flash:max'), {
      modelName: '@makers/deepseek-v4-flash',
      reasoningEffort: 'max',
    })
  })

  test('parseGatewayModel returns no effort for a bare model and ignores unknown suffixes', () => {
    const bare = parseGatewayModel('@makers/deepseek-v4-flash')
    assert.equal(bare.modelName, '@makers/deepseek-v4-flash')
    assert.equal(bare.reasoningEffort, undefined)
    const unknown = parseGatewayModel('@makers/deepseek-v4-flash:abc')
    assert.equal(unknown.modelName, '@makers/deepseek-v4-flash:abc')
    assert.equal(unknown.reasoningEffort, undefined)
    const empty = parseGatewayModel('')
    assert.equal(empty.modelName, '')
    assert.equal(empty.reasoningEffort, undefined)
    const ws = parseGatewayModel('   ')
    assert.equal(ws.modelName, '')
    assert.equal(ws.reasoningEffort, undefined)
    // A leading ':' is not a suffix either — the model name must not be emptied.
    const leading = parseGatewayModel(':none')
    assert.equal(leading.modelName, ':none')
    assert.equal(leading.reasoningEffort, undefined)
  })

  test('requireGatewayEnv strips the suffix into reasoningEffort', () => {
    const parsed = requireGatewayEnv(makeContext({
      env: { ...gatewayEnv(), AI_GATEWAY_MODEL: '@makers/deepseek-v4-flash:max' },
    }))
    assert.equal(parsed.model, '@makers/deepseek-v4-flash')
    assert.equal(parsed.reasoningEffort, 'max')
  })

  test('buildChatBody sends the stripped model plus reasoning_effort for a suffixed gateway', () => {
    const body = buildChatBody(
      { apiKey: 'k', baseUrl: 'https://gateway.test', model: '@makers/deepseek-v4-flash', reasoningEffort: 'none' },
      [{ role: 'user', content: 'hi' }],
      undefined,
      0.6,
      undefined,
      false,
    )
    assert.equal(body.model, '@makers/deepseek-v4-flash')
    assert.equal(body.reasoning_effort, 'none')
  })

  test('buildChatBody omits reasoning_effort when no suffix was configured', () => {
    const body = buildChatBody(
      { apiKey: 'k', baseUrl: 'https://gateway.test', model: '@makers/deepseek-v4-flash' },
      [{ role: 'user', content: 'hi' }],
      undefined,
      0.6,
      undefined,
      false,
    )
    assert.equal(body.model, '@makers/deepseek-v4-flash')
    assert.ok(!('reasoning_effort' in body), 'reasoning_effort must be absent without a suffix')
  })

  test('chatCompletion sends model without the suffix and reasoning_effort:none when configured', async () => {
    const bodies: Array<Record<string, unknown>> = []
    mock.method(globalThis, 'fetch', async (_input: unknown, init?: RequestInit) => {
      if (init?.body) bodies.push(JSON.parse(String(init.body)) as Record<string, unknown>)
      return new Response(JSON.stringify({ choices: [{ message: { content: 'ok' } }] }), { status: 200 })
    })

    await chatCompletion({
      context: makeContext({ env: { ...gatewayEnv(), AI_GATEWAY_MODEL: '@makers/deepseek-v4-flash:none' } }),
      conversationId: 'eo-test',
      messages: [{ role: 'user', content: 'go' }],
      maxTurns: 1,
    })

    assert.ok(bodies.length >= 1)
    assert.equal(bodies[0]?.model, '@makers/deepseek-v4-flash')
    assert.equal(bodies[0]?.reasoning_effort, 'none')
  })

  test('chatCompletion sends reasoning_effort:max and no suffix in model when configured', async () => {
    const bodies: Array<Record<string, unknown>> = []
    mock.method(globalThis, 'fetch', async (_input: unknown, init?: RequestInit) => {
      if (init?.body) bodies.push(JSON.parse(String(init.body)) as Record<string, unknown>)
      return new Response(JSON.stringify({ choices: [{ message: { content: 'ok' } }] }), { status: 200 })
    })

    await chatCompletion({
      context: makeContext({ env: { ...gatewayEnv(), AI_GATEWAY_MODEL: '@makers/deepseek-v4-flash:max' } }),
      conversationId: 'eo-test',
      messages: [{ role: 'user', content: 'go' }],
      maxTurns: 1,
    })

    assert.ok(bodies.length >= 1)
    assert.equal(bodies[0]?.model, '@makers/deepseek-v4-flash')
    assert.equal(bodies[0]?.reasoning_effort, 'max')
  })

  test('chatCompletion sends no reasoning_effort and the full model when no suffix is configured', async () => {
    const bodies: Array<Record<string, unknown>> = []
    mock.method(globalThis, 'fetch', async (_input: unknown, init?: RequestInit) => {
      if (init?.body) bodies.push(JSON.parse(String(init.body)) as Record<string, unknown>)
      return new Response(JSON.stringify({ choices: [{ message: { content: 'ok' } }] }), { status: 200 })
    })

    await chatCompletion({
      context,
      conversationId: 'eo-test',
      messages: [{ role: 'user', content: 'go' }],
      maxTurns: 1,
    })

    assert.ok(bodies.length >= 1)
    assert.equal(bodies[0]?.model, '@makers/deepseek-v4-flash')
    assert.ok(!('reasoning_effort' in (bodies[0] ?? {})), 'reasoning_effort must be absent without a suffix')
  })

  test('chatCompletion ignores an unknown suffix and sends the full model name unchanged', async () => {
    const bodies: Array<Record<string, unknown>> = []
    mock.method(globalThis, 'fetch', async (_input: unknown, init?: RequestInit) => {
      if (init?.body) bodies.push(JSON.parse(String(init.body)) as Record<string, unknown>)
      return new Response(JSON.stringify({ choices: [{ message: { content: 'ok' } }] }), { status: 200 })
    })

    await chatCompletion({
      context: makeContext({ env: { ...gatewayEnv(), AI_GATEWAY_MODEL: '@makers/deepseek-v4-flash:abc' } }),
      conversationId: 'eo-test',
      messages: [{ role: 'user', content: 'go' }],
      maxTurns: 1,
    })

    assert.ok(bodies.length >= 1)
    assert.equal(bodies[0]?.model, '@makers/deepseek-v4-flash:abc')
    assert.ok(!('reasoning_effort' in (bodies[0] ?? {})), 'reasoning_effort must be absent for an unknown suffix')
  })

  test('streaming request carries reasoning_effort:none from a :none suffix', async () => {
    const bodies: Array<Record<string, unknown>> = []
    mock.method(globalThis, 'fetch', async (_input: unknown, init?: RequestInit) => {
      if (init?.body) bodies.push(JSON.parse(String(init.body)) as Record<string, unknown>)
      return sseStreamResponse([sseData({ choices: [{ delta: { content: 'ok' } }] }), 'data: [DONE]\n\n'])
    })

    await streamChatCompletion({
      context: makeContext({ env: { ...gatewayEnv(), AI_GATEWAY_MODEL: '@makers/deepseek-v4-flash:none' } }),
      conversationId: 'eo-test',
      messages: [{ role: 'user', content: 'go' }],
      onDelta: () => {},
    })

    assert.ok(bodies.length >= 1)
    assert.equal(bodies[0]?.stream, true)
    assert.equal(bodies[0]?.model, '@makers/deepseek-v4-flash')
    assert.equal(bodies[0]?.reasoning_effort, 'none')
  })
})

describe('vision degradation helpers', () => {
  test('isVisionUnsupportedError matches image/vision/multimodal error text', () => {
    assert.equal(isVisionUnsupportedError(new Error('AI gateway HTTP 400: image_url is not supported by this model')), true)
    assert.equal(isVisionUnsupportedError(new Error('AI gateway HTTP 400: this model does not support vision input')), true)
    assert.equal(isVisionUnsupportedError(new Error('AI gateway HTTP 400: multimodal input is not supported')), true)
    assert.equal(isVisionUnsupportedError(new Error('AI gateway HTTP 400: data:image content rejected')), true)
    // Non-image errors must NOT trigger the degradation retry.
    assert.equal(isVisionUnsupportedError(new Error('AI gateway HTTP 400: messages[1]: missing field name')), false)
    // P2-6: the old broad `not support`/`image` pattern also matched unrelated
    // errors; the converged regex must keep them out.
    assert.equal(isVisionUnsupportedError(new Error('AI gateway HTTP 400: temperature not supported by this model')), false)
    assert.equal(isVisionUnsupportedError(new Error('AI gateway HTTP 400: unsupported parameter "max_tokens"')), false)
    assert.equal(isVisionUnsupportedError(new Error('AI gateway HTTP 503: upstream busy')), false)
    // A vision token on a non-400 status must NOT trigger the retry either.
    assert.equal(isVisionUnsupportedError(new Error('AI gateway HTTP 500: vision input failed')), false)
  })

  test('stripImageContent keeps plain text and collapses arrays to their text parts', () => {
    assert.equal(stripImageContent('plain'), 'plain')
    assert.equal(
      stripImageContent([
        { type: 'text', text: '标题' },
        { type: 'image_url', image_url: { url: 'data:image/png;base64,AA' } },
      ]),
      '标题',
    )
    // Image-only content falls back to a placeholder.
    assert.equal(
      stripImageContent([{ type: 'image_url', image_url: { url: 'data:image/png;base64,AA' } }]),
      '（图片已省略，当前模型不支持视觉输入）',
    )
  })

  test('degradeVisionMessages appends the system note and strips image parts everywhere else', () => {
    const degraded = degradeVisionMessages([
      { role: 'system', content: 'persona' },
      {
        role: 'user',
        content: [
          { type: 'text', text: '看图' },
          { type: 'image_url', image_url: { url: 'data:image/png;base64,AA' } },
        ],
      },
      { role: 'assistant', content: '早前回复' },
    ])
    assert.match(degraded[0]?.content as string, /当前模型不支持视觉输入/)
    // The image-bearing user row is back to text only.
    assert.equal(degraded[1]?.content, '看图')
    // Plain messages pass through untouched.
    assert.equal(degraded[2]?.content, '早前回复')
  })
})

