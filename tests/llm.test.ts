/**
 * Tests for the bounded LLM tool loop (`_llm.ts`): abort checkpoints that
 * cover tool execution (P2-1), final-turn tool calls not being dropped (P2-2),
 * and `arguments` arriving as either a JSON string or an object (P2-9).
 */
import { afterEach, describe, test } from 'node:test'
import assert from 'node:assert/strict'
import { mock } from 'node:test'
import { chatCompletion } from '../agents/_llm.ts'
import { gatewayEnv, makeContext } from './_helpers.ts'

afterEach(() => {
  mock.restoreAll()
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
  test('sends tools wrapped as { type: "function", function: { name, description, parameters } }', async () => {
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
    // The real AI Gateway 400s on the flat { name, description, parameters }
    // shape (`tools[0].type is invalid or missing`), so assert the exact
    // OpenAI-compatible wrapper that `singleCall` sends.
    assert.deepEqual(bodies[0]?.tools, [
      {
        type: 'function',
        function: { name: 'echo', description: 'echo args back', parameters: {} },
      },
    ])
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
