/**
 * Tests for the heartbeat endpoint (`agents/heartbeat.ts`).
 *
 * The gateway fetch is mocked to return one or more payloads (tool-call rounds
 * then a final text). Blob access is injected with an in-memory store. All
 * mocks are restored in `afterEach`.
 *
 * Since heartbeat is free-form with no mood/energy parsing, the tests cover:
 * pure-text turns (zero sandbox), workspace tool turns (short-turn budget +
 * snapshot), blob tool turns (persistence), running without a sandbox while
 * the AI asks for a workspace tool, default degradation on empty/errored
 * output, MEMORY.md seeding on first use, and the system prompt carrying the
 * dynamic wall clock + the AI's memory (never MOOD/ENERGY parse hints).
 */
import { afterEach, describe, test } from 'node:test'
import assert from 'node:assert/strict'
import { mock } from 'node:test'
import { injectBlobStoreForTesting } from '../agents/_blob-tools.ts'
import { onRequest as heartbeatOnRequest, runHeartbeat, DECISION_SYSTEM } from '../agents/heartbeat.ts'
import { dateKey, INITIAL_MEMORY_SEED } from '../agents/_memory.ts'
import { SELF_ID, type Env } from '../agents/_shared.ts'
import { SELF_STATE_KEY, type AgentState } from '../agents/_state.ts'
import {
  gatewayEnv,
  makeContext,
  makeMockBlobStore,
  makeMockSandbox,
  makeMockStore,
  type MockStore,
} from './_helpers.ts'

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

function mockGatewayError(status = 500): void {
  mock.method(globalThis, 'fetch', async () => new Response('boom', { status }))
}

function defaultSelfState(): AgentState {
  return { lastActivityAt: 0, created: 1 }
}

function withGatewayEnv(store: MockStore, sandbox?: ReturnType<typeof makeMockSandbox>): ReturnType<typeof makeContext> {
  return makeContext({ store, sandbox, env: gatewayEnv() as Env })
}

function lastLogKind(store: MockStore): string | undefined {
  const last = store.messageLog.at(-1)
  return (last?.metadata as { logKind?: string } | undefined)?.logKind
}

function savedSelfState(store: MockStore): AgentState {
  return store.stateMap.get(SELF_ID)?.get(SELF_STATE_KEY) as AgentState
}

afterEach(() => {
  mock.restoreAll()
  injectBlobStoreForTesting(null)
})

describe('POST /heartbeat', () => {
  test('pure text output (no tools): zero sandbox, one heartbeat log, state updated', async () => {
    const store = makeMockStore({ [SELF_ID]: { [SELF_STATE_KEY]: defaultSelfState() } })
    const sandbox = makeMockSandbox()
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const context = withGatewayEnv(store, sandbox)
    const gateway = mockGateway(llmTextResponse('安静地想了想今天的路，觉得一切还好。'))

    const result = await runHeartbeat(context)

    assert.equal(gateway.calls, 1)
    assert.equal(result.turns, 1)
    assert.equal(result.toolCount, 0)
    assert.equal(result.touchedSandbox, false)
    assert.equal(result.savedFiles, 0)
    assert.ok(result.text.includes('安静地想了想'))

    // Trigger message (kind=heartbeat) + the assistant heartbeat log.
    assert.equal(store.messageLog.length, 2)
    assert.equal(store.messageLog[0]?.role, 'user')
    assert.equal((store.messageLog[0]?.metadata as { kind?: string } | undefined)?.kind, 'heartbeat')
    assert.equal(lastLogKind(store), 'heartbeat')

    // State only tracks lastActivityAt (mood/energy are no longer persisted).
    const saved = savedSelfState(store)
    assert.ok(saved.lastActivityAt > 0)
    assert.equal('mood' in saved, false)
    assert.equal('energy' in saved, false)
    assert.equal(Object.hasOwn(saved, 'created'), true)

    // Sandbox untouched: the pure-text path never calls ensureWorkspace.
    assert.equal(sandbox.filesMap.size, 0)
  })

  test('workspace tool call: short-turn budget runs and workspace is snapshotted to Blob', async () => {
    const store = makeMockStore({ [SELF_ID]: { [SELF_STATE_KEY]: defaultSelfState() } })
    const sandbox = makeMockSandbox()
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const context = withGatewayEnv(store, sandbox)
    const gateway = mockGateway(
      llmToolCallResponse('workspace_write', { path: 'src/index.ts', content: 'export const x = 1;\n' }),
      llmTextResponse('把入口文件写好了，今天先到这里。'),
    )

    const result = await runHeartbeat(context)

    assert.equal(gateway.calls, 2)
    assert.ok(result.turns >= 2)
    assert.equal(result.toolCount, 1)
    assert.equal(result.touchedSandbox, true)
    assert.ok(result.savedFiles >= 1)

    // File mirrored to sandbox AND persisted to Blob under the self workspace.
    assert.equal(sandbox.filesMap.get('projects/eo-self/workspace/src/index.ts'), 'export const x = 1;\n')
    assert.equal(blob.blobMap.get('projects/eo-self/workspace/src/index.ts'), 'export const x = 1;\n')
    assert.equal(lastLogKind(store), 'heartbeat')

    // The tool call is persisted into the shared history as a kind='tool' record.
    const toolRecord = store.messageLog.find((message) =>
      (message.metadata as { kind?: string } | undefined)?.kind === 'tool')
    assert.ok(toolRecord, 'tool call is recorded in the store history')
    assert.equal(toolRecord?.role, 'assistant')
    assert.equal((toolRecord?.metadata as { toolName?: string } | undefined)?.toolName, 'workspace_write')
    assert.match(toolRecord?.content ?? '', /workspace_write/)

    const saved = savedSelfState(store)
    assert.ok(saved.lastActivityAt > 0)
  })

  test('blob tool call: persists to Blob without touching the sandbox', async () => {
    const store = makeMockStore({ [SELF_ID]: { [SELF_STATE_KEY]: defaultSelfState() } })
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const context = withGatewayEnv(store)
    const gateway = mockGateway(
      llmToolCallResponse('blob_write', { key: 'memory/NOTES.md', content: '- 今天决定先完成注册页。\n' }),
      llmTextResponse('把今天的决定记进记忆了。'),
    )

    const result = await runHeartbeat(context)

    assert.equal(gateway.calls, 2)
    assert.equal(result.toolCount, 1)
    assert.equal(result.touchedSandbox, false)
    assert.equal(blob.blobMap.get('memory/NOTES.md'), '- 今天决定先完成注册页。\n')
    assert.equal(lastLogKind(store), 'heartbeat')

    const saved = savedSelfState(store)
    assert.ok(saved.lastActivityAt > 0)
  })

  test('without a sandbox the AI can still fall back to text: tool error, no crash', async () => {
    const store = makeMockStore({ [SELF_ID]: { [SELF_STATE_KEY]: defaultSelfState() } })
    const context = withGatewayEnv(store)
    const gateway = mockGateway(
      llmToolCallResponse('workspace_list', {}),
      llmTextResponse('沙箱不可用，那就只写几句思考。'),
    )

    const result = await runHeartbeat(context)

    assert.equal(gateway.calls, 2)
    assert.equal(result.toolCount, 1)
    // The workspace tool was attempted but could not touch a sandbox; the
    // heartbeat must not throw, and should persist the AI's fallback text.
    assert.equal(result.touchedSandbox, true)
    assert.equal(result.savedFiles, 0)
    assert.ok(result.text.includes('沙箱不可用'))
    assert.equal(lastLogKind(store), 'heartbeat')

    const saved = savedSelfState(store)
    assert.ok(saved.lastActivityAt > 0)
  })

  test('onRequest returns 200 when the no-sandbox AI falls back to text', async () => {
    const store = makeMockStore({ [SELF_ID]: { [SELF_STATE_KEY]: defaultSelfState() } })
    mockGateway(
      llmToolCallResponse('workspace_list', {}),
      llmTextResponse('没有沙箱，先思考。'),
    )

    const res = await heartbeatOnRequest(withGatewayEnv(store))
    assert.equal(res.status, 200)
    const body = (await res.json()) as { ok: boolean; touchedSandbox: boolean; text: string }
    assert.equal(body.ok, true)
    assert.equal(body.touchedSandbox, true)
    assert.ok(body.text.includes('没有沙箱'))
  })

  test('empty output degrades to a fallback log entry and still updates state', async () => {
    const store = makeMockStore({ [SELF_ID]: { [SELF_STATE_KEY]: defaultSelfState() } })
    const context = withGatewayEnv(store)
    const gateway = mockGateway(llmTextResponse(''))

    const result = await runHeartbeat(context)

    assert.equal(gateway.calls, 1)
    assert.ok(result.text.length > 0)
    assert.equal(store.messageLog.length, 2) // trigger + assistant fallback
    assert.ok(store.messageLog[1]?.content.length > 0)

    const saved = savedSelfState(store)
    assert.ok(saved.lastActivityAt > 0)
  })

  test('gateway failure: runHeartbeat rejects, onRequest maps to a stable 500', async () => {
    const store = makeMockStore({ [SELF_ID]: { [SELF_STATE_KEY]: defaultSelfState() } })
    mockGatewayError(503)

    const res = await heartbeatOnRequest(withGatewayEnv(store))
    assert.equal(res.status, 500)
    const body = (await res.json()) as { ok: boolean; error: string }
    assert.equal(body.ok, false)
    assert.match(body.error, /AI gateway HTTP 503/)
  })

  test('LLM 503 with zero workspace calls: sandbox is never activated (no commands.run)', async () => {
    const store = makeMockStore({ [SELF_ID]: { [SELF_STATE_KEY]: defaultSelfState() } })
    const sandbox = makeMockSandbox()
    const runCount = { value: 0 }
    const originalRun = sandbox.commands.run.bind(sandbox.commands)
    mock.method(sandbox.commands, 'run', async (cmd: string, opts?: { cwd?: string; timeout?: number }) => {
      runCount.value += 1
      return originalRun(cmd, opts)
    })
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const context = withGatewayEnv(store, sandbox)
    mockGatewayError(503)

    await assert.rejects(() => runHeartbeat(context), /AI gateway HTTP 503/)

    // A pure-text heartbeat that failed at the LLM must not warm up the
    // sandbox just to snapshot nothing (P2-3).
    assert.equal(runCount.value, 0)
    assert.equal(sandbox.filesMap.size, 0)
  })

  test('LLM failure AFTER a workspace_* call: sandbox is snapshotted (partial progress saved)', async () => {
    const store = makeMockStore({ [SELF_ID]: { [SELF_STATE_KEY]: defaultSelfState() } })
    const sandbox = makeMockSandbox()
    const runCount = { value: 0 }
    const originalRun = sandbox.commands.run.bind(sandbox.commands)
    mock.method(sandbox.commands, 'run', async (cmd: string, opts?: { cwd?: string; timeout?: number }) => {
      runCount.value += 1
      return originalRun(cmd, opts)
    })
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const context = withGatewayEnv(store, sandbox)
    let fetchCount = 0
    mock.method(globalThis, 'fetch', async () => {
      fetchCount += 1
      if (fetchCount === 1) {
        return new Response(
          JSON.stringify(llmToolCallResponse('workspace_write', {
            path: 'draft.ts',
            content: 'export const draft = 1;\n',
          })),
          { status: 200, headers: { 'content-type': 'application/json' } },
        )
      }
      return new Response('boom', { status: 503 })
    })

    await assert.rejects(() => runHeartbeat(context), /AI gateway HTTP 503/)

    // A workspace tool DID run this turn, so the failure path snapshots the
    // sandbox to capture command-generated files too.
    assert.ok(runCount.value >= 1)
    // The written file survives in Blob (mirrored on write + snapshot).
    assert.equal(blob.blobMap.get('projects/eo-self/workspace/draft.ts'), 'export const draft = 1;\n')
  })

  test('onRequest returns a 200 JSON envelope with text/turns/touchedSandbox', async () => {
    const store = makeMockStore({ [SELF_ID]: { [SELF_STATE_KEY]: defaultSelfState() } })
    mockGateway(llmTextResponse('今天心情不错，留个脚印。'))

    const res = await heartbeatOnRequest(withGatewayEnv(store))
    assert.equal(res.status, 200)
    const body = (await res.json()) as { ok: boolean; text: string; turns: number; touchedSandbox: boolean }
    assert.equal(body.ok, true)
    assert.ok(body.text.includes('今天心情不错'))
    assert.equal(body.turns, 1)
    assert.equal(body.touchedSandbox, false)
    assert.equal('mood' in body, false)
    assert.equal('energy' in body, false)
  })

  test('onRequest surfaces 500 when the store is missing (never crashes)', async () => {
    const context = makeContext({ env: gatewayEnv() as Env })
    mockGateway(llmTextResponse('想写点什么。'))

    const res = await heartbeatOnRequest(context)
    assert.equal(res.status, 500)
    const body = (await res.json()) as { ok: boolean; error: string }
    assert.equal(body.ok, false)
    assert.match(body.error, /Store is not available/)
  })

  test('auto-compacts an overgrown context before the decision turn', async () => {
    const store = makeMockStore({ [SELF_ID]: { [SELF_STATE_KEY]: defaultSelfState() } })
    for (let i = 0; i < 6000; i += 1) {
      store.addMessage(SELF_ID, { role: 'assistant', content: `old-${i}`, metadata: { logKind: 'think' } })
    }
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const context = withGatewayEnv(store)
    // Payload[0] feeds the compact LLM call, payload[1] feeds the decision turn.
    const gateway = mockGateway(
      llmTextResponse('早期记录被压缩。'),
      llmTextResponse('醒来，一切还好。'),
    )

    const result = await runHeartbeat(context)

    assert.equal(gateway.calls, 2) // compact + decision
    assert.ok(result.text.includes('醒来，一切还好。'))
    const messages = store.logs.get(SELF_ID) ?? []
    // 6000 - 1200 compacted + 1 summary + 1 trigger user + 1 heartbeat log
    assert.equal(messages.length, 4803)
    assert.ok(messages.some((message) => (message.metadata as { kind?: string } | undefined)?.kind === 'summary'))
    assert.equal(lastLogKind(store), 'heartbeat')
  })

  test('request body is a standard messages array: history entries + the wake trigger', async () => {
    const store = makeMockStore({ [SELF_ID]: { [SELF_STATE_KEY]: defaultSelfState() } })
    store.addMessage(SELF_ID, { role: 'assistant', content: '早先折叠进摘要的内容', metadata: { kind: 'summary' } })
    for (let i = 0; i < 30; i += 1) {
      store.addMessage(SELF_ID, { role: 'assistant', content: `log-${i}`, metadata: { logKind: 'think' } })
    }
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const context = withGatewayEnv(store)
    const bodies: Array<Record<string, unknown>> = []
    mock.method(globalThis, 'fetch', async (_input: unknown, init?: RequestInit) => {
      if (init?.body) bodies.push(JSON.parse(String(init.body)) as Record<string, unknown>)
      return new Response(JSON.stringify(llmTextResponse('照常生活。')), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      })
    })

    await runHeartbeat(context)

    // Usage is tiny, so no compact call: bodies[0] is the decision turn.
    assert.ok(bodies.length >= 1)
    const decision = bodies[0] as { messages: Array<{ role: string; content: string }> }
    assert.ok(Array.isArray(decision.messages))
    // One system message, then the full history as independent role entries.
    assert.ok(decision.messages.some((message) => message.role === 'system'))
    // The wake trigger is its own user message (not a clamped context blob).
    const trigger = decision.messages.find((message) => message.role === 'user')
    assert.ok(trigger, 'decision request has a user (wake) message')
    assert.match(trigger?.content ?? '', /heartbeat 醒来/)
    // History entries are independent messages — an early log a ".slice(-15)"
    // window would have dropped is present.
    assert.ok(decision.messages.some((message) => message.role === 'assistant' && message.content === 'log-0'))
    assert.ok(decision.messages.some((message) => message.role === 'assistant' && message.content === 'log-29'))
    // The compact summary message is in place BEFORE the recent stream, with a
    // [system][compact] identity prefix so the model does not read it as speech.
    const summaryIndex = decision.messages.findIndex((message) => message.content.includes('早先折叠进摘要的内容'))
    const log0Index = decision.messages.findIndex((message) => message.content === 'log-0')
    assert.ok(summaryIndex >= 0 && summaryIndex < log0Index)
    assert.match(decision.messages[summaryIndex]?.content ?? '', /^\[system\]\[compact\] /)
    // The wake trigger is marked with [system][heartbeat] but stays role 'user'.
    assert.match(trigger?.content ?? '', /^\[system\]\[heartbeat\] /)
    // The user trigger carries the wake text but no memory section (memory is in system).
    assert.doesNotMatch(trigger?.content ?? '', /## 我的记忆/)
  })

  test('system prompt = dynamic wall clock + MEMORY.md self, no MOOD/ENERGY hints', async () => {
    const store = makeMockStore({ [SELF_ID]: { [SELF_STATE_KEY]: defaultSelfState() } })
    const blob = makeMockBlobStore({ 'memory/MEMORY.md': '# 我的记忆\n我是小蓝，喜欢安静地写代码。' })
    injectBlobStoreForTesting(blob)
    const context = withGatewayEnv(store)
    const bodies: Array<Record<string, unknown>> = []
    mock.method(globalThis, 'fetch', async (_input: unknown, init?: RequestInit) => {
      if (init?.body) bodies.push(JSON.parse(String(init.body)) as Record<string, unknown>)
      return new Response(JSON.stringify(llmTextResponse('今天适合休息。')), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      })
    })

    await runHeartbeat(context)

    assert.ok(bodies.length >= 1)
    const decision = bodies[0] as { messages: Array<{ role: string; content: string }> }
    const system = decision.messages.find((message) => message.role === 'system')
    assert.ok(system, 'decision request has a system message')
    // Dynamic wall clock only (no fake mood/energy/project injected).
    assert.match(system.content, /现在是 20\d{2}-\d{2}-\d{2} 星期[一二三四五六日] \d{2}:\d{2}/)
    assert.match(system.content, /## 我的记忆（MEMORY\.md）/)
    assert.match(system.content, /我是小蓝，喜欢安静地写代码。/)
    // The heartbeat identity line and quiet-life guidance are present.
    assert.match(system.content, /你不是聊天机器人/)
    assert.match(system.content, /沉默也是合法的活法/)
    assert.match(system.content, /把这段独处过好/)
    // The [system]-prefix identity guidance explains heartbeat + compact history.
    assert.match(system.content, /\[system\]\[heartbeat\]/)
    assert.match(system.content, /\[system\]\[compact\]/)
    assert.match(system.content, /不是用户说的/)
    // No mood/energy output markers are requested or parsed anymore.
    assert.doesNotMatch(system.content, /MOOD\s*:|ENERGY\s*:/)
    // The old menu-style option enumeration is gone.
    assert.doesNotMatch(system.content, /你可以：/)
    // Existing memory is left untouched (never overwritten by the seed).
    assert.equal(blob.blobMap.get('memory/MEMORY.md'), '# 我的记忆\n我是小蓝，喜欢安静地写代码。')
  })

  test('DECISION_SYSTEM explains the identity of heartbeat and compact history', () => {
    assert.match(DECISION_SYSTEM, /不是用户说的/)
    assert.match(DECISION_SYSTEM, /\[system\]\[heartbeat\]/)
    assert.match(DECISION_SYSTEM, /\[system\]\[compact\]/)
    assert.match(DECISION_SYSTEM, /没有 \[system\] 前缀的消息才是真实的对话/)
    // The explanation sits at the top, before the "you are not a chatbot" line.
    assert.ok(DECISION_SYSTEM.indexOf('[system]') < DECISION_SYSTEM.indexOf('你不是聊天机器人'))
  })

  test('the system message sent to the gateway is unique and first', async () => {
    const store = makeMockStore({ [SELF_ID]: { [SELF_STATE_KEY]: defaultSelfState() } })
    store.addMessage(SELF_ID, { role: 'user', content: '早先的问题', metadata: {} })
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const context = withGatewayEnv(store)
    const bodies: Array<Record<string, unknown>> = []
    mock.method(globalThis, 'fetch', async (_input: unknown, init?: RequestInit) => {
      if (init?.body) bodies.push(JSON.parse(String(init.body)) as Record<string, unknown>)
      return new Response(JSON.stringify(llmTextResponse('照常生活。')), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      })
    })

    await runHeartbeat(context)

    assert.ok(bodies.length >= 1)
    const decision = bodies[0] as { messages: Array<{ role: string; content: string }> }
    const systems = decision.messages.filter((message) => message.role === 'system')
    assert.equal(systems.length, 1)
    // The single system message (persona + decision guidance) is index 0, before
    // every history entry — loadMessages skips stored system rows.
    assert.equal(decision.messages[0]?.role, 'system')
    assert.equal(decision.messages[0], systems[0])
  })

  test('first heartbeat seeds MEMORY.md with the birth guide, once', async () => {
    const store = makeMockStore({ [SELF_ID]: { [SELF_STATE_KEY]: defaultSelfState() } })
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const context = withGatewayEnv(store)
    const gateway = mockGateway(llmTextResponse('我刚醒来，还不认识自己。先留一句。'))

    await runHeartbeat(context)
    assert.equal(gateway.calls, 1)

    assert.ok(blob.blobMap.has('memory/MEMORY.md'))
    const memory = blob.blobMap.get('memory/MEMORY.md') ?? ''
    assert.equal(memory, INITIAL_MEMORY_SEED)
    assert.ok(memory.includes('MEMORY.md'))
    assert.ok(memory.includes('我是谁'))
    assert.ok(memory.includes('我认识的人'))
    assert.ok(memory.includes('我学会的事'))

    // A second heartbeat never overwrites the (now user-owned) memory.
    await runHeartbeat(context)
    assert.equal(blob.blobMap.get('memory/MEMORY.md'), memory)
  })

  test('diary is not injected into the prompt, but the output is still appended to today\'s diary', async () => {
    const store = makeMockStore({ [SELF_ID]: { [SELF_STATE_KEY]: defaultSelfState() } })
    const blob = makeMockBlobStore({ 'memory/daily/2026-09-17.md': 'SECRET_DIARY_CONTENT' })
    injectBlobStoreForTesting(blob)
    const context = withGatewayEnv(store)
    const bodies: Array<Record<string, unknown>> = []
    mock.method(globalThis, 'fetch', async (_input: unknown, init?: RequestInit) => {
      if (init?.body) bodies.push(JSON.parse(String(init.body)) as Record<string, unknown>)
      return new Response(JSON.stringify(llmTextResponse('今天的输出，留进日记。')), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      })
    })

    const result = await runHeartbeat(context)

    assert.ok(result.text.includes('今天的输出，留进日记。'))
    // The gateway request must never carry the diary content or a diary section.
    const allPromptText = bodies.map((body) => JSON.stringify(body)).join('\n')
    assert.ok(!allPromptText.includes('SECRET_DIARY_CONTENT'))
    assert.ok(!allPromptText.includes('## 最近日记'))
    // The heartbeat still appends its output to today's diary file.
    const todayKey = `memory/daily/${dateKey(new Date())}.md`
    const written = blob.blobMap.get(todayKey) ?? ''
    assert.ok(written.includes('今天的输出，留进日记。'))
    assert.equal(store.messageLog.length, 2) // wake trigger + assistant reply
    assert.equal(lastLogKind(store), 'heartbeat')
  })
})