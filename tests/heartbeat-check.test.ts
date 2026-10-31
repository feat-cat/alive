/**
 * Tests for the manual heartbeat diagnostics endpoint (`agents/heartbeat-check.ts`).
 *
 * It must reuse `runHeartbeat` UNCHANGED — the whole point of /heartbeat-check
 * is that a manual trigger exercises the SAME logic as the scheduled heartbeat
 * (auto-compact, wake-trigger message, LLM loop, tools, diary, state) and
 * surfaces the full HeartbeatResult envelope. The gateway fetch is mocked the
 * same way as the heartbeat tests.
 */
import { afterEach, describe, test } from 'node:test'
import assert from 'node:assert/strict'
import { mock } from 'node:test'
import { injectBlobStoreForTesting } from '../agents/_blob-tools.ts'
import { onRequest as heartbeatCheckOnRequest } from '../agents/heartbeat-check.ts'
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

function defaultSelfState(): AgentState {
  return { lastActivityAt: 0, created: 1 }
}

function withGatewayEnv(store: MockStore): ReturnType<typeof makeContext> {
  return makeContext({ store, env: gatewayEnv() as Env })
}

afterEach(() => {
  mock.restoreAll()
  injectBlobStoreForTesting(null)
})

describe('GET /heartbeat-check', () => {
  test('manually triggers a full heartbeat and returns the complete HeartbeatResult envelope', async () => {
    const store = makeMockStore({ [SELF_ID]: { [SELF_STATE_KEY]: defaultSelfState() } })
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const gateway = mockGateway(llmTextResponse('手动触发的心跳，一切正常。'))

    const res = await heartbeatCheckOnRequest(withGatewayEnv(store))

    assert.equal(res.status, 200)
    assert.equal(gateway.calls, 1)
    const body = (await res.json()) as {
      ok: boolean
      text: string
      turns: number
      toolCount: number
      touchedSandbox: boolean
      savedFiles: number
      now: string
    }
    assert.equal(body.ok, true)
    assert.ok(body.text.includes('手动触发的心跳'))
    assert.equal(body.turns, 1)
    assert.equal(body.toolCount, 0)
    assert.equal(body.touchedSandbox, false)
    assert.equal(body.savedFiles, 0)
    assert.match(body.now, /^\d{4}-\d{2}-\d{2}T/)
    // The heartbeat ran end-to-end: wake trigger + assistant heartbeat log.
    assert.equal(store.messageLog.length, 2)
    // State was updated exactly like a scheduled heartbeat.
    const state = store.stateMap.get(SELF_ID)?.get(SELF_STATE_KEY) as AgentState | undefined
    assert.ok(state && state.lastActivityAt > 0)
  })

  test('the trigger writes the same history as the scheduled heartbeat (kind=heartbeat)', async () => {
    const store = makeMockStore({ [SELF_ID]: { [SELF_STATE_KEY]: defaultSelfState() } })
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    mockGateway(llmTextResponse('留一句脚印。'))

    await heartbeatCheckOnRequest(withGatewayEnv(store))

    const trigger = store.messageLog[0]
    assert.equal(trigger?.role, 'user')
    assert.equal((trigger?.metadata as { kind?: string } | undefined)?.kind, 'heartbeat')
    const last = store.messageLog.at(-1)
    assert.equal((last?.metadata as { logKind?: string } | undefined)?.logKind, 'heartbeat')
  })

  test('tool-using manual heartbeat reports toolCount / savedFiles like the scheduled run', async () => {
    const store = makeMockStore({ [SELF_ID]: { [SELF_STATE_KEY]: defaultSelfState() } })
    const sandbox = makeMockSandbox()
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const context = makeContext({ store, sandbox, env: gatewayEnv() as Env })
    const gateway = mockGateway(
      {
        choices: [{
          message: {
            content: '',
            tool_calls: [
              { id: 'call_1', type: 'function', function: { name: 'blob_write', arguments: JSON.stringify({ key: 'memory/NOTES.md', content: '- 手动触发的记录。\n' }) } },
            ],
          },
        }],
      },
      llmTextResponse('把记录写好了。'),
    )

    const res = await heartbeatCheckOnRequest(context)

    assert.equal(res.status, 200)
    assert.equal(gateway.calls, 2)
    const body = (await res.json()) as { ok: boolean; toolCount: number; touchedSandbox: boolean; savedFiles: number }
    assert.equal(body.ok, true)
    assert.equal(body.toolCount, 1)
    assert.equal(body.touchedSandbox, false)
    assert.equal(body.savedFiles, 0)
    assert.equal(blob.blobMap.get('memory/NOTES.md'), '- 手动触发的记录。\n')
    // The tool call is recorded into the shared history as a kind='tool' row.
    const toolRecord = store.messageLog.find((message) =>
      (message.metadata as { kind?: string } | undefined)?.kind === 'tool')
    assert.ok(toolRecord, 'tool call is recorded in the store history')
  })

  test('gateway failure maps to a stable 500 (never crashes)', async () => {
    const store = makeMockStore({ [SELF_ID]: { [SELF_STATE_KEY]: defaultSelfState() } })
    mock.method(globalThis, 'fetch', async () => new Response('boom', { status: 503 }))

    const res = await heartbeatCheckOnRequest(withGatewayEnv(store))
    assert.equal(res.status, 500)
    const body = (await res.json()) as { ok: boolean; error: string }
    assert.equal(body.ok, false)
    assert.match(body.error, /AI gateway HTTP 503/)
  })

  test('missing store surfaces a 500 with a clear message', async () => {
    const context = makeContext({ env: gatewayEnv() as Env })
    mockGateway(llmTextResponse('想写点什么。'))

    const res = await heartbeatCheckOnRequest(context)
    assert.equal(res.status, 500)
    const body = (await res.json()) as { ok: boolean; error: string }
    assert.equal(body.ok, false)
    assert.match(body.error, /Store is not available/)
  })
})
