/**
 * Optional bearer-token auth (ALIVE_AUTH_TOKEN).
 *
 * Covers the pure helpers (authToken / requireAuth) and the endpoint wiring:
 * when ALIVE_AUTH_TOKEN is unset everything stays open; when set, /chat,
 * /stop and /history require `Authorization: Bearer <token>` and return 401
 * on a missing / wrong / case-mismatched header. /heartbeat is deliberately
 * NOT gated (EdgeOne schedules wake it without a token), so it must still POST
 * successfully even when a token is configured.
 */
import { afterEach, describe, mock, test } from 'node:test'
import assert from 'node:assert/strict'
import { injectBlobStoreForTesting } from '../agents/_blob-tools.ts'
import { onRequest as chatOnRequest } from '../agents/chat.ts'
import { onRequest as stopOnRequest } from '../agents/stop.ts'
import { onRequest as historyOnRequest } from '../agents/history.ts'
import { onRequest as heartbeatOnRequest } from '../agents/heartbeat.ts'
import { authToken, requireAuth, SELF_ID, type Env } from '../agents/_shared.ts'
import { SELF_STATE_KEY } from '../agents/_state.ts'
import {
  gatewayEnv,
  makeContext,
  makeMockBlobStore,
  makeMockStore,
} from './_helpers.ts'

function llmTextResponse(content: string): unknown {
  return { choices: [{ message: { content } }] }
}

/** Gateway env plus an optional bearer token. */
function authedEnv(token: string): Env {
  return { ...(gatewayEnv() as Env), ALIVE_AUTH_TOKEN: token }
}

afterEach(() => {
  mock.restoreAll()
  injectBlobStoreForTesting(null)
})

describe('authToken / requireAuth', () => {
  test('authToken returns the trimmed token from context.env', () => {
    const ctx = makeContext({ env: { ALIVE_AUTH_TOKEN: '  top-secret  ' } })
    assert.equal(authToken(ctx), 'top-secret')
  })

  test('authToken returns empty when unset or whitespace-only', () => {
    assert.equal(authToken(makeContext({ env: {} })), '')
    assert.equal(authToken(makeContext({ env: { ALIVE_AUTH_TOKEN: '   ' } })), '')
  })

  test('requireAuth allows when no token is configured (open mode)', () => {
    assert.equal(requireAuth(makeContext({ env: {} })), null)
  })

  test('requireAuth allows an exact lowercase Bearer header', () => {
    const ctx = makeContext({ env: { ALIVE_AUTH_TOKEN: 'secret' }, headers: { authorization: 'Bearer secret' } })
    assert.equal(requireAuth(ctx), null)
  })

  test('requireAuth accepts the header under the capitalized key', () => {
    const ctx = makeContext({ env: { ALIVE_AUTH_TOKEN: 'secret' }, headers: { Authorization: 'Bearer secret' } })
    assert.equal(requireAuth(ctx), null)
  })

  test('requireAuth rejects a wrong token with a 401 JSON response', async () => {
    const ctx = makeContext({ env: { ALIVE_AUTH_TOKEN: 'secret' }, headers: { authorization: 'Bearer wrong' } })
    const res = requireAuth(ctx)
    assert.ok(res)
    assert.equal(res.status, 401)
    const body = (await res.json()) as { ok: boolean; error: string }
    assert.equal(body.ok, false)
    assert.equal(body.error, 'Unauthorized')
  })

  test('requireAuth rejects a missing header with 401', () => {
    const res = requireAuth(makeContext({ env: { ALIVE_AUTH_TOKEN: 'secret' } }))
    assert.ok(res)
    assert.equal(res.status, 401)
  })

  test('requireAuth rejects a case-mismatched token value', () => {
    const ctx = makeContext({ env: { ALIVE_AUTH_TOKEN: 'Secret' }, headers: { authorization: 'Bearer secret' } })
    const res = requireAuth(ctx)
    assert.ok(res)
    assert.equal(res.status, 401)
  })
})

describe('endpoint auth wiring', () => {
  test('POST /chat returns 401 without a Bearer header when token is set', async () => {
    const res = await chatOnRequest(makeContext({
      store: makeMockStore(),
      env: authedEnv('secret'),
      body: { message: '你好' },
    }))
    assert.equal(res.status, 401)
  })

  test('POST /chat returns 401 with a wrong Bearer token', async () => {
    const res = await chatOnRequest(makeContext({
      store: makeMockStore(),
      env: authedEnv('secret'),
      body: { message: '你好' },
      headers: { authorization: 'Bearer wrong' },
    }))
    assert.equal(res.status, 401)
  })

  test('POST /chat returns 200 with the correct Bearer header', async () => {
    const store = makeMockStore()
    mock.method(globalThis, 'fetch', async () => new Response(
      JSON.stringify(llmTextResponse('你好呀，有什么可以帮你？')),
      { status: 200, headers: { 'content-type': 'application/json' } },
    ))
    const res = await chatOnRequest(makeContext({
      store,
      env: authedEnv('secret'),
      body: { message: '你好' },
      headers: { authorization: 'Bearer secret' },
    }))
    assert.equal(res.status, 200)
    const body = (await res.json()) as { ok: boolean; reply: string }
    assert.equal(body.ok, true)
    assert.equal(body.reply, '你好呀，有什么可以帮你？')
  })

  test('POST /stop returns 401 without a Bearer header when token is set', async () => {
    const res = await stopOnRequest(makeContext({ env: authedEnv('secret'), body: { conversation_id: 'eo-self' } }))
    assert.equal(res.status, 401)
  })

  test('POST /stop returns 200 with the correct Bearer header', async () => {
    const res = await stopOnRequest(makeContext({
      env: authedEnv('secret'),
      body: { conversation_id: 'eo-self' },
      headers: { authorization: 'Bearer secret' },
    }))
    assert.equal(res.status, 200)
    const body = (await res.json()) as { ok: boolean; aborted: boolean }
    assert.equal(body.ok, true)
    assert.equal(body.aborted, false)
  })

  test('GET /history returns 401 without a Bearer header when token is set', async () => {
    const res = await historyOnRequest(makeContext({ env: authedEnv('secret'), url: '/history' }))
    assert.equal(res.status, 401)
  })

  test('GET /history returns 200 with the correct Bearer header', async () => {
    injectBlobStoreForTesting(makeMockBlobStore())
    const res = await historyOnRequest(makeContext({
      env: authedEnv('secret'),
      url: '/history',
      headers: { authorization: 'Bearer secret' },
    }))
    assert.equal(res.status, 200)
    const body = (await res.json()) as { ok: boolean; messages: unknown[] }
    assert.equal(body.ok, true)
    assert.deepEqual(body.messages, [])
  })

  test('POST /heartbeat stays public even with a token configured (no Bearer header)', async () => {
    const store = makeMockStore({ [SELF_ID]: { [SELF_STATE_KEY]: { lastActivityAt: 0, created: 1 } } })
    injectBlobStoreForTesting(makeMockBlobStore())
    mock.method(globalThis, 'fetch', async () => new Response(
      JSON.stringify(llmTextResponse('醒来，一切还好。')),
      { status: 200, headers: { 'content-type': 'application/json' } },
    ))
    const res = await heartbeatOnRequest(makeContext({ store, env: authedEnv('secret') }))
    assert.equal(res.status, 200)
  })
})