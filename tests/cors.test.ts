/**
 * CORS headers on every JSON response (web frontend cross-origin calls).
 *
 * `jsonOk` / `jsonError` are the single exit point for every endpoint (chat,
 * stop, history, heartbeat), so asserting them here covers all endpoints.
 */
import { describe, test } from 'node:test'
import assert from 'node:assert/strict'
import { jsonError, jsonOk } from '../agents/_shared.ts'

describe('CORS headers on every JSON response', () => {
  test('jsonOk carries access-control-allow-origin: * plus common preflight headers', () => {
    const res = jsonOk({ ping: true })
    assert.equal(res.headers.get('access-control-allow-origin'), '*')
    assert.equal(res.headers.get('access-control-allow-methods'), 'GET,POST,OPTIONS')
    assert.equal(res.headers.get('access-control-allow-headers'), 'content-type,authorization')
  })

  test('jsonError carries access-control-allow-origin: * plus common preflight headers', () => {
    const res = jsonError(401, 'Unauthorized')
    assert.equal(res.headers.get('access-control-allow-origin'), '*')
    assert.equal(res.headers.get('access-control-allow-methods'), 'GET,POST,OPTIONS')
    assert.equal(res.headers.get('access-control-allow-headers'), 'content-type,authorization')
  })
})