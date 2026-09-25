/**
 * Tests for the persisted agent state layer (`_state.ts`) using a mock
 * `store.state` KV.
 */
import { describe, test } from 'node:test'
import assert from 'node:assert/strict'
import {
  SELF_STATE_KEY,
  clearState,
  defaultState,
  getSelfState,
  getState,
  updateState,
  type AgentState,
} from '../agents/_state.ts'
import { SELF_ID } from '../agents/_shared.ts'
import { makeContext, makeMockStore } from './_helpers.ts'

describe('defaultState', () => {
  test('returns a stable baseline with only lastActivityAt and created', () => {
    const state = defaultState()
    assert.equal(state.lastActivityAt, 0)
    assert.ok(typeof state.created === 'number' && state.created > 0)
    // Mood/energy/project are not state anymore — they belong to MEMORY.md.
    assert.equal('mood' in state, false)
    assert.equal('energy' in state, false)
    assert.equal('currentProject' in state, false)
    assert.equal('projectStep' in state, false)
    assert.equal('lastDreamAt' in state, false)
  })
})

describe('getState', () => {
  test('returns a stored valid state as-is', async () => {
    const stored: AgentState = {
      lastActivityAt: 1,
      created: 3,
    }
    const store = makeMockStore({ [SELF_ID]: { [SELF_STATE_KEY]: stored } })
    const state = await getState(makeContext({ store }), SELF_ID)
    assert.deepEqual(state, stored)
  })

  test('falls back to defaultState when nothing is stored', async () => {
    const store = makeMockStore()
    const state = await getState(makeContext({ store }), SELF_ID)
    // Shape matches defaultState (lastActivityAt=0, created = a fresh epoch).
    // The exact `created` value is not asserted against a second
    // `defaultState()` call: each call captures its own `Date.now()`, so the
    // two could differ by 1ms across a millisecond boundary (pre-existing
    // timing race in this test).
    assert.equal(state.lastActivityAt, 0)
    assert.ok(Number.isFinite(state.created) && state.created > 0)
  })

  test('falls back to defaultState for malformed stored values', async () => {
    const store = makeMockStore({ [SELF_ID]: { [SELF_STATE_KEY]: { lastActivityAt: 'x', nope: true } } })
    const state = await getState(makeContext({ store }), SELF_ID)
    assert.equal(state.lastActivityAt, defaultState().lastActivityAt)
  })

  test('rejects when the store is unavailable', async () => {
    await assert.rejects(getState(makeContext({}), SELF_ID), /Store is not available/)
  })
})

describe('updateState', () => {
  test('merges the patch, persists it and preserves created', async () => {
    const store = makeMockStore({ [SELF_ID]: { [SELF_STATE_KEY]: { ...defaultState(), created: 111 } } })
    const next = await updateState(makeContext({ store }), SELF_ID, { lastActivityAt: 55 })
    assert.equal(next.lastActivityAt, 55)
    assert.equal(next.created, 111)
    const persisted = store.stateMap.get(SELF_ID)?.get(SELF_STATE_KEY) as AgentState
    assert.equal(persisted.lastActivityAt, 55)
    assert.equal(persisted.created, 111)
  })
})

describe('SELF state (single shared conversation)', () => {
  test('getSelfState returns what updateState persisted', async () => {
    const store = makeMockStore()
    const context = makeContext({ store })

    await updateState(context, SELF_ID, { lastActivityAt: 100 })

    const self = await getSelfState(context)
    assert.equal(self.lastActivityAt, 100)
    const persisted = store.stateMap.get(SELF_ID)?.get(SELF_STATE_KEY) as AgentState | undefined
    assert.equal(persisted?.lastActivityAt, 100)
  })
})

describe('clearState', () => {
  test('deletes the state key', async () => {
    const store = makeMockStore({ [SELF_ID]: { [SELF_STATE_KEY]: defaultState() } })
    await clearState(makeContext({ store }), SELF_ID)
    assert.equal(store.stateMap.get(SELF_ID)?.has(SELF_STATE_KEY), false)
  })
})