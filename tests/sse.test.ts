/**
 * Tests for the SSE transport helper (`_shared.ts#createSSEResponse`) —
 * P2-7 client-disconnect robustness. A cancelled controller must stop the
 * generator loop, and a generator error racing a disconnect must never escape
 * through `controller.enqueue` (enqueue on a cancelled stream throws).
 */
import { describe, test } from 'node:test'
import assert from 'node:assert/strict'
import { createSSEResponse, sseEvent } from '../agents/_shared.ts'

async function drain(reader: ReadableStreamDefaultReader<Uint8Array>): Promise<void> {
  for (;;) {
    const { done } = await reader.read()
    if (done) break
  }
}

describe('createSSEResponse client disconnect (P2-7)', () => {
  test('cancel() resolves and the response stays drainable without throwing', async () => {
    async function* gen(): AsyncGenerator<string> {
      yield sseEvent({ type: 'ai_response', content: 'a' })
      yield sseEvent({ type: 'ai_response', content: 'b' })
      yield sseEvent({ type: 'ai_response', content: 'c' })
    }
    const res = createSSEResponse(gen)
    const reader = res.body?.getReader()
    assert.ok(reader)

    const first = await reader.read()
    assert.equal(first.done, false)
    assert.ok(first.value && first.value.byteLength > 0)

    // Client abandons the stream mid-flight — cancel() must resolve cleanly.
    await assert.doesNotReject(() => reader.cancel())
    // A later read reports the stream is closed (done), not a throw.
    const after = await reader.read()
    assert.equal(after.done, true)
  })

  test('a generator error after cancel() is swallowed — no crash on enqueue', async () => {
    let released = false
    async function* gen(): AsyncGenerator<string> {
      yield sseEvent({ type: 'ai_response', content: 'a' })
      // Spin until the test releases us AFTER cancelling the reader.
      while (!released) {
        await new Promise((resolve) => setTimeout(resolve, 5))
      }
      throw new Error('boom after disconnect')
    }

    const res = createSSEResponse(gen)
    const reader = res.body?.getReader()
    assert.ok(reader)

    const first = await reader.read()
    assert.equal(first.done, false)

    // Client disconnects, then the producer tries to continue and throws.
    await reader.cancel()
    released = true
    // Give the start loop a tick to hit the throwing step after the disconnect.
    await new Promise((resolve) => setTimeout(resolve, 30))

    // Draining the cancelled stream must never reject — the loop observed the
    // cancelled flag and never attempted the error frame on the dead controller.
    await assert.doesNotReject(() => drain(reader))
  })
})
