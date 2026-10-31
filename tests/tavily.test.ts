/**
 * Tests for the Tavily executors (_tavily.ts): web_search validation
 * (notably the mutually-exclusive time filters) and the web_fetch extract tool.
 *
 * `globalThis.fetch` is mocked per test and restored in `afterEach` — no
 * network dependency, mirroring the rest of the suite.
 */
import { afterEach, describe, test } from 'node:test'
import assert from 'node:assert/strict'
import { mock } from 'node:test'
import { executeWebFetch, executeWebSearch } from '../agents/_tavily.ts'

const API_KEY = 'test-tavily-key'

afterEach(() => {
  mock.restoreAll()
})

describe('executeWebSearch time filters', () => {
  test('rejects time_range combined with start_date/end_date as mutually exclusive', async () => {
    const result = await executeWebSearch(API_KEY, { query: 'x', time_range: 'week', start_date: '2026-09-01' })
    assert.equal(result.isError, true)
    assert.match(result.content, /mutually exclusive/)

    const endOnly = await executeWebSearch(API_KEY, { query: 'x', time_range: 'month', end_date: '2026-09-30' })
    assert.equal(endOnly.isError, true)
    assert.match(endOnly.content, /mutually exclusive/)
  })

  test('accepts time_range alone or start/end dates alone (validation passes before the empty-key check)', async () => {
    // Empty API key stops execution before any fetch, so these only exercise
    // the time-filter validation: they must NOT hit the mutual-exclusion guard.
    const rangeOnly = await executeWebSearch('', { query: 'x', time_range: 'day' })
    assert.equal(rangeOnly.isError, true)
    assert.match(rangeOnly.content, /not configured/)
    assert.ok(!rangeOnly.content.includes('mutually exclusive'), 'time_range alone is not a mutual-exclusion error')

    const datesOnly = await executeWebSearch('', { query: 'x', start_date: '2026-09-01', end_date: '2026-09-30' })
    assert.equal(datesOnly.isError, true)
    assert.match(datesOnly.content, /not configured/)
    assert.ok(!datesOnly.content.includes('mutually exclusive'), 'dates alone are not a mutual-exclusion error')
  })
})

describe('executeWebFetch input validation', () => {
  test('rejects missing urls', async () => {
    const result = await executeWebFetch(API_KEY, {})
    assert.equal(result.isError, true)
    assert.match(result.content, /urls/)
  })

  test('rejects a non-array urls value', async () => {
    const result = await executeWebFetch(API_KEY, { urls: 'https://example.com' })
    assert.equal(result.isError, true)
    assert.match(result.content, /urls/)
  })

  test('rejects an empty / non-string urls array', async () => {
    const empty = await executeWebFetch(API_KEY, { urls: [] })
    assert.equal(empty.isError, true)
    assert.match(empty.content, /urls/)

    const nonString = await executeWebFetch(API_KEY, { urls: ['https://example.com', 42] })
    assert.equal(nonString.isError, true)
    assert.match(nonString.content, /urls/)

    const blank = await executeWebFetch(API_KEY, { urls: ['   '] })
    assert.equal(blank.isError, true)
    assert.match(blank.content, /urls/)
  })

  test('rejects more than 5 urls', async () => {
    const urls = Array.from({ length: 6 }, (_, i) => `https://example.com/${i + 1}`)
    const result = await executeWebFetch(API_KEY, { urls })
    assert.equal(result.isError, true)
    assert.match(result.content, /at most 5/)
  })

  test('rejects an out-of-range / non-integer max_chars', async () => {
    const tooSmall = await executeWebFetch(API_KEY, { urls: ['https://example.com'], max_chars: 50 })
    assert.equal(tooSmall.isError, true)
    assert.match(tooSmall.content, /max_chars/)

    const tooBig = await executeWebFetch(API_KEY, { urls: ['https://example.com'], max_chars: 9000 })
    assert.equal(tooBig.isError, true)
    assert.match(tooBig.content, /max_chars/)

    const fractional = await executeWebFetch(API_KEY, { urls: ['https://example.com'], max_chars: 1500.5 })
    assert.equal(fractional.isError, true)
    assert.match(fractional.content, /max_chars/)
  })

  test('returns isError when TAVILY_API_KEY is empty', async () => {
    const result = await executeWebFetch('   ', { urls: ['https://example.com'] })
    assert.equal(result.isError, true)
    assert.match(result.content, /TAVILY_API_KEY/)
  })
})

describe('executeWebFetch success formatting', () => {
  test('formats a successful extract result and sends the expected /extract request', async () => {
    const captured: { url: string; init: RequestInit } = { url: '', init: {} }
    mock.method(globalThis, 'fetch', async (input: unknown, init?: RequestInit) => {
      captured.url = String(input)
      captured.init = init ?? {}
      return new Response(
        JSON.stringify({
          results: [
            { url: 'https://example.com/page', title: 'Example Page', content: 'Some page body text.' },
          ],
          failed_results: [],
        }),
        { status: 200, headers: { 'content-type': 'application/json' } },
      )
    })

    const result = await executeWebFetch(API_KEY, { urls: ['https://example.com/page'], max_chars: 500 })
    assert.equal(result.isError, undefined)
    assert.match(result.content, /Example Page/)
    assert.match(result.content, /Some page body text/)

    assert.equal(captured.url, 'https://api.tavily.com/extract')
    const headers = captured.init.headers as Record<string, string>
    assert.equal(headers.Authorization, `Bearer ${API_KEY}`)
    assert.equal(headers['Content-Type'], 'application/json')
    const body = JSON.parse(String(captured.init.body)) as { urls: string[]; max_chars: number }
    assert.deepEqual(body.urls, ['https://example.com/page'])
    assert.equal(body.max_chars, 500)
  })

  test('truncates long page content to max_chars with an ellipsis', async () => {
    mock.method(globalThis, 'fetch', async () =>
      new Response(
        JSON.stringify({
          results: [{ url: 'https://example.com/long', title: 'Long Page', content: 'A'.repeat(200) }],
          failed_results: [],
        }),
        { status: 200 },
      ))

    const result = await executeWebFetch(API_KEY, { urls: ['https://example.com/long'], max_chars: 100 })
    assert.equal(result.isError, undefined)
    assert.ok(result.content.includes('…'))
    assert.ok(!result.content.includes('A'.repeat(150)), 'content beyond max_chars must be cut off')
  })

  test('marks failed URLs from failed_results', async () => {
    mock.method(globalThis, 'fetch', async () =>
      new Response(
        JSON.stringify({
          results: [{ url: 'https://example.com/ok', title: 'OK', content: 'fine' }],
          failed_results: [{ url: 'https://example.com/bad', error: '404 Not Found' }],
        }),
        { status: 200 },
      ))

    const result = await executeWebFetch(API_KEY, { urls: ['https://example.com/ok', 'https://example.com/bad'] })
    assert.equal(result.isError, undefined)
    assert.match(result.content, /https:\/\/example\.com\/bad/)
    assert.match(result.content, /failed/)
    assert.match(result.content, /404 Not Found/)
  })
})

describe('executeWebFetch failures', () => {
  test('returns isError when fetch throws (network failure)', async () => {
    mock.method(globalThis, 'fetch', async () => {
      throw new Error('socket hang up')
    })
    const result = await executeWebFetch(API_KEY, { urls: ['https://example.com/page'] })
    assert.equal(result.isError, true)
    assert.match(result.content, /socket hang up/)
  })

  test('returns isError when the extract backend responds non-200', async () => {
    mock.method(globalThis, 'fetch', async () => new Response('bad gateway', { status: 502, statusText: 'Bad Gateway' }))
    const result = await executeWebFetch(API_KEY, { urls: ['https://example.com/page'] })
    assert.equal(result.isError, true)
    assert.match(result.content, /HTTP 502/)
  })
})
