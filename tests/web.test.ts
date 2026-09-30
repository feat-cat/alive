/**
 * Tests for the web frontend (`web/index.html`) loaded in a Node `vm`
 * sandbox with a minimal DOM stub. The page's inline script defines
 * `parseSseEvents`, which must recognize the new `reasoning_delta` SSE event
 * (DeepSeek thinking) alongside `ai_response` / `error_message` / `DONE`.
 *
 * The vm sandbox runs the page's `init()` (file:// protocol → the
 * `renderNeedServer` branch, so no fetch/network is touched), then exposes
 * `window.__alive` exactly like a browser would.
 */
import { describe, test } from 'node:test'
// Loose assert (not /strict): events are created inside a vm realm whose
// Object.prototype differs from the test realm, so deepStrictEqual would reject
// them as "same structure but not reference-equal". Loose deepEqual compares
// own properties only, which is exactly what we want here.
import assert from 'node:assert'
import { readFileSync } from 'node:fs'
import * as path from 'node:path'
import * as vm from 'node:vm'
import { fileURLToPath } from 'node:url'

interface FakeEl {
  textContent: string
  className: string
  value: string
  hidden: boolean
  disabled: boolean
  scrollTop: number
  scrollHeight: number
  clientHeight: number
  style: Record<string, string>
  appendChild(child: unknown): unknown
  addEventListener(): void
  removeEventListener(): void
  click(): void
}

function makeFakeEl(): FakeEl {
  return {
    textContent: '',
    className: '',
    value: '',
    hidden: false,
    disabled: false,
    scrollTop: 0,
    scrollHeight: 0,
    clientHeight: 0,
    style: {},
    appendChild(child: unknown) {
      return child
    },
    addEventListener() {},
    removeEventListener() {},
    click() {},
  }
}

interface SseEvent {
  type: string
  content?: string
  streamed?: boolean
}

interface AliveApi {
  parseSseEvents(text: string): SseEvent[]
  [key: string]: unknown
}

const INDEX_HTML_PATH = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  '..',
  '..',
  'web',
  'index.html',
)

const INDEX_HTML = readFileSync(INDEX_HTML_PATH, 'utf8')

/** Extract the single inline `<script>` block from index.html. */
const PAGE_SCRIPT = (() => {
  const open = INDEX_HTML.indexOf('<script>')
  const close = INDEX_HTML.indexOf('</script>')
  assert.ok(open >= 0 && close > open, 'web/index.html must contain an inline <script> block')
  return INDEX_HTML.slice(open + '<script>'.length, close)
})()

/** Load the page script in a vm context with a DOM stub and return `window.__alive`. */
function loadAliveApi(): AliveApi {
  const sandbox = {
    console,
    setTimeout,
    clearTimeout,
    location: { protocol: 'file:' },
    document: {
      getElementById: () => makeFakeEl(),
      createElement: () => makeFakeEl(),
      createTextNode: (text: string) => ({ text }),
      createDocumentFragment: () => makeFakeEl(),
    },
  }
  const ctx = vm.createContext(sandbox)
  ;(ctx as Record<string, unknown>).window = ctx
  vm.runInContext(PAGE_SCRIPT, ctx, { filename: 'web/index.html' })
  const api = (ctx as Record<string, unknown>).__alive
  assert.ok(api && typeof api === 'object', 'window.__alive must be exposed after init()')
  return api as AliveApi
}

const alive = loadAliveApi()

describe('web frontend parseSseEvents (reasoning_delta)', () => {
  test('parses reasoning_delta events alongside ai_response and DONE', () => {
    const text =
      'data: {"type":"reasoning_delta","content":"先想想"}\n\n' +
      'data: {"type":"reasoning_delta","content":"再判断"}\n\n' +
      'data: {"type":"ai_response","content":"你好","streamed":true}\n\n' +
      'data: [DONE]\n\n'
    const events = alive.parseSseEvents(text)
    assert.deepEqual(events, [
      { type: 'reasoning_delta', content: '先想想' },
      { type: 'reasoning_delta', content: '再判断' },
      { type: 'ai_response', content: '你好', streamed: true },
      { type: 'DONE' },
    ])
  })

  test('a pure ai_response stream yields no reasoning_delta events', () => {
    const text = 'data: {"type":"ai_response","content":"ok","streamed":true}\n\n'
    const events = alive.parseSseEvents(text)
    assert.ok(events.every((event) => event.type !== 'reasoning_delta'))
  })

  test('ignores unknown event types (e.g. the periodic ping frame)', () => {
    const text = 'data: {"type":"ping","ts":1234567890}\n\n'
    const events = alive.parseSseEvents(text)
    assert.deepEqual(events, [])
  })

  test('parses tool_call and tool_result events alongside ai_response and DONE', () => {
    const text =
      'data: {"type":"tool_call","name":"web_search","arguments":{"query":"x"}}\n\n' +
      'data: {"type":"tool_result","name":"web_search","content":"Web search is not configured."}\n\n' +
      'data: {"type":"ai_response","content":"我查完了。","streamed":true}\n\n' +
      'data: [DONE]\n\n'
    const events = alive.parseSseEvents(text)
    assert.deepEqual(events, [
      { type: 'tool_call', name: 'web_search', arguments: { query: 'x' } },
      { type: 'tool_result', name: 'web_search', content: 'Web search is not configured.' },
      { type: 'ai_response', content: '我查完了。', streamed: true },
      { type: 'DONE' },
    ])
  })

  test('a pure ai_response stream yields no tool events', () => {
    const text = 'data: {"type":"ai_response","content":"ok","streamed":true}\n\n'
    const events = alive.parseSseEvents(text)
    assert.ok(events.every((event) => event.type !== 'tool_call' && event.type !== 'tool_result'))
  })
})
