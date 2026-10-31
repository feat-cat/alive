/**
 * Tests for the MCP client (_mcp.ts) and its integration into the tool
 * registry (_tools.ts buildToolsAsync).
 *
 * `globalThis.fetch` is mocked per test (node:test `mock.method`) so no real
 * network is touched; short injected timeouts keep the suite fast. The
 * per-instance MCP cache is cleared after each test via `clearMcpCacheForTesting`.
 */
import { afterEach, describe, test, mock } from 'node:test'
import assert from 'node:assert/strict'
import {
  buildMcpTools,
  clearMcpCacheForTesting,
  connectMcpServer,
  discoverMcpClients,
  parseMcpServers,
} from '../agents/_mcp.ts'
import { buildToolsAsync } from '../agents/_tools.ts'
import { SELF_ID } from '../agents/_shared.ts'
import { makeContext } from './_helpers.ts'

interface MockMcpTool {
  name: string
  description?: string
  inputSchema?: Record<string, unknown>
}

interface MockMcpServer {
  url: string
  sessionId?: string
  tools: MockMcpTool[]
  callText?: Record<string, string>
  callError?: Record<string, boolean>
  /** When true, fetch rejects (network failure). */
  fail?: boolean
  /** When set, every request to this server returns this HTTP status. */
  status?: number
  /** When true, the fetch never resolves (times out via the injected timeout). */
  hang?: boolean
}

interface RecordedRequest {
  url: string
  method: string
  headers: Record<string, string>
  body: Record<string, unknown>
}

function makeMcpFetch(servers: MockMcpServer[], log: RecordedRequest[] = []): typeof fetch {
  return (async (input: unknown, init?: RequestInit) => {
    const url = String(input)
    const headers = (init?.headers ?? {}) as Record<string, string>
    const bodyText = typeof init?.body === 'string' ? init.body : ''
    const body = bodyText ? (JSON.parse(bodyText) as Record<string, unknown>) : {}
    const method = typeof body.method === 'string' ? body.method : ''
    log.push({ url, method, headers, body })
    const server = servers.find((s) => s.url === url)
    if (!server) return new Response('not found', { status: 404 })
    if (server.fail) throw new Error('socket hang up')
    if (server.status !== undefined && server.status !== 200) {
      return new Response('server error', { status: server.status, statusText: 'Server Error' })
    }
    if (server.hang) {
      return new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => reject(new Error('aborted')), { once: true })
      })
    }
    const id = typeof body.id === 'number' ? body.id : undefined
    const jsonRpc = (payload: Record<string, unknown>): string => JSON.stringify({ jsonrpc: '2.0', id, ...payload })
    if (method === 'initialize') {
      const respHeaders: Record<string, string> = { 'Content-Type': 'application/json' }
      if (server.sessionId) respHeaders['Mcp-Session-Id'] = server.sessionId
      return new Response(
        jsonRpc({ result: { protocolVersion: '2025-03-26', capabilities: {}, serverInfo: { name: 'mock', version: '1' } } }),
        { status: 200, headers: respHeaders },
      )
    }
    if (method === 'notifications/initialized') {
      return new Response(jsonRpc({ result: {} }), { status: 200, headers: { 'Content-Type': 'application/json' } })
    }
    if (method === 'tools/list') {
      return new Response(jsonRpc({ result: { tools: server.tools } }), { status: 200, headers: { 'Content-Type': 'application/json' } })
    }
    if (method === 'tools/call') {
      const params = (body.params ?? {}) as { name?: string }
      const toolName = params.name ?? ''
      const isError = server.callError?.[toolName] === true
      const text = server.callText?.[toolName] ?? 'ok'
      const result: Record<string, unknown> = { content: [{ type: 'text', text }] }
      if (isError) result.isError = true
      return new Response(jsonRpc({ result }), { status: 200, headers: { 'Content-Type': 'application/json' } })
    }
    return new Response('unknown method', { status: 400 })
  }) as unknown as typeof fetch
}

afterEach(() => {
  mock.restoreAll()
  clearMcpCacheForTesting()
})

describe('parseMcpServers', () => {
  test('returns [] for undefined / empty / whitespace', () => {
    assert.deepEqual(parseMcpServers(undefined), [])
    assert.deepEqual(parseMcpServers(''), [])
    assert.deepEqual(parseMcpServers('   '), [])
  })

  test('splits comma-separated URLs, skips invalid segments, defaults name to host', () => {
    const configs = parseMcpServers('https://a.test/mcp, https://b.test/mcp,not-a-url,,ftp://c.test/mcp')
    assert.equal(configs.length, 2)
    assert.deepEqual(configs[0], { url: 'https://a.test/mcp', name: 'a.test' })
    assert.deepEqual(configs[1], { url: 'https://b.test/mcp', name: 'b.test' })
  })

  test('uses hostname even when the path is empty', () => {
    const configs = parseMcpServers('https://gh.example')
    assert.equal(configs.length, 1)
    assert.equal(configs[0].name, 'gh.example')
    assert.equal(configs[0].url, 'https://gh.example')
  })
})

describe('discoverMcpClients', () => {
  test('handshakes (initialize -> session id -> initialized -> tools/list) and maps tools', async () => {
    const log: RecordedRequest[] = []
    mock.method(globalThis, 'fetch', makeMcpFetch(
      [
        {
          url: 'https://mcp.test/server',
          sessionId: 'sess-1',
          tools: [
            { name: 'get_issue', description: 'Get an issue', inputSchema: { type: 'object', properties: { id: { type: 'string' } } } },
            { name: 'list_issues', description: 'List issues' },
          ],
        },
      ],
      log,
    ))
    const context = makeContext({ env: { ALIVE_MCP_SERVERS: 'https://mcp.test/server' } })
    const clients = await discoverMcpClients(context, undefined, 500)

    assert.equal(clients.length, 1)
    assert.equal(clients[0].config.name, 'mcp.test')
    assert.equal(clients[0].tools.length, 2)
    assert.deepEqual(
      clients[0].tools.map((t) => t.name),
      ['get_issue', 'list_issues'],
    )
    assert.equal(clients[0].tools[0].description, 'Get an issue')
    assert.equal(clients[0].tools[0].parameters.type, 'object')

    // Handshake ordering: initialize, then notification, then tools/list.
    const methods = log.map((r) => r.method)
    assert.ok(methods.includes('initialize'))
    assert.ok(methods.includes('notifications/initialized'))
    assert.ok(methods.includes('tools/list'))
    // The session id from initialize's response header is sent on later requests.
    const listReq = log.find((r) => r.method === 'tools/list')
    assert.equal(listReq?.headers['Mcp-Session-Id'], 'sess-1')
    const initReq = log.find((r) => r.method === 'initialize')
    assert.equal(initReq?.headers['Mcp-Session-Id'], undefined)
    // The notification carries no id (JSON-RPC notification).
    const notifReq = log.find((r) => r.method === 'notifications/initialized')
    assert.equal(notifReq?.body.id, undefined)
  })

  test('caches the client per URL so a second discovery does not reconnect', async () => {
    const log: RecordedRequest[] = []
    mock.method(globalThis, 'fetch', makeMcpFetch(
      [{ url: 'https://mcp.test/cache', tools: [{ name: 'ping' }] }],
      log,
    ))
    const context = makeContext({ env: { ALIVE_MCP_SERVERS: 'https://mcp.test/cache' } })
    const first = await discoverMcpClients(context, undefined, 500)
    const second = await discoverMcpClients(context, undefined, 500)
    assert.equal(first.length, 1)
    assert.equal(second.length, 1)
    // Only one initialize round-trip happened.
    assert.equal(log.filter((r) => r.method === 'initialize').length, 1)
    assert.equal(first[0], second[0])
  })

  test('skips an unreachable server (fetch throws) and keeps the healthy one', async () => {
    mock.method(globalThis, 'fetch', makeMcpFetch([
      { url: 'https://bad.test/mcp', fail: true, tools: [] },
      { url: 'https://good.test/mcp', tools: [{ name: 'ping' }] },
    ]))
    const context = makeContext({ env: { ALIVE_MCP_SERVERS: 'https://bad.test/mcp,https://good.test/mcp' } })
    const clients = await discoverMcpClients(context, undefined, 500)
    assert.equal(clients.length, 1)
    assert.equal(clients[0].config.url, 'https://good.test/mcp')
    assert.deepEqual(clients[0].tools.map((t) => t.name), ['ping'])
  })

  test('skips a server returning non-2xx', async () => {
    mock.method(globalThis, 'fetch', makeMcpFetch([
      { url: 'https://down.test/mcp', status: 502, tools: [] },
      { url: 'https://up.test/mcp', tools: [{ name: 'ping' }] },
    ]))
    const context = makeContext({ env: { ALIVE_MCP_SERVERS: 'https://down.test/mcp,https://up.test/mcp' } })
    const clients = await discoverMcpClients(context, undefined, 500)
    assert.equal(clients.length, 1)
    assert.equal(clients[0].config.url, 'https://up.test/mcp')
  })

  test('skips a server that times out (short injected timeout)', async () => {
    mock.method(globalThis, 'fetch', makeMcpFetch([
      { url: 'https://slow.test/mcp', hang: true, tools: [] },
      { url: 'https://fast.test/mcp', tools: [{ name: 'ping' }] },
    ]))
    const context = makeContext({ env: { ALIVE_MCP_SERVERS: 'https://slow.test/mcp,https://fast.test/mcp' } })
    const clients = await discoverMcpClients(context, undefined, 40)
    assert.equal(clients.length, 1)
    assert.equal(clients[0].config.url, 'https://fast.test/mcp')
  })

  test('returns [] when ALIVE_MCP_SERVERS is unset', async () => {
    const context = makeContext({ env: {} })
    const clients = await discoverMcpClients(context, undefined, 500)
    assert.deepEqual(clients, [])
  })
})

describe('MCP tool execution (tools/call)', () => {
  test('returns text content and forwards the session id + arguments', async () => {
    const log: RecordedRequest[] = []
    mock.method(globalThis, 'fetch', makeMcpFetch(
      [{ url: 'https://mcp.test/x', sessionId: 'sess-2', tools: [{ name: 'echo' }], callText: { echo: 'hello world' } }],
      log,
    ))
    const client = await connectMcpServer({ url: 'https://mcp.test/x', name: 'mcp.test' }, undefined, 500)
    const result = await client.call('echo', { message: 'hi', count: 3 })
    assert.equal(result.isError, undefined)
    assert.equal(result.content, 'hello world')

    const callReq = log.find((r) => r.method === 'tools/call')
    assert.equal(callReq?.headers['Mcp-Session-Id'], 'sess-2')
    assert.deepEqual(callReq?.body.params, { name: 'echo', arguments: { message: 'hi', count: 3 } })
  })

  test('truncates an oversized result to MCP_TOOL_RESULT_MAX + ellipsis', async () => {
    const long = 'A'.repeat(5000)
    mock.method(globalThis, 'fetch', makeMcpFetch([
      { url: 'https://mcp.test/x', tools: [{ name: 'big' }], callText: { big: long } },
    ]))
    const client = await connectMcpServer({ url: 'https://mcp.test/x', name: 'mcp.test' }, undefined, 500)
    const result = await client.call('big', {})
    assert.equal(result.content.length, 4000 + '…(truncated)'.length)
    assert.ok(result.content.endsWith('…(truncated)'))
    assert.ok(result.content.startsWith('A'.repeat(4000)))
    assert.ok(!result.content.includes('A'.repeat(4100)), 'content beyond the cap must be cut')
  })

  test('returns isError when the server flags the call', async () => {
    mock.method(globalThis, 'fetch', makeMcpFetch([
      { url: 'https://mcp.test/x', tools: [{ name: 'boom' }], callError: { boom: true }, callText: { boom: 'server says no' } },
    ]))
    const client = await connectMcpServer({ url: 'https://mcp.test/x', name: 'mcp.test' }, undefined, 500)
    const result = await client.call('boom', {})
    assert.equal(result.isError, true)
    assert.equal(result.content, 'server says no')
  })
})

describe('buildMcpTools', () => {
  test('namespaces every tool as <serverName>_<toolName> (no conflict with handwritten names)', async () => {
    mock.method(globalThis, 'fetch', makeMcpFetch([
      { url: 'https://gh.test/mcp', tools: [{ name: 'web_search', description: 'server-side search' }] },
    ]))
    const context = makeContext({ env: { ALIVE_MCP_SERVERS: 'https://gh.test/mcp' } })
    const clients = await discoverMcpClients(context, undefined, 500)
    const defs = buildMcpTools(clients)
    assert.equal(defs.length, 1)
    // Prefixing means the MCP tool can never shadow the handwritten web_search.
    assert.equal(defs[0].name, 'gh.test_web_search')
    assert.notEqual(defs[0].name, 'web_search')
    assert.equal(defs[0].description, 'server-side search')
  })

  test('merges tools from multiple servers with distinct prefixes', async () => {
    mock.method(globalThis, 'fetch', makeMcpFetch([
      { url: 'https://a.test/mcp', tools: [{ name: 'one' }, { name: 'two' }] },
      { url: 'https://b.test/mcp', tools: [{ name: 'three' }] },
    ]))
    const context = makeContext({ env: { ALIVE_MCP_SERVERS: 'https://a.test/mcp,https://b.test/mcp' } })
    const clients = await discoverMcpClients(context, undefined, 500)
    const defs = buildMcpTools(clients)
    assert.deepEqual(
      defs.map((d) => d.name).sort(),
      ['a.test_one', 'a.test_two', 'b.test_three'].sort(),
    )
  })

  test('runner returns isError text when the MCP call throws (never throws)', async () => {
    mock.method(globalThis, 'fetch', makeMcpFetch([
      { url: 'https://mcp.test/x', tools: [{ name: 'broken' }] },
    ]))
    const clients = await discoverMcpClients(
      makeContext({ env: { ALIVE_MCP_SERVERS: 'https://mcp.test/x' } }),
      undefined,
      500,
    )
    const defs = buildMcpTools(clients)
    // Force a transport failure by making the server unreachable on the call.
    mock.restoreAll()
    mock.method(globalThis, 'fetch', async () => {
      throw new Error('network down')
    })
    const result = await defs[0].run({})
    assert.equal(result.isError, true)
    assert.match(result.content, /MCP 工具 mcp\.test_broken 调用失败/)
    assert.match(result.content, /network down/)
  })
})

describe('buildToolsAsync integration', () => {
  test('injects MCP tools into the registry alongside the handwritten 20', async () => {
    mock.method(globalThis, 'fetch', makeMcpFetch([
      {
        url: 'https://mcp.test/x',
        tools: [{ name: 'ping', description: 'ping desc', inputSchema: { type: 'object', properties: { n: { type: 'number' } } } }],
        callText: { ping: 'pong' },
      },
    ]))
    const context = makeContext({ env: { ALIVE_MCP_SERVERS: 'https://mcp.test/x' } })
    const tools = await buildToolsAsync({ context, conversationId: SELF_ID })
    const names = tools.definitions.map((d) => d.name)

    assert.ok(names.includes('mcp.test_ping'), 'MCP tool is injected')
    assert.ok(names.includes('blob_read'), 'handwritten blob tools remain')
    assert.ok(names.includes('workspace_list'), 'handwritten workspace tools remain')
    assert.ok(names.includes('web_search'), 'handwritten web_search remains')
    assert.ok(names.includes('decline_response'), 'handwritten decline_response remains')
    assert.ok(names.includes('diary_append'), 'handwritten diary remains')
    assert.ok(names.includes('chatlog_read'), 'handwritten chatlog remains')

    const pingDef = tools.definitions.find((d) => d.name === 'mcp.test_ping')
    assert.equal(pingDef?.description, 'ping desc')
    assert.equal(pingDef?.parameters.type, 'object')

    const result = await tools.run('mcp.test_ping', { n: 1 })
    assert.equal(result.isError, undefined)
    assert.equal(result.content, 'pong')
  })

  test('degrades to the handwritten 20 when MCP discovery fails', async () => {
    mock.method(globalThis, 'fetch', makeMcpFetch([{ url: 'https://down.test/mcp', fail: true, tools: [] }]))
    const context = makeContext({ env: { ALIVE_MCP_SERVERS: 'https://down.test/mcp' } })
    const tools = await buildToolsAsync({ context, conversationId: SELF_ID })
    const names = tools.definitions.map((d) => d.name)
    assert.ok(names.includes('blob_read'), 'handwritten tools survive a failed MCP discovery')
    assert.ok(names.includes('decline_response'))
    assert.ok(!names.some((n) => n.includes('down.test')), 'no MCP tool leaked in')
    assert.ok(tools.definitions.length >= 20)
  })

  test('does not block when ALIVE_MCP_SERVERS is unset', async () => {
    const context = makeContext({ env: {} })
    const tools = await buildToolsAsync({ context, conversationId: SELF_ID })
    assert.ok(tools.definitions.length >= 20)
    assert.ok(tools.definitions.some((d) => d.name === 'blob_read'))
  })
})
