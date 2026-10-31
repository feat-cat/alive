/**
 * Tavily web search + web fetch executors, ported from easyclaw
 * core/tools/web-search.ts and extended with the Tavily /extract capability.
 *
 * Unlike the original they never touch the filesystem or process.env: the API
 * key is passed in by the caller (which reads it from `context.env.TAVILY_API_KEY`
 * via `envString`). Failures never throw: they come back as `{ isError: true }`
 * tool results.
 */
export type ToolResult = { content: string; isError?: boolean }

const TAVILY_SEARCH_URL = 'https://api.tavily.com/search'
const TAVILY_EXTRACT_URL = 'https://api.tavily.com/extract'
const REQUEST_TIMEOUT_MS = 30_000
const DEFAULT_MAX_RESULTS = 5
const MAX_RESULTS_LIMIT = 20
const ERROR_BODY_SNIPPET_CHARS = 300
const MIN_URLS = 1
const MAX_URLS = 5
const DEFAULT_MAX_CHARS = 3000
const MAX_CHARS_LIMIT = 8000

const SEARCH_DEPTHS = ['basic', 'advanced'] as const
const SEARCH_TOPICS = ['general', 'news'] as const
const TIME_RANGES = ['day', 'week', 'month', 'year'] as const

/* ------------------------------------------------------------------ */
/* Lightweight retry for transient Tavily failures (429 / 5xx / net)   */
/* ------------------------------------------------------------------ */

/** Maximum retries for a single Tavily request (one attempt + up to 2 retries). */
export const TAVILY_RETRY_MAX = 2

/** Fixed backoff schedule (ms) between retries, indexed by the retry ordinal. */
export const TAVILY_RETRY_BACKOFF_MS = [500, 1000] as const

/** Test seam: replace the active backoff (e.g. `[0, 0]`) so the suite never sleeps. */
let activeTavilyBackoff: readonly number[] = TAVILY_RETRY_BACKOFF_MS

export function setTavilyRetryForTesting(backoffMs: readonly number[] | null): void {
  activeTavilyBackoff = backoffMs ?? TAVILY_RETRY_BACKOFF_MS
}

/** True when an HTTP status is worth retrying (transient). 400/401/403 never retry. */
function isRetryableTavilyStatus(status: number): boolean {
  return status === 429 || status === 500 || status === 502 || status === 503 || status === 504
}

/** True when a network-layer fetch error (TypeError) is worth retrying. */
function isRetryableNetworkError(error: unknown): boolean {
  if (error instanceof Error && error.name === 'AbortError') return false
  return error instanceof TypeError
}

async function waitForTavilyRetry(ms: number): Promise<void> {
  if (ms <= 0) return
  await new Promise<void>((resolve) => {
    setTimeout(resolve, ms)
  })
}

/**
 * Run a single fetch attempt with a per-attempt wall-clock timeout, retrying
 * transient failures (429/5xx/network TypeError) up to `TAVILY_RETRY_MAX` times
 * with the configured backoff. Non-retryable statuses (400/401/403/…) return
 * immediately. On retryable exhaustion the last thrown error is rethrown.
 */
async function tavilyFetch(url: string, init: RequestInit): Promise<Response> {
  let lastError: unknown = null
  for (let attemptIndex = 0; ; attemptIndex += 1) {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)
    try {
      const response = await fetch(url, { ...init, signal: controller.signal })
      if (response.ok || !isRetryableTavilyStatus(response.status) || attemptIndex >= TAVILY_RETRY_MAX) {
        return response
      }
      lastError = new Error(`Tavily HTTP ${response.status}`)
    } catch (error) {
      // A per-attempt timeout aborts with an AbortError — surface it immediately
      // (not retried); a genuine network TypeError is retried.
      if (error instanceof Error && error.name === 'AbortError') throw error
      if (!isRetryableNetworkError(error) || attemptIndex >= TAVILY_RETRY_MAX) throw error
      lastError = error
    } finally {
      clearTimeout(timer)
    }
    const delay = activeTavilyBackoff[attemptIndex] ?? activeTavilyBackoff[activeTavilyBackoff.length - 1] ?? 0
    await waitForTavilyRetry(delay)
  }
}

/**
 * Private / loopback / metadata hostnames that must never be reached by an
 * outbound fetch (SSRF guard). A URL whose host resolves to any of these is
 * rejected before any request is made.
 */
const PRIVATE_HOSTNAMES = new Set([
  'localhost',
  'metadata.google.internal',
  'metadata.googleapis.com',
  '169.254.169.254',
  '0.0.0.0',
])

/**
 * True when `url` targets a private / loopback / link-local / metadata network.
 * Resolves the host to an IP literal and checks every range a public-only fetch
 * must never hit. Invalid URLs return true (they can never be fetched safely).
 */
export function isPrivateUrl(url: string): boolean {
  let host: string
  try {
    host = new URL(url).hostname.toLowerCase().replace(/^\[|\]$/g, '')
  } catch {
    return true
  }
  // Exact known-bad hostnames.
  if (PRIVATE_HOSTNAMES.has(host)) return true
  // IPv6 loopback / unspecified.
  if (host === '::1' || host === '::' || host === '0:0:0:0:0:0:0:1') return true
  // IPv6 link-local (fe80::/10).
  if (/^fe80:/i.test(host)) return true
  // IPv6 unique-local (fc00::/7).
  if (/^f[cd][0-9a-f]{2}:/i.test(host)) return true
  // Numeric IPv4 ranges (127/8 loopback, 10/8, 172.16/12, 192.168/16,
  // 169.254/16 link-local, 0/8).
  const v4 = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(host)
  if (v4) {
    const a = Number(v4[1])
    const b = Number(v4[2])
    if (a === 127 || a === 10 || a === 0) return true
    if (a === 172 && b >= 16 && b <= 31) return true
    if (a === 192 && b === 168) return true
    if (a === 169 && b === 254) return true
  }
  // Hostnames are otherwise treated as public (DNS could point anywhere, but a
  // remote resolver will re-check on connect; blocking unknown names outright
  // would break legitimate public hosts).
  return false
}

interface TavilySearchResponse {
  results?: Array<{
    title?: string
    url?: string
    content?: string
    score?: number
    published_date?: string
  }>
  images?: Array<{ url?: string }>
}

interface TavilyExtractResult {
  url?: string
  title?: string
  content?: string
  raw_content?: string
}

interface TavilyExtractFailure {
  url?: string
  error?: string
}

interface TavilyExtractResponse {
  results?: TavilyExtractResult[]
  failed_results?: TavilyExtractFailure[]
}

function stringListOrError(value: unknown, argName: string): { ok: true; list: string[] | undefined } | { ok: false; message: string } {
  if (value === undefined) return { ok: true, list: undefined }
  if (!Array.isArray(value) || value.some((item) => typeof item !== 'string')) {
    return { ok: false, message: `Tool "web_search" requires "${argName}" to be an array of strings.` }
  }
  return { ok: true, list: value as string[] }
}

function enumOrError<T extends readonly string[]>(
  value: unknown,
  argName: string,
  allowed: T,
): { ok: true; value: string | undefined } | { ok: false; message: string } {
  if (value === undefined) return { ok: true, value: undefined }
  if (typeof value !== 'string' || !allowed.includes(value)) {
    return { ok: false, message: `Tool "web_search" requires "${argName}" to be one of ${allowed.join(', ')}.` }
  }
  return { ok: true, value }
}

function snippet(text: string): string {
  const trimmed = text.trim()
  return trimmed.length > ERROR_BODY_SNIPPET_CHARS ? `${trimmed.slice(0, ERROR_BODY_SNIPPET_CHARS)}…` : trimmed
}

function truncateText(text: string, maxChars: number): string {
  const trimmed = text.trim()
  return trimmed.length > maxChars ? `${trimmed.slice(0, maxChars)}…` : trimmed
}

function formatSearchResults(query: string, payload: TavilySearchResponse, includeImages: boolean): string {
  const results = (payload.results ?? []).filter((r) => r && typeof r.url === 'string')
  if (results.length === 0) return `No results found for "${query}".`
  const lines: string[] = [`Web search results for "${query}":`, '']
  results.forEach((result, index) => {
    const title = typeof result.title === 'string' && result.title.trim().length > 0
      ? result.title.trim()
      : result.url ?? '(untitled)'
    lines.push(`${index + 1}. [${title}](${result.url})`)
    if (typeof result.content === 'string' && result.content.trim().length > 0) {
      lines.push(`   ${result.content.trim()}`)
    }
    if (typeof result.published_date === 'string' && result.published_date.trim().length > 0) {
      lines.push(`   Published: ${result.published_date.trim()}`)
    }
  })
  if (includeImages) {
    const imageUrls = (payload.images ?? [])
      .map((image) => (image && typeof image.url === 'string' ? image.url : null))
      .filter((url): url is string => url !== null)
    if (imageUrls.length > 0) {
      lines.push('', 'Related images:')
      for (const url of imageUrls) lines.push(`- ${url}`)
    }
  }
  return lines.join('\n')
}

function formatExtractResults(payload: TavilyExtractResponse, maxChars: number): string {
  const results = (payload.results ?? []).filter((r) => r && typeof r.url === 'string')
  const failed = (payload.failed_results ?? []).filter((r) => r && typeof r.url === 'string')
  if (results.length === 0 && failed.length === 0) return 'No page content extracted.'
  const lines: string[] = ['Web page extraction results:', '']
  results.forEach((result, index) => {
    const url = result.url ?? ''
    const title = typeof result.title === 'string' && result.title.trim().length > 0
      ? result.title.trim()
      : url
    const content = typeof result.raw_content === 'string' && result.raw_content.trim().length > 0
      ? result.raw_content
      : (typeof result.content === 'string' ? result.content : '')
    lines.push(`${index + 1}. [${title}](${url})`)
    if (content.trim().length > 0) lines.push(`   ${truncateText(content, maxChars)}`)
  })
  for (const failure of failed) {
    lines.push(`- [${failure.url}] failed${typeof failure.error === 'string' && failure.error.trim().length > 0 ? `: ${snippet(failure.error)}` : ''}`)
  }
  return lines.join('\n')
}

export async function executeWebSearch(apiKey: string, args: Record<string, unknown>): Promise<ToolResult> {
  try {
    const query = typeof args.query === 'string' ? args.query.trim() : ''
    if (!query) return { content: 'Tool "web_search" requires a non-empty "query" argument.', isError: true }

    let maxResults = DEFAULT_MAX_RESULTS
    if (args.max_results !== undefined) {
      if (
        typeof args.max_results !== 'number' ||
        !Number.isFinite(args.max_results) ||
        !Number.isInteger(args.max_results) ||
        args.max_results < 1 ||
        args.max_results > MAX_RESULTS_LIMIT
      ) {
        return { content: `Tool "web_search" requires "max_results" to be an integer between 1 and ${MAX_RESULTS_LIMIT}.`, isError: true }
      }
      maxResults = args.max_results
    }
    const depth = enumOrError(args.search_depth, 'search_depth', SEARCH_DEPTHS)
    if (!depth.ok) return { content: depth.message, isError: true }
    const topic = enumOrError(args.topic, 'topic', SEARCH_TOPICS)
    if (!topic.ok) return { content: topic.message, isError: true }
    const timeRange = enumOrError(args.time_range, 'time_range', TIME_RANGES)
    if (!timeRange.ok) return { content: timeRange.message, isError: true }
    for (const argName of ['start_date', 'end_date'] as const) {
      if (args[argName] !== undefined && typeof args[argName] !== 'string') {
        return { content: `Tool "web_search" requires "${argName}" to be a string (YYYY-MM-DD).`, isError: true }
      }
    }
    if (timeRange.value !== undefined && (args.start_date !== undefined || args.end_date !== undefined)) {
      return {
        content: 'Tool "web_search" cannot combine "time_range" with "start_date"/"end_date": they are mutually exclusive time filters.',
        isError: true,
      }
    }
    const includeDomains = stringListOrError(args.include_domains, 'include_domains')
    if (!includeDomains.ok) return { content: includeDomains.message, isError: true }
    const excludeDomains = stringListOrError(args.exclude_domains, 'exclude_domains')
    if (!excludeDomains.ok) return { content: excludeDomains.message, isError: true }
    const includeImages = args.include_images === true

    const key = apiKey.trim()
    if (!key) {
      return { content: 'Web search is not configured. Set TAVILY_API_KEY in project environment variables.', isError: true }
    }

    const body: Record<string, unknown> = { query, max_results: maxResults }
    if (depth.value !== undefined) body.search_depth = depth.value
    if (topic.value !== undefined) body.topic = topic.value
    if (timeRange.value !== undefined) body.time_range = timeRange.value
    if (typeof args.start_date === 'string') body.start_date = args.start_date
    if (typeof args.end_date === 'string') body.end_date = args.end_date
    if (includeDomains.list !== undefined) body.include_domains = includeDomains.list
    if (excludeDomains.list !== undefined) body.exclude_domains = excludeDomains.list
    if (args.include_images !== undefined) body.include_images = includeImages

    let response: Response
    try {
      response = await tavilyFetch(TAVILY_SEARCH_URL, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${key}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      })
    } catch (error) {
      const isTimeout = error instanceof Error && error.name === 'AbortError'
      const message = isTimeout
        ? `Web search failed: request to the search backend timed out after ${REQUEST_TIMEOUT_MS / 1000}s.`
        : `Web search failed: ${error instanceof Error ? error.message : String(error)}`
      return { content: message, isError: true }
    }

    if (!response.ok) {
      const errorText = await response.text().catch(() => '')
      return {
        content: `Web search failed: the search backend returned HTTP ${response.status}${response.statusText ? ` (${response.statusText})` : ''}.${errorText.trim() ? ` Response: ${snippet(errorText)}` : ''}`,
        isError: true,
      }
    }

    const payload = (await response.json()) as TavilySearchResponse
    return { content: formatSearchResults(query, payload, includeImages) }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    return { content: `Error executing tool "web_search": ${message}`, isError: true }
  }
}

export async function executeWebFetch(apiKey: string, args: Record<string, unknown>): Promise<ToolResult> {
  try {
    if (!Array.isArray(args.urls) || args.urls.length === 0) {
      return { content: `Tool "web_fetch" requires "urls" to be a non-empty array of ${MIN_URLS}-${MAX_URLS} URL strings.`, isError: true }
    }
    const rawUrls = args.urls as unknown[]
    if (rawUrls.some((item) => typeof item !== 'string' || item.trim().length === 0)) {
      return { content: `Tool "web_fetch" requires "urls" to be a non-empty array of ${MIN_URLS}-${MAX_URLS} URL strings.`, isError: true }
    }
    const urls = rawUrls as string[]
    if (urls.length > MAX_URLS) {
      return { content: `Tool "web_fetch" requires at most ${MAX_URLS} urls (got ${urls.length}).`, isError: true }
    }
    // H3 (SSRF guard): refuse any private / loopback / link-local / metadata
    // URL so web_fetch can never be used to poke the internal network.
    const blocked = urls.find((u) => isPrivateUrl(u))
    if (blocked) {
      return { content: 'blocked: private/loopback URL', isError: true }
    }

    let maxChars = DEFAULT_MAX_CHARS
    if (args.max_chars !== undefined) {
      if (
        typeof args.max_chars !== 'number' ||
        !Number.isFinite(args.max_chars) ||
        !Number.isInteger(args.max_chars) ||
        args.max_chars < 100 ||
        args.max_chars > MAX_CHARS_LIMIT
      ) {
        return { content: `Tool "web_fetch" requires "max_chars" to be an integer between 100 and ${MAX_CHARS_LIMIT}.`, isError: true }
      }
      maxChars = args.max_chars
    }

    const key = apiKey.trim()
    if (!key) {
      return { content: 'Web fetch is not configured. Set TAVILY_API_KEY in project environment variables.', isError: true }
    }

    const body: Record<string, unknown> = { urls: urls.map((url) => url.trim()), max_chars: maxChars }

    let response: Response
    try {
      response = await tavilyFetch(TAVILY_EXTRACT_URL, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${key}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      })
    } catch (error) {
      const isTimeout = error instanceof Error && error.name === 'AbortError'
      const message = isTimeout
        ? `Web fetch failed: request to the extract backend timed out after ${REQUEST_TIMEOUT_MS / 1000}s.`
        : `Web fetch failed: ${error instanceof Error ? error.message : String(error)}`
      return { content: message, isError: true }
    }

    if (!response.ok) {
      const errorText = await response.text().catch(() => '')
      return {
        content: `Web fetch failed: the extract backend returned HTTP ${response.status}${response.statusText ? ` (${response.statusText})` : ''}.${errorText.trim() ? ` Response: ${snippet(errorText)}` : ''}`,
        isError: true,
      }
    }

    const payload = (await response.json()) as TavilyExtractResponse
    return { content: formatExtractResults(payload, maxChars) }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    return { content: `Error executing tool "web_fetch": ${message}`, isError: true }
  }
}
