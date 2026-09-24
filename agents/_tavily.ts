/**
 * Tavily web search executor, ported from easyclaw core/tools/web-search.ts.
 *
 * Unlike the original it never touches the filesystem or process.env: the API
 * key is passed in by the caller (which reads it from `context.env.TAVILY_API_KEY`
 * via `envString`). Failures never throw: they come back as `{ isError: true }`
 * tool results.
 */
export type ToolResult = { content: string; isError?: boolean }

const TAVILY_SEARCH_URL = 'https://api.tavily.com/search'
const REQUEST_TIMEOUT_MS = 30_000
const DEFAULT_MAX_RESULTS = 5
const MAX_RESULTS_LIMIT = 20
const ERROR_BODY_SNIPPET_CHARS = 300

const SEARCH_DEPTHS = ['basic', 'advanced'] as const
const SEARCH_TOPICS = ['general', 'news'] as const
const TIME_RANGES = ['day', 'week', 'month', 'year'] as const

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

    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)
    let response: Response
    try {
      response = await fetch(TAVILY_SEARCH_URL, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${key}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
        signal: controller.signal,
      })
    } catch (error) {
      const message = controller.signal.aborted
        ? `Web search failed: request to the search backend timed out after ${REQUEST_TIMEOUT_MS / 1000}s.`
        : `Web search failed: ${error instanceof Error ? error.message : String(error)}`
      return { content: message, isError: true }
    } finally {
      clearTimeout(timer)
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
