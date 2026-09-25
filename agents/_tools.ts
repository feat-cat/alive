/**
 * Unified tool registry for the heartbeat turn (and anything else that opts in).
 *
 * Registers Blob tools (persistent, sandbox-independent), diary tools
 * (diary_append — timestamped append; diary_read / diary_search — opt-in
 * recall), chatlog retrieval tools (chatlog_search / chatlog_read — the
 * complete, never-compacted conversation archive), workspace tools
 * (sandbox-backed, auto-mirrored to Blob) and the Tavily web_search executor.
 * A pure-text heartbeat never calls any of them, so it stays sandbox-free.
 */
import { envString, type MakersContext } from './_shared.ts'
import type { LlmToolDef, ToolRunner } from './_llm.ts'
import {
  blobDeleteTool,
  blobEditTool,
  blobFromSandboxTool,
  blobListTool,
  blobReadTool,
  blobToSandboxTool,
  blobWriteTool,
} from './_blob-tools.ts'
import {
  ensureWorkspace,
  editWorkspaceFiles,
  listWorkspaceFiles,
  readWorkspaceFile,
  runWorkspaceCommand,
  workspaceRoot,
  writeWorkspaceFile,
} from './_workspace-tools.ts'
import { executeWebSearch } from './_tavily.ts'
import {
  appendDailyLog,
  dateFromDay,
  listDailyFiles,
  readChatlogFile,
  readDailyFile,
  readRecentChatlog,
  searchChatlog,
  searchDaily,
} from './_memory.ts'

export interface ToolContext {
  context: MakersContext
  conversationId: string
  /** Turn-level abort signal (play's controller) so tools stop on timeout. */
  signal?: AbortSignal
  /** Turn deadline (epoch ms). Long-running tools clamp their own timeout to fit. */
  deadlineAt?: number
  /**
   * Fired (synchronously) when a sandbox-backed `workspace_*` tool is actually
   * invoked. Lets the turn track "did we touch the sandbox?" at the tool-call
   * point — needed by heartbeat's failure path so it only snapshots when a
   * workspace tool ran, even when the LLM call itself throws mid-loop and
   * `result.toolResults` is unavailable.
   */
  onWorkspaceTool?: (name: string) => void
}

/** Clamp a requested tool timeout (seconds) to the remaining turn budget. */
function clampTimeoutToDeadline(requestedSeconds: number, deadlineAt: number | undefined): number {
  const clamped = Math.min(Math.max(Math.round(requestedSeconds), 1), 300)
  if (deadlineAt === undefined) return clamped
  const remainingSeconds = Math.floor((deadlineAt - Date.now()) / 1000)
  if (remainingSeconds <= 0) return 1
  return Math.min(clamped, remainingSeconds)
}

const stringSchema = (description: string): Record<string, unknown> => ({
  type: 'string',
  description,
})
const optionalString = (description: string): Record<string, unknown> => ({
  type: 'string',
  description,
})

const BLOB_TOOLS: Array<{
  name: string
  description: string
  parameters: Record<string, unknown>
  run: (tc: ToolContext, args: Record<string, unknown>) => Promise<{ content: string; isError?: boolean }>
}> = [
  {
    name: 'blob_read',
    description: 'Read a persisted text file by key. Agent-global keys start with "memory/"; other keys are scoped to this conversation.',
    parameters: { key: stringSchema('Blob key to read') },
    run: (tc, args) => blobReadTool(tc.context, tc.conversationId, args),
  },
  {
    name: 'blob_write',
    description: 'Create or replace a persisted text file by key. The write is immediately visible to the next read.',
    parameters: {
      key: stringSchema('Blob key to write'),
      content: stringSchema('Full text content'),
    },
    run: (tc, args) => blobWriteTool(tc.context, tc.conversationId, args),
  },
  {
    name: 'blob_edit',
    description: 'Edit one persisted text file using the apply_patch Begin Patch format. Supports exactly one file per call.',
    parameters: {
      key: stringSchema('Blob key to edit'),
      input: stringSchema('Patch text with *** Begin Patch / *** End Patch'),
    },
    run: (tc, args) => blobEditTool(tc.context, tc.conversationId, args),
  },
  {
    name: 'blob_list',
    description: 'List persisted blob keys under a prefix.',
    parameters: { prefix: optionalString('Optional key prefix, e.g. "workspace/"') },
    run: (tc, args) => blobListTool(tc.context, tc.conversationId, args),
  },
  {
    name: 'blob_delete',
    description: 'Delete a persisted blob key (idempotent).',
    parameters: { key: stringSchema('Blob key to delete') },
    run: (tc, args) => blobDeleteTool(tc.context, tc.conversationId, args),
  },
  {
    name: 'blob_to_sandbox',
    description: 'Copy a persisted blob file into the sandbox workspace (e.g. to resume project work).',
    parameters: {
      key: stringSchema('Blob key to read'),
      sandbox_path: optionalString('Target sandbox path relative to workspace root (default: key)'),
    },
    run: (tc, args) => blobToSandboxTool(tc.context, tc.conversationId, workspaceRoot(tc.conversationId), args),
  },
  {
    name: 'blob_from_sandbox',
    description: 'Copy a sandbox workspace file back into persisted Blob.',
    parameters: {
      key: stringSchema('Blob key to write'),
      sandbox_path: optionalString('Sandbox path relative to workspace root (default: key)'),
    },
    run: (tc, args) => blobFromSandboxTool(tc.context, tc.conversationId, workspaceRoot(tc.conversationId), args),
  },
]

/**
 * Diary tools: zero-sandbox, pure strong-consistency Blob. `diary_append`
 * writes through `appendDailyLog` (timestamped, never overwrites); the reads
 * are opt-in because the diary is NOT auto-injected into the heartbeat prompt.
 */
const DIARY_TOOLS: Array<{
  name: string
  description: string
  parameters: Record<string, unknown>
  run: (tc: ToolContext, args: Record<string, unknown>) => Promise<{ content: string; isError?: boolean }>
}> = [
  {
    name: 'diary_append',
    description: "Append a diary entry to today's diary file (memory/daily/YYYY-MM-DD.md). Auto-appends with timestamp; never overwrites previous entries. Use this to write your diary/journal.",
    parameters: {
      content: stringSchema('Diary entry text to append'),
      day: optionalString('Date in YYYY-MM-DD format (optional; defaults to today)'),
    },
    run: async (tc, args) => {
      try {
        const content = typeof args.content === 'string' ? args.content.trim() : ''
        if (!content) return { content: 'diary_append requires "content".', isError: true }
        const rawDay = typeof args.day === 'string' ? args.day.trim() : ''
        let at: Date
        if (rawDay) {
          const parsed = dateFromDay(rawDay)
          if (parsed === null) {
            return { content: 'diary_append "day" must be a valid YYYY-MM-DD date.', isError: true }
          }
          at = parsed
        } else {
          at = new Date()
        }
        const key = await appendDailyLog(tc.context, content, at)
        return { content: `Appended diary entry to ${key}.` }
      } catch (error) {
        return { content: error instanceof Error ? error.message : String(error), isError: true }
      }
    },
  },
  {
    name: 'diary_read',
    description: 'Read one diary day (memory/daily/YYYY-MM-DD.md), or omit "day" to list the most recent diary files.',
    parameters: {
      day: optionalString('Date in YYYY-MM-DD format, or a full memory/daily/… key (optional; defaults to listing recent days)'),
    },
    run: async (tc, args) => {
      try {
        const rawDay = typeof args.day === 'string' ? args.day.trim() : ''
        if (!rawDay) {
          const keys = await listDailyFiles(tc.context, 7)
          return { content: keys.length === 0 ? 'No diary entries yet.' : keys.join('\n') }
        }
        const day = normalizeDailyDay(rawDay)
        if (day === null) {
          return { content: 'diary_read "day" must be YYYY-MM-DD or a memory/daily/… key.', isError: true }
        }
        const content = await readDailyFile(tc.context, day)
        if (content === null) return { content: `No diary for ${day}.`, isError: true }
        return { content: `## ${day}\n${content}` }
      } catch (error) {
        return { content: error instanceof Error ? error.message : String(error), isError: true }
      }
    },
  },
  {
    name: 'diary_search',
    description: 'Case-insensitive keyword search across recent diary files; returns per-day matching snippets.',
    parameters: {
      keyword: stringSchema('Keyword to search for in recent diary files'),
      days: { type: 'number', description: 'How many recent days to search (default 14, clamped 1-90)' },
    },
    run: async (tc, args) => {
      try {
        const keyword = typeof args.keyword === 'string' ? args.keyword.trim() : ''
        if (!keyword) return { content: 'diary_search requires "keyword".', isError: true }
        const requested = typeof args.days === 'number' ? args.days : 14
        const days = Math.min(Math.max(Math.round(requested), 1), 90)
        const hits = await searchDaily(tc.context, keyword, days)
        if (hits.length === 0) return { content: `No diary matches for "${keyword}".` }
        const blocks = hits.map((hit) => `## ${hit.day}\n${hit.snippets.map((line) => `- ${line}`).join('\n')}`)
        return { content: blocks.join('\n\n') }
      } catch (error) {
        return { content: error instanceof Error ? error.message : String(error), isError: true }
      }
    },
  },
]

/** Accept a bare YYYY-MM-DD or a full daily blob key; return the date or null. */
function normalizeDailyDay(value: string): string | null {
  const stem = value.trim().replace(/^memory\/daily\//, '').replace(/\.md$/, '')
  // Round-trip validation (rejects impossible dates like 2026-02-31), same as
  // diary_append and readDailyFile so all read/write paths share one rule.
  return dateFromDay(stem) ? stem : null
}

/**
 * Chatlog retrieval tools: zero-sandbox, pure strong-consistency Blob reads of
 * the complete conversation archive. The archive is append-only and never
 * touched by compact, so these tools see the FULL original history — unlike the
 * model's context, which compact folds into summaries.
 */
const CHATLOG_TOOLS: Array<{
  name: string
  description: string
  parameters: Record<string, unknown>
  run: (tc: ToolContext, args: Record<string, unknown>) => Promise<{ content: string; isError?: boolean }>
}> = [
  {
    name: 'chatlog_search',
    description: 'Case-insensitive keyword search across the recent chatlog archive (the complete, never-compacted conversation history). Returns per-day matching snippets.',
    parameters: {
      keyword: stringSchema('Keyword to search for in the chatlog archive'),
      days: { type: 'number', description: 'How many recent days to search (default 30, clamped 1-90)' },
    },
    run: async (tc, args) => {
      try {
        const keyword = typeof args.keyword === 'string' ? args.keyword.trim() : ''
        if (!keyword) return { content: 'chatlog_search requires "keyword".', isError: true }
        const requested = typeof args.days === 'number' ? args.days : 30
        const days = Math.min(Math.max(Math.round(requested), 1), 90)
        const hits = await searchChatlog(tc.context, tc.conversationId, keyword, days)
        if (hits.length === 0) return { content: `No chatlog matches for "${keyword}".` }
        const blocks = hits.map((hit) => `## ${hit.day}\n${hit.snippets.map((line) => `- ${line}`).join('\n')}`)
        return { content: blocks.join('\n\n') }
      } catch (error) {
        return { content: error instanceof Error ? error.message : String(error), isError: true }
      }
    },
  },
  {
    name: 'chatlog_read',
    description: 'Read one chatlog archive day (chatlog/YYYY-MM-DD.md), or omit "day" to read the most recent N days of the complete conversation history (heartbeat triggers, tool calls, compact summaries and all).',
    parameters: {
      day: optionalString('Date in YYYY-MM-DD format, or a full chatlog/… key (optional)'),
      days: { type: 'number', description: 'How many recent days to read when "day" is omitted (default 7, clamped 1-90)' },
    },
    run: async (tc, args) => {
      try {
        const rawDay = typeof args.day === 'string' ? args.day.trim() : ''
        if (rawDay) {
          const day = normalizeChatlogDay(rawDay)
          if (day === null) {
            return { content: 'chatlog_read "day" must be YYYY-MM-DD or a chatlog/… key.', isError: true }
          }
          const content = await readChatlogFile(tc.context, tc.conversationId, day)
          if (content === null) return { content: `No chatlog for ${day}.`, isError: true }
          return { content: `## ${day}\n${content}` }
        }
        const requested = typeof args.days === 'number' ? args.days : 7
        const days = Math.min(Math.max(Math.round(requested), 1), 90)
        const content = await readRecentChatlog(tc.context, tc.conversationId, days)
        return { content: content || 'No chatlog entries yet.' }
      } catch (error) {
        return { content: error instanceof Error ? error.message : String(error), isError: true }
      }
    },
  },
]

/** Accept a bare YYYY-MM-DD or a full chatlog blob key; return the date or null. */
function normalizeChatlogDay(value: string): string | null {
  const stem = value.trim().replace(/^chatlog\//, '').replace(/\.md$/, '')
  // Round-trip validation (rejects impossible dates like 2026-02-31), matching
  // readChatlogFile and the diary read/write paths.
  return dateFromDay(stem) ? stem : null
}

const WORKSPACE_TOOLS: Array<{
  name: string
  description: string
  parameters: Record<string, unknown>
  run: (tc: ToolContext, args: Record<string, unknown>) => Promise<{ content: string; isError?: boolean }>
}> = [
  {
    name: 'workspace_list',
    description: 'List files in the coding workspace (paths relative to workspace root).',
    parameters: {},
    run: async (tc) => {
      try {
        const root = await ensureWorkspace(tc.context, tc.conversationId)
        const items = await listWorkspaceFiles(tc.context, tc.conversationId, root)
        return { content: JSON.stringify({ root, items: items.slice(0, 200) }) }
      } catch (error) {
        return { content: error instanceof Error ? error.message : String(error), isError: true }
      }
    },
  },
  {
    name: 'workspace_read',
    description: 'Read one UTF-8 source file from the workspace using a relative path.',
    parameters: { path: stringSchema('Relative path in the workspace') },
    run: async (tc, args) => {
      try {
        const path = typeof args.path === 'string' ? args.path : ''
        if (!path) return { content: 'workspace_read requires "path".', isError: true }
        const root = await ensureWorkspace(tc.context, tc.conversationId)
        const file = await readWorkspaceFile(tc.context, tc.conversationId, root, path)
        return { content: JSON.stringify(file) }
      } catch (error) {
        return { content: error instanceof Error ? error.message : String(error), isError: true }
      }
    },
  },
  {
    name: 'workspace_write',
    description: 'Create or replace one complete UTF-8 source file in the workspace. Use one call per file.',
    parameters: {
      path: stringSchema('Relative path in the workspace'),
      content: stringSchema('Complete file content'),
    },
    run: async (tc, args) => {
      try {
        const path = typeof args.path === 'string' ? args.path : ''
        const content = typeof args.content === 'string' ? args.content : ''
        if (!path) return { content: 'workspace_write requires "path".', isError: true }
        const root = await ensureWorkspace(tc.context, tc.conversationId)
        const result = await writeWorkspaceFile(tc.context, tc.conversationId, root, path, content)
        return { content: JSON.stringify(result) }
      } catch (error) {
        return { content: error instanceof Error ? error.message : String(error), isError: true }
      }
    },
  },
  {
    name: 'workspace_edit',
    description: 'Apply a multi-file apply_patch envelope (*** Begin Patch / *** End Patch) to the workspace.',
    parameters: { input: stringSchema('Patch text with *** Begin Patch / *** End Patch') },
    run: async (tc, args) => {
      try {
        const input = typeof args.input === 'string' ? args.input : ''
        if (!input) return { content: 'workspace_edit requires "input".', isError: true }
        const root = await ensureWorkspace(tc.context, tc.conversationId)
        const result = await editWorkspaceFiles(tc.context, tc.conversationId, root, input)
        return { content: result.text }
      } catch (error) {
        return { content: error instanceof Error ? error.message : String(error), isError: true }
      }
    },
  },
  {
    name: 'workspace_run_command',
    description: 'Run a shell command in the workspace. Use for dependency installs, builds, tests and diagnostics.',
    parameters: {
      command: stringSchema('Shell command'),
      timeout: { type: 'number', description: 'Timeout in seconds (1-300, default 60)' },
    },
    run: async (tc, args) => {
      try {
        const command = typeof args.command === 'string' ? args.command : ''
        if (!command) return { content: 'workspace_run_command requires "command".', isError: true }
        const timeout = typeof args.timeout === 'number' ? args.timeout : 60
        // Bound the command by the turn deadline so a long build can never
        // overshoot the play timeout (abort checkpoints live in _llm/_tools).
        const boundedTimeout = clampTimeoutToDeadline(timeout, tc.deadlineAt)
        const root = await ensureWorkspace(tc.context, tc.conversationId)
        const result = await runWorkspaceCommand(tc.context, tc.conversationId, root, command, boundedTimeout)
        return { content: JSON.stringify(result) }
      } catch (error) {
        return { content: error instanceof Error ? error.message : String(error), isError: true }
      }
    },
  },
]

const SEARCH_TOOL: { name: string; description: string; parameters: Record<string, unknown>; run: (tc: ToolContext, args: Record<string, unknown>) => Promise<{ content: string; isError?: boolean }> } = {
  name: 'web_search',
  description: 'Search the web via Tavily. Returns titled result snippets. Requires TAVILY_API_KEY.',
  parameters: {
    query: stringSchema('Search query'),
    max_results: { type: 'number', description: 'Max results 1-20 (default 5)' },
    search_depth: { type: 'string', description: 'basic or advanced' },
    topic: { type: 'string', description: 'general or news' },
    time_range: { type: 'string', description: 'day/week/month/year' },
    start_date: { type: 'string', description: 'YYYY-MM-DD' },
    end_date: { type: 'string', description: 'YYYY-MM-DD' },
  },
  run: (tc, args) => executeWebSearch(envString(tc.context, 'TAVILY_API_KEY'), args),
}

export interface ToolRegistry {
  definitions: LlmToolDef[]
  run: ToolRunner
}

/** Build the full registry for a request (e.g. a heartbeat turn). */
export function buildTools(tc: ToolContext): ToolRegistry {
  const entries = [...BLOB_TOOLS, ...WORKSPACE_TOOLS, ...DIARY_TOOLS, ...CHATLOG_TOOLS, SEARCH_TOOL]
  const definitions: LlmToolDef[] = entries.map((entry) => ({
    name: entry.name,
    description: entry.description,
    parameters: entry.parameters,
  }))
  const byName = new Map(entries.map((entry) => [entry.name, entry]))
  const run: ToolRunner = async (name, args, signal) => {
    if (signal?.aborted || tc.signal?.aborted) {
      const abort = new Error('Tool execution aborted')
      abort.name = 'AbortError'
      throw abort
    }
    const tool = byName.get(name)
    if (!tool) return { content: `Unknown tool: ${name}.`, isError: true }
    // Mark sandbox-backed calls before executing so a throw from the LLM loop
    // later can still know that the sandbox was actually touched.
    if (name.startsWith('workspace_')) tc.onWorkspaceTool?.(name)
    return tool.run(tc, args)
  }
  return { definitions, run }
}


