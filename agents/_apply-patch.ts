/**
 * Self-contained apply_patch parser + applier (OpenAI "Begin Patch" envelope).
 *
 * Schema (reference: easyclaw core/tools/catalog.ts + openclaw apply-patch):
 *
 *   *** Begin Patch
 *   *** Add File: path/to/file
 *   +line 1
 *   +line 2
 *   *** Update File: src/app.ts
 *   @@ optional change context
 *   -old line
 *   +new line
 *   *** Delete File: obsolete.txt
 *   *** End Patch
 *
 * This module is pure and testable: filesystem access goes through the injected
 * `PatchFs` adapter (Blob-backed or sandbox-backed), and every path is
 * normalized relative to the workspace root (no `..`, no absolute paths).
 */
export interface PatchFs {
  read(path: string): Promise<string>
  write(path: string, content: string): Promise<void>
  delete(path: string): Promise<void>
  exists(path: string): Promise<boolean>
  makeDir(path: string): Promise<void>
}

export interface UpdateChunk {
  changeContext?: string
  oldLines: string[]
  newLines: string[]
  contextOldIndexes: Array<number | undefined>
  isEndOfFile: boolean
}

export type Hunk =
  | { kind: 'add'; path: string; contents: string }
  | { kind: 'delete'; path: string }
  | { kind: 'update'; path: string; movePath?: string; chunks: UpdateChunk[] }

export interface ApplyPatchSummary {
  added: string[]
  modified: string[]
  deleted: string[]
  moved: string[]
}

export interface ApplyPatchResult {
  summary: ApplyPatchSummary
  text: string
  noOp: boolean
}

const BEGIN_PATCH_MARKER = '*** Begin Patch'
const END_PATCH_MARKER = '*** End Patch'
const ADD_FILE_MARKER = '*** Add File: '
const DELETE_FILE_MARKER = '*** Delete File: '
const UPDATE_FILE_MARKER = '*** Update File: '
const MOVE_TO_MARKER = '*** Move to: '
const EOF_MARKER = '*** End of File'
const CHANGE_CONTEXT_MARKER = '@@ '
const EMPTY_CHANGE_CONTEXT_MARKER = '@@'

/** Normalize a patch header path to a safe relative path, or null when unsafe. */
export function normalizePatchPath(value: string): string | null {
  const raw = value.trim().replaceAll('\\', '/').replace(/^\.\//, '')
  if (!raw || raw.startsWith('/') || raw.includes('\0')) return null
  const parts = raw.split('/')
  if (parts.some((part) => !part || part === '.' || part === '..')) return null
  return parts.join('/')
}

export function parsePatchText(input: string): Hunk[] {
  const trimmed = input.trim()
  if (!trimmed) throw new Error('Invalid patch: input is empty.')
  const lines = trimmed.split(/\r?\n/)
  const first = lines[0]?.trim()
  const last = lines[lines.length - 1]?.trim()
  if (first !== BEGIN_PATCH_MARKER || last !== END_PATCH_MARKER) {
    throw new Error("Patch must start with '*** Begin Patch' and end with '*** End Patch'.")
  }
  const body = lines.slice(1, -1)
  const hunks: Hunk[] = []
  let index = 0
  while (index < body.length) {
    const { hunk, consumed } = parseOneHunk(body.slice(index))
    hunks.push(hunk)
    index += consumed
  }
  return hunks
}

function parseOneHunk(lines: string[]): { hunk: Hunk; consumed: number } {
  const firstLine = lines[0]?.trim()
  if (!firstLine) throw new Error('Invalid patch: empty hunk.')
  if (firstLine.startsWith(ADD_FILE_MARKER)) {
    const targetPath = firstLine.slice(ADD_FILE_MARKER.length)
    let contents = ''
    let consumed = 1
    for (const addLine of lines.slice(1)) {
      if (addLine.startsWith('+')) {
        contents += `${addLine.slice(1)}\n`
        consumed += 1
      } else {
        break
      }
    }
    return { hunk: { kind: 'add', path: targetPath, contents }, consumed }
  }
  if (firstLine.startsWith(DELETE_FILE_MARKER)) {
    return { hunk: { kind: 'delete', path: firstLine.slice(DELETE_FILE_MARKER.length) }, consumed: 1 }
  }
  if (firstLine.startsWith(UPDATE_FILE_MARKER)) {
    const targetPath = firstLine.slice(UPDATE_FILE_MARKER.length)
    let remaining = lines.slice(1)
    let consumed = 1
    let movePath: string | undefined
    const moveCandidate = remaining[0]?.trim()
    if (moveCandidate?.startsWith(MOVE_TO_MARKER)) {
      movePath = moveCandidate.slice(MOVE_TO_MARKER.length)
      remaining = remaining.slice(1)
      consumed += 1
    }
    const chunks: UpdateChunk[] = []
    while (remaining.length > 0) {
      const candidate = remaining[0]?.trim()
      if (candidate === undefined) break
      if (candidate === '') {
        remaining = remaining.slice(1)
        consumed += 1
        continue
      }
      if (candidate.startsWith('***')) break
      const parsed = parseUpdateChunk(remaining, chunks.length === 0)
      chunks.push(parsed.chunk)
      remaining = remaining.slice(parsed.consumed)
      consumed += parsed.consumed
    }
    if (chunks.length === 0) {
      throw new Error(`Invalid patch hunk: update for '${targetPath}' is empty.`)
    }
    return { hunk: { kind: 'update', path: targetPath, movePath, chunks }, consumed }
  }
  throw new Error(
    `Invalid patch hunk: '${firstLine}' is not a valid header. ` +
      `Valid headers: '${ADD_FILE_MARKER}{path}', '${DELETE_FILE_MARKER}{path}', '${UPDATE_FILE_MARKER}{path}'.`,
  )
}

function parseUpdateChunk(lines: string[], allowMissingContext: boolean): { chunk: UpdateChunk; consumed: number } {
  let changeContext: string | undefined
  let startIndex = 0
  const firstLine = lines[0]
  if (firstLine === EMPTY_CHANGE_CONTEXT_MARKER) {
    startIndex = 1
  } else if (firstLine?.startsWith(CHANGE_CONTEXT_MARKER)) {
    changeContext = firstLine.slice(CHANGE_CONTEXT_MARKER.length)
    startIndex = 1
  } else if (!allowMissingContext) {
    throw new Error(`Invalid patch hunk: expected '@@' context marker, got '${firstLine}'.`)
  }

  const chunk: UpdateChunk = {
    changeContext,
    oldLines: [],
    newLines: [],
    contextOldIndexes: [],
    isEndOfFile: false,
  }
  let parsedLines = 0
  for (const line of lines.slice(startIndex)) {
    if (line === EOF_MARKER) {
      chunk.isEndOfFile = true
      parsedLines += 1
      break
    }
    const marker = line[0]
    if (marker === ' ') {
      chunk.contextOldIndexes.push(chunk.oldLines.length)
      chunk.oldLines.push(line.slice(1))
      chunk.newLines.push(line.slice(1))
      parsedLines += 1
      continue
    }
    if (marker === '+') {
      chunk.contextOldIndexes.push(undefined)
      chunk.newLines.push(line.slice(1))
      parsedLines += 1
      continue
    }
    if (marker === '-') {
      chunk.oldLines.push(line.slice(1))
      parsedLines += 1
      continue
    }
    if (parsedLines === 0) {
      throw new Error(
        `Invalid patch hunk: unexpected line '${line}'. Every line must start with ' ', '+' or '-'.`,
      )
    }
    break
  }
  if (parsedLines === 0) {
    throw new Error('Invalid patch hunk: update chunk is empty.')
  }
  return { chunk, consumed: startIndex + parsedLines }
}

function normalizeUpdateComparison(content: string): string {
  const normalized = content.replace(/\r\n/g, '\n').replace(/\r/g, '\n')
  if (normalized.length === 0 || normalized.endsWith('\n')) return normalized
  return `${normalized}\n`
}

const DASH_PUNCTUATION = /[\u2010-\u2015\u2212]/g
const SINGLE_QUOTE_PUNCTUATION = /[\u2018-\u201B]/g
const DOUBLE_QUOTE_PUNCTUATION = /[\u201C-\u201F]/g
const SPACE_PUNCTUATION = /[\u00A0\u2002-\u200A\u202F\u205F\u3000]/g

function normalizePunctuation(value: string): string {
  return value
    .replace(DASH_PUNCTUATION, '-')
    .replace(SINGLE_QUOTE_PUNCTUATION, "'")
    .replace(DOUBLE_QUOTE_PUNCTUATION, '"')
    .replace(SPACE_PUNCTUATION, ' ')
}

type Normalizer = (value: string) => string

const NORMALIZERS: Normalizer[] = [
  (value: string) => value,
  (value: string) => value.trimEnd(),
  (value: string) => value.trim(),
  (value: string) => normalizePunctuation(value.trim()),
]

function linesMatch(
  lines: string[],
  pattern: string[],
  start: number,
  normalize: Normalizer,
): boolean {
  for (let idx = 0; idx < pattern.length; idx += 1) {
    const line = lines[start + idx]
    const expected = pattern[idx]
    if (line === undefined || expected === undefined) return false
    if (normalize(line) !== normalize(expected)) return false
  }
  return true
}

function seekSequence(
  lines: string[],
  pattern: string[],
  start: number,
  eof: boolean,
): number | null {
  if (pattern.length === 0) return start
  if (pattern.length > lines.length) return null
  const maxStart = lines.length - pattern.length
  const searchStart = eof && lines.length >= pattern.length ? maxStart : Math.min(start, maxStart)
  if (searchStart > maxStart) return null
  for (const normalize of NORMALIZERS) {
    for (let i = searchStart; i <= maxStart; i += 1) {
      if (linesMatch(lines, pattern, i, normalize)) return i
    }
  }
  return null
}

function keepContextBytes(params: {
  originalLines: string[]
  matchIndex: number
  patternLength: number
  newSlice: string[]
  contextOldIndexes: Array<number | undefined>
}): string[] {
  const { originalLines, matchIndex, patternLength, newSlice, contextOldIndexes } = params
  return newSlice.map((line, index) => {
    const oldIndex = contextOldIndexes[index]
    if (oldIndex === undefined || oldIndex >= patternLength) return line
    const original = originalLines[matchIndex + oldIndex]
    return original === undefined ? line : original
  })
}

function computeReplacements(
  originalLines: string[],
  filePath: string,
  chunks: UpdateChunk[],
): Array<[number, number, string[]]> {
  const replacements: Array<[number, number, string[]]> = []
  let lineIndex = 0
  for (const chunk of chunks) {
    if (chunk.changeContext) {
      const ctxIndex = seekSequence(originalLines, [chunk.changeContext], lineIndex, false)
      if (ctxIndex === null) {
        throw new Error(`Failed to find context '${chunk.changeContext}' in ${filePath}.`)
      }
      lineIndex = ctxIndex + 1
    }
    if (chunk.oldLines.length === 0) {
      const insertionIndex =
        chunk.changeContext && !chunk.isEndOfFile
          ? lineIndex
          : originalLines.length > 0 && originalLines[originalLines.length - 1] === ''
            ? originalLines.length - 1
            : originalLines.length
      replacements.push([insertionIndex, 0, chunk.newLines])
      lineIndex = insertionIndex
      continue
    }
    let pattern = chunk.oldLines
    let newSlice = chunk.newLines
    let found = seekSequence(originalLines, pattern, lineIndex, chunk.isEndOfFile)
    if (found === null && pattern[pattern.length - 1] === '') {
      pattern = pattern.slice(0, -1)
      if (newSlice.length > 0 && newSlice[newSlice.length - 1] === '') newSlice = newSlice.slice(0, -1)
      found = seekSequence(originalLines, pattern, lineIndex, chunk.isEndOfFile)
    }
    if (found === null) {
      throw new Error(`Failed to find expected lines in ${filePath}:\n${chunk.oldLines.join('\n')}`)
    }
    replacements.push([
      found,
      pattern.length,
      keepContextBytes({
        originalLines,
        matchIndex: found,
        patternLength: pattern.length,
        newSlice,
        contextOldIndexes: chunk.contextOldIndexes,
      }),
    ])
    lineIndex = found + pattern.length
  }
  replacements.sort((a, b) => a[0] - b[0])
  return replacements
}

function applyReplacements(lines: string[], replacements: Array<[number, number, string[]]>): string[] {
  const result = [...lines]
  for (const [startIndex, oldLen, newLines] of [...replacements].reverse()) {
    for (let i = 0; i < oldLen; i += 1) {
      if (startIndex < result.length) result.splice(startIndex, 1)
    }
    for (const [i, line] of newLines.entries()) {
      result.splice(startIndex + i, 0, line)
    }
  }
  return result
}

function applyUpdateHunk(fs: PatchFs, filePath: string, chunks: UpdateChunk[]): Promise<string> {
  return (async () => {
    const originalContents = await fs.read(filePath).catch((error: unknown) => {
      throw new Error(
        `Failed to read file to update ${filePath}: ${error instanceof Error ? error.message : String(error)}`,
      )
    })
    const originalLines = normalizeUpdateComparison(originalContents).split('\n')
    if (originalLines[originalLines.length - 1] === '') originalLines.pop()
    const replacements = computeReplacements(originalLines, filePath, chunks)
    let newLines = applyReplacements(originalLines, replacements)
    if (newLines.length === 0 || newLines[newLines.length - 1] !== '') newLines = [...newLines, '']
    return newLines.join('\n')
  })()
}

/**
 * Apply a full patch envelope through the provided fs adapter. Every affected
 * path is normalized and must resolve inside the workspace root.
 */
export async function applyPatch(input: string, fs: PatchFs): Promise<ApplyPatchResult> {
  const hunks = parsePatchText(input)
  if (hunks.length === 0) throw new Error('No files were modified.')

  const summary: ApplyPatchSummary = { added: [], modified: [], deleted: [], moved: [] }
  const seen = {
    added: new Set<string>(),
    modified: new Set<string>(),
    deleted: new Set<string>(),
    moved: new Set<string>(),
  }
  let changed = false

  const record = (bucket: keyof ApplyPatchSummary, path: string): void => {
    if (seen[bucket].has(path)) return
    seen[bucket].add(path)
    summary[bucket].push(path)
  }

  for (const hunk of hunks) {
    if (hunk.kind === 'add') {
      const target = normalizePatchPath(hunk.path)
      if (!target) throw new Error(`Invalid add path: '${hunk.path}'.`)
      const parent = target.split('/').slice(0, -1).join('/')
      if (parent) await fs.makeDir(parent)
      if (await fs.exists(target)) {
        const existing = await fs.read(target)
        if (normalizeUpdateComparison(existing) === normalizeUpdateComparison(hunk.contents)) continue
        throw new Error(
          `File '${target}' already exists. Use '${UPDATE_FILE_MARKER}${target}' to change it, ` +
            `or delete it earlier in the same patch.`,
        )
      }
      await fs.write(target, hunk.contents)
      changed = true
      record('added', target)
      continue
    }

    if (hunk.kind === 'delete') {
      const target = normalizePatchPath(hunk.path)
      if (!target) throw new Error(`Invalid delete path: '${hunk.path}'.`)
      await fs.delete(target)
      changed = true
      record('deleted', target)
      continue
    }

    const target = normalizePatchPath(hunk.path)
    if (!target) throw new Error(`Invalid update path: '${hunk.path}'.`)
    const applied = await applyUpdateHunk(fs, target, hunk.chunks)

    if (hunk.movePath) {
      const moveTarget = normalizePatchPath(hunk.movePath)
      if (!moveTarget) throw new Error(`Invalid move path: '${hunk.movePath}'.`)
      const moveParent = moveTarget.split('/').slice(0, -1).join('/')
      if (moveParent) await fs.makeDir(moveParent)
      await fs.write(moveTarget, applied)
      await fs.delete(target)
      changed = true
      record('moved', moveTarget)
      continue
    }

    const existing = await fs.read(target)
    if (normalizeUpdateComparison(existing) === normalizeUpdateComparison(applied)) continue
    await fs.write(target, applied)
    changed = true
    record('modified', target)
  }

  const noOp = !changed
  return {
    summary,
    text: noOp
      ? 'No changes made.'
      : ['Success. Updated the following files:']
          .concat(summary.added.map((file) => `A ${file}`))
          .concat(summary.modified.map((file) => `M ${file}`))
          .concat(summary.deleted.map((file) => `D ${file}`))
          .concat(summary.moved.map((file) => `R ${file}`))
          .join('\n'),
    noOp,
  }
}
