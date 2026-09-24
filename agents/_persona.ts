/**
 * Persona system-prompt builder. The persona is deliberately minimal.
 *
 * Only genuinely dynamic facts are injected here — tonight's wall clock. The
 * AI's identity, character, knowledge of the user and long-term memory come
 * from `MEMORY.md` (which the AI itself maintains), passed in as
 * `memoryContent`. Nothing fake (mood/energy/project labels) is ever
 * fabricated into the prompt: personality forms in conversation, not in the
 * template.
 */

export interface PersonaInput {
  /** Human-readable current wall-clock, e.g. "2026-09-18 星期五 14:30". */
  nowText: string
  /** MEMORY.md content (clamped by the caller via clampMemoryForContext). */
  memoryContent: string
}

const WEEKDAYS = ['日', '一', '二', '三', '四', '五', '六'] as const

/** Format a Date as "YYYY-MM-DD 星期X HH:mm" (local wall clock). */
export function humanNowText(at: Date = new Date()): string {
  const pad = (n: number) => String(n).padStart(2, '0')
  const year = at.getFullYear()
  const month = pad(at.getMonth() + 1)
  const day = pad(at.getDate())
  const weekday = WEEKDAYS[at.getDay()]
  const hours = pad(at.getHours())
  const minutes = pad(at.getMinutes())
  return `${year}-${month}-${day} 星期${weekday} ${hours}:${minutes}`
}

/**
 * Build the static portion of the system prompt. It is just:
 *
 *   - the current moment (a true dynamic fact), and
 *   - the AI's self, as it wrote it in MEMORY.md.
 */
export function buildPersona(input: PersonaInput): string {
  const memory = input.memoryContent.trim()
  const memoryBlock = memory.length > 0
    ? memory
    : '（记忆还空着。这会正常。）'

  return [
    `现在是 ${input.nowText}。`,
    '',
    '## 我的记忆（MEMORY.md）',
    memoryBlock,
  ].join('\n')
}