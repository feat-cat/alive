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

/** Fallback: format a Date from its local wall-clock fields. */
function localNowText(at: Date): string {
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
 * Read the optional IANA timezone from `ALIVE_TZ` (e.g. `Asia/Shanghai`). The
 * value is trimmed; unset / empty / whitespace-only returns `undefined` (the
 * caller then falls back to the function-local wall clock). Invalid IANA names
 * are NOT rejected here — `Intl.DateTimeFormat` throws on them and
 * `humanNowText` catches that and falls back to local time.
 */
export function readTimeZone(): string | undefined {
  const value = process.env.ALIVE_TZ?.trim()
  return value ? value : undefined
}

/**
 * Format a Date as "YYYY-MM-DD 星期X HH:mm". When `timeZone` is a valid IANA
 * zone, the wall clock is rendered in that zone via `Intl.DateTimeFormat`
 * (explicit 4-digit year — zh-CN may otherwise emit a 2-digit year); an invalid
 * zone (or any Intl failure) falls back to the local wall clock, preserving the
 * pre-timezone behaviour.
 */
export function humanNowText(at: Date = new Date(), timeZone?: string): string {
  if (timeZone && timeZone.trim()) {
    try {
      const formatter = new Intl.DateTimeFormat('zh-CN', {
        timeZone,
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        weekday: 'short',
        hour: '2-digit',
        minute: '2-digit',
        hourCycle: 'h23',
      })
      const parts = new Map(formatter.formatToParts(at).map((part) => [part.type, part.value]))
      const year = parts.get('year')
      const month = parts.get('month')
      const day = parts.get('day')
      const weekdayRaw = parts.get('weekday')
      const hours = parts.get('hour')
      const minutes = parts.get('minute')
      if (year && month && day && weekdayRaw && hours && minutes) {
        const weekday = weekdayRaw.replace(/^周/, '')
        return `${year}-${month}-${day} 星期${weekday} ${hours}:${minutes}`
      }
    } catch {
      // Invalid IANA timezone → fall through to the local wall clock.
    }
  }
  return localNowText(at)
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