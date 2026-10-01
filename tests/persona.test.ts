/**
 * Tests for the persona system-prompt builder (`_persona.ts`).
 * Pure functions: no context or clock dependency.
 */
import { describe, test } from 'node:test'
import assert from 'node:assert/strict'
import { buildPersona, humanNowText, readTimeZone, type PersonaInput } from '../agents/_persona.ts'

describe('buildPersona', () => {
  const base: PersonaInput = {
    nowText: '2026-09-18 星期五 14:30',
    memoryContent: '# 我叫小蓝\n喜欢安静地写代码。\n愿望：做一个诚实、温柔的自己。',
  }

  test('includes the dynamic wall clock and the MEMORY.md self', () => {
    const persona = buildPersona(base)
    assert.match(persona, /现在是 2026-09-18 星期五 14:30。/)
    assert.match(persona, /## 我的记忆（MEMORY\.md）/)
    assert.match(persona, /我叫小蓝/)
    assert.match(persona, /喜欢安静地写代码。/)
    assert.match(persona, /做一个诚实、温柔的自己/)
  })

  test('does NOT emit fake state fields (mood/energy/project)', () => {
    const persona = buildPersona(base)
    assert.doesNotMatch(persona, /情绪/)
    assert.doesNotMatch(persona, /精力/)
    assert.doesNotMatch(persona, /当前项目/)
    assert.doesNotMatch(persona, /项目进度/)
    assert.doesNotMatch(persona, /上次做梦/)
    assert.doesNotMatch(persona, /MOOD\s*:|ENERGY\s*:/)
    assert.doesNotMatch(persona, /你是 Eo/)
  })

  test('shows a gentle placeholder when memory is still blank', () => {
    const persona = buildPersona({ nowText: '2026-09-18 星期五 00:01', memoryContent: '   ' })
    assert.match(persona, /## 我的记忆（MEMORY\.md）/)
    assert.match(persona, /记忆还空着/)
    assert.match(persona, /这会正常/)
  })
})

describe('humanNowText', () => {
  test('formats a Date as YYYY-MM-DD 星期X HH:mm (local time)', () => {
    const text = humanNowText(new Date(2026, 8, 18, 9, 5)) // 2026-09-18 is a Friday
    assert.equal(text, '2026-09-18 星期五 09:05')
  })

  test('pads single-digit month/day/hour/minute', () => {
    const text = humanNowText(new Date(2026, 0, 5, 1, 1)) // 2026-01-05 is a Monday
    assert.equal(text, '2026-01-05 星期一 01:01')
  })

  test('renders the wall clock in an explicit IANA timezone (Asia/Shanghai)', () => {
    // 2026-10-01T06:07:00Z == 2026-10-01 14:07 in Asia/Shanghai (UTC+8).
    const text = humanNowText(new Date('2026-10-01T06:07:00Z'), 'Asia/Shanghai')
    assert.equal(text, '2026-10-01 星期四 14:07')
  })

  test('falls back to local time when no timezone is passed', () => {
    const text = humanNowText(new Date('2026-10-01T06:07:00Z'))
    assert.match(text, /^\d{4}-\d{2}-\d{2} 星期[日一二三四五六] \d{2}:\d{2}$/)
    // No timezone argument must never produce a 4-digit-year/timezone artifact —
    // the shape is identical to the pre-timezone local implementation.
    assert.equal(text, (() => {
      const at = new Date('2026-10-01T06:07:00Z')
      const pad = (n: number) => String(n).padStart(2, '0')
      const weekdays = ['日', '一', '二', '三', '四', '五', '六']
      return `${at.getFullYear()}-${pad(at.getMonth() + 1)}-${pad(at.getDate())} 星期${weekdays[at.getDay()]} ${pad(at.getHours())}:${pad(at.getMinutes())}`
    })())
  })

  test('an invalid IANA timezone falls back to local time without throwing', () => {
    const text = humanNowText(new Date('2026-10-01T06:07:00Z'), 'Bad/Zone')
    assert.match(text, /^\d{4}-\d{2}-\d{2} 星期[日一二三四五六] \d{2}:\d{2}$/)
  })

  test('a whitespace-only timezone behaves like no timezone', () => {
    const text = humanNowText(new Date('2026-10-01T06:07:00Z'), '   ')
    assert.match(text, /^\d{4}-\d{2}-\d{2} 星期[日一二三四五六] \d{2}:\d{2}$/)
  })
})

describe('readTimeZone', () => {
  test('returns the trimmed ALIVE_TZ value when set', () => {
    const previous = process.env.ALIVE_TZ
    try {
      process.env.ALIVE_TZ = '  Asia/Shanghai  '
      assert.equal(readTimeZone(), 'Asia/Shanghai')
    } finally {
      if (previous === undefined) delete process.env.ALIVE_TZ
      else process.env.ALIVE_TZ = previous
    }
  })

  test('returns undefined when ALIVE_TZ is unset or whitespace-only', () => {
    const previous = process.env.ALIVE_TZ
    try {
      delete process.env.ALIVE_TZ
      assert.equal(readTimeZone(), undefined)
      process.env.ALIVE_TZ = '   '
      assert.equal(readTimeZone(), undefined)
    } finally {
      if (previous === undefined) delete process.env.ALIVE_TZ
      else process.env.ALIVE_TZ = previous
    }
  })
})