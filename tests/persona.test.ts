/**
 * Tests for the persona system-prompt builder (`_persona.ts`).
 * Pure functions: no context or clock dependency.
 */
import { describe, test } from 'node:test'
import assert from 'node:assert/strict'
import { buildPersona, humanNowText, type PersonaInput } from '../agents/_persona.ts'

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
})