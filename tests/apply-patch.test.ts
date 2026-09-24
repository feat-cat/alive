/**
 * Pure logic tests for the apply_patch "Begin Patch" envelope parser/applier.
 * All filesystem access goes through an in-memory `PatchFs` adapter.
 */
import { describe, test } from 'node:test'
import assert from 'node:assert/strict'
import { applyPatch, normalizePatchPath, parsePatchText, type PatchFs } from '../agents/_apply-patch.ts'

function makeFs(initial: Record<string, string> = {}): { fs: PatchFs; files: Map<string, string> } {
  const files = new Map<string, string>(Object.entries(initial))
  const fs: PatchFs = {
    read: async (path) => {
      const content = files.get(path)
      if (content === undefined) throw new Error(`ENOENT: ${path}`)
      return content
    },
    write: async (path, content) => {
      files.set(path, content)
    },
    delete: async (path) => {
      files.delete(path)
    },
    exists: async (path) => files.has(path),
    makeDir: async () => undefined,
  }
  return { fs, files }
}

const ADD_PATCH = `*** Begin Patch
*** Add File: hello.txt
+Hello, world!
*** End Patch`

describe('applyPatch — add', () => {
  test('adds a single new file and reports it in the summary', async () => {
    const { fs, files } = makeFs()
    const result = await applyPatch(ADD_PATCH, fs)
    assert.equal(result.noOp, false)
    assert.deepEqual(result.summary.added, ['hello.txt'])
    assert.equal(files.get('hello.txt'), 'Hello, world!\n')
    assert.ok(result.text.includes('A hello.txt'))
  })

  test('creates parent directories for nested add paths', async () => {
    const { fs, files } = makeFs()
    const patch = `*** Begin Patch
*** Add File: src/lib/util.ts
+export const value = 1
*** End Patch`
    const result = await applyPatch(patch, fs)
    assert.deepEqual(result.summary.added, ['src/lib/util.ts'])
    assert.equal(files.get('src/lib/util.ts'), 'export const value = 1\n')
  })

  test('add that already exists with identical content is a no-op', async () => {
    const { fs } = makeFs({ 'hello.txt': 'Hello, world!\n' })
    const result = await applyPatch(ADD_PATCH, fs)
    assert.equal(result.noOp, true)
    assert.equal(result.text, 'No changes made.')
  })

  test('add that already exists with different content throws', async () => {
    const { fs } = makeFs({ 'hello.txt': 'Different\n' })
    await assert.rejects(applyPatch(ADD_PATCH, fs), /already exists/)
  })
})

describe('applyPatch — update', () => {
  test('replaces old lines with new lines (with @@ context)', async () => {
    const { fs, files } = makeFs({ 'file.txt': 'a\nb\nc\n' })
    const patch = `*** Begin Patch
*** Update File: file.txt
@@ a
-b
+B
*** End Patch`
    const result = await applyPatch(patch, fs)
    assert.deepEqual(result.summary.modified, ['file.txt'])
    assert.equal(files.get('file.txt'), 'a\nB\nc\n')
    assert.ok(result.text.includes('M file.txt'))
  })

  test('supports a hunk without a @@ context marker', async () => {
    const { fs, files } = makeFs({ 'file.txt': 'a\nb\nc\n' })
    const patch = `*** Begin Patch
*** Update File: file.txt
-b
+B
*** End Patch`
    await applyPatch(patch, fs)
    assert.equal(files.get('file.txt'), 'a\nB\nc\n')
  })

  test('ambiguous (multi-match) old lines resolve to the first occurrence', async () => {
    // Documented behavior: an ambiguous pattern replaces the first match only.
    const { fs, files } = makeFs({ 'file.txt': 'a\nb\na\nb\n' })
    const patch = `*** Begin Patch
*** Update File: file.txt
-a
+x
*** End Patch`
    await applyPatch(patch, fs)
    assert.equal(files.get('file.txt'), 'x\nb\na\nb\n')
  })

  test('update that changes nothing is a no-op', async () => {
    const { fs } = makeFs({ 'file.txt': 'a\nb\n' })
    const patch = `*** Begin Patch
*** Update File: file.txt
-b
+b
*** End Patch`
    const result = await applyPatch(patch, fs)
    assert.equal(result.noOp, true)
    assert.equal(result.text, 'No changes made.')
  })
})

describe('applyPatch — delete', () => {
  test('deletes an existing file', async () => {
    const { fs, files } = makeFs({ 'old.txt': 'byebye\n' })
    const patch = `*** Begin Patch
*** Delete File: old.txt
*** End Patch`
    const result = await applyPatch(patch, fs)
    assert.deepEqual(result.summary.deleted, ['old.txt'])
    assert.equal(files.has('old.txt'), false)
    assert.ok(result.text.includes('D old.txt'))
  })
})

describe('applyPatch — multi-file', () => {
  test('add + update + delete in one envelope', async () => {
    const { fs, files } = makeFs({ 'b.txt': 'old\n', 'c.txt': 'gone\n' })
    const patch = `*** Begin Patch
*** Add File: a.txt
+alpha
*** Update File: b.txt
-old
+new
*** Delete File: c.txt
*** End Patch`
    const result = await applyPatch(patch, fs)
    assert.deepEqual(result.summary.added, ['a.txt'])
    assert.deepEqual(result.summary.modified, ['b.txt'])
    assert.deepEqual(result.summary.deleted, ['c.txt'])
    assert.equal(files.get('a.txt'), 'alpha\n')
    assert.equal(files.get('b.txt'), 'new\n')
    assert.equal(files.has('c.txt'), false)
  })
})

describe('applyPatch — error paths', () => {
  test('empty input throws', async () => {
    const { fs } = makeFs()
    await assert.rejects(applyPatch('', fs), /input is empty/)
  })

  test('missing Begin/End markers throws', async () => {
    const { fs } = makeFs()
    await assert.rejects(
      applyPatch('*** Add File: x.txt\n+x\n', fs),
      /must start with '.* Begin Patch' and end with '.* End Patch'/,
    )
  })

  test('update that cannot find expected lines throws', async () => {
    const { fs } = makeFs({ 'file.txt': 'a\nb\n' })
    const patch = `*** Begin Patch
*** Update File: file.txt
-zzz
+x
*** End Patch`
    await assert.rejects(applyPatch(patch, fs), /Failed to find expected lines/)
  })

  test('update with a @@ context that is missing throws', async () => {
    const { fs } = makeFs({ 'file.txt': 'a\nb\n' })
    const patch = `*** Begin Patch
*** Update File: file.txt
@@ missing-context
-a
+x
*** End Patch`
    await assert.rejects(applyPatch(patch, fs), /Failed to find context/)
  })

  test('unsafe add paths (parent traversal) are rejected', async () => {
    const { fs } = makeFs()
    const patch = `*** Begin Patch
*** Add File: ../evil.txt
+bad
*** End Patch`
    await assert.rejects(applyPatch(patch, fs), /Invalid add path/)
  })

  test('unsafe update paths are rejected', async () => {
    const { fs } = makeFs({ 'safe.txt': 'a\n' })
    const patch = `*** Begin Patch
*** Update File: /etc/passwd
-a
+b
*** End Patch`
    await assert.rejects(applyPatch(patch, fs), /Invalid update path/)
  })
})

describe('parsePatchText', () => {
  test('parses an add hunk', () => {
    const hunks = parsePatchText(ADD_PATCH)
    assert.equal(hunks.length, 1)
    assert.equal(hunks[0]?.kind, 'add')
  })

  test('parses update hunks with multiple chunks', () => {
    const patch = `*** Begin Patch
*** Update File: file.txt
@@ one
-a
+b
@@ two
-c
+d
*** End Patch`
    const hunks = parsePatchText(patch)
    assert.equal(hunks.length, 1)
    if (hunks[0]?.kind !== 'update') throw new Error('expected update hunk')
    assert.equal(hunks[0].chunks.length, 2)
  })
})

describe('normalizePatchPath', () => {
  test('accepts relative paths and strips leading ./', () => {
    assert.equal(normalizePatchPath('./a/b.ts'), 'a/b.ts')
    assert.equal(normalizePatchPath('a/b.ts'), 'a/b.ts')
  })
  test('rejects absolute, empty and traversal segments', () => {
    assert.equal(normalizePatchPath('/abs'), null)
    assert.equal(normalizePatchPath('a/../b'), null)
    assert.equal(normalizePatchPath(''), null)
    assert.equal(normalizePatchPath('a//b'), null)
  })
})
