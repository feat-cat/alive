/**
 * Tests for the Blob persistence helpers (`_blob-tools.ts`): key scoping,
 * normalization and the injectable store mock read/write path.
 */
import { afterEach, describe, test } from 'node:test'
import assert from 'node:assert/strict'
import {
  blobDeleteTool,
  blobEditTool,
  blobWriteTool,
  deleteBlobKey,
  injectBlobStoreForTesting,
  listBlobKeys,
  normalizeBlobKey,
  readBlobText,
  scopedBlobKey,
  writeBlobText,
} from '../agents/_blob-tools.ts'
import { SELF_ID } from '../agents/_shared.ts'
import { makeContext, makeMockBlobStore } from './_helpers.ts'

afterEach(() => {
  injectBlobStoreForTesting(null)
})

describe('normalizeBlobKey', () => {
  test('normalizes separators and returns a clean relative key', () => {
    assert.equal(normalizeBlobKey('workspace/a/b.ts'), 'workspace/a/b.ts')
    assert.equal(normalizeBlobKey('workspace\\a\\b.ts'), 'workspace/a/b.ts')
    assert.equal(normalizeBlobKey('  memory/MEMORY.md  '), 'memory/MEMORY.md')
  })
  test('rejects escaping, absolute and empty keys', () => {
    assert.equal(normalizeBlobKey(''), null)
    assert.equal(normalizeBlobKey('  '), null)
    assert.equal(normalizeBlobKey('/abs'), null)
    assert.equal(normalizeBlobKey('../evil'), null)
    assert.equal(normalizeBlobKey('a/./b'), null)
    assert.equal(normalizeBlobKey('a/../b'), null)
  })
})

describe('scopedBlobKey', () => {
  test('keeps memory/ keys global (not conversation-scoped)', () => {
    assert.equal(scopedBlobKey(SELF_ID, 'memory/MEMORY.md'), 'memory/MEMORY.md')
    assert.equal(scopedBlobKey('eo-play', 'memory/DREAMS.md'), 'memory/DREAMS.md')
  })
  test('scopes other keys under projects/<conversation>/', () => {
    assert.equal(scopedBlobKey(SELF_ID, 'workspace/src/a.ts'), 'projects/eo-self/workspace/src/a.ts')
    assert.equal(scopedBlobKey('eo-play', 'notes.md'), 'projects/eo-play/notes.md')
  })
  test('throws on invalid keys', () => {
    assert.throws(() => scopedBlobKey(SELF_ID, '../evil'), /Invalid blob key/)
  })
})

describe('injected blob store read/write', () => {
  test('writeBlobText then readBlobText round-trips through the mock store', async () => {
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const context = makeContext({})

    await writeBlobText(context, SELF_ID, 'workspace/hello.txt', 'hi')
    assert.equal(blob.blobMap.get('projects/eo-self/workspace/hello.txt'), 'hi')

    const text = await readBlobText(context, SELF_ID, 'workspace/hello.txt')
    assert.equal(text, 'hi')
  })

  test('memory/ keys bypass the project scope in the mock store', async () => {
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const context = makeContext({})

    await writeBlobText(context, SELF_ID, 'memory/MEMORY.md', 'persist')
    assert.equal(blob.blobMap.get('memory/MEMORY.md'), 'persist')
  })

  test('listBlobKeys returns scoped keys for a prefix', async () => {
    const blob = makeMockBlobStore({
      'projects/eo-self/workspace/a.ts': '1',
      'projects/eo-self/workspace/sub/b.ts': '2',
      'projects/eo-play/workspace/c.ts': '3',
      'memory/MEMORY.md': '4',
    })
    injectBlobStoreForTesting(blob)
    const context = makeContext({})

    const keys = await listBlobKeys(context, SELF_ID, 'workspace')
    assert.deepEqual(keys.sort(), [
      'projects/eo-self/workspace/a.ts',
      'projects/eo-self/workspace/sub/b.ts',
    ])
  })

  test('deleteBlobKey removes the scoped key', async () => {
    const blob = makeMockBlobStore({ 'projects/eo-self/notes.md': 'x' })
    injectBlobStoreForTesting(blob)
    await deleteBlobKey(makeContext({}), SELF_ID, 'notes.md')
    assert.equal(blob.blobMap.has('projects/eo-self/notes.md'), false)
  })
})

describe('blob write/edit tools', () => {
  test('blobWriteTool requires a key', async () => {
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const result = await blobWriteTool(makeContext({}), SELF_ID, { content: 'x' })
    assert.equal(result.isError, true)
    assert.match(result.content, /requires "key"/)
  })

  test('blobEditTool errors when the key does not exist', async () => {
    const blob = makeMockBlobStore()
    injectBlobStoreForTesting(blob)
    const patch = `*** Begin Patch
*** Update File: notes.md
-old
+new
*** End Patch`
    const result = await blobEditTool(makeContext({}), SELF_ID, { key: 'memory/notes.md', input: patch })
    assert.equal(result.isError, true)
    assert.match(result.content, /not found/)
  })

  test('blobEditTool applies a one-file patch through the injected store', async () => {
    const blob = makeMockBlobStore({ 'memory/notes.md': 'old line\n' })
    injectBlobStoreForTesting(blob)
    const patch = `*** Begin Patch
*** Update File: notes.md
-old line
+new line
*** End Patch`
    const result = await blobEditTool(makeContext({}), SELF_ID, { key: 'memory/notes.md', input: patch })
    assert.equal(result.isError, undefined)
    assert.equal(blob.blobMap.get('memory/notes.md'), 'new line\n')
  })

  test('blobEditTool rejects patches touching more than one file', async () => {
    const blob = makeMockBlobStore({ 'memory/notes.md': 'a\nb\n' })
    injectBlobStoreForTesting(blob)
    const patch = `*** Begin Patch
*** Update File: notes.md
-a
+x
*** Update File: other.txt
-b
+y
*** End Patch`
    const result = await blobEditTool(makeContext({}), SELF_ID, { key: 'memory/notes.md', input: patch })
    assert.equal(result.isError, true)
    assert.match(result.content, /exactly one file/)
  })

  test('blobDeleteTool deletes a scoped key', async () => {
    const blob = makeMockBlobStore({ 'projects/eo-self/notes.md': 'x' })
    injectBlobStoreForTesting(blob)
    const result = await blobDeleteTool(makeContext({}), SELF_ID, { key: 'notes.md' })
    assert.equal(result.isError, undefined)
    assert.equal(blob.blobMap.has('projects/eo-self/notes.md'), false)
  })
})
