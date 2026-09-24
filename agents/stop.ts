/**
 * POST /stop — abort an active run for a conversation.
 *
 * Delegates to the platform `context.utils.abortActiveRun(conversationId)` when
 * it exists (older/newer runtimes may omit it) and reports whether an abort was
 * actually signalled. Mirrors deepseek-harness/agents/stop.ts.
 */
import {
  asMakersContext,
  bodyValue,
  errorResponse,
  jsonError,
  jsonOk,
  requireAuth,
  type MakersContext,
} from './_shared.ts'

export type StopResult = {
  aborted: boolean
}

export async function runStop(context: MakersContext, conversationId: string): Promise<StopResult> {
  const result = await context.utils?.abortActiveRun?.(conversationId)
  return { aborted: result?.aborted === true }
}

export async function onRequest(context: any): Promise<Response> {
  const ctx = asMakersContext(context)
  const denied = requireAuth(ctx)
  if (denied) return denied
  try {
    const conversationId = bodyValue(ctx, 'conversation_id')
    if (!conversationId.trim()) {
      return jsonError(400, 'conversation_id is required')
    }
    const result = await runStop(ctx, conversationId.trim())
    return jsonOk({ conversation_id: conversationId.trim(), aborted: result.aborted })
  } catch (error) {
    return errorResponse(error)
  }
}
