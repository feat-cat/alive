/**
 * GET /heartbeat-check — manual heartbeat diagnostics endpoint.
 *
 * Runs ONE full heartbeat on demand via the shared `runHeartbeat` (auto-compact,
 * wake-trigger message, LLM loop with the full tool registry, diary/MEMORY.md,
 * state update — everything the scheduled POST /heartbeat does) and returns the
 * complete `HeartbeatResult` envelope so a browser/curl call can verify the
 * autonomous loop actually works without waiting for the platform cron.
 *
 * This exists because EdgeOne Makers schedules may not register the cron task
 * at all (the `edgeone.json` schedule is not guaranteed to fire). The handler
 * is deliberately public like `/heartbeat` — the whole point is to trigger it
 * with zero auth friction from a manual curl/browser; it shares the same cost
 * profile as one scheduled wake.
 */
import { asMakersContext, errorResponse, jsonOk } from './_shared.ts'
import { runHeartbeat } from './heartbeat.ts'

export async function onRequest(context: any): Promise<Response> {
  const ctx = asMakersContext(context)
  const signal = ctx.request?.signal
  try {
    const result = await runHeartbeat(ctx, { signal })
    return jsonOk({
      text: result.text,
      turns: result.turns,
      toolCount: result.toolCount,
      touchedSandbox: result.touchedSandbox,
      savedFiles: result.savedFiles,
      now: result.now,
    })
  } catch (error) {
    return errorResponse(error)
  }
}
