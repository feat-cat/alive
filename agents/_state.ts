/**
 * Agent state persisted via `store.state` (conversation-scoped persistent KV).
 * Heartbeat and chat share the fixed SELF_ID conversation, so there is exactly
 * one autonomous persona — no legacy PLAY conversation.
 *
 * State is deliberately minimal: only the last-activity timestamp and the
 * created time survive here. Mood/energy/project are NOT state — they were
 * fake (labels the model wrote for itself and fed back in), so the AI now
 * writes its identity, character and goals into MEMORY.md instead.
 */
import { SELF_ID, type MakersContext } from './_shared.ts'

/** State key for the self persona (fixed SELF_ID conversation). */
export const SELF_STATE_KEY = 'agent_state_self'

export interface AgentState {
  /** Last heartbeat timestamp (ms epoch); 0 = never. */
  lastActivityAt: number
  /** First-seen timestamp (ms epoch), preserved on every update. */
  created: number
}

export function defaultState(): AgentState {
  return {
    lastActivityAt: 0,
    created: Date.now(),
  }
}

export function isAgentState(value: unknown): value is AgentState {
  if (!value || typeof value !== 'object') return false
  const record = value as Record<string, unknown>
  return (
    typeof record.lastActivityAt === 'number' &&
    typeof record.created === 'number'
  )
}

function requireStore(context: MakersContext): NonNullable<MakersContext['store']> {
  if (!context.store) throw new Error('Store is not available in this context.')
  return context.store
}

export async function getState(context: MakersContext, conversationId: string): Promise<AgentState> {
  const store = requireStore(context)
  const stored = await store.state.get<unknown>(SELF_STATE_KEY, conversationId)
  if (isAgentState(stored)) return stored
  return defaultState()
}

/** Merge a partial patch into persisted state and return the merged value. */
export async function updateState(context: MakersContext, conversationId: string, patch: Partial<AgentState>): Promise<AgentState> {
  const store = requireStore(context)
  const current = await getState(context, conversationId)
  const merged: AgentState = { ...current, ...patch, created: current.created }
  await store.state.set(SELF_STATE_KEY, merged, conversationId)
  return merged
}

export async function clearState(context: MakersContext, conversationId: string): Promise<void> {
  const store = requireStore(context)
  await store.state.delete(SELF_STATE_KEY, conversationId)
}

/** Quick way for callers to read the self state without naming the id. */
export async function getSelfState(context: MakersContext): Promise<AgentState> {
  return getState(context, SELF_ID)
}
