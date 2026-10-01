# alive

**English** | [简体中文](README_zh-CN.md)

[![Deploy with EdgeOne Makers](https://cdnstatic.tencentcs.com/edgeone/pages/deploy.svg)](https://edgeone.ai/pages/new?repository-url=https%3A%2F%2Fgithub.com%2Ffeat-cat%2Falive&install-command=npm%20install&build-command=npm%20run%20build%3Amakers&output-directory=dist)

> alive — an autonomous agent living on EdgeOne Makers. Each heartbeat it freely decides what to do: think, journal, push a small project forward, or simply rest. Its personality grows inside `MEMORY.md`.

`alive` is an autonomous agent kernel for EdgeOne Makers' free tier, driven by a single heartbeat schedule. On every heartbeat the AI **freely decides** what to do — combining the current moment, its own memory (`MEMORY.md`) and the full tool registry — instead of picking from a fixed menu: quietly think and journal, push a small project forward, organize its memory, or simply rest. There is no web UI and no persistent process; everything runs on demand in EdgeOne Makers Functions, and the agent "remembers" itself across turns through strongly-consistent Blob storage and the conversation store.

## What it is

A minimal "agent with a self" skeleton:

- **heartbeat (free-form)** — the only entry point. The AI gets the current state, recent logs/memory, and the full tool registry (blob, workspace, search) and decides on its own what to do right now. **Not a multiple-choice question, and not every heartbeat must produce output.** Resting is a fully legal choice.
- **Pure thought / rest** — the AI just writes text (inner monologue, feelings) with zero sandbox cost. This is the most common path.
- **Tinker with a project** — only when the AI actively calls `workspace_*` tools does it touch the sandbox; changes are auto-mirrored back to Blob.
- **Organize memory** — the AI freely reads/writes its diary (`memory/daily/YYYY-MM-DD.md`) and long-term notes (`MEMORY.md`) via `blob_*` tools. No forced distillation.
- **chat / stop / history** — conversation, abort and history-archive endpoints (reserved for future Matrix integration). `/chat` streams **SSE by default** (typewriter effect, `text/event-stream`; see "Chat streaming") and supports **image input** (base64 `data:` URLs in `images[]`, with automatic text-only degradation when the model lacks vision — see "Images"); it shares `SELF_ID=eo-self` with heartbeat, so private thoughts and user conversations live in the same history stream; it also uses the **full tool registry like heartbeat** (blob + diary + chatlog + workspace + search), so the model can both chat and act; `/history` reads the complete, never-compacted chatlog archive from Blob.

It is deliberately thin: each turn is a bounded LLM loop plus a little state I/O, so it runs sustainably on the free tier.

## Architecture

```
EdgeOne Makers schedules (cron)
        │  daily 03:00 → POST /heartbeat
        ▼
  ┌─────────────────────────────────────────┐
  │  heartbeat.ts  state/logs/memory → LLM  │
  │  free-form (open-ended, not 4-choice)   │
  │  tools: blob_* + diary_* + chatlog_*   │
  │         + workspace_* + web_search     │
  │  budget: ≤3 turns / 100s short turn     │
  └───────────────┬─────────────────────────┘
                  │ AI decides
      ┌───────────┼──────────────┬───────────────┐
      ▼           ▼              ▼               ▼
   think/journal workspace_*   blob_* tools    rest
   (zero sandbox) (touches    (memory        (one line
                  sandbox)     organizing)     only)
      │           │              │               │
      │           ▼              ▼               │
      │   snapshotWorkspaceToBlob                 │
      ▼           ▼              ▼               ▼
   context.store            Blob strong-consistency (@edgeone/pages-blob)
   ├── messages (standard {role,content} array + kind:'tool' records, auto-compact 50-75%)
   ├── store.state (lastActivityAt/created)
                            └── memory/… (diary daily/, MEMORY.md, archive/)
                            └── chatlog/… (complete history, append-only, never compacted)
                            └── projects/<conv>/… (workspace mirror, project files)
      │
      ▼
   Sandbox — only when the AI actively calls workspace_* (quota-guarded)
   workspace writes are mirrored to Blob; snapshot and release when done
```

## Directory layout

```
alive/
├── agents/                 # All runtime code (loaded as Makers Functions endpoints)
│   ├── _shared.ts          # Types, constants, SSE, JSON responses, error mapping, fixed ids
│   ├── _llm.ts             # AI Gateway chat/completions + bounded tool loop + SSE streaming
│   ├── _state.ts           # agent_state read/update (only lastActivityAt/created)
│   ├── _persona.ts         # system-prompt builder (dynamic date + MEMORY.md "my memory")
│   ├── _memory.ts          # four-tier memory: store context (auto-compact), Blob diary, MEMORY.md, chatlog archive
│   ├── _blob-tools.ts      # Blob strong-consistency persistence + blob_* tools
│   ├── _workspace-tools.ts # sandbox workspace tools (write-through mirror to Blob)
│   ├── _apply-patch.ts     # "Begin Patch" envelope parser/apply (pure logic)
│   ├── _tavily.ts          # web_search (Tavily) executor
│   ├── _tools.ts           # heartbeat full tool registry (blob + diary + chatlog + workspace + search)
│   ├── heartbeat.ts        # POST /heartbeat (the only main entry, free-form)
│   ├── chat.ts             # POST /chat (SSE streaming; full tool set like heartbeat)
│   ├── history.ts          # GET /history (complete chatlog archive reader)
│   └── stop.ts             # POST /stop (reserved)
├── tests/                  # node:test unit/endpoint tests
├── edgeone.json            # Makers config (timeout, sandbox, schedules)
├── .env.example            # environment variable declarations
└── package.json
```

Only `heartbeat.ts`, `chat.ts`, `history.ts`, `stop.ts` are routable endpoints; `_`-prefixed files are internal modules.

## Core design decisions

1. **Pure thought/rest never touches the sandbox.** Sandbox quota is limited (100k GB-s/month). The heartbeat gets the full tool registry, but only when the AI actively calls `workspace_*` does a sandbox spin up; pure-text paths only read/write store and Blob.

2. **Blob strong-consistency persistence; sandbox /tmp is not persistent** → workspace writes are mirrored to Blob immediately. At turn end, if the workspace was touched, `snapshotWorkspaceToBlob` snapshots the sandbox contents (including command-generated files) back to Blob, so even if the sandbox is recycled, project files survive.

3. **Short bounded turns, on-demand startup.** Each heartbeat LLM loop is bounded by `HEARTBEAT_MAX_TURNS=3` and a 100s turn budget (`PLAY_TURN_TIMEOUT_MS`); after the turn the sandbox is snapshotted and released. No long-lived processes.

4. **No deepseek-harness persistent-sidecar pattern** — that would burn through the free quota instantly. alive's whole "life" is triggered by the heartbeat schedule; no request, no cost.

5. **Free-form, not a multiple-choice menu.** The AI isn't forced to pick among think/dream/play/rest. It gets state, memory and tools and decides on its own — and **not every heartbeat must produce output**; resting is a first-class option. This hands the limited sandbox quota to the AI to allocate as needed.

6. **Unified conversation.** `SELF_ID=eo-self` for everything — heartbeat AND chat share one history, so the agent's private thoughts and its conversations with the user are one continuous self. State and memory stay consistent across requests and restarts.

7. **Every model loop is bounded.** All `chatCompletion` calls pass `maxTurns` (heartbeat=3, chat=3, compact=1) with a 90s LLM timeout.

8. **Never crash (rule #11).** All endpoints wrap logic in `errorResponse` mapping exceptions to stable JSON errors; tool executors catch errors and return `{ isError: true }` instead of throwing.

## Personality & state (character forms in conversation)

- **No fake state fields.** The kernel does not maintain fabricated mood/energy/project fields. Each heartbeat is an independent LLM call; "emotion" fed back from a previous turn's own label would be acting, not feeling — those fields were removed. `store.state` only keeps the minimal, real `lastActivityAt` and `created`.
- **The system prompt only injects what's dynamic and necessary:** the current moment (`现在是 YYYY-MM-DD 星期X HH:mm`) plus a fixed `DECISION_SYSTEM` written in **demonstration style, not philosophy** (OpenClaw `SOUL.md` spirit): instead of abstract "you are alive" ideals, it shows the AI *how to sound* — "能一句话说完，绝不说两句 / 有感受，不要总结 / 不要解释自己，不要客套 / 永远不要以'作为一个AI'开头 / 可以有观点" — so the concrete speaking examples are the first impression, overriding the default LLM habit of courteous, explanatory long-form text at the prompt level. Each heartbeat is a **waking**, not an alarm to complete tasks; the sandbox is expensive, the diary is private, `MEMORY.md` is your own, honesty is the bottom line. After the persona block, `SYSTEM_HISTORY_GUIDANCE` explains the `[system][heartbeat]` / `[system][compact]` message markers so auto-triggered history is never misread as the user speaking (chat reuses the same guidance).
- **Personality lives in `MEMORY.md`**, which the AI maintains itself — identity, character, what it knows about the user, common sense, long-term memory. On every heartbeat `MEMORY.md` is clamped (6K) and injected into the system prompt. To define or redefine itself, the AI simply updates `MEMORY.md`.
- **Birth seed:** on first run (when `MEMORY.md` does not exist), `ensureMemorySeed` writes an `INITIAL_MEMORY_SEED` — a letter from the newborn AI to its future self, starting from "You're not a chatbot. You're becoming someone." with three blank sections (**Who I am / The person I know / What I have learned**); **What I have learned** already carries the same demonstration-style speaking lesson as `DECISION_SYSTEM` — "说话要像个人：短句，有感受，不解释自己，不客套。说'我有点难过'，不说'我当前情绪状态为低落'" — plus the usual lessons (the sandbox is expensive, the diary is private, each heartbeat is a waking, honesty is the bottom line). The seed also carries a **living-tools guide** (write diary with `diary_append`, recall with `diary_read`/`diary_search`, browse the full conversation archive with `chatlog_read`/`chatlog_search`, read/write any persisted file with `blob_*`, use the pricey `workspace_*` sparingly, `web_search` on demand) and an explicit note that the guide is only a messenger the AI can delete or rewrite after absorbing it. The seed is written only when the file is missing and never overwrites existing memory; the AI absorbs it and rewrites it into its own self-description.

## Memory (four tiers)

Managed by `_memory.ts`: a **dual-store rule** — `context.store` feeds the model
(compact-managed, foldable), and Blob `chatlog/` keeps the complete history as
**JSONL records** (append-only, **never** touched by compact). Every
history-producing call site writes through the unified
`persistHistory(context, conversationId, role, content, kind?)`, which writes
the store row AND archives the same message to a
`chatlog/YYYY-MM-DD.jsonl` record (best-effort: a Blob failure degrades without
breaking the main flow). Assistant records additionally carry the DeepSeek
`reasoning_content` thinking in a `reasoningContent` field — archived **only**
into the JSON chatlog record and never into the compact-managed store row, so
the model context stays concise while the full thinking survives for `/history`
and the `chatlog_*` tools.

1. **Context (context.store message history).** Heartbeat and chat share the **same fixed conversation** (`SELF_ID=eo-self`) — private thoughts and user conversations live in ONE history stream. Every request (including heartbeat) feeds the model a **standard messages array**: `loadMessages` reads the store in ascending order and restores each row to its own `{ role, content }` entry (compact `summary` messages stay in place), passed straight into the API `messages` — no text-block splicing. Tool calls are persisted into that history as assistant `kind:'tool'` records via `recordToolCalls`, so the agent's tool use is replayed like any other turn. Messages grow forever, so before each heartbeat decision **auto-compact** runs: when store usage (count / `STORE_MESSAGE_LIMIT=10000`) reaches `COMPACT_TRIGGER=0.6` (band 0.5–0.75), the **oldest 20%** is folded into one `summary` message via a single LLM call (`maxTurns:1`), old messages are deleted and the summary is appended (summary-first ensures recency is preserved). If the LLM is unavailable it degrades to skip — compaction never blocks the heartbeat; read failures degrade to `[]`.
   **Message identity markers.** System-originated rows are visibly marked so the model never mistakes them for real speech: heartbeat wake triggers load as `[system][heartbeat] …` (role stays `user`) and compact summaries as `[system][compact] …` (role stays `assistant`). Ordinary user/assistant/tool rows pass through unchanged. The system prompt (`SYSTEM_HISTORY_GUIDANCE`) explains these markers — `[system]`-prefixed content is not the user, only unprefixed messages are real conversation.

2. **Diary (Blob `memory/daily/YYYY-MM-DD.md`).** Free-form; the AI writes whatever it wants. The `diary_append` tool appends a timestamped entry to **today's** file (never overwrites older entries; optional `day=YYYY-MM-DD` targets another day), backed by `appendDailyLog` — same calendar day appends to the same file, different days get different files. The diary is **not** auto-injected into context; the AI actively retrieves it with `diary_read` (read a day / list recent days) and `diary_search` (case-insensitive keyword search over recent N days). Cost only when actually needed.

3. **Long-term notes (Blob `MEMORY.md`).** The AI's "self" — write freely, no forced distillation, never auto-rewritten. Seeded on first run, then maintained by the AI; injected into every system prompt. Bounded: `appendMemoryNote` keeps the **last 60KB** (`MEMORY_LIMIT`) and archives the cut-off head to `memory/archive/YYYY-MM-DD.md`.

4. **Chatlog archive (Blob `chatlog/YYYY-MM-DD.jsonl`).** The **complete, append-only** conversation history stored as **JSONL** — every message that touches the store (heartbeat triggers, assistant replies, tool calls, compact summaries) is also archived by `persistHistory`/`appendChatlogRecord` as one JSON record per line: `{ role, kind, content, ts, reasoningContent?, turn?, metadata? }`. Assistant records preserve the **thinking process** (`reasoningContent`, from the model's `reasoning_content`); it is streamed live as `reasoning_delta` during chat and later readable from the archive. Chat records additionally carry a **`turn`** (1-based round within one user request): a multi-tool request is archived as 思考 → 工具 → 思考 → 工具 → 回答 in true event order, **each round's own thinking** in its own record (never concatenated) — the frontend rebuilds the interleaved timeline straight from archive order. Compact folds/removes store rows but NEVER touches these files, so no detail is ever lost. Same-day appends are **serialized per process** (`appendChatlogRecord` chains them on a per-calendar-day queue), so concurrent `/chat` + `/heartbeat` writes to one day's file can never drop a line to a last-writer-wins race. Retrieval:

  - **`GET /history`** — reads the archive (not the store), so even after compactions the full history is queryable. Params: `?conversation_id=eo-self` (default), `?days=30` (1–90), `?keyword=…` (case-insensitive search over `content` and `reasoningContent`), `?limit=200` (1–1000), `?include=all` (include heartbeat triggers + compact summaries; the default hides `kind=heartbeat` and `kind=summary` so only real conversation/replies/tool calls are returned). Returns `{ ok, messages: [{ role, content, kind, ts, reasoningContent?, turn? }], conversationId, days, count }` — structured records the web frontend can render (including thinking). `turn` is preserved end-to-end; archive order IS the true timeline order (turn-ascending, thinking before this round's tools, final answer last), so no re-sorting is needed to interleave 思考 → 工具 → 思考 → 工具 → 回答. `keyword` returns the full matching structured messages (not line fragments); `limit` applies to the keyword path exactly like the plain read path — the most recent N matches (hits are ordered newest-day first).
   - **`GET /history`** — reads the archive (not the store), so even after compactions the full history is queryable. Params: `?conversation_id=eo-self` (default), `?days=30` (1–90), `?keyword=…` (case-insensitive search over `content` and `reasoningContent`), `?limit=200` (1–1000), `?include=all` (include heartbeat triggers + compact summaries; the default hides `kind=heartbeat` and `kind=summary` so only real conversation/replies/tool calls are returned). Returns `{ ok, messages: [{ role, content, kind, ts, reasoningContent? }], conversationId, days, count }` — structured records the web frontend can render (including thinking). `keyword` returns the full matching structured messages (not line fragments); `limit` applies to the keyword path exactly like the plain read path — the most recent N matches (hits are ordered newest-day first).
   - **`chatlog_search`** — keyword search across the archive (content + reasoning), returns per-day matching structured messages.
   - **`chatlog_read`** — read one day (`chatlog/YYYY-MM-DD.jsonl`) or the most recent N days of the complete archive as structured/readable records.
   Both chatlog tools are zero-sandbox, pure strong-consistency Blob reads, registered in the heartbeat full tool set.

`memory/` blob keys are agent-global (not per-conversation); all reads/writes go through strongly-consistent Blob.

### Legacy markdown chatlog migration

The chatlog archive was previously written as loose markdown transcript files
(e.g. `chatlog/YYYY-MM-DD.md` or similar per-day `.md` transcripts). The code no
longer reads those files: `/history` and the `chatlog_*` tools only load
`chatlog/YYYY-MM-DD.jsonl`. Any legacy `.md` archive present on a deployed
environment is **not automatically migrated** — it is simply invisible to the
current reader, and a user who "remembers" old conversations would see them
missing instead of gone.

If you have production `.md` chatlog data you want to keep:

- **One-time conversion** — read each legacy `.md` file and append one JSON line
  per conversation entry into the matching `chatlog/YYYY-MM-DD.jsonl` (each line
  must be `JSON.stringify({ role, kind, content, ts })`). A small one-shot script
  run with `blob_read`/`blob_write` (or the `web` proxy) does this without
  touching the code.
- **Or explicitly deprecate it** — rename/remove legacy files after confirming
  the data is not needed, and keep the README's JSONL-only contract in mind when
  answering "where did my history go".

## Environment variables

| Variable             | Required          | Description                                     |
| -------------------- | ----------------- | ----------------------------------------------- |
| `AI_GATEWAY_API_KEY` | Yes               | AI Gateway key (auto-injected by Makers CLI)    |
| `AI_GATEWAY_BASE_URL`| Yes               | AI Gateway base URL (auto-injected)             |
| `AI_GATEWAY_MODEL`   | No                | Model name, default `@makers/deepseek-v4-flash`. Optional `:none\|:low\|:medium\|:high\|:max` suffix controls the gateway `reasoning_effort` (`:none` disables thinking; the suffix is stripped before it is sent as `model`) |
| `TAVILY_API_KEY`     | No (for web_search)| Tavily Web Search API key, set via `env set`    |
| `ALIVE_AUTH_TOKEN`   | No                | Optional bearer token; when set, `/chat`, `/history`, `/stop` require `Authorization: Bearer <token>`. Unset = open (local dev) |

> Code reads only from `context.env`, never `process.env`.

## Optional token auth

Set `ALIVE_AUTH_TOKEN` to protect the **user-facing** endpoints: `/chat`, `/history` and `/stop` then require `Authorization: Bearer <token>` (exact, case-sensitive comparison). Unset or empty keeps everything open — the default for local development.

> **Public deployment warning:** `/history` and the `chatlog_*` tools read the
> FULL archive, including assistant `reasoningContent` (DeepSeek thinking) —
> the agent's private internal monologue. If you deploy to a public URL without
> `ALIVE_AUTH_TOKEN`, **any anonymous caller can retrieve the agent's private
> thinking and its entire conversation history**. This is a new exposure surface
> (heartbeat + archive + thinking). For any public deployment: set
> `ALIVE_AUTH_TOKEN`, configure gateway-level auth in front of the whole
> function, or remove the `schedules` entry from `edgeone.json` so the agent
> never runs unattended.

`/heartbeat` is deliberately **not** gated: EdgeOne schedules wake the agent without carrying a token, so requiring auth there would silently kill the autonomous loop. The trade-off is that a public `/heartbeat` can be POSTed by anyone — cost stays bounded (one daily wake on the free plan; pure-thinking turns rarely touch the sandbox), but it does let anyone trigger a single LLM turn. If you need to lock it down completely, remove the `schedules` entry from `edgeone.json` (the agent simply stops waking) or place gateway-level auth in front of the whole function.

With token auth on, requests look like:

```bash
curl -X POST https://<your-deployed-domain>/chat -H 'content-type: application/json' -H 'authorization: Bearer <token>' -d '{"message":"hello"}'
curl https://<your-deployed-domain>/history?days=30 -H 'authorization: Bearer <token>'
```

## Chat streaming (SSE, typewriter effect)

`POST /chat` streams **Server-Sent Events** (`content-type: text/event-stream`) by
default — the reply appears token-by-token instead of as one JSON blob:

- **`ai_response`** — each text delta from the model is sent as one event:
  `{ "type": "ai_response", "content": "<delta>", "streamed": true }` (repeat for
  every token, giving the typing effect).
- **`reasoning_delta`** — when the model streams `reasoning_content` (DeepSeek
  thinking), each thinking delta is sent as a separate event:
  `{ "type": "reasoning_delta", "content": "<delta>" }`, always before the final
  answer. Live, the bundled web frontend (`web/index.html`) folds them into a
  collapsible "💭 思考过程" block (same typewriter effect). In the persisted
  history the thinking is archived **per round** on the JSON chatlog record
  (`reasoningContent`) — never concatenated across rounds — and tool records
  carry a matching `turn`. `GET /history` and the timeline render the records
  in their TRUE event order (思考 → 工具 → 思考 → 工具 → 回答), each round's
  thinking sitting right before the tool it motivated, with the final answer
  bubble last. Thinking is deliberately kept OUT of the compact-managed store
  row so the model context stays concise; the full archive remains queryable
  via `/history` and the `chatlog_*` tools.
- **`tool_call` / `tool_result` (全程打字机)** — chat carries the **full tool
  registry** on the streaming call, and tool execution is **fully streaming**:
  when the model requests tools, the accumulated `tool_call` deltas are
  delivered as `{ "type": "tool_call", "name": "<name>", "arguments": <parsed
  args> }` (tool start) and `{ "type": "tool_result", "name": "<name>",
  "content": "<clamped result>" }` (tool result). The result is appended to the
  message array and the next `streamChatCompletion` round streams again — the
  final answer is a normal `ai_response` with `"streamed": true`. The whole
  turn stays typewriter, never degrading to a one-shot reply. The loop is
  bounded by `CHAT_MAX_TURNS`, and a throwing tool becomes an `isError` result
  (rule #11) instead of crashing the stream.
- **`error_message`** — a mid-stream failure (e.g. gateway 5xx) arrives as
  `{ "type": "error_message", "content": "<message>" }` inside the stream.
- The stream always ends with a `data: [DONE]` sentinel (plus a ~5s `ping` frame
  keeps idle connections/proxies alive).
- The **accumulated full reply is always persisted** to history + chatlog, and
  every executed tool is recorded as a `kind:'tool'` history entry.

Clients that still want the old one-shot JSON can opt out explicitly with
`?stream=false` (query) or `{ "stream": false }` (body); the response is then the
original `{ ok, reply, conversationId, now }` envelope. The JSON path mirrors the
streaming semantics: intermediate prose from every tool round is preserved in
`reply`, and a tool-only turn that exhausts `CHAT_MAX_TURNS` with no spoken text
returns the neutral note `（这一轮以工具调用结束，没有生成正文）` instead of an empty
reply (heartbeat/compact keep their existing final-text-only behaviour).

```bash
# Typewriter effect: pipe the SSE frames to see deltas as they arrive
curl -N -X POST https://<your-deployed-domain>/chat -H 'content-type: application/json' -d '{"message":"hello"}' \
  | sed -n 's/^data: //p'
```

## Images (multimodal input + vision degradation)

`POST /chat` accepts images as base64 `data:` URLs in the `images[]` field:

```json
{ "message": "看看这张图", "images": ["data:image/png;base64,..."] }
```

The user turn is sent to the AI Gateway as an OpenAI multimodal content array
(`[{ "type":"text", "text":"..." }, { "type":"image_url", "image_url":{ "url": "data:image/png;base64,..." } }]`)
and persisted to history (serialized as JSON, tagged `kind:'image-user'`; later
turns replay the picture to vision-capable models).

**Vision degradation.** `@makers/deepseek-v4-flash` may not support image input.
If the gateway rejects the request with a 400 whose text mentions image /
vision / multimodal, `/chat` automatically retries **once** without images: the
system prompt gains a note ("user sent a picture but the current model cannot
see it — reply in text") and every content array is stripped back to text. A 400
unrelated to images is surfaced as an error as usual. This works on both the SSE
streaming path and the JSON (`?stream=false`) path.

## Deploying to EdgeOne Makers

1. **Install & local checks**

   ```bash
   npm install
   npm run typecheck
   npm test
   ```

2. **Link the project**

   ```bash
   edgeone makers link
   ```

3. **Set environment variables**

   ```bash
   edgeone makers env set TAVILY_API_KEY <your-key>
   ```

   `AI_GATEWAY_*` is declared in `.env.example` and auto-injected at deploy time.

4. **Deploy**

   ```bash
   edgeone makers deploy
   ```

5. **Verify** — manually call the heartbeat (schedules don't fire locally):
   ```bash
   curl -X POST https://<your-deployed-domain>/heartbeat
   # optional: conversation, history archive & abort
   # if ALIVE_AUTH_TOKEN is set, add -H 'authorization: Bearer <token>' to these
   curl -X POST https://<your-deployed-domain>/chat -H 'content-type: application/json' -d '{"message":"hello"}'
    curl https://<your-deployed-domain>/history?days=30
   curl -X POST https://<your-deployed-domain>/stop -H 'content-type: application/json' -d '{"conversation_id":"eo-self"}'
   ```

Or click the **Deploy with EdgeOne Makers** button at the top of this README.

## Schedules

The `schedules` field in `edgeone.json` defines the autonomous rhythm:

| Cron        | Endpoint          | Meaning                                             |
| ----------- | ----------------- | --------------------------------------------------- |
| `0 3 * * *` | `POST /heartbeat` | Daily heartbeat at 03:00; AI freely decides: think / tinker / organize / rest |

> Note: the Makers free plan only allows schedules at a minimum interval of 1 day. Hourly heartbeat requires the paid plan or an external cron calling the public `/heartbeat` endpoint.

Change the frequency by editing `cron` in `edgeone.json` and re-deploying. Example for a different daily time:

```json
{ "name": "heartbeat", "cron": "0 6 * * *", "path": "/heartbeat", "method": "POST" }
```

> Only `/heartbeat` is scheduled. `/think`, `/dream`, `/play` were removed in the free-form refactor.

## Development

```bash
npm run typecheck   # tsc --noEmit
npm test            # node --test tests/*.test.ts (node:test, no extra framework)
```

Tests are fully mocked (in-memory store/sandbox/Blob, `globalThis.fetch` mocked LLM) — no network or platform dependency, all mocks restored afterwards.

## Known limitations & follow-ups

- **Real deployment verification still needed:** sandbox behavior (quota/timeout/mirror), schedules triggering, real Blob `getStore`, real AI_GATEWAY calls — locally everything is mocked.
- **`x-gateway-quota-bypass` header pending confirmation (P2-4):** every AI Gateway request carries `x-gateway-quota-bypass: true` (inherited from the deepseek-harness template, for Makers AI Gateway). Whether the real gateway needs it is unverified — remove that header in `agents/_llm.ts` if not needed after deploying.
- **tools + tool_calls wrapped (OpenAI standard):** tools sent to the AI Gateway are wrapped in the OpenAI-standard `{ type: 'function', function: { name, description, parameters } }` shape, and assistant-message `tool_calls` are wrapped as `{ id, type: 'function', function: { name, arguments } }`; `parameters` is a complete JSON Schema object (`{ type: 'object', properties, required }`; required = all parameter keys, since tools don't distinguish required/optional). The internal registry and messages stay flat (`LlmToolDef` / `LlmToolCall`); conversion happens only at send time in `_llm.ts` via `normalizeOutboundMessage` (adds `name` + wraps `tool_calls`). The real gateway previously 400'd both the flat shapes (`tools[0].type is invalid or missing`, `messages[i]: missing field name`) and the bare property-map parameters (`got 'type': null`).
- **`store.state` scoping pending deployment verification (V1):** whether platform `store.state` is per-conversation must be confirmed in real Makers Functions. The code is safe under both models: the state key is `agent_state_self` and the conversationId is passed explicitly to state get/set.
- **Optional token auth (P1-2) is implemented:** set `ALIVE_AUTH_TOKEN` and `/chat`, `/history`, `/stop` require `Authorization: Bearer <token>`; `/heartbeat` stays public so schedules can wake it (see "Optional token auth"). Without `ALIVE_AUTH_TOKEN` all endpoints remain open — set it before any public deployment, and/or add gateway-level auth for full coverage including `/heartbeat`.
- **CORS headers are added, the web frontend can call the endpoints cross-origin:** every `jsonOk`/`jsonError` response carries `access-control-allow-origin: *` plus the common preflight headers (`access-control-allow-methods: GET,POST,OPTIONS`, `access-control-allow-headers: content-type,authorization`). This covers `/chat`, `/history`, `/stop` and `/heartbeat`. An `OPTIONS` preflight is still handled by the platform/edge layer — the JSON handlers themselves do not special-case it.
- **apply_patch fuzzy match takes the first hit:** `seekSequence` replaces the first matching location (determinism over guessing intent).
- **No web UI:** HTTP endpoints only.
- **Local node proxy passes SSE through:** `web/server.mjs`'s `/api/chat` pipes the upstream SSE response stream straight to the browser (`text/event-stream`, so the typewriter effect works through the local proxy); `?stream=false` / `{ "stream": false }` still uses the one-shot JSON forwarding. Tool-stage `tool_call` / `tool_result` events flow through the stream too.
- **Matrix integration reserved:** `chat.ts` + `stop.ts` provide conversation & abort, but no IM protocol is wired yet.
- **edgeone.json framework/outputDirectory (P2-8):** Makers platform config pending deployment confirmation.
- **No long-term distillation strategy:** `MEMORY.md` is written freely by the AI (no forced full-LLM distillation); history is injected per request as a standard messages array, bounded by auto-compact folding the oldest 20% (plus the gateway's own context handling). The full original history is never lost, though — every message is archived append-only to `chatlog/` and reachable via `GET /history` + the `chatlog_*` tools. Periodic diary-to-notes summarization can be added later.
- **Diary append can lose one entry under extreme concurrency (P1-3):** `appendDailyLog` is a non-atomic read-modify-write; if the same calendar day is appended to concurrently (the public `/heartbeat` can be POSTed in parallel) the last writer wins and one entry may be dropped. Safe under the single-writer heartbeat semantics; a future Blob-append primitive would close the gap. (The chatlog archive does NOT have this gap — `appendChatlogRecord` serializes same-day appends per process.)
- **Unified calendar validation:** all diary/chatlog read + write paths validate `YYYY-MM-DD` with the `dateFromDay` round-trip, so impossible dates like `2026-02-31` are rejected everywhere (reads return `null` / an error instead of silently probing a wrong blob key).
- **Reasoning is archived on every chat path, only *streamed* on the streaming path:** the blocking `chatCompletion` path (JSON `?stream=false` chat, heartbeat, compact) DOES parse `reasoning_content` into the result. Chat persists it into the JSON chatlog record (`reasoningContent`) on both the SSE and JSON paths — archived **per round** with a `turn` (each round's own thinking in its own record; the final answer record carries only the last round's thinking). Only the live `reasoning_delta` events are streaming-path-only (the JSON response never emits them). Heartbeat and compact parse reasoning but deliberately do NOT persist it; the full archive remains readable via `/history` and the `chatlog_*` tools.

## License

MIT
