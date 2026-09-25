# alive

[English](README.md) | **简体中文**

[![使用 EdgeOne Makers 部署](https://cdnstatic.tencentcs.com/edgeone/pages/deploy.svg)](https://edgeone.ai/pages/new?repository-url=https%3A%2F%2Fgithub.com%2Ffeat-cat%2Falive&install-command=npm%20install&build-command=npm%20run%20build%3Amakers&output-directory=dist)

> alive —— 一个活在 EdgeOne Makers 上的自主 agent。它每次心跳自由决定要做什么：思考、写日记、推进小项目，或单纯休息。性格在 `MEMORY.md` 里慢慢长成。

`alive` 是一个运行在 EdgeOne Makers 免费版上的自主 agent 内核，由**单个 heartbeat 定时触发器**驱动：每次 heartbeat，AI **自由发挥**——结合当前时刻、自己的记忆（`MEMORY.md`）和完整工具集，自己决定此刻要做什么，而不是从固定菜单里选：安静思考写日记、动手推进小项目、整理记忆，或直接休息。它没有 Web UI、没有常驻进程，全部由 EdgeOne Makers Functions 在请求到来时按需运行，靠强一致 Blob 和 conversation store 在回合之间"记得"自己。

## 项目是什么

一个最小的"有自我感"的 agent 骨架：

- **heartbeat（自由发挥）**——唯一的主入口。给 AI 当前状态 + 最近日志/记忆 + 完整工具集（blob、workspace、search），让它像人一样自己决定此刻做什么。**不是四选一的选择题，也不是每次都必须有产出**。休息是完全合法的选择。
- **纯思考 / 休息**——AI 只输出文字（内心日志、感受），零 sandbox 成本。这是最常见的路径。
- **动手玩项目**——只有当 AI 主动调用 `workspace_*` 工具时才触沙箱，改动自动镜像回 Blob。
- **整理记忆**——AI 用 `blob_*` 工具自由读写日记（`memory/daily/YYYY-MM-DD.md`）与长期笔记（`MEMORY.md`），想写就写，不做强制蒸馏。
- **chat / stop / history**——对话、中止与历史归档端点（为未来 Matrix 接入准备）。chat 与 heartbeat 共用 `SELF_ID=eo-self` 同一份历史——AI 私下的思考与和用户的对话是一体的；`/history` 从 Blob 读取完整、永不 compact 的 chatlog 归档。

它刻意做得很"瘦"：每个回合只是有限次 LLM 调用 + 少量状态读写，因此在免费额度下可持续运行。

## 架构图（文字版）

```
EdgeOne Makers schedules (cron)
        │  每天 03:00 → POST /heartbeat
        ▼
  ┌─────────────────────────────────────────┐
  │  heartbeat.ts  读状态/日志/记忆 → LLM     │
  │  自由发挥（开放题，不是四选一）            │
  │  工具集：blob_* + diary_* + chatlog_*  │
  │         + workspace_* + web_search     │
  │  预算：≤3 轮 / 100s 短回合                │
  └───────────────┬─────────────────────────┘
                  │ AI 自己决定
      ┌───────────┼──────────────┬───────────────┐
      ▼           ▼              ▼               ▼
  纯思考/日记   workspace_* 工具  blob_* 工具    休息
  (零 sandbox)  (才触发 sandbox) (记忆整理)    (只留一句话)
      │           │              │               │
      │           ▼              ▼               │
      │   snapshotWorkspaceToBlob（快照回镜像）     │
      ▼           ▼              ▼               ▼
   context.store            Blob 强一致持久化（@edgeone/pages-blob）
   ├── messages（标准 {role,content} 数组 + kind:'tool' 记录，auto-compact 50-75%）
   ├── store.state（lastActivityAt/created）
                            └── memory/…（日记 daily/、MEMORY.md、archive/）
                            └── chatlog/…（完整历史，append-only，永不 compact）
                            └── projects/<conv>/…（workspace 镜像、项目文件）
      │
      ▼
   Sandbox —— 仅当 AI 主动调用 workspace_* 时按需启动（配额保护）
   workspace 写入即镜像 Blob，跑完快照释放
```

## 目录结构

```
alive/
├── agents/                 # 全部运行时代码（Makers Functions 按端点加载）
│   ├── _shared.ts          # 类型、常量、SSE、JSON 响应、错误映射、固定会话 id
│   ├── _llm.ts             # AI Gateway chat/completions + 有界工具循环
│   ├── _state.ts           # agent_state 读取/更新（仅 lastActivityAt/created）
│   ├── _persona.ts         # 系统提示词构建（动态日期 + MEMORY.md "我的记忆"）
│   ├── _memory.ts          # 四层记忆：store 上下文(auto-compact)、Blob 日记、MEMORY.md、chatlog 归档
│   ├── _blob-tools.ts      # Blob 强一致持久化 + blob_* 工具
│   ├── _workspace-tools.ts # 沙箱 workspace 工具（写入即镜像 Blob）
│   ├── _apply-patch.ts     # "Begin Patch" 信封解析与应用（纯逻辑）
│   ├── _tavily.ts          # web_search（Tavily）执行器
│   ├── _tools.ts           # heartbeat 完整工具注册表（blob + diary + chatlog + workspace + search）
│   ├── heartbeat.ts        # POST /heartbeat（唯一主入口，AI 自由发挥）
│   ├── chat.ts             # POST /chat（预留）
│   ├── history.ts          # GET /history（完整 chatlog 归档读取）
│   └── stop.ts             # POST /stop（预留）
├── tests/                  # node:test 单元/端点测试
├── edgeone.json            # Makers 配置（timeout、sandbox、schedules）
├── .env.example            # 环境变量声明
└── package.json
```

对外可路由的端点只有 `heartbeat.ts` / `chat.ts` / `history.ts` / `stop.ts`；`_` 前缀文件均为内部模块。

## 核心设计决策与铁律

1. **纯思考/休息绝不碰 sandbox**。沙箱配额有限（10 万 GB-s/月）。heartbeat 给 AI 完整工具集，但只有它**主动调用 `workspace_*` 工具**时才申请沙箱；纯文本路径只读写 store 与 Blob。

2. **Blob 强一致持久化，沙箱 /tmp 不持久** → workspace 写入即镜像 Blob。回合结束时若动用过 workspace，`snapshotWorkspaceToBlob` 把沙箱内容（含命令生成的产物）快照回 Blob，即使沙箱被回收，项目文件也不丢失。

3. **单回合短时限、按需启动**。每次 heartbeat 的 LLM 循环受 `HEARTBEAT_MAX_TURNS=3` 和 100s 回合预算（`PLAY_TURN_TIMEOUT_MS`）约束；跑完立即快照并释放沙箱。不做长驻进程。

4. **不复制 deepseek-harness 的常驻 sidecar 模式**——那会瞬间烧光免费配额。alive 的全部"生活"由 heartbeat schedules 触发，无请求即无成本。

5. **自由发挥而非选择题**。不再让 AI 在 think/dream/play/rest 里四选一，而是把状态、记忆和完整工具交给它，自己决定做什么——**不是每次 heartbeat 都必须有产出**，休息是一等公民。有限的 sandbox 配额完全交给 AI 按需分配。

6. **统一会话**。heartbeat 与 chat 共用 `SELF_ID=eo-self` 同一份历史——AI 私下的思考与和用户的对话是一体的，跨请求、跨重启状态与记忆保持一致。

7. **模型循环永远有上限**。所有 `chatCompletion` 都传 `maxTurns`（heartbeat=3，chat=3，compact=1），配合 90s LLM 超时。

8. **永不崩溃（规则 #11）**。所有端点用 `errorResponse` 把异常映射成稳定的 JSON 错误；工具执行器捕获错误并返回 `{ isError: true }` 而非抛出。

## 人格与状态（性格在对话中形成）

- **没有假状态字段**。内核不维护伪造的情绪/精力/项目字段。每次 heartbeat 都是一次独立的 LLM 调用——所谓"情绪"不过是上次调用自己写的标签再喂回来，是表演不是感受，这类字段已全部删除。`store.state` 只保留**最少且真实**的 `lastActivityAt` 和 `created`。
- **系统提示词只注入动态且必要的内容**：当前时刻（`现在是 YYYY-MM-DD 星期X HH:mm`），加上固定的 `DECISION_SYSTEM`——它告诉 AI：**你不是聊天机器人，你在成为某个人**。每一次 heartbeat 都是一次**醒来**，不是被闹钟叫去完成任务；沙箱很贵、日记是自己的私处、`MEMORY.md` 是自己的、诚实是底线，这些"只属于它的现实"以简短形式写在这里。引导文案开头（`SYSTEM_HISTORY_GUIDANCE`）专门说明 `[system][heartbeat]` / `[system][compact]` 消息标记的含义，避免把系统自动触发的内容误当成用户发言（chat 复用同一段说明）。
- **人格主体来自 `MEMORY.md`**，由 AI 自己维护——身份、性格、关于用户的了解、常识、长期记忆。每次 heartbeat 都会把 `MEMORY.md`（clamp 6K）强制注入系统提示词。想定义或修改自己，AI 直接更新 `MEMORY.md` 即可。
- **出生引导**：首次运行时（`MEMORY.md` 不存在），`ensureMemorySeed` 写入一份 `INITIAL_MEMORY_SEED`——新生的 AI 写给未来自己的一封信，从 "You're not a chatbot. You're becoming someone." 开始，留出三个空白小节（**我是谁 / 我认识的人 / 我学会的事**）等它自己长出来；另带一节**生活工具引导**（`diary_append` 写日记、`diary_read`/`diary_search` 回顾、`chatlog_read`/`chatlog_search` 翻完整聊天记录、`blob_*` 读写任意持久化文件、`workspace_*` 很贵少用、`web_search` 按需搜索）和一段**可删除说明**——引导只是信使，读完后可以删掉或重写成自己的记忆。种子只在文件缺失时写入，**绝不覆盖**既有记忆；AI 吸收后会把它改写成自己的自我描述。

## 记忆层（四层）

由 `_memory.ts` 统一维护，核心是**双层存储约定**：`context.store` 负责喂模型（可被 compact 折叠），Blob `chatlog/` 负责保存完整历史（append-only，**永不**被 compact 触碰）。所有产生历史的调用点都走统一入口 `persistHistory(context, conversationId, role, content, kind?)`——既写 store 也归档同一消息（best-effort：Blob 失败降级，不影响主流程）。

1. **上下文（context.store 消息历史）**：heartbeat 与 chat 共用**同一固定会话**（`SELF_ID=eo-self`）——私下的思考与和用户的对话在**同一份历史**里，是一体的。每次请求（包括 heartbeat）都以**标准 messages 数组**喂给模型：`loadMessages` 按 asc 读取整个 store，把每条消息还原成独立的 `{ role, content }` 条目（compact 的 `summary` 摘要按原样作为消息保留），直接作为 API `messages` 传入——不再拼接 clamp 文本块。工具调用通过 `recordToolCalls` 以 assistant `kind:'tool'` 记录落盘进历史，之后的回合像普通轮次一样重放。消息会无限增长，因此每次 heartbeat 决策前会先跑 **auto-compact**：当 store 占用（条数 / `STORE_MESSAGE_LIMIT=10000`）达到 `COMPACT_TRIGGER=0.6`（区间 0.5~0.75）时，只把**最旧的 20%** 用一次 LLM 调用（`maxTurns:1`）折叠成一条 `summary` 消息，删除旧消息并把摘要追加进消息流（摘要在前、保最新）。LLM 不可用时**降级跳过**，不阻塞 heartbeat；读取失败降级返回 `[]`。
   **消息身份标记。** 系统自动产生的历史会被明确标记，避免模型误认为是真实对话：heartbeat 唤醒触发的消息加载为 `[system][heartbeat] …`（角色仍是 `user`），compact 摘要为 `[system][compact] …`（角色仍是 `assistant`），普通 user/assistant/tool 消息原样不动。系统提示词（`SYSTEM_HISTORY_GUIDANCE`）会说明这些标记——带 `[system]` 前缀的内容不是用户说的，只有不带前缀的消息才是真实对话。

2. **日记（Blob `memory/daily/YYYY-MM-DD.md`）**：AI 自由写，想写什么写什么。写用 `diary_append` 工具（自动追加**今天** `memory/daily/YYYY-MM-DD.md`、带时间戳、**绝不覆盖**旧条目；可选 `day=YYYY-MM-DD` 指定其他日期），底层是 `appendDailyLog`——同一天追加同一文件、不同日期写不同文件。**日记不自动注入上下文**：AI 需要回顾时用 `diary_read`（读某天 / 省略参数列最近日记）和 `diary_search`（最近 N 天日记里大小写不敏感的关键词检索）工具主动检索——成本只在真正需要时发生。

3. **长期笔记（Blob `MEMORY.md`）**：AI 的"自我"，想写就写，不做强制蒸馏，从不自动改写。首次由 `ensureMemorySeed` 写入出生引导，之后 AI 自己维护；每次系统提示词强制注入。有界：`appendMemoryNote` 超过 `MEMORY_LIMIT=60KB` 时**保留末尾 60KB**（最新笔记不丢），被截掉的旧内容追加到 `memory/archive/YYYY-MM-DD.md`。

4. **聊天记录归档（Blob `chatlog/YYYY-MM-DD.md`）**：**完整、追加式**的对话历史——凡是写过 store 的消息（heartbeat 触发、AI 回复、工具调用、compact 摘要）都会由 `persistHistory` / `appendChatlog` 一并归档成一行带时间戳的记录。compact 只会折叠/删除 store 里的消息，**绝不触碰**这些文件，所以任何细节都不会丢失。读取方式：
   - **`GET /history`**——读归档（不是 store），因此即使被 compact 折叠过，完整历史依然可查。参数：`?conversation_id=eo-self`（默认）、`?days=30`（1–90）、`?keyword=…`（大小写不敏感搜索）、`?limit=200`（1–1000）、`?include=all`（包含 heartbeat 触发 + compact 摘要；默认会隐藏 `kind=heartbeat` 与 `kind=summary`，只返回真实对话/回复/工具调用）。返回 `{ ok, messages: [{ role, content, kind, ts }], conversationId, days, count }`。**`keyword` 路径返回的是归档中的命中行片段，而非完整消息**（无法解析成整行的片段标记为 `kind:'search'`、`ts` 为空）；`limit` 对 keyword 路径同样生效。
   - **`chatlog_search`**——跨归档关键词搜索，返回按天分组的命中片段。
   - **`chatlog_read`**——读某一天（`chatlog/YYYY-MM-DD.md`）或最近 N 天的完整归档原文。
   两个 chatlog 工具都是零 sandbox、纯强一致 Blob 读取，并注册进 heartbeat 的完整工具集。

`memory/` 前缀的 blob key 是 agent 全局的（不按会话加前缀），所有读写走强一致 Blob。

## 环境变量

| 变量                  | 必填                  | 说明                                         |
| --------------------- | --------------------- | -------------------------------------------- |
| `AI_GATEWAY_API_KEY`  | 是                    | AI Gateway 密钥（Makers CLI 部署时自动注入） |
| `AI_GATEWAY_BASE_URL` | 是                    | AI Gateway base URL（自动注入）              |
| `AI_GATEWAY_MODEL`    | 否                    | 模型名，默认 `@makers/deepseek-v4-flash`     |
| `TAVILY_API_KEY`      | 否（web_search 需要） | Tavily Web Search API 密钥，需手动 `env set` |
| `ALIVE_AUTH_TOKEN`    | 否                    | 可选 Bearer Token；设置后 `/chat`、`/history`、`/stop` 需要 `Authorization: Bearer <token>`。不设置则全部开放（本地开发默认） |

> 注意：代码只从 `context.env` 读取，绝不读 `process.env`。

## 可选 Token 鉴权

设置 `ALIVE_AUTH_TOKEN` 可保护**面向用户**的端点：设置后 `/chat`、`/history`、`/stop` 需要 `Authorization: Bearer <token>`（精确、大小写敏感比较）。不设置或为空则全部开放——这是本地开发的默认状态。

`/heartbeat` **刻意不做鉴权**：EdgeOne schedules 触发唤醒时不会携带 token，若在这里要求鉴权会静默掐断自主循环。代价是公开的 `/heartbeat` 可以被任何人 POST——成本仍然有界（免费版下每天一次唤醒；纯思考回合几乎不碰沙箱），但确实允许任何人触发一次 LLM 调用。若要彻底关闭它，请删除 `edgeone.json` 中的 `schedules` 条目（agent 将不再被唤醒），或在函数前再加网关级认证。

开启 Token 鉴权后，请求形如：

```bash
curl -X POST https://<你的部署域名>/chat -H 'content-type: application/json' -H 'authorization: Bearer <token>' -d '{"message":"你好"}'
curl https://<你的部署域名>/history?days=30 -H 'authorization: Bearer <token>'
```

## 部署步骤（EdgeOne Makers）

1. **安装与本地检查**

   ```bash
   npm install
   npm run typecheck
   npm test
   ```

2. **关联项目**

   ```bash
   edgeone makers link
   ```

3. **设置环境变量**

   ```bash
   edgeone makers env set TAVILY_API_KEY <你的key>
   ```

   `AI_GATEWAY_*` 已在 `.env.example` 中声明，部署时由 CLI 自动注入，无需手动设置。

4. **部署**

   ```bash
   edgeone makers deploy
   ```

5. **验证**：手动调用 heartbeat（schedules 不会在本地/测试环境真实触发）

   ```bash
   curl -X POST https://<你的部署域名>/heartbeat
   # 可选：对话、历史归档与中止
   # 若设置了 ALIVE_AUTH_TOKEN，需给这些请求加上 -H 'authorization: Bearer <token>'
   curl -X POST https://<你的部署域名>/chat -H 'content-type: application/json' -d '{"message":"你好"}'
   curl https://<你的部署域名>/history?days=30
   curl -X POST https://<你的部署域名>/stop -H 'content-type: application/json' -d '{"conversation_id":"eo-self"}'
   ```

也可以直接点击本 README 顶部的 **使用 EdgeOne Makers 部署** 按钮。

## Schedules

`edgeone.json` 的 `schedules` 字段定义自主节奏：

| Cron        | 端点              | 含义                                                     |
| ----------- | ----------------- | -------------------------------------------------------- |
| `0 3 * * *` | `POST /heartbeat` | 每天凌晨 3 点 heartbeat，AI 自由发挥：思考/玩项目/整理记忆/休息 |

> 注意：Makers 免费版 schedules 最小间隔为 1 天（86400s），每小时 heartbeat 需要付费版，或由外部 cron（如 GitHub Actions）定时调用公开的 `/heartbeat` 端点。

修改频率：编辑 `edgeone.json` 中的 `cron` 后重新 `edgeone makers deploy`。例如改为每天早晨 6 点：

```json
{ "name": "heartbeat", "cron": "0 6 * * *", "path": "/heartbeat", "method": "POST" }
```

> schedules 只有 `/heartbeat` 一个；`/think`、`/dream`、`/play` 端点已随自由发挥改造移除。

## 开发

```bash
npm run typecheck   # tsc --noEmit
npm test            # node --test tests/*.test.ts（node:test，无需额外框架）
```

测试全部使用 mock（内存 store / sandbox / Blob，`globalThis.fetch` mock LLM），不依赖网络或真实平台，运行完毕后恢复所有 mock。

## 已知限制与后续

- **需真实部署验证**：sandbox 真实行为（配额/超时/镜像）、schedules 触发、Blob 真实 `getStore`、AI_GATEWAY 真实调用——本地测试均为 mock。
- **`x-gateway-quota-bypass` 请求头待部署确认（P2-4）**：每次 AI Gateway 请求都带 `x-gateway-quota-bypass: true`（自 deepseek-harness 模板继承，供 Makers AI Gateway 识别）。真实网关是否需要此头尚未验证——部署时若不需要应删除 `agents/_llm.ts` 中的该 header 行。
- **tools + parameters 已按 OpenAI JSON Schema 包装**：发给 AI Gateway 的 tools 已包装为 `{ type: 'function', function: { name, description, parameters } }`，且 `parameters` 为完整 JSON Schema 对象（`{ type: 'object', properties, required }`；工具定义不区分必填/可选，因此 required 为全部参数键）。内部注册表保持扁平 `LlmToolDef`，仅发送时在 `_llm.ts` 转换；真实网关此前会因扁平结构返回 400（`tools[0].type is invalid or missing`），也会因裸属性对象参数返回 400（`got 'type': null`）。
- **`store.state` 作用域待部署验证（V1）**：平台 `store.state` 是否按会话隔离需在真实 Makers Functions 中确认。代码按"两种模型都安全"实现：状态 key 为 `agent_state_self`，把 conversationId 显式传给 state get/set。
- **可选 Token 鉴权（P1-2）已实现**：设置 `ALIVE_AUTH_TOKEN` 后 `/chat`、`/history`、`/stop` 需要 `Authorization: Bearer <token>`；`/heartbeat` 保持公开以便 schedules 唤醒它（见"可选 Token 鉴权"）。不设置则全部端点仍开放——公开部署前建议设置它，并/或在网关层加认证以覆盖 `/heartbeat` 在内。
- **apply_patch 模糊匹配取首个命中**：`seekSequence` 在多个可替换位置时替换第一个匹配（确定性优先于"猜测意图"）。
- **未做 Web UI**：当前只有 HTTP 端点，没有管理界面。
- **Matrix 接入预留**：`chat.ts` + `stop.ts` 已具备对话与中止能力，但尚未接入任何即时通讯协议。
- **edgeone.json framework/outputDirectory（P2-8）**：Makers 平台配置待部署确认，暂不改动。
- **无长期蒸馏策略**：`MEMORY.md` 由 AI 用 `blob_*` 自由读写（想写就写），不做全量 LLM 蒸馏；历史每次请求以标准 messages 数组注入，靠 auto-compact 折叠最旧 20% 控制长度（其余由网关上下文处理）。完整原始历史永不会丢——每条消息都以 append-only 方式归档进 `chatlog/`，可通过 `GET /history` 与 `chatlog_*` 工具查看。后续可升级为定期归纳日记为长期笔记。
- **日记追加在极端并发下可能丢一条（P1-3）**：`appendDailyLog` 是非原子的读-改-写；同一日历日被并发追加时（公开 `/heartbeat` 可被并发 POST）最后写入者胜，可能丢掉一条记录。单写者（每小时一次 heartbeat）语义下安全；后续可引入 Blob append 原语修复。
- **统一日期校验口径**：所有日记 / chatlog 读写路径都用 `dateFromDay` 的 round-trip 校验 `YYYY-MM-DD`，`2026-02-31` 这类不存在的日期在所有入口都会被拒绝（读路径返回 `null` / 错误，而不是去探测一个错误的 blob key）。

## 许可证

MIT
