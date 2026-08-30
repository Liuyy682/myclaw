# myclaw

一个极简的个人助手 Agent MVP。

单轮对话：

```bash
python -m myclaw "你好"
python -m myclaw --session work "你好"
```

交互式运行：

```bash
python -m myclaw
```

以本地 HTTP Gateway 和 WebUI 运行：

```bash
python -m myclaw gateway
python -m myclaw gateway --host 127.0.0.1 --port 8765
```

打开：

```text
http://127.0.0.1:8765/
```

## 本机 Docker 一键部署

仅在本机使用时，可以用 Docker Compose 构建并启动 Gateway 与内置 WebUI：
需要已安装并运行 Docker Engine 或 Docker Desktop，以及 Docker Compose v2（`docker compose` 插件）。

```bash
./scripts/docker-deploy.sh
```

首次运行会在项目根目录创建 `.env`（不会覆盖已有文件）。编辑其中的
`OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL` 和 `MYCLAW_PORT` 后，重新运行部署脚本即可。
没有 API Key 时会使用内置 Fake Provider，服务仍可启动但不会调用真实模型。

后续启动已构建的容器（不重新构建镜像）：

```bash
./scripts/docker-start.sh
```

默认只将 `127.0.0.1:${MYCLAW_PORT:-8765}` 发布到本机，访问
`http://127.0.0.1:8765/`。运行数据保存在两个 Docker named volume：`/data`
（`MYCLAW_WORKSPACE`，会话、记忆、任务、Cron、安全和观测 SQLite 数据）以及
`/workspace`（Agent 文件和 Shell 工具工作区）。查看状态和日志、停止服务：

```bash
docker compose ps
docker compose logs -f myclaw
docker compose stop
```

删除容器但保留数据：

```bash
docker compose down
```

删除容器并永久删除上述数据卷（谨慎执行）：

```bash
docker compose down -v
```

容器以非 root 用户运行，不挂载 Docker socket，也不使用特权模式。普通 Docker
环境中 bubblewrap 的用户命名空间通常不可用，因此本机 Compose 配置显式设置
`MYCLAW_REQUIRE_EXEC_SANDBOX=false`；`exec` 会在容器隔离范围内以降级模式执行，
并可能访问容器网络。这是本机部署便利性设置，不能当作公网安全边界；如需公网服务，
还必须另行设计认证、TLS、网络隔离和高风险工具策略。

React WebUI 已构建并包含在 Python 包中。开发前端时，请在一个终端运行 Gateway、在另一个终端运行 Vite 开发服务器：

```bash
cd webui
npm install
npm run dev
```

后端测试按领域存放在 `tests/` 的子目录中。完整测试和定向测试可分别运行：

```bash
python -m pytest
python -m pytest tests/agent/test_dispatcher.py
python -m pytest tests/gateway/test_gateway.py
```

`npm run build` 会将生产环境资源输出到 `myclaw/web/dist`；该目录只包含构建产物，请勿手工修改。
HTTP API 与 SSE 事件约定见 [docs/backend-api.md](docs/backend-api.md)。

## 代码目录与导入边界

后端按运行领域组织：`agent/` 负责对话执行，`tools/` 负责内置工具，
`providers/` 负责模型适配，`session/`、`memory/`、`tasks/` 和 `cron/`
负责持久化能力。HTTP Gateway 进一步拆分为：

```text
myclaw/gateway/
├── server.py   # 生命周期、路由、SSE
├── http.py     # HTTP 请求解析和响应
└── static.py   # WebUI 静态资源
```

根包不再聚合导出运行时类型，请从对应领域显式导入：

```python
from myclaw.agent import AgentLoop
from myclaw.gateway.server import run_gateway
from myclaw.providers import FakeProvider
from myclaw.tools import ToolRegistry
```

WebUI 同样按 `app/`、`features/` 和 `shared/` 组织；业务组件应通过
`shared/api/gateway.ts` 和 `shared/types/gateway.ts` 使用 Gateway 合约。

WebUI 通过 `/api/messages` 发送消息，从 `/api/events` 接收流式回复，并通过
`/api/sessions` 列出已保存的对话。助手输出会先以非终止的 `message_delta`
SSE 事件持续推送，再以包含完整回复的终止 `message` 事件结束。选择历史会话后，
系统会载入其中的用户/助手消息；后续轮次会携带该会话的 `session_key`，因此
Gateway 会话和 CLI 会话都可以恢复。若要发送字面量的一次性消息 `gateway`，
请使用 `python -m myclaw -- gateway`。

交互式命令：

- `/resume`：列出带有自动生成标题的 CLI 会话。
- `/resume <name>`：切换到指定会话；若不存在则创建。
- `/new`：创建并切换到新会话。
- `/clear`：清除当前会话历史。
- `/status`：显示当前会话处于空闲、运行中或排队状态。
- `/stop`：取消当前正在执行的轮次，并保留恢复检查点状态。

除非传入 `--session`，否则交互模式会从一个自动生成的新会话开始。

未配置 `OPENAI_API_KEY` 时，CLI 会使用本地假 Provider，因此 MVP 可离线运行。若要使用真实的 OpenAI 兼容端点，请在项目根目录创建 `.env`：

```env
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

已有的 Shell 环境变量优先于 `.env` 中的值。

## 本地运行监控

日志、Trace 和聚合指标默认写入工作区的
`observability/observability.db`，Gateway WebUI 左侧的“运行监控”可查看请求概览、
Span 瀑布图和结构化日志。默认不保存提示词、回复正文、工具参数值或工具结果。

```env
MYCLAW_OBSERVABILITY_ENABLED=1
MYCLAW_LOG_LEVEL=INFO
MYCLAW_LOG_FORMAT=console
MYCLAW_OBSERVABILITY_RETENTION_DAYS=7
MYCLAW_OBSERVABILITY_MAX_MB=100
```

设置 `MYCLAW_LOG_FORMAT=json` 可输出 JSON 控制台日志；关闭观测时将
`MYCLAW_OBSERVABILITY_ENABLED` 设为 `0`。Token 只使用模型端点返回的真实 usage，
缺失时监控页显示为未知。

可选的空闲压缩可通过以下配置启用：

```env
MYCLAW_IDLE_COMPACT_AFTER_MINUTES=15
```

默认值为 `0`，表示禁用自动压缩。启用后，空闲会话在恢复前会被总结，且会话文件中仅保留最近的对话尾部。

## Dream：记忆整合

Dream 是一个在主对话循环之外运行的定时两阶段任务，用于将压缩后的历史流
（`memory/history.jsonl`）整理为长期记忆文件。通过分钟级间隔启用：

```env
MYCLAW_DREAM_INTERVAL_MINUTES=120
```

默认值为 `0`，即禁用 Dream。启用后，调度器会在每次空闲检查时判断间隔是否已到、
是否存在未消费的历史记录；满足条件时会在后台运行一次：

- **阶段 1（分析，无工具）**：读取新增历史和当前记忆文件，生成要新增或清理事实的纯文本清单。
- **校验**：拒绝格式错误的指令、无效来源 ID、同一文件内的增删冲突，以及被分配到多个文件的重复新增项。
- **阶段 2（文件编辑 Agent）**：仅根据已校验的清单做增量修改，文件工具的作用域限制在 `memory/` 目录。

`memory/` 下的三个 Markdown 文件职责互斥：

- `SOUL.md`：仅记录用户明确要求的助手人格、语气和长期行为准则；不得包含用户、项目或任务事实。
- `USER.md`：仅记录跨项目稳定的用户身份、习惯、沟通或工作偏好；不得包含仓库、技术或任务状态。
- `MEMORY.md`：项目知识、仓库、任务、技术决策、运行配置、定时任务和运行状态的唯一归属。

每个持久事实只有一个权威归属。Dream 按
`MEMORY -> USER -> SOUL -> skip` 的顺序路由候选信息；临时状态、一次性请求、
执行输出和不确定事实会被跳过。每次运行还会审计现有文件：只有当等价的
`MEMORY.md` 副本已存在，或可由当前历史批次中的有效来源新增时，才会删除放错文件的
项目/任务副本。这样后续 Dream 运行可以清理旧的误分类，同时不会静默丢弃缺乏依据的事实。

`USER.md` 与 `MEMORY.md` 会一起注入长期记忆块。每条 `MEMORY.md` 事实都带有
指向 `history.jsonl` 来源记录的 `⟨id⟩` 标签；记录拥有稳定 ID，因此即使历史流随后被截断，
这些指针仍然有效。游标 `memory/.dream_cursor` 记录最近消费的位置，并始终在一个批次处理后推进，
因此失败或全部被校验拒绝的周期都不会卡在同一批记录上。

### 版本追踪

每次改变记忆文件的整合都会提交到 `memory/` 目录中的 Git 仓库，提供可审计、可回退的历史。
仓库会在首次提交时初始化（使用本地 `myclaw-dream` 身份，不影响你的全局 Git 配置），且只追踪
记忆文件；`sessions/` 独立保存。提交标题采用
`dream: <time>, N change(s)` 格式，正文保存阶段 1 的清单，因此每个提交都能说明记忆变更原因。

两个交互式命令（仅 CLI）：

- `/dream`：立即运行一次整合，不等待定时器。
- `/dream-log [N]`：查看最近的整合提交，默认显示 10 条。

如需回退，请直接在记忆仓库中使用 Git，例如：
`git -C ~/.myclaw/workspace/memory revert <hash>`。

## Agent Eval

仓库提供一套旁路、规则评分的真实模型评测，不修改 Agent 主执行链路。内置数据集包含
60 个隔离场景，覆盖工具调用、仓库任务、多轮上下文、checkpoint 中断恢复、安全边界
和协作工具；每次运行只会写入临时工作区。

```bash
python -m myclaw.evals \
  --dataset evals/datasets/live_core.jsonl \
  --profile live \
  --repeat 3 \
  --output eval-results/live
```

`live` 使用现有 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 和 `OPENAI_MODEL` Provider 配置，缺少 API Key 时会直接失败而不会降级为
Fake Provider。可通过 `--case 'tool-*'` 或重复传入 `--tag` 过滤场景。输出包括逐次运行的
`runs.jsonl`、聚合 `summary.json` 和 `report.md`；任务成功率与普通 pytest 机制回归通过率
必须分开解释。`evals/manifests/deterministic.yaml` 维护了现有确定性测试与能力声明的映射。

## Skills

助手会在启动时扫描 `${MYCLAW_WORKSPACE}/skills/*/SKILL.md`（默认位于
`~/.myclaw/workspace/skills`）。每个技能使用 YAML frontmatter 描述名称、
用途和可选平台，正文保存仅在任务需要时才加载的操作指引：

```markdown
---
name: code-review
description: Review code structure, call chains, and implementation risks
platforms: [linux, darwin]
---

Inspect the real entrypoints and trace calls before drawing conclusions.
```

`name` 必须是最多 64 个字符的小写 ASCII slug，`description` 必填，
`platforms` 可省略或使用 `linux`、`darwin`、`windows`。单个 `SKILL.md`
最大为 12,000 bytes。格式错误、路径越界、当前平台不兼容或名称重复的技能
不会启用，也不会影响其他技能和助手启动。

启动时只有有效技能的名称和描述会进入模型上下文；当技能与任务匹配时，模型可调用
`skill_load` 读取正文。修改技能元数据或新增技能后需要重启进程，当前版本不支持热重载。
可通过逗号分隔的环境变量禁用指定技能：

```bash
MYCLAW_DISABLED_SKILLS=code-review,legacy-skill
```

Skills 当前只提供指令加载，不安装依赖、不执行技能代码，也不改变内置工具或 MCP 配置。

## MCP 服务器

助手可以挂载外部 [MCP](https://modelcontextprotocol.io) 服务器提供的工具。在工作区中
（与 `sessions/` 目录同级）放置可选的 `mcp.json`，列出需要通过 stdio 启动的服务器：

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]
    }
  }
}
```

启动时，系统会启动每台服务器、发现其工具，并以
`mcp__filesystem__read_file` 这类带命名空间的名称注册，因此不会与内置工具冲突。
进程退出时会正常关闭服务器。缺失或无效的 `mcp.json` 会被忽略。

## 沙箱 Shell 执行

当 [bubblewrap](https://github.com/containers/bubblewrap)（`bwrap`）可用时，
`exec` 工具会在其沙箱中运行 Shell 命令：

- 宿主文件系统以只读方式挂载，只有工作区可写；
- PID、IPC 和 UTS 命名空间相互隔离，父进程退出时子进程也会结束；
- 除非调用传入 `allow_network: true`，否则网络默认禁用。

如果缺少 `bwrap` 或用户命名空间不可用（例如部分容器和 CI 环境），`exec` 会回退为直接运行命令，
并保留命令黑名单（`rm -rf`、`mkfs`、`dd of=`、fork bomb 等）和相对工作区的 `cwd` 检查作为第二道防线。
结果中包含 `sandboxed` 标志，用于说明实际采用的执行路径。

## 持久化任务图

任务会持久化到 `tasks/tasks.json`，可跨会话保留。存储层
（`myclaw/tasks/store.py`，通过 `task_create` / `task_list` /
`task_get` / `task_update` 工具暴露）保证以下三件事：

- **单向状态机**：状态只能沿固定方向流转：

  ```text
  pending ──▶ in_progress ──▶ completed
     │             │
     │             ├──▶ blocked ──▶ in_progress
     ▼             ▼
  cancelled    cancelled
  ```

  `completed` 与 `cancelled` 是终态。非法跳转会被拒绝，例如
  `completed → in_progress`，或从 `pending` 直接跳到 `completed`。
  将状态设置为当前值是幂等空操作。读取旧版任务文件时，历史 `open` / `done`
  值会映射为 `pending` / `completed`。

- **依赖关系**：任务可声明 `depends_on: [<id>, ...]`。引用的 ID 必须存在，依赖图通过
  深度优先检查保证无环；只要任一依赖尚未完成，任务就不能进入 `in_progress` 或 `completed`。

- **多实例安全**：每次修改都会对 `tasks/tasks.json.lock` 获取独占 `flock`，并执行一次新的
  “加载 → 修改 → 原子写入”流程，避免并发进程相互覆盖更新。写入本身采用与会话和 cron 存储相同的
  `tmp + os.replace` 原子写入模式。
