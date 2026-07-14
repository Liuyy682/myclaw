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

React WebUI 已构建并包含在 Python 包中。开发前端时，请在一个终端运行 Gateway、在另一个终端运行 Vite 开发服务器：

```bash
cd webui
npm install
npm run dev
```

`npm run build` 会将生产环境资源输出到 `myclaw/web/dist`。HTTP API 与 SSE 事件约定见 [docs/backend-api.md](docs/backend-api.md)。

The WebUI posts messages to `/api/messages`, receives streamed replies from
`/api/events`, and lists saved conversations from `/api/sessions`. Assistant
output streams as non-terminal `message_delta` SSE events followed by one
terminal `message` event containing the complete response. Selecting a history
entry loads its user/assistant messages and sends future turns with that entry's
`session_key`, so gateway sessions and CLI sessions can both be resumed. To send
the literal one-shot message `gateway`, use `python -m myclaw -- gateway`.

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

可选的空闲压缩可通过以下配置启用：

```env
MYCLAW_IDLE_COMPACT_AFTER_MINUTES=15
```

默认值为 `0`，表示禁用自动压缩。启用后，空闲会话在恢复前会被总结，且会话文件中仅保留最近的对话尾部。

## Dream：记忆整合

Dream is a periodic, two-phase job that runs off the main conversation loop and
distills the compacted history stream (`memory/history.jsonl`) into long-term
memory files. Enable it with an interval in minutes:

```env
MYCLAW_DREAM_INTERVAL_MINUTES=120
```

The default is `0`, which disables Dream. When enabled, the dispatcher checks on
each idle tick whether the interval has elapsed and there are unconsumed history
entries; if so it runs one cycle in the background:

- **Phase 1 (analysis, no tools)** reads the new history entries plus the current
  memory files and emits a plain-text checklist of facts to add or prune.
- **Phase 2 (file-editing agent)** applies that checklist with incremental edits,
  using file tools scoped to the `memory/` directory.

Three markdown files under `memory/` hold the distilled memory:

- `SOUL.md` — the assistant's persona and tone (injected ahead of the base prompt).
- `USER.md` — user identity and preferences.
- `MEMORY.md` — project knowledge and facts.

`USER.md` and `MEMORY.md` are injected together as the long-term memory block.
Each `MEMORY.md` fact carries a `⟨id⟩` tag pointing back to its source entry in
`history.jsonl`; entries carry a stable id so these pointers survive even if the
stream is later truncated. A cursor at `memory/.dream_cursor` tracks the last
consumed entry and always advances past a processed batch, so a failed cycle
never wedges the system on the same entries.

### 版本追踪

Each consolidation that changes the memory files is committed to a git
repository in the `memory/` directory, giving the memory an auditable,
revertible history. The repo is initialised on the first commit (with a local
`myclaw-dream` identity, leaving your global git config untouched) and tracks
only the memory files — `sessions/` is separate. Commit messages use a
`dream: <time>, N change(s)` subject with the Phase 1 checklist as the body, so
each commit records why the memory changed.

Two interactive commands (CLI only):

- `/dream` — run a consolidation cycle now instead of waiting for the timer.
- `/dream-log [N]` — show the most recent consolidation commits (default 10).

To revert, use git directly against the memory repo, for example
`git -C ~/.myclaw/workspace/memory revert <hash>`.

## MCP 服务器

The assistant can attach tools from external [MCP](https://modelcontextprotocol.io)
servers. Drop an optional `mcp.json` in the workspace (next to the `sessions/`
directory) listing stdio servers:

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

On startup each server is launched, its tools are discovered, and they are
registered under namespaced names like `mcp__filesystem__read_file` so they
never collide with built-in tools. Servers are shut down cleanly when the
process exits. A missing or invalid `mcp.json` is ignored.

## 沙箱 Shell 执行

The `exec` tool runs shell commands inside a [bubblewrap](https://github.com/containers/bubblewrap)
(`bwrap`) sandbox when it is available:

- the host filesystem is mounted read-only, and only the workspace is writable;
- PID, IPC, and UTS namespaces are isolated, and the child dies with the parent;
- network is disabled unless the call passes `allow_network: true`.

If `bwrap` is missing or user namespaces are unavailable (some containers and CI),
`exec` falls back to running the command directly, keeping a command blacklist
(`rm -rf`, `mkfs`, `dd of=`, fork bombs, ...) and workspace-relative `cwd`
checks as a second line of defense. The result includes a `sandboxed` flag
indicating which path was used.

## 持久化任务图

Tasks persist to `tasks/tasks.json` and survive across sessions. The store
(`myclaw/tasks/store.py`, exposed through the `task_create` / `task_list` /
`task_get` / `task_update` tools) enforces three things:

- **One-way state machine.** Status flows in a single direction:

  ```text
  pending ──▶ in_progress ──▶ completed
     │             │
     │             ├──▶ blocked ──▶ in_progress
     ▼             ▼
  cancelled    cancelled
  ```

  `completed` and `cancelled` are terminal. Illegal jumps (for example
  `completed → in_progress`, or skipping straight from `pending` to
  `completed`) are rejected. Setting a status to itself is an idempotent no-op.
  Legacy `open` / `done` values from older task files are mapped to
  `pending` / `completed` on read.

- **Dependency links.** A task can declare `depends_on: [<id>, ...]`. Referenced
  ids must exist, the dependency graph is kept acyclic (cycles are rejected via a
  depth-first check), and a task cannot move to `in_progress` or `completed`
  while any dependency is still unfinished.

- **Multi-instance safety.** Every mutation takes an exclusive `flock` over
  `tasks/tasks.json.lock` and runs a fresh load → mutate → atomic-write cycle, so
  concurrent processes cannot lose each other's updates. The write itself uses the
  same `tmp + os.replace` atomic pattern as the session and cron stores.
