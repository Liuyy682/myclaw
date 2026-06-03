# myclaw

Minimal personal assistant agent MVP.

Run one turn:

```bash
python -m myclaw "hello"
python -m myclaw --session work "hello"
```

Run interactively:

```bash
python -m myclaw
```

Run as a local HTTP gateway with WebUI:

```bash
python -m myclaw gateway
python -m myclaw gateway --host 127.0.0.1 --port 8765
```

Open:

```text
http://127.0.0.1:8765/
```

The WebUI posts messages to `/api/messages`, receives streamed replies from
`/api/events`, and lists saved conversations from `/api/sessions`. Assistant
output streams as non-terminal `message_delta` SSE events followed by one
terminal `message` event containing the complete response. Selecting a history
entry loads its user/assistant messages and sends future turns with that entry's
`session_key`, so gateway sessions and CLI sessions can both be resumed. To send
the literal one-shot message `gateway`, use `python -m myclaw -- gateway`.

Interactive commands:

- `/resume` lists CLI sessions with their generated titles.
- `/resume <name>` switches to the named session, creating it if needed.
- `/new` creates and switches to a new session.
- `/clear` clears the current session history.
- `/status` shows whether the current session is idle, running, or queued.
- `/stop` cancels the current in-flight turn and keeps recovery checkpoint state.

Interactive mode starts in a new generated session unless `--session` is provided.

Without `OPENAI_API_KEY`, the CLI uses a local fake provider so the MVP runs offline.
For a real OpenAI-compatible endpoint, create `.env` in the project root:

```env
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

Existing shell environment variables take priority over `.env` values.

Optional idle compaction can be enabled with:

```env
MYCLAW_IDLE_COMPACT_AFTER_MINUTES=15
```

The default is `0`, which disables auto-compact. When enabled, idle sessions are
summarized before they are resumed and only the recent conversation tail is kept
in the session file.

## MCP servers

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

## Sandboxed shell execution

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
