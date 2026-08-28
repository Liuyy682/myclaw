from __future__ import annotations

import json
import logging
import shutil
from contextlib import AsyncExitStack
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

from myclaw.mcp.tool import McpServerConfig, McpTool

if TYPE_CHECKING:
    from myclaw.tools import ToolRegistry

logger = logging.getLogger(__name__)

MCP_CONFIG_FILENAME = "mcp.json"


def resolve_mcp_command(command: str) -> str:
    """Resolve a stdio command to the executable path used for approval.

    Bare commands that cannot currently be found are retained verbatim so a
    missing executable remains an ordinary connection error rather than being
    silently rewritten relative to the current working directory.
    """

    value = str(command)
    resolved = shutil.which(value)
    if resolved:
        return str(Path(resolved).expanduser().resolve())

    candidate = Path(value).expanduser()
    if candidate.is_absolute() or candidate.parent != Path(".") or value.startswith((".", "~")):
        return str(candidate.resolve())
    return value


def normalize_mcp_config(config: McpServerConfig) -> dict[str, Any]:
    """Return the canonical, secret-bearing config used only for hashing.

    The resulting mapping must not be persisted or printed: its env values
    are intentionally included so changing a credential invalidates approval.
    """

    return {
        "name": str(config.name),
        "command": resolve_mcp_command(config.command),
        "args": [str(argument) for argument in config.args],
        "env": {
            str(key): str(value)
            for key, value in sorted(config.env.items(), key=lambda item: str(item[0]))
        },
    }


def mcp_config_hash(config: McpServerConfig) -> str:
    """Hash the complete canonical MCP server configuration."""

    payload = json.dumps(
        normalize_mcp_config(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


class McpManager:
    """Connect to stdio MCP servers and expose their tools to a ToolRegistry."""

    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self._tools: list[McpTool] = []
        self._connected = False

    @property
    def tools(self) -> list[McpTool]:
        return list(self._tools)

    async def connect(self, configs: list[McpServerConfig]) -> None:
        if self._connected:
            return
        self._connected = True
        for config in configs:
            try:
                await self._connect_server(config)
            except Exception:
                logger.exception("Failed to connect MCP server %s", config.name)

    async def _connect_server(self, config: McpServerConfig) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=config.resolved_command,
            args=list(config.args),
            env=dict(config.env) or None,
        )
        read, write = await self._stack.enter_async_context(stdio_client(params))
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        listed = await session.list_tools()
        for remote in listed.tools:
            self._tools.append(
                McpTool(
                    session=session,
                    server_name=config.name,
                    remote_name=remote.name,
                    description=remote.description or "",
                    input_schema=remote.inputSchema,
                    config_hash=config.config_hash,
                )
            )

    def register_into(self, registry: ToolRegistry) -> None:
        for tool in self._tools:
            registry.register(tool)

    async def aclose(self) -> None:
        await self._stack.aclose()
        self._tools.clear()
        self._connected = False


def load_mcp_configs(workspace: Path | str | None) -> list[McpServerConfig]:
    """Read optional <workspace>/mcp.json. Missing or invalid file -> no servers."""
    if workspace is None:
        return []
    path = Path(workspace).expanduser() / MCP_CONFIG_FILENAME
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read MCP config at %s", path)
        return []
    servers = raw.get("mcpServers") if isinstance(raw, dict) else None
    if not isinstance(servers, dict):
        return []
    configs: list[McpServerConfig] = []
    for name, spec in servers.items():
        if not isinstance(spec, dict):
            continue
        command = spec.get("command")
        if not isinstance(command, str) or not command:
            continue
        args = spec.get("args")
        env = spec.get("env")
        configs.append(
            McpServerConfig(
                name=str(name),
                command=command,
                args=[str(arg) for arg in args] if isinstance(args, list) else [],
                env={str(k): str(v) for k, v in env.items()} if isinstance(env, dict) else {},
            )
        )
    return configs
