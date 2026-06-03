from __future__ import annotations

import json
import logging
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any

from myclaw.mcp.tool import McpServerConfig, McpTool

if TYPE_CHECKING:
    from myclaw.tools import ToolRegistry

logger = logging.getLogger(__name__)

MCP_CONFIG_FILENAME = "mcp.json"


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
            command=config.command,
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
