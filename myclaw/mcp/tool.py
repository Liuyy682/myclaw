from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class McpServerConfig:
    """Describe a stdio MCP server to launch and connect to."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


def mcp_tool_name(server: str, tool: str) -> str:
    """Namespaced tool name to avoid collisions across servers."""
    return f"mcp__{server}__{tool}"


class McpTool:
    """Adapt a remote MCP tool to the local Tool protocol."""

    read_only = False
    exclusive = False

    def __init__(
        self,
        session: Any,
        server_name: str,
        remote_name: str,
        description: str,
        input_schema: dict[str, Any] | None,
    ) -> None:
        self._session = session
        self._server_name = server_name
        self._remote_name = remote_name
        self._description = description or f"MCP tool {remote_name} from {server_name}"
        self._parameters = input_schema or {"type": "object", "properties": {}}
        self._name = mcp_tool_name(server_name, remote_name)

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    async def execute(self, **kwargs: Any) -> str:
        result = await self._session.call_tool(self._remote_name, kwargs)
        return self._flatten_result(result)

    @staticmethod
    def _flatten_result(result: Any) -> str:
        is_error = bool(getattr(result, "isError", False))
        parts: list[str] = []
        for block in getattr(result, "content", None) or []:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
                continue
            data = getattr(block, "data", None)
            if data is not None:
                parts.append(str(data))
        if not parts:
            structured = getattr(result, "structuredContent", None)
            if structured is not None:
                try:
                    parts.append(json.dumps(structured, ensure_ascii=False))
                except TypeError:
                    parts.append(str(structured))
        body = "\n".join(parts) if parts else "(empty)"
        return f"Error: {body}" if is_error else body
