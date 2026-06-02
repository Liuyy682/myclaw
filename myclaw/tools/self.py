from __future__ import annotations

from typing import Any

from myclaw.tools.base import Tool, get_current_tool_context


class MyTool(Tool):
    read_only = True
    exclusive = False

    @property
    def name(self) -> str:
        return "my"

    @property
    def description(self) -> str:
        return "Inspect the current myclaw runtime context and available built-in tool names."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        context = get_current_tool_context()
        return {
            "session_key": context.session_key,
            "channel": context.channel,
            "chat_id": context.chat_id,
            "metadata": dict(context.metadata),
            "workspace": str(context.workspace) if context.workspace is not None else "",
            "tools": list(context.tool_names),
        }
