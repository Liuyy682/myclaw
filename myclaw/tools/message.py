from __future__ import annotations

from typing import Any

from myclaw.tools.base import Tool, get_current_tool_context


class MessageTool(Tool):
    read_only = False
    exclusive = False

    @property
    def name(self) -> str:
        return "message"

    @property
    def description(self) -> str:
        return "Prepare a message for the current channel or an explicitly supplied destination."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Message content"},
                "channel": {"type": "string", "description": "Optional destination channel"},
                "chat_id": {"type": "string", "description": "Optional destination chat id"},
            },
            "required": ["content"],
        }

    async def execute(
        self,
        content: str | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | str:
        if content is None or not str(content).strip():
            return "Error: content is required"
        context = get_current_tool_context()
        return {
            "status": "queued",
            "channel": channel or context.channel,
            "chat_id": chat_id or context.chat_id,
            "content": str(content),
        }
