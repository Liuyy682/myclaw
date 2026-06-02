from __future__ import annotations

from typing import Any

from myclaw.tools.base import Tool, get_current_tool_context


class SpawnTool(Tool):
    read_only = False
    exclusive = False

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return "Record a subtask request. This simplified implementation does not start a real subagent."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Subtask prompt"},
                "name": {"type": "string", "description": "Optional subtask name"},
            },
            "required": ["prompt"],
        }

    async def execute(self, prompt: str | None = None, name: str | None = None, **kwargs: Any) -> dict[str, Any] | str:
        if prompt is None or not str(prompt).strip():
            return "Error: prompt is required"
        context = get_current_tool_context()
        return {
            "status": "stubbed",
            "message": "spawn recorded but not executed",
            "name": name or "subtask",
            "prompt": str(prompt),
            "session_key": context.session_key,
        }
