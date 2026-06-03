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
        return "Run a subtask in an isolated sub-agent and return its final answer."

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
        if context.spawn is None:
            return "Error: sub-agent execution is not available in this context"
        result = await context.spawn(str(prompt), name)
        return {
            "status": "completed",
            "name": name or "subtask",
            "result": result,
        }
