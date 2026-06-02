from __future__ import annotations

from typing import Any

from myclaw.tools.base import Tool, get_current_tool_context


class AskUserTool(Tool):
    read_only = False
    exclusive = True

    @property
    def name(self) -> str:
        return "ask_user"

    @property
    def description(self) -> str:
        return "Ask the user a follow-up question. This simplified tool records the request without pausing execution."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Question to ask the user"},
                "choices": {"type": "array", "items": {"type": "string"}, "description": "Optional choices"},
            },
            "required": ["question"],
        }

    async def execute(self, question: str | None = None, choices: list[str] | None = None, **kwargs: Any) -> dict[str, Any] | str:
        if not question or not question.strip():
            return "Error: question is required"
        context = get_current_tool_context()
        return {
            "status": "awaiting_user",
            "question": question.strip(),
            "choices": [str(choice) for choice in choices or []],
            "session_key": context.session_key,
        }
