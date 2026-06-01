from __future__ import annotations

from typing import Any

from myclaw.memory import MemoryStore
from myclaw.tools.base import Tool


class MemoryWriteTool(Tool):
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    @property
    def name(self) -> str:
        return "remember"

    @property
    def description(self) -> str:
        return (
            "Persist a concise long-term memory for future conversations. "
            "Use only for stable user preferences, facts, or decisions the user wants remembered."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Concise memory content to persist"},
            },
            "required": ["content"],
        }

    async def execute(self, content: str | None = None, **kwargs: Any) -> str:
        try:
            return self.store.remember("" if content is None else content)
        except ValueError as exc:
            return f"Error: {exc}"
