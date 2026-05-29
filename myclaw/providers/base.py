from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from myclaw.agent.types import Message


@dataclass(slots=True)
class ToolCallRequest:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_openai_tool_call(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


@dataclass(slots=True)
class LLMResponse:
    content: str
    final: bool = True
    stop_reason: str = "completed"
    tool_calls: list[ToolCallRequest] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def should_execute_tools(self) -> bool:
        return self.has_tool_calls and self.stop_reason in {"tool_calls", "stop", "completed"}


class LLMProvider(Protocol):
    model: str

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> str | LLMResponse:
        """Return an assistant response for the given messages."""
