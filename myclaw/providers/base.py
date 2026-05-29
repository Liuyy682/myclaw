from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from myclaw.agent.types import Message


@dataclass(slots=True)
class LLMResponse:
    content: str
    final: bool = True
    stop_reason: str = "completed"


class LLMProvider(Protocol):
    model: str

    async def complete(self, messages: list[Message]) -> str | LLMResponse:
        """Return an assistant response for the given messages."""
