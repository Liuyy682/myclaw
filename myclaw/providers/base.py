from __future__ import annotations

from typing import Protocol

from myclaw.agent.types import Message


class LLMProvider(Protocol):
    model: str

    async def complete(self, messages: list[Message]) -> str:
        """Return an assistant response for the given messages."""
