from __future__ import annotations

from typing import Protocol

Message = dict[str, str]


class LLMProvider(Protocol):
    model: str

    async def complete(self, messages: list[Message]) -> str:
        """Return an assistant response for the given messages."""
