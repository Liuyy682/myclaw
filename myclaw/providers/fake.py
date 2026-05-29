from __future__ import annotations

from dataclasses import dataclass

from typing import Any

from myclaw.providers.base import Message


@dataclass(slots=True)
class FakeProvider:
    prefix: str = "Echo"
    model: str = "fake"

    async def complete(self, messages: list[Message], *, tools: list[dict[str, Any]] | None = None) -> str:
        last_user = next(
            (message["content"] for message in reversed(messages) if message.get("role") == "user"),
            "",
        )
        return f"{self.prefix}: {last_user}"
