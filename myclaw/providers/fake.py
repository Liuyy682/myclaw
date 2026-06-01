from __future__ import annotations

from dataclasses import dataclass

from typing import Any

from myclaw.config import DEFAULT_FAKE_PROVIDER_PREFIX, FAKE_PROVIDER_MODEL
from myclaw.providers.base import Message


@dataclass(slots=True)
class FakeProvider:
    prefix: str = DEFAULT_FAKE_PROVIDER_PREFIX
    model: str = FAKE_PROVIDER_MODEL

    async def complete(self, messages: list[Message], *, tools: list[dict[str, Any]] | None = None) -> str:
        last_user = next(
            (message["content"] for message in reversed(messages) if message.get("role") == "user"),
            "",
        )
        return f"{self.prefix}: {last_user}"

    async def stream_complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        delta_callback=None,
    ) -> str:
        content = await self.complete(messages, tools=tools)
        if delta_callback is not None:
            for index in range(0, len(content), 4):
                await delta_callback(content[index:index + 4])
        return content
