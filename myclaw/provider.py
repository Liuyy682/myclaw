from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

Message = dict[str, str]


class LLMProvider(Protocol):
    model: str

    async def complete(self, messages: list[Message]) -> str:
        """Return an assistant response for the given messages."""


@dataclass(slots=True)
class FakeProvider:
    prefix: str = "Echo"
    model: str = "fake"

    async def complete(self, messages: list[Message]) -> str:
        last_user = next(
            (message["content"] for message in reversed(messages) if message.get("role") == "user"),
            "",
        )
        return f"{self.prefix}: {last_user}"


@dataclass(slots=True)
class OpenAICompatibleProvider:
    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    timeout: int = 120

    async def complete(self, messages: list[Message]) -> str:
        return await asyncio.to_thread(self._complete_sync, messages)

    def _complete_sync(self, messages: list[Message]) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self._chat_completions_url(),
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"LLM request failed: HTTP {exc.code} {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

        data = json.loads(body)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("LLM response did not include choices[0].message.content") from exc
        if not isinstance(content, str):
            raise RuntimeError("LLM response content was not text")
        return content

    def _chat_completions_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"
