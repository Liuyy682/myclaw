from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from myclaw.providers.base import LLMResponse, Message, ToolCallRequest


@dataclass(slots=True)
class OpenAICompatibleProvider:
    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    timeout: int = 120

    async def complete(self, messages: list[Message], *, tools: list[dict[str, Any]] | None = None) -> str | LLMResponse:
        return await asyncio.to_thread(self._complete_sync, messages, tools)

    def _complete_sync(self, messages: list[Message], tools: list[dict[str, Any]] | None = None) -> str | LLMResponse:
        request_body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            request_body["tools"] = tools

        payload = json.dumps(request_body).encode("utf-8")
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

        return self._parse_response(json.loads(body))

    def _parse_response(self, data: dict[str, Any]) -> str | LLMResponse:
        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("LLM response did not include choices[0].message") from exc

        raw_tool_calls = message.get("tool_calls") or []
        if raw_tool_calls:
            tool_calls = self._parse_tool_calls(raw_tool_calls)
            content = message.get("content") or ""
            if not isinstance(content, str):
                raise RuntimeError("LLM response content was not text")
            return LLMResponse(
                content=content,
                final=False,
                stop_reason=str(choice.get("finish_reason") or "tool_calls"),
                tool_calls=tool_calls,
            )

        try:
            content = message["content"]
        except KeyError as exc:
            raise RuntimeError("LLM response did not include choices[0].message.content") from exc
        if not isinstance(content, str):
            raise RuntimeError("LLM response content was not text")
        return content

    @staticmethod
    def _parse_tool_calls(raw_tool_calls: Any) -> list[ToolCallRequest]:
        if not isinstance(raw_tool_calls, list):
            raise RuntimeError("LLM response tool_calls was not a list")

        tool_calls: list[ToolCallRequest] = []
        for raw_tool_call in raw_tool_calls:
            if not isinstance(raw_tool_call, dict):
                raise RuntimeError("LLM response tool call was not an object")
            function = raw_tool_call.get("function")
            if not isinstance(function, dict):
                raise RuntimeError("LLM response tool call did not include function")

            raw_arguments = function.get("arguments", "{}")
            if isinstance(raw_arguments, str):
                try:
                    arguments = json.loads(raw_arguments or "{}")
                except json.JSONDecodeError as exc:
                    raise RuntimeError("LLM tool call arguments were not valid JSON") from exc
            elif isinstance(raw_arguments, dict):
                arguments = raw_arguments
            else:
                raise RuntimeError("LLM tool call arguments were not valid JSON")
            if not isinstance(arguments, dict):
                raise RuntimeError("LLM tool call arguments were not a JSON object")

            call_id = raw_tool_call.get("id")
            name = function.get("name")
            if not isinstance(call_id, str) or not call_id:
                raise RuntimeError("LLM response tool call did not include id")
            if not isinstance(name, str) or not name:
                raise RuntimeError("LLM response tool call did not include function.name")
            tool_calls.append(ToolCallRequest(id=call_id, name=name, arguments=arguments))
        return tool_calls

    def _chat_completions_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"
