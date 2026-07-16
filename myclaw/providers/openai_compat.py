from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from myclaw.config import DEFAULT_OPENAI_BASE_URL, DEFAULT_OPENAI_TIMEOUT_SECONDS
from myclaw.providers.base import LLMResponse, LLMUsage, Message, ToolCallRequest


@dataclass(slots=True)
class OpenAICompatibleProvider:
    api_key: str
    model: str
    base_url: str = DEFAULT_OPENAI_BASE_URL
    timeout: int = DEFAULT_OPENAI_TIMEOUT_SECONDS

    async def complete(self, messages: list[Message], *, tools: list[dict[str, Any]] | None = None) -> str | LLMResponse:
        return await asyncio.to_thread(self._complete_sync, messages, tools)

    async def stream_complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        delta_callback=None,
    ) -> str | LLMResponse:
        if delta_callback is None:
            return await asyncio.to_thread(self._stream_complete_sync, messages, tools, None)

        loop = asyncio.get_running_loop()
        deltas: asyncio.Queue[str] = asyncio.Queue()

        def emit_delta(delta: str) -> None:
            loop.call_soon_threadsafe(deltas.put_nowait, delta)

        task = asyncio.create_task(asyncio.to_thread(self._stream_complete_sync, messages, tools, emit_delta))
        while True:
            if task.done() and deltas.empty():
                break
            try:
                delta = await asyncio.wait_for(deltas.get(), timeout=0.05)
            except asyncio.TimeoutError:
                continue
            await delta_callback(delta)
        return await task

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

    def _stream_complete_sync(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        emit_delta: Callable[[str], None] | None = None,
    ) -> str | LLMResponse:
        request_body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
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
                return self._parse_stream_response(response, emit_delta)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"LLM request failed: HTTP {exc.code} {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

    def _parse_response(self, data: dict[str, Any]) -> str | LLMResponse:
        usage = self._parse_usage(data.get("usage"))
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
                usage=usage,
            )

        try:
            content = message["content"]
        except KeyError as exc:
            raise RuntimeError("LLM response did not include choices[0].message.content") from exc
        if not isinstance(content, str):
            raise RuntimeError("LLM response content was not text")
        if usage is not None:
            return LLMResponse(content=content, usage=usage)
        return content

    def _parse_stream_response(self, response: Any, emit_delta: Callable[[str], None] | None) -> str | LLMResponse:
        content_parts: list[str] = []
        tool_call_parts: dict[int, dict[str, Any]] = {}
        finish_reason = ""
        usage: LLMUsage | None = None

        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line or not line.startswith("data:"):
                continue
            data_text = line[len("data:"):].strip()
            if data_text == "[DONE]":
                break
            try:
                data = json.loads(data_text)
            except (json.JSONDecodeError, TypeError) as exc:
                raise RuntimeError("LLM stream response did not include choices[0].delta") from exc
            parsed_usage = self._parse_usage(data.get("usage")) if isinstance(data, dict) else None
            if parsed_usage is not None:
                usage = parsed_usage
            choices = data.get("choices") if isinstance(data, dict) else None
            if not choices:
                if parsed_usage is not None:
                    continue
                raise RuntimeError("LLM stream response did not include choices[0].delta")
            try:
                choice = choices[0]
            except (IndexError, TypeError) as exc:
                raise RuntimeError("LLM stream response did not include choices[0].delta") from exc

            if choice.get("finish_reason"):
                finish_reason = str(choice["finish_reason"])
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                raise RuntimeError("LLM stream response delta was not an object")

            content = delta.get("content")
            if isinstance(content, str) and content:
                content_parts.append(content)
                if emit_delta is not None:
                    emit_delta(content)
            elif content is not None and not isinstance(content, str):
                raise RuntimeError("LLM stream response content delta was not text")

            raw_tool_calls = delta.get("tool_calls") or []
            if raw_tool_calls:
                self._accumulate_stream_tool_calls(tool_call_parts, raw_tool_calls)

        content = "".join(content_parts)
        if tool_call_parts:
            tool_calls = self._parse_tool_calls(
                [tool_call_parts[index] for index in sorted(tool_call_parts)]
            )
            return LLMResponse(
                content=content,
                final=False,
                stop_reason=finish_reason or "tool_calls",
                tool_calls=tool_calls,
                usage=usage,
            )
        if usage is not None:
            return LLMResponse(content=content, usage=usage)
        return content

    @staticmethod
    def _parse_usage(raw_usage: Any) -> LLMUsage | None:
        if not isinstance(raw_usage, dict):
            return None

        def token(name: str) -> int | None:
            value = raw_usage.get(name)
            return value if isinstance(value, int) and value >= 0 else None

        prompt = token("prompt_tokens")
        completion = token("completion_tokens")
        total = token("total_tokens")
        if prompt is None and completion is None and total is None:
            return None
        return LLMUsage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)

    @staticmethod
    def _accumulate_stream_tool_calls(tool_call_parts: dict[int, dict[str, Any]], raw_tool_calls: Any) -> None:
        if not isinstance(raw_tool_calls, list):
            raise RuntimeError("LLM stream response tool_calls delta was not a list")

        for raw_tool_call in raw_tool_calls:
            if not isinstance(raw_tool_call, dict):
                raise RuntimeError("LLM stream response tool call delta was not an object")
            index = raw_tool_call.get("index")
            if not isinstance(index, int):
                raise RuntimeError("LLM stream response tool call delta did not include index")

            tool_call = tool_call_parts.setdefault(
                index,
                {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
            )
            call_id = raw_tool_call.get("id")
            if isinstance(call_id, str) and call_id:
                tool_call["id"] = call_id

            function = raw_tool_call.get("function")
            if function is None:
                continue
            if not isinstance(function, dict):
                raise RuntimeError("LLM stream response tool call delta function was not an object")

            name = function.get("name")
            if isinstance(name, str) and name:
                tool_call["function"]["name"] = name
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                tool_call["function"]["arguments"] += arguments
            elif arguments is not None:
                raise RuntimeError("LLM stream response tool call arguments delta was not text")

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
