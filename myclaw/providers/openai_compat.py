from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

from myclaw.config import (
    DEFAULT_LLM_CIRCUIT_FAILURE_THRESHOLD,
    DEFAULT_LLM_CIRCUIT_OPEN_SECONDS,
    DEFAULT_LLM_MAX_RETRIES,
    DEFAULT_LLM_RETRY_BASE_DELAY_SECONDS,
    DEFAULT_LLM_RETRY_MAX_DELAY_SECONDS,
    DEFAULT_LLM_TOTAL_TIMEOUT_SECONDS,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_TIMEOUT_SECONDS,
)
from myclaw.providers.base import LLMResponse, LLMServiceUnavailableError, LLMUsage, Message, ToolCallRequest


logger = logging.getLogger(__name__)
_RETRYABLE_HTTP_CODES = frozenset({408, 409, 425, 429})


@dataclass(frozen=True, slots=True)
class LLMResilienceConfig:
    """Bounds retries and protects one provider instance from an outage."""

    max_retries: int = DEFAULT_LLM_MAX_RETRIES
    request_timeout_seconds: float = DEFAULT_OPENAI_TIMEOUT_SECONDS
    total_timeout_seconds: float = DEFAULT_LLM_TOTAL_TIMEOUT_SECONDS
    base_delay_seconds: float = DEFAULT_LLM_RETRY_BASE_DELAY_SECONDS
    max_delay_seconds: float = DEFAULT_LLM_RETRY_MAX_DELAY_SECONDS
    circuit_failure_threshold: int = DEFAULT_LLM_CIRCUIT_FAILURE_THRESHOLD
    circuit_open_seconds: float = DEFAULT_LLM_CIRCUIT_OPEN_SECONDS

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.circuit_failure_threshold < 1:
            raise ValueError("circuit_failure_threshold must be positive")
        for name in (
            "request_timeout_seconds",
            "total_timeout_seconds",
            "base_delay_seconds",
            "max_delay_seconds",
            "circuit_open_seconds",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("max_delay_seconds must be greater than or equal to base_delay_seconds")


class _RequestError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds


class _StreamAttemptError(RuntimeError):
    def __init__(self, cause: BaseException, *, delta_emitted: bool) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.delta_emitted = delta_emitted


@dataclass(slots=True)
class OpenAICompatibleProvider:
    api_key: str
    model: str
    base_url: str = DEFAULT_OPENAI_BASE_URL
    timeout: int = DEFAULT_OPENAI_TIMEOUT_SECONDS
    resilience: LLMResilienceConfig = field(default_factory=LLMResilienceConfig)
    _circuit_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _circuit_state: str = field(default="closed", init=False, repr=False)
    _consecutive_failures: int = field(default=0, init=False, repr=False)
    _open_until: float = field(default=0.0, init=False, repr=False)
    _half_open_probe_active: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        # Keep the original ``timeout=...`` constructor argument effective for
        # callers that predate the resilience configuration.
        if (
            self.timeout != DEFAULT_OPENAI_TIMEOUT_SECONDS
            and self.resilience.request_timeout_seconds == DEFAULT_OPENAI_TIMEOUT_SECONDS
        ):
            self.resilience = replace(self.resilience, request_timeout_seconds=float(self.timeout))

    async def complete(self, messages: list[Message], *, tools: list[dict[str, Any]] | None = None) -> str | LLMResponse:
        return await self._run_with_resilience(
            lambda timeout: asyncio.to_thread(self._complete_sync, messages, tools, timeout)
        )

    async def stream_complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        delta_callback=None,
    ) -> str | LLMResponse:
        if delta_callback is None:
            return await self._run_with_resilience(
                lambda timeout: asyncio.to_thread(self._stream_complete_sync, messages, tools, None, timeout)
            )

        return await self._run_with_resilience(
            lambda timeout: self._stream_attempt(messages, tools, delta_callback, timeout),
            stop_retrying=lambda exc: isinstance(exc, _StreamAttemptError) and exc.delta_emitted,
        )

    async def _stream_attempt(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        delta_callback,
        timeout: float,
    ) -> str | LLMResponse:
        loop = asyncio.get_running_loop()
        deltas: asyncio.Queue[str] = asyncio.Queue()
        delta_emitted = False

        def emit_delta(delta: str) -> None:
            loop.call_soon_threadsafe(deltas.put_nowait, delta)

        task = asyncio.create_task(asyncio.to_thread(self._stream_complete_sync, messages, tools, emit_delta, timeout))
        try:
            while True:
                if task.done() and deltas.empty():
                    break
                try:
                    delta = await asyncio.wait_for(deltas.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    continue
                await delta_callback(delta)
                delta_emitted = True
            return await task
        except asyncio.CancelledError:
            task.cancel()
            raise
        except Exception as exc:
            raise _StreamAttemptError(exc, delta_emitted=delta_emitted) from exc

    async def _run_with_resilience(self, attempt, *, stop_retrying=None) -> str | LLMResponse:
        await self._acquire_circuit_permission()
        deadline = time.monotonic() + self.resilience.total_timeout_seconds
        last_error: BaseException | None = None

        for retry_index in range(self.resilience.max_retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                result = await attempt(min(self.resilience.request_timeout_seconds, remaining))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                error = self._unwrap_attempt_error(exc)
                last_error = error
                if not self._is_retryable(error):
                    await self._record_non_retryable_result()
                    raise error
                if stop_retrying is not None and stop_retrying(exc):
                    break
                if retry_index >= self.resilience.max_retries:
                    break
                delay = self._retry_delay(error, retry_index)
                if time.monotonic() + delay >= deadline:
                    break
                logger.warning(
                    "llm_resilience event=retry attempt=%s delay_seconds=%.3f error_type=%s",
                    retry_index + 1,
                    delay,
                    type(error).__name__,
                )
                await asyncio.sleep(delay)
            else:
                await self._record_success()
                return result

        await self._record_retryable_failure()
        logger.warning(
            "llm_resilience event=retry_exhausted attempts=%s error_type=%s",
            self.resilience.max_retries + 1,
            type(last_error).__name__ if last_error is not None else "DeadlineExceeded",
        )
        raise LLMServiceUnavailableError() from last_error

    @staticmethod
    def _unwrap_attempt_error(exc: BaseException) -> BaseException:
        return exc.cause if isinstance(exc, _StreamAttemptError) else exc

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        return isinstance(exc, _RequestError) and exc.retryable

    def _retry_delay(self, exc: BaseException, retry_index: int) -> float:
        delay = min(self.resilience.max_delay_seconds, self.resilience.base_delay_seconds * (2 ** retry_index))
        if isinstance(exc, _RequestError) and exc.retry_after_seconds is not None:
            delay = max(delay, exc.retry_after_seconds)
        return delay + random.uniform(0.0, delay * 0.25)

    async def _acquire_circuit_permission(self) -> None:
        now = time.monotonic()
        async with self._circuit_lock:
            if self._circuit_state == "closed":
                return
            if self._circuit_state == "half_open":
                logger.warning("llm_resilience event=circuit_rejected_half_open")
                raise LLMServiceUnavailableError()
            if now < self._open_until:
                logger.warning("llm_resilience event=circuit_rejected")
                raise LLMServiceUnavailableError()
            if self._half_open_probe_active:
                logger.warning("llm_resilience event=circuit_rejected_half_open")
                raise LLMServiceUnavailableError()
            self._circuit_state = "half_open"
            self._half_open_probe_active = True
            logger.info("llm_resilience event=circuit_half_open")

    async def _record_success(self) -> None:
        async with self._circuit_lock:
            recovered = self._circuit_state != "closed" or self._consecutive_failures
            self._circuit_state = "closed"
            self._consecutive_failures = 0
            self._open_until = 0.0
            self._half_open_probe_active = False
            if recovered:
                logger.info("llm_resilience event=circuit_recovered")

    async def _record_non_retryable_result(self) -> None:
        await self._record_success()

    async def _record_retryable_failure(self) -> None:
        now = time.monotonic()
        async with self._circuit_lock:
            if self._circuit_state == "half_open":
                should_open = True
            else:
                self._consecutive_failures += 1
                should_open = self._consecutive_failures >= self.resilience.circuit_failure_threshold
            self._half_open_probe_active = False
            if not should_open:
                return
            self._circuit_state = "open"
            self._open_until = now + self.resilience.circuit_open_seconds
            logger.warning(
                "llm_resilience event=circuit_open consecutive_failures=%s open_seconds=%.3f",
                self._consecutive_failures,
                self.resilience.circuit_open_seconds,
            )

    def _complete_sync(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
    ) -> str | LLMResponse:
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
            with urllib.request.urlopen(request, timeout=timeout if timeout is not None else self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise self._http_error(exc) from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise _RequestError(f"LLM request failed: {getattr(exc, 'reason', exc)}", retryable=True) from exc

        return self._parse_response(json.loads(body))

    def _stream_complete_sync(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        emit_delta: Callable[[str], None] | None = None,
        timeout: float | None = None,
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
            with urllib.request.urlopen(request, timeout=timeout if timeout is not None else self.timeout) as response:
                return self._parse_stream_response(response, emit_delta)
        except urllib.error.HTTPError as exc:
            raise self._http_error(exc) from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise _RequestError(f"LLM request failed: {getattr(exc, 'reason', exc)}", retryable=True) from exc

    @staticmethod
    def _http_error(exc: urllib.error.HTTPError) -> _RequestError:
        retryable = exc.code in _RETRYABLE_HTTP_CODES or 500 <= exc.code <= 599
        retry_after = _retry_after_seconds(exc.headers.get("Retry-After") if exc.headers is not None else None)
        return _RequestError(
            f"LLM request failed: HTTP {exc.code} {exc.reason}",
            retryable=retryable,
            retry_after_seconds=retry_after,
        )

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


def _retry_after_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=UTC)
    return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())
