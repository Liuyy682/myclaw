from __future__ import annotations

from typing import Any

from myclaw.observability.runtime import ObservabilityRuntime
from myclaw.providers.base import LLMResponse, Message


class ObservedProvider:
    """Provider wrapper that records metadata without retaining message content."""

    def __init__(self, provider: Any, observability: ObservabilityRuntime) -> None:
        self._provider = provider
        self._observability = observability
        self.model = provider.model

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> str | LLMResponse:
        return await self._observe_call(self._provider.complete, messages, tools=tools)

    def __getattr__(self, name: str):
        if name != "stream_complete":
            return getattr(self._provider, name)
        stream_complete = getattr(self._provider, "stream_complete", None)
        if not callable(stream_complete):
            raise AttributeError(name)

        async def observed_stream(
            messages: list[Message],
            *,
            tools: list[dict[str, Any]] | None = None,
            delta_callback=None,
        ) -> str | LLMResponse:
            return await self._observe_call(
                stream_complete,
                messages,
                tools=tools,
                delta_callback=delta_callback,
                streaming=True,
            )

        return observed_stream

    async def _observe_call(
        self,
        callback,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None,
        delta_callback=None,
        streaming: bool = False,
    ) -> str | LLMResponse:
        attributes = {
            "model": self.model,
            "streaming": streaming,
            "message_count": len(messages),
            "input_chars": sum(len(str(message.get("content") or "")) for message in messages),
            "tool_definition_count": len(tools or []),
        }
        with self._observability.span("llm.complete", "llm", attributes=attributes) as span:
            if streaming:
                response = await callback(messages, tools=tools, delta_callback=delta_callback)
            else:
                response = await callback(messages, tools=tools)
            content = response.content if isinstance(response, LLMResponse) else response
            span.set_attribute("output_chars", len(content))
            if isinstance(response, LLMResponse):
                span.set_attribute("stop_reason", response.stop_reason)
                span.set_attribute("tool_call_count", len(response.tool_calls))
                if response.usage is not None:
                    span.set_usage(
                        response.usage.prompt_tokens,
                        response.usage.completion_tokens,
                        response.usage.total_tokens,
                    )
            return response
