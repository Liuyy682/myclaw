from __future__ import annotations

from myclaw.agent.types import AgentRunResult, AgentRunSpec, Message
from myclaw.providers.base import LLMProvider, LLMResponse
from myclaw.tools import ToolCallRequest, ToolRegistry


class AgentRunner:
    """Run one model execution without owning product-layer history."""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        working_messages = [dict(message) for message in spec.messages]
        generated: list[Message] = []
        last_assistant_content = ""

        for iteration in range(spec.max_iterations):
            try:
                response = await self._complete(
                    spec,
                    [dict(message) for message in working_messages],
                )
            except Exception as exc:
                error = str(exc)
                assistant_text = f"Error: {error}"
                return AgentRunResult(
                    content=assistant_text,
                    messages=[self._assistant_message(assistant_text)],
                    stop_reason="error",
                    error=error,
                )

            llm_response = self._normalize_response(response)
            if llm_response.should_execute_tools:
                assistant_message = self._assistant_tool_call_message(
                    llm_response.content,
                    llm_response.tool_calls,
                )
                generated.append(assistant_message)
                working_messages.append(assistant_message)
                await self._emit_checkpoint(
                    spec,
                    phase="awaiting_tools",
                    iteration=iteration,
                    messages=generated,
                    pending_tool_calls=[
                        tool_call.to_openai_tool_call()
                        for tool_call in llm_response.tool_calls
                    ],
                )
                registry = spec.tools or ToolRegistry()
                for index, tool_call in enumerate(llm_response.tool_calls):
                    await self._emit_progress(
                        spec,
                        event="tool_started",
                        iteration=iteration,
                        tool_call=tool_call,
                        index=index + 1,
                        total=len(llm_response.tool_calls),
                    )
                    tool_result = await registry.execute(
                        tool_call,
                        max_result_chars=spec.max_tool_result_chars,
                        context=spec.tool_context,
                    )
                    await self._emit_progress(
                        spec,
                        event="tool_completed",
                        iteration=iteration,
                        tool_call=tool_call,
                        index=index + 1,
                        total=len(llm_response.tool_calls),
                    )
                    tool_message = self._tool_message(tool_call, tool_result)
                    generated.append(tool_message)
                    working_messages.append(tool_message)
                    pending_tool_calls = [
                        pending.to_openai_tool_call()
                        for pending in llm_response.tool_calls[index + 1:]
                    ]
                    await self._emit_checkpoint(
                        spec,
                        phase="tools_completed" if not pending_tool_calls else "tools_in_progress",
                        iteration=iteration,
                        messages=generated,
                        pending_tool_calls=pending_tool_calls,
                    )
                continue

            assistant_message = self._assistant_message(llm_response.content)
            generated.append(assistant_message)
            working_messages.append(assistant_message)
            last_assistant_content = assistant_message["content"]

            if llm_response.final:
                return AgentRunResult(
                    content=assistant_message["content"],
                    messages=generated,
                    stop_reason=llm_response.stop_reason,
                )

        return AgentRunResult(
            content=last_assistant_content,
            messages=generated,
            stop_reason="max_iterations",
        )

    @staticmethod
    def _assistant_message(content: str) -> Message:
        return {"role": "assistant", "content": content}

    @staticmethod
    def _assistant_tool_call_message(content: str, tool_calls: list[ToolCallRequest]) -> Message:
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": [tool_call.to_openai_tool_call() for tool_call in tool_calls],
        }

    @staticmethod
    def _tool_message(tool_call: ToolCallRequest, content: str) -> Message:
        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.name,
            "content": content,
        }

    @staticmethod
    def _tool_definitions(registry: ToolRegistry | None) -> list[dict] | None:
        if registry is None or len(registry) == 0:
            return None
        return registry.definitions()

    async def _complete(self, spec: AgentRunSpec, messages: list[Message]) -> str | LLMResponse:
        tools = self._tool_definitions(spec.tools)
        stream_complete = getattr(self.provider, "stream_complete", None)
        if spec.stream_callback is not None and callable(stream_complete):
            return await stream_complete(
                messages,
                tools=tools,
                delta_callback=spec.stream_callback,
            )
        return await self.provider.complete(messages, tools=tools)

    @staticmethod
    async def _emit_checkpoint(
        spec: AgentRunSpec,
        *,
        phase: str,
        iteration: int,
        messages: list[Message],
        pending_tool_calls: list[dict],
    ) -> None:
        if spec.checkpoint_callback is None:
            return
        await spec.checkpoint_callback(
            {
                "phase": phase,
                "iteration": iteration,
                "messages": [dict(message) for message in messages],
                "pending_tool_calls": [dict(tool_call) for tool_call in pending_tool_calls],
            }
        )

    @staticmethod
    async def _emit_progress(
        spec: AgentRunSpec,
        *,
        event: str,
        iteration: int,
        tool_call: ToolCallRequest,
        index: int,
        total: int,
    ) -> None:
        if spec.progress_callback is None:
            return
        await spec.progress_callback(
            {
                "event": event,
                "iteration": iteration,
                "tool_call_id": tool_call.id,
                "tool_name": tool_call.name,
                "index": index,
                "total": total,
            }
        )

    @staticmethod
    def _normalize_response(response: str | LLMResponse) -> LLMResponse:
        if isinstance(response, LLMResponse):
            if response.tool_calls:
                content = response.content.strip()
                return LLMResponse(
                    content=content,
                    final=False,
                    stop_reason=response.stop_reason if response.stop_reason else "tool_calls",
                    tool_calls=list(response.tool_calls),
                )
            content = response.content.strip() if response.content.strip() else "(empty response)"
            return LLMResponse(
                content=content,
                final=response.final,
                stop_reason=response.stop_reason if response.final else "continue",
            )
        content = response.strip() if response.strip() else "(empty response)"
        return LLMResponse(content=content)
