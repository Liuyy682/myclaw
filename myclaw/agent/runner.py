from __future__ import annotations

from myclaw.agent.types import AgentRunResult, AgentRunSpec, Message
from myclaw.providers.base import LLMProvider, LLMResponse


class AgentRunner:
    """Run one model execution without owning product-layer history."""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        working_messages = [dict(message) for message in spec.messages]
        generated: list[Message] = []

        for _ in range(spec.max_iterations):
            try:
                response = await self.provider.complete([dict(message) for message in working_messages])
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
            assistant_message = self._assistant_message(llm_response.content)
            generated.append(assistant_message)
            working_messages.append(assistant_message)

            if llm_response.final:
                return AgentRunResult(
                    content=assistant_message["content"],
                    messages=generated,
                    stop_reason=llm_response.stop_reason,
                )

        return AgentRunResult(
            content=generated[-1]["content"] if generated else "",
            messages=generated,
            stop_reason="max_iterations",
        )

    @staticmethod
    def _assistant_message(content: str) -> Message:
        return {"role": "assistant", "content": content}

    @staticmethod
    def _normalize_response(response: str | LLMResponse) -> LLMResponse:
        if isinstance(response, LLMResponse):
            content = response.content.strip() if response.content.strip() else "(empty response)"
            return LLMResponse(
                content=content,
                final=response.final,
                stop_reason=response.stop_reason if response.final else "continue",
            )
        content = response.strip() if response.strip() else "(empty response)"
        return LLMResponse(content=content)
