from __future__ import annotations

from myclaw.agent.types import AgentRunResult, AgentRunSpec, Message
from myclaw.providers.base import LLMProvider


class AgentRunner:
    """Run one model execution without owning product-layer history."""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        try:
            content = await self.provider.complete([dict(message) for message in spec.messages])
        except Exception as exc:
            error = str(exc)
            assistant_text = f"Error: {error}"
            return AgentRunResult(
                content=assistant_text,
                messages=[self._assistant_message(assistant_text)],
                stop_reason="error",
                error=error,
            )

        assistant_text = content.strip() if content.strip() else "(empty response)"
        return AgentRunResult(
            content=assistant_text,
            messages=[self._assistant_message(assistant_text)],
        )

    @staticmethod
    def _assistant_message(content: str) -> Message:
        return {"role": "assistant", "content": content}
