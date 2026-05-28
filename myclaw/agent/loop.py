from __future__ import annotations

from myclaw.agent.runner import AgentRunner
from myclaw.agent.types import AgentConfig, AgentRunSpec, Message, RunResult
from myclaw.providers.base import LLMProvider


class AgentLoop:
    """Own user turns and in-memory conversation history."""

    def __init__(self, provider: LLMProvider, config: AgentConfig | None = None) -> None:
        self.provider = provider
        self.config = config or AgentConfig()
        if self.config.max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self.runner = AgentRunner(provider)
        self._messages: list[Message] = self._initial_messages(self.config)

    @property
    def messages(self) -> list[Message]:
        return [dict(message) for message in self._messages]

    async def process(self, text: str) -> RunResult:
        user_text = text.strip()
        if not user_text:
            raise ValueError("user input cannot be empty")

        self._messages.append({"role": "user", "content": user_text})

        result = await self.runner.run(
            AgentRunSpec(
                messages=self.messages,
                model=self.config.model or self.provider.model,
                max_iterations=self.config.max_turns,
            )
        )
        self._messages.extend(dict(message) for message in result.messages)
        return RunResult(
            content=result.content,
            messages=self.messages,
            model=self.config.model or self.provider.model,
        )

    @staticmethod
    def _initial_messages(config: AgentConfig) -> list[Message]:
        messages: list[Message] = []
        if config.system_prompt.strip():
            messages.append({"role": "system", "content": config.system_prompt.strip()})
        messages.extend(dict(message) for message in config.history)
        return messages
