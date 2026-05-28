from __future__ import annotations

from myclaw.agent.runner import AgentRunner
from myclaw.agent.types import AgentConfig, AgentRunSpec, Message, RunResult
from myclaw.providers.base import LLMProvider
from myclaw.session import Session, SessionManager


class AgentLoop:
    """Own user turns and in-memory conversation history."""

    def __init__(
        self,
        provider: LLMProvider,
        config: AgentConfig | None = None,
        *,
        session: Session | None = None,
        session_manager: SessionManager | None = None,
    ) -> None:
        self.provider = provider
        self.config = config or AgentConfig()
        if self.config.max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self.runner = AgentRunner(provider)
        self.session = session
        self.session_manager = session_manager
        self._messages: list[Message] = self._initial_messages(self.config)

    @property
    def messages(self) -> list[Message]:
        return [dict(message) for message in self._messages]

    async def process(self, text: str) -> RunResult:
        user_text = text.strip()
        if not user_text:
            raise ValueError("user input cannot be empty")

        user_message = {"role": "user", "content": user_text}
        self._messages.append(user_message)

        result = await self.runner.run(
            AgentRunSpec(
                messages=self.messages,
                model=self.config.model or self.provider.model,
                max_iterations=self.config.max_turns,
            )
        )
        self._messages.extend(dict(message) for message in result.messages)
        self._persist_turn(user_text, result.messages)
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

    def _persist_turn(self, user_text: str, assistant_messages: list[Message]) -> None:
        if self.session is None or self.session_manager is None:
            return

        self.session.add_message("user", user_text)
        for message in assistant_messages:
            if message.get("role") == "assistant":
                self.session.add_message("assistant", message.get("content", ""))
        self.session_manager.save(self.session)
