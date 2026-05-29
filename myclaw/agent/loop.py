from __future__ import annotations

from myclaw.agent.runner import AgentRunner
from myclaw.agent.types import AgentConfig, AgentRunSpec, Message, RunResult
from myclaw.providers.base import LLMProvider
from myclaw.session import Session, SessionManager
from myclaw.tools import ToolRegistry


class AgentLoop:
    """Run one user turn against the session selected by the inbound message."""

    def __init__(
        self,
        provider: LLMProvider,
        config: AgentConfig | None = None,
        *,
        session_manager: SessionManager,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.provider = provider
        self.config = config or AgentConfig()
        if self.config.max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self.runner = AgentRunner(provider)
        self.session_manager = session_manager
        self.tool_registry = tool_registry

    async def run(self, text: str, *, session_key: str) -> RunResult:
        user_text = text.strip()
        if not user_text:
            raise ValueError("user input cannot be empty")

        session = self.session_manager.get_or_create(session_key)
        messages = self._messages_for_run(session, user_text)

        result = await self.runner.run(
            AgentRunSpec(
                messages=messages,
                model=self.config.model or self.provider.model,
                max_iterations=self.config.max_turns,
                tools=self.tool_registry,
            )
        )
        self._persist_turn(session, user_text, result.messages)
        run_messages = messages + [dict(message) for message in result.messages]
        return RunResult(
            content=result.content,
            messages=run_messages,
            model=self.config.model or self.provider.model,
        )

    def _messages_for_run(self, session: Session, user_text: str) -> list[Message]:
        messages = self._initial_messages(self.config)
        messages.extend(
            {"role": message["role"], "content": message["content"]}
            for message in session.messages
            if message.get("role") in {"user", "assistant"}
        )
        messages.append({"role": "user", "content": user_text})
        return messages

    @staticmethod
    def _initial_messages(config: AgentConfig) -> list[Message]:
        messages: list[Message] = []
        if config.system_prompt.strip():
            messages.append({"role": "system", "content": config.system_prompt.strip()})
        messages.extend(dict(message) for message in config.history)
        return messages

    def _persist_turn(self, session: Session, user_text: str, assistant_messages: list[Message]) -> None:
        session.add_message("user", user_text)
        for message in assistant_messages:
            if message.get("role") == "assistant":
                session.add_message("assistant", message.get("content", ""))
        self.session_manager.save(session)
