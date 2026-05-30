from __future__ import annotations

from myclaw.agent.context import ContextBuilder
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
        if self.config.max_tool_result_chars < 1:
            raise ValueError("max_tool_result_chars must be at least 1")
        self.context_builder = ContextBuilder()
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
        return self.context_builder.build_messages(self.config, session.messages, user_text)

    def _persist_turn(self, session: Session, user_text: str, assistant_messages: list[Message]) -> None:
        session.add_message("user", user_text)
        for message in assistant_messages:
            role = message.get("role")
            content = message.get("content", "")
            if role not in {"assistant", "tool"} or not isinstance(content, str):
                continue
            if role == "assistant" and not content and not message.get("tool_calls"):
                continue
            fields = {
                key: message[key]
                for key in ("tool_calls", "tool_call_id", "name")
                if key in message
            }
            session.add_message(role, content, **fields)
        self.session_manager.save(session)
