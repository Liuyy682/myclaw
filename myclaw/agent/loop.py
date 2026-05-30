from __future__ import annotations

from datetime import datetime
from typing import Any

from myclaw.agent.context import ContextBuilder
from myclaw.agent.runner import AgentRunner
from myclaw.agent.types import AgentConfig, AgentRunSpec, Message, ProgressCallback, RunResult
from myclaw.providers.base import LLMProvider
from myclaw.session import Session, SessionManager
from myclaw.tools import ToolRegistry


class AgentLoop:
    """Run one user turn against the session selected by the inbound message."""

    _PENDING_USER_TURN_KEY = "pending_user_turn"
    _RUNTIME_CHECKPOINT_KEY = "runtime_checkpoint"
    _SESSION_TITLE_KEY = "title"
    _PENDING_USER_ERROR = "Error: Task interrupted before a response was generated."
    _PENDING_TOOL_ERROR = "Error: Task interrupted before this tool finished."

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

    async def run(
        self,
        text: str,
        *,
        session_key: str,
        progress_callback: ProgressCallback | None = None,
    ) -> RunResult:
        user_text = text.strip()
        if not user_text:
            raise ValueError("user input cannot be empty")

        session = self.session_manager.get_or_create(session_key)
        if self._restore_incomplete_turn(session):
            self.session_manager.save(session)
        messages = self._messages_for_run(session, user_text)
        self._mark_pending_user_turn(session, user_text)

        result = await self.runner.run(
            AgentRunSpec(
                messages=messages,
                model=self.config.model or self.provider.model,
                max_iterations=self.config.max_turns,
                tools=self.tool_registry,
                max_tool_result_chars=self.config.max_tool_result_chars,
                checkpoint_callback=lambda payload: self._set_runtime_checkpoint(session, payload),
                progress_callback=progress_callback,
            )
        )
        self._persist_turn(session, result.messages)
        self._clear_pending_user_turn(session)
        self._clear_runtime_checkpoint(session)
        await self._ensure_session_title(session)
        self.session_manager.save(session)
        run_messages = messages + [dict(message) for message in result.messages]
        return RunResult(
            content=result.content,
            messages=run_messages,
            model=self.config.model or self.provider.model,
        )

    def reset_session(self, session_key: str) -> None:
        self.session_manager.reset(session_key)

    async def _ensure_session_title(self, session: Session) -> None:
        if not self.config.auto_title or session.metadata.get(self._SESSION_TITLE_KEY):
            return

        title = ""
        if self.provider.model != "fake":
            try:
                title = self._clean_session_title(await self.provider.complete(self._title_messages(session)))
            except Exception:
                title = ""
        session.metadata[self._SESSION_TITLE_KEY] = title or self._fallback_session_title(session)

    @staticmethod
    def _title_messages(session: Session) -> list[Message]:
        transcript_lines = []
        for message in session.messages:
            role = message.get("role")
            content = message.get("content")
            if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                transcript_lines.append(f"{role}: {content.strip()}")
            if len(transcript_lines) >= 6:
                break
        transcript = "\n".join(transcript_lines)[:2000]
        return [
            {
                "role": "system",
                "content": (
                    "Generate a concise chat title in the user's language. "
                    "Return only the title, without quotes or punctuation."
                ),
            },
            {"role": "user", "content": transcript},
        ]

    @classmethod
    def _clean_session_title(cls, response: Any) -> str:
        if isinstance(response, str):
            content = response
        else:
            content = getattr(response, "content", "")
        if not isinstance(content, str):
            return ""
        title = content.strip().strip("\"'` ")
        if ":" in title and title.lower().split(":", 1)[0] in {"title", "标题"}:
            title = title.split(":", 1)[1].strip()
        return cls._truncate_title(title)

    @classmethod
    def _fallback_session_title(cls, session: Session) -> str:
        for message in session.messages:
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                return cls._truncate_title(message["content"].strip()) or "Untitled"
        return "Untitled"

    @staticmethod
    def _truncate_title(title: str, limit: int = 60) -> str:
        title = " ".join(title.split())
        if len(title) <= limit:
            return title
        return title[: limit - 3].rstrip() + "..."

    def _messages_for_run(self, session: Session, user_text: str) -> list[Message]:
        return self.context_builder.build_messages(self.config, session.messages, user_text)

    def _persist_turn(self, session: Session, assistant_messages: list[Message]) -> None:
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

    def _mark_pending_user_turn(self, session: Session, user_text: str) -> None:
        session.add_message("user", user_text)
        session.metadata[self._PENDING_USER_TURN_KEY] = True
        self.session_manager.save(session)

    async def _set_runtime_checkpoint(self, session: Session, payload: dict[str, Any]) -> None:
        session.metadata[self._RUNTIME_CHECKPOINT_KEY] = dict(payload)
        self.session_manager.save(session)

    def _clear_pending_user_turn(self, session: Session) -> None:
        session.metadata.pop(self._PENDING_USER_TURN_KEY, None)

    def _clear_runtime_checkpoint(self, session: Session) -> None:
        session.metadata.pop(self._RUNTIME_CHECKPOINT_KEY, None)

    def _restore_incomplete_turn(self, session: Session) -> bool:
        if self._restore_runtime_checkpoint(session):
            return True
        return self._restore_pending_user_turn(session)

    def _restore_pending_user_turn(self, session: Session) -> bool:
        if not session.metadata.get(self._PENDING_USER_TURN_KEY):
            return False
        if session.messages and session.messages[-1].get("role") == "user":
            session.add_message("assistant", self._PENDING_USER_ERROR)
        else:
            session.updated_at = datetime.now()
        self._clear_pending_user_turn(session)
        return True

    def _restore_runtime_checkpoint(self, session: Session) -> bool:
        checkpoint = session.metadata.get(self._RUNTIME_CHECKPOINT_KEY)
        if not isinstance(checkpoint, dict):
            return False

        restored_messages = self._checkpoint_messages(checkpoint)
        overlap = self._checkpoint_overlap(session.messages, restored_messages)
        session.messages.extend(restored_messages[overlap:])
        session.updated_at = datetime.now()
        self._clear_pending_user_turn(session)
        self._clear_runtime_checkpoint(session)
        return True

    def _checkpoint_messages(self, checkpoint: dict[str, Any]) -> list[Message]:
        restored: list[Message] = []
        raw_messages = checkpoint.get("messages") or []
        if isinstance(raw_messages, list):
            for message in raw_messages:
                restored_message = self._restorable_checkpoint_message(message)
                if restored_message is not None:
                    restored.append(restored_message)

        fulfilled = {
            str(message["tool_call_id"])
            for message in restored
            if message.get("role") == "tool" and message.get("tool_call_id")
        }
        raw_pending = checkpoint.get("pending_tool_calls") or []
        if isinstance(raw_pending, list):
            for tool_call in raw_pending:
                message = self._pending_tool_message(tool_call, fulfilled)
                if message is not None:
                    restored.append(message)
        return restored

    @staticmethod
    def _restorable_checkpoint_message(message: Any) -> Message | None:
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        content = message.get("content")
        if role not in {"assistant", "tool"} or not isinstance(content, str):
            return None
        restored: Message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        for key in ("tool_calls", "tool_call_id", "name"):
            if key in message:
                restored[key] = message[key]
        return restored

    def _pending_tool_message(self, tool_call: Any, fulfilled: set[str]) -> Message | None:
        if not isinstance(tool_call, dict):
            return None
        tool_call_id = tool_call.get("id")
        if not isinstance(tool_call_id, str) or not tool_call_id or tool_call_id in fulfilled:
            return None
        function = tool_call.get("function")
        name = "tool"
        if isinstance(function, dict) and isinstance(function.get("name"), str) and function["name"]:
            name = function["name"]
        fulfilled.add(tool_call_id)
        return {
            "role": "tool",
            "content": self._PENDING_TOOL_ERROR,
            "timestamp": datetime.now().isoformat(),
            "tool_call_id": tool_call_id,
            "name": name,
        }

    @classmethod
    def _checkpoint_overlap(cls, existing: list[Message], restored: list[Message]) -> int:
        max_overlap = min(len(existing), len(restored))
        for size in range(max_overlap, 0, -1):
            left = existing[-size:]
            right = restored[:size]
            if all(
                cls._checkpoint_message_key(candidate) == cls._checkpoint_message_key(restored_message)
                for candidate, restored_message in zip(left, right)
            ):
                return size
        return 0

    @staticmethod
    def _checkpoint_message_key(message: Message) -> tuple[Any, ...]:
        return (
            message.get("role"),
            message.get("content"),
            message.get("tool_call_id"),
            message.get("name"),
            message.get("tool_calls"),
        )
