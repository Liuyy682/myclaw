from __future__ import annotations

import asyncio
from collections.abc import Collection, Callable, Coroutine
from datetime import datetime
import logging
from typing import Any

from myclaw.agent.context import CONTEXT_SUMMARY_METADATA_KEY, ContextBudgetManager
from myclaw.agent.types import AgentConfig, Message
from myclaw.memory import MemoryStore
from myclaw.session import Session, SessionManager

AUTO_COMPACT_PENDING_SUMMARY_METADATA_KEY = "auto_compact_pending_summary"

logger = logging.getLogger(__name__)


class AutoCompactManager:
    """Proactively summarize idle session history and keep a recent suffix."""

    def __init__(
        self,
        session_manager: SessionManager,
        context_budget: ContextBudgetManager,
        memory_store: MemoryStore,
        config: AgentConfig,
        *,
        model: str,
    ) -> None:
        self.session_manager = session_manager
        self.context_budget = context_budget
        self.memory_store = memory_store
        self.config = config
        self.model = model
        self._archiving: set[str] = set()
        self._archive_events: dict[str, asyncio.Event] = {}

    @property
    def enabled(self) -> bool:
        return self.config.idle_compact_after_minutes > 0

    def check_expired(
        self,
        schedule_background: Callable[[Coroutine[Any, Any, bool]], None],
        active_session_keys: Collection[str] = (),
    ) -> None:
        if not self.enabled:
            return

        active = set(active_session_keys)
        now = datetime.now()
        for session in self.session_manager.list_sessions():
            key = session.key
            if not key or key in active or key in self._archiving:
                continue
            if not self._is_expired(session.updated_at, now):
                continue
            self._start_archiving(key)
            try:
                schedule_background(self._compact_archiving_session(key))
            except Exception:
                self._finish_archiving(key)
                raise

    async def prepare_session(self, session_key: str) -> bool:
        await self._wait_for_archiving(session_key)
        changed = False
        if self.enabled:
            session = self.session_manager.get_or_create(session_key)
            if self._is_expired(session.updated_at):
                changed = await self.compact_session(session_key) or changed

        session = self.session_manager.get_or_create(session_key)
        if self._consume_pending_summary(session):
            self.session_manager.save(session)
            changed = True
        return changed

    async def compact_session(self, session_key: str) -> bool:
        if session_key in self._archiving:
            return False
        self._start_archiving(session_key)
        try:
            return await self._compact_session(session_key)
        except Exception:
            logger.exception("Auto-compact failed for %s", session_key)
            return False
        finally:
            self._finish_archiving(session_key)

    async def _compact_archiving_session(self, session_key: str) -> bool:
        try:
            return await self._compact_session(session_key)
        except Exception:
            logger.exception("Auto-compact failed for %s", session_key)
            return False
        finally:
            self._finish_archiving(session_key)

    async def _compact_session(self, session_key: str) -> bool:
        session = self.session_manager.get_or_create(session_key)
        summary = ContextBudgetManager.summary_metadata(
            session.messages,
            session.metadata.get(CONTEXT_SUMMARY_METADATA_KEY),
        )
        existing_summary = self._summary_content(summary)
        covered_count = int(summary["covered_message_count"]) if summary else 0
        archive_messages, kept_messages = self._split_archive_and_keep(session.messages, covered_count)

        if not archive_messages and covered_count == 0:
            session.updated_at = datetime.now()
            self.session_manager.save(session)
            return False

        summary_content = existing_summary
        if archive_messages:
            summary_content = await self.context_budget.summarize_archive(
                existing_summary,
                archive_messages,
                self.config,
                model=self.model,
            )
            if summary_content.strip() == "(nothing)":
                summary_content = existing_summary
            elif summary_content.strip():
                self.memory_store.append_history(summary_content)

        session.messages = kept_messages
        session.metadata.pop(AUTO_COMPACT_PENDING_SUMMARY_METADATA_KEY, None)
        if archive_messages and summary_content.strip():
            session.metadata.pop(CONTEXT_SUMMARY_METADATA_KEY, None)
            self._store_pending_summary(session, summary_content)
        elif summary_content.strip():
            self.context_budget.store_summary(session, summary_content, 0, model=self.model)
        else:
            session.metadata.pop(CONTEXT_SUMMARY_METADATA_KEY, None)
        self.session_manager.save(session)
        return True

    def _consume_pending_summary(self, session: Session) -> bool:
        pending = session.metadata.pop(AUTO_COMPACT_PENDING_SUMMARY_METADATA_KEY, None)
        if not isinstance(pending, dict):
            return pending is not None

        content = pending.get("content")
        if not isinstance(content, str) or not content.strip():
            return True

        summary = ContextBudgetManager.summary_metadata(
            session.messages,
            session.metadata.get(CONTEXT_SUMMARY_METADATA_KEY),
        )
        existing_summary = self._summary_content(summary)
        summary_content = self._merge_summary(existing_summary, content.strip())
        self.context_budget.store_summary(session, summary_content, 0, model=self.model)
        return True

    def _store_pending_summary(self, session: Session, content: str) -> None:
        session.metadata[AUTO_COMPACT_PENDING_SUMMARY_METADATA_KEY] = {
            "content": content.strip(),
            "updated_at": datetime.now().isoformat(),
        }

    def _start_archiving(self, session_key: str) -> None:
        self._archiving.add(session_key)
        self._archive_events[session_key] = asyncio.Event()

    def _finish_archiving(self, session_key: str) -> None:
        self._archiving.discard(session_key)
        event = self._archive_events.pop(session_key, None)
        if event is not None:
            event.set()

    async def _wait_for_archiving(self, session_key: str) -> None:
        event = self._archive_events.get(session_key)
        if event is not None:
            await event.wait()

    def _split_archive_and_keep(
        self,
        messages: list[Message],
        covered_count: int,
    ) -> tuple[list[Message], list[Message]]:
        covered_count = max(0, min(covered_count, len(messages)))
        tail = list(messages[covered_count:])
        if len(tail) <= self.config.auto_compact_recent_messages:
            return [], tail

        target = len(tail) - self.config.auto_compact_recent_messages
        boundary = self._recent_suffix_boundary(tail, target)
        if boundary is None or boundary <= 0:
            return [], tail
        return tail[:boundary], tail[boundary:]

    @staticmethod
    def _recent_suffix_boundary(messages: list[Message], target: int) -> int | None:
        boundaries = [
            index
            for index in range(1, len(messages))
            if messages[index].get("role") == "user"
        ]
        for boundary in boundaries:
            if boundary >= target:
                return boundary
        if boundaries:
            return boundaries[-1]
        return None

    def _is_expired(self, value: datetime | str | None, now: datetime | None = None) -> bool:
        if not self.enabled or value is None:
            return False
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value)
            except ValueError:
                return False
        return ((now or datetime.now()) - value).total_seconds() >= self.config.idle_compact_after_minutes * 60

    @staticmethod
    def _summary_content(summary: dict[str, Any] | None) -> str:
        if not summary:
            return ""
        content = summary.get("content")
        return content.strip() if isinstance(content, str) else ""

    @staticmethod
    def _merge_summary(existing_summary: str, pending_summary: str) -> str:
        existing = existing_summary.strip()
        pending = pending_summary.strip()
        if not existing:
            return pending
        if existing == pending:
            return existing
        return f"{existing}\n\n{pending}"
