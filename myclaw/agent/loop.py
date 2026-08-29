from __future__ import annotations

from datetime import datetime
import contextlib
import logging
from typing import Any
import uuid

from myclaw.agent.autocompact import AutoCompactManager
from myclaw.agent.context import (
    CONTEXT_SUMMARY_METADATA_KEY,
    ContextBudgetManager,
    ContextBuilder,
    TokenEstimator,
)
from myclaw.agent.dream import DreamManager
from myclaw.agent.runner import AgentRunner
from myclaw.agent.types import AgentConfig, AgentRunSpec, Message, ProgressCallback, RunResult, StreamCallback
from myclaw.cron import CronStore
from myclaw.memory import MemoryStore
from myclaw.observability import ObservabilityConfig, ObservabilityRuntime, SpanHandle, current_trace_context
from myclaw.providers.base import LLMProvider
from myclaw.session import Session, SessionManager, TranscriptStore
from myclaw.skills import SkillCatalog
from myclaw.tools import ToolRegistry
from myclaw.tools.base import AskCallback, ToolRuntimeContext, get_current_tool_context


_CRON_LOCAL_READ_TOOLS = (
    "read_file",
    "list_dir",
    "grep",
    "glob",
    "task_list",
    "task_get",
    "my",
    "skill_load",
)


logger = logging.getLogger(__name__)


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
        skill_catalog: SkillCatalog | None = None,
        observability: ObservabilityRuntime | None = None,
    ) -> None:
        self.provider = provider
        self.observability = observability or ObservabilityRuntime(
            session_manager.workspace,
            ObservabilityConfig(enabled=False),
        )
        self.config = config or AgentConfig()
        if self.config.max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        if self.config.max_tool_result_chars < 1:
            raise ValueError("max_tool_result_chars must be at least 1")
        if self.config.max_context_messages < 1:
            raise ValueError("max_context_messages must be at least 1")
        if self.config.max_context_tokens < 1:
            raise ValueError("max_context_tokens must be at least 1")
        if self.config.context_summary_max_chars < 1:
            raise ValueError("context_summary_max_chars must be at least 1")
        if self.config.context_summary_chunk_tokens < 1:
            raise ValueError("context_summary_chunk_tokens must be at least 1")
        if self.config.idle_compact_after_minutes < 0:
            raise ValueError("idle_compact_after_minutes must be at least 0")
        if self.config.dream_interval_minutes < 0:
            raise ValueError("dream_interval_minutes must be at least 0")
        if self.config.auto_compact_recent_messages < 1:
            raise ValueError("auto_compact_recent_messages must be at least 1")
        if self.config.observation_reflection_batch_size < 1:
            raise ValueError("observation_reflection_batch_size must be at least 1")
        if self.config.observation_memory_max_tokens < 1:
            raise ValueError("observation_memory_max_tokens must be at least 1")
        self.context_builder = ContextBuilder()
        self.context_budget = ContextBudgetManager(provider, self.context_builder)
        self.runner = AgentRunner(provider)
        self.session_manager = session_manager
        self.transcript = TranscriptStore(
            session_manager.workspace,
            safe_key=SessionManager.safe_key,
        )
        self.memory_store = MemoryStore(session_manager.workspace)
        self.cron_store = CronStore(session_manager.workspace)
        self.auto_compact = AutoCompactManager(
            session_manager,
            self.context_budget,
            self.memory_store,
            self.config,
            model=self.config.model or self.provider.model,
            observability=self.observability,
        )
        self.tool_registry = tool_registry
        self.skill_catalog = skill_catalog
        self.dream = DreamManager(
            session_manager,
            provider,
            self.memory_store,
            self.config,
            model=self.config.model or self.provider.model,
            observability=self.observability,
            tool_registry=self.tool_registry,
        )
        self.observation_store = None
        self.observation_worker = None
        if self.config.observation_memory_enabled:
            from myclaw.agent.observation import ObservationMemoryWorker
            from myclaw.memory import ObservationMemoryStore
            from myclaw.tools import RecallTool

            self.observation_store = ObservationMemoryStore(session_manager.workspace)
            self.observation_worker = ObservationMemoryWorker(
                provider,
                self.observation_store,
                batch_size=self.config.observation_reflection_batch_size,
            )
            if self.tool_registry is not None and self.tool_registry.get("recall") is None:
                self.tool_registry.register(RecallTool(self.observation_store))

    async def run(
        self,
        text: str,
        *,
        session_key: str,
        channel: str = "cli",
        chat_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        progress_callback: ProgressCallback | None = None,
        stream_callback: StreamCallback | None = None,
        ask_callback: AskCallback | None = None,
        tool_context: ToolRuntimeContext | None = None,
    ) -> RunResult:
        user_text = text.strip()
        if not user_text:
            raise ValueError("user input cannot be empty")

        trace_id = str((metadata or {}).get("trace_id") or "") or None
        trace_scope = (
            contextlib.nullcontext(SpanHandle())
            if current_trace_context() is not None
            else self.observability.trace(
                "agent.request",
                "conversation",
                request_id=str((metadata or {}).get("request_id") or ""),
                session_key=session_key,
                channel=channel,
                model=self.config.model or self.provider.model,
                trace_id=trace_id,
            )
        )
        with trace_scope as trace:
            with self.observability.span("agent.turn", "agent") as turn:
                result = await self._run_turn(
                    user_text,
                    session_key=session_key,
                    channel=channel,
                    chat_id=chat_id,
                    metadata=metadata,
                    progress_callback=progress_callback,
                    stream_callback=stream_callback,
                    ask_callback=ask_callback,
                    tool_context=tool_context,
                )
                if result.error:
                    turn.set_error(result.error, error_type="AgentRunError")
                    trace.set_error(result.error, error_type="AgentRunError")
                turn.set_attribute("stop_reason", result.stop_reason)
                return result

    async def _run_turn(
        self,
        user_text: str,
        *,
        session_key: str,
        channel: str,
        chat_id: str | None,
        metadata: dict[str, Any] | None,
        progress_callback: ProgressCallback | None,
        stream_callback: StreamCallback | None,
        ask_callback: AskCallback | None,
        tool_context: ToolRuntimeContext | None,
    ) -> RunResult:

        with self.observability.span("context.prepare", "agent") as prepare_span:
            session = self.session_manager.get_or_create(session_key)
            if self._restore_incomplete_turn(session):
                self.session_manager.save(session)
            compacted = False
            if self._observation_compaction_safe(session):
                compacted = await self.auto_compact.prepare_session(session_key)
            if compacted:
                session = self.session_manager.get_or_create(session_key)
            memory_text = self.memory_store.read_memory()
            user_memory_text = self.memory_store.read_user()
            soul_text = self.memory_store.read_soul()
            skills_text = self.skill_catalog.render_for_prompt() if self.skill_catalog is not None else ""
            session_memory_text = self._observation_memory_text(session_key)
            summarized = await self.context_budget.ensure_budget(
                session,
                self.config,
                user_text,
                model=self.config.model or self.provider.model,
                memory_text=memory_text,
                user_text=user_memory_text,
                soul_text=soul_text,
                skills_text=skills_text,
                session_memory_text=session_memory_text,
                archive_history=self.memory_store.append_history,
                fast_summary=lambda count: self._fast_observation_summary(
                    session_key, session.messages[:count]
                ),
            )
            if summarized:
                self.session_manager.save(session)
            prepare_span.set_attribute("auto_compacted", compacted)
            prepare_span.set_attribute("context_summarized", summarized)
            prepare_span.set_attribute("history_messages", len(session.messages))
            prepare_span.set_attribute("memory_chars", len(memory_text))
        with self.observability.span("context.build", "agent") as build_span:
            messages = self._messages_for_run(
                session,
                user_text,
                memory_text,
                skills_text,
                session_memory_text,
                user_memory_text,
                soul_text,
            )
            build_span.set_attribute("message_count", len(messages))
            build_span.set_attribute("input_chars", len(user_text))
        with self.observability.span("session.persist_input", "storage"):
            turn_id = uuid.uuid4().hex if self.observation_store is not None else None
            user_message = self._mark_pending_user_turn(session, user_text, turn_id=turn_id)
            self.transcript.append(session_key, user_message)

        runtime_context = self._tool_runtime_context(
            session_key,
            channel,
            chat_id,
            metadata,
            ask_callback,
            base_context=tool_context,
        )
        runtime_registry = self._registry_for_context(runtime_context)
        result = await self.runner.run(
            AgentRunSpec(
                messages=messages,
                model=self.config.model or self.provider.model,
                max_iterations=self.config.max_turns,
                tools=runtime_registry,
                max_tool_result_chars=self.config.max_tool_result_chars,
                tool_context=runtime_context,
                checkpoint_callback=lambda payload: self._set_runtime_checkpoint(session, payload),
                progress_callback=progress_callback,
                stream_callback=stream_callback,
            )
        )
        with self.observability.span("session.persist_output", "storage") as persist_span:
            persisted_messages = self._persist_turn(session, result.messages, turn_id=turn_id)
            self.transcript.append_many(session_key, persisted_messages)
            self._clear_pending_user_turn(session)
            self._clear_runtime_checkpoint(session)
            await self._ensure_session_title(session)
            self.session_manager.save(session)
            self._enqueue_observation_turn(
                session_key,
                turn_id,
                [user_message, *persisted_messages],
                metadata,
            )
            persist_span.set_attribute("generated_messages", len(result.messages))
        run_messages = messages + [dict(message) for message in result.messages]
        return RunResult(
            content=result.content,
            messages=run_messages,
            model=self.config.model or self.provider.model,
            stop_reason=result.stop_reason,
            error=result.error,
        )

    def _registry_for_context(self, context: ToolRuntimeContext) -> ToolRegistry | None:
        parent = self.tool_registry
        if parent is None:
            return None
        allowed = getattr(context, "allowed_tools", None)
        if allowed is None:
            return parent
        allowed_names = {str(name) for name in allowed}
        if allowed_names == set(parent.tool_names):
            return parent
        constructor_values = {
            "security_store": _registry_security_value(parent, "security_store"),
            "policy_gate": _registry_security_value(parent, "policy_gate"),
            "tool_timeout_seconds": _registry_security_value(parent, "tool_timeout_seconds"),
            "max_result_chars": _registry_security_value(parent, "max_result_chars"),
        }
        try:
            scoped = ToolRegistry(**constructor_values)
        except TypeError:
            scoped = ToolRegistry()
        for name in parent.tool_names:
            if name in allowed_names:
                tool = parent.get(name)
                if tool is not None:
                    scoped.register(tool)
        _copy_registry_security_components(parent, scoped)
        return scoped

    def reset_session(self, session_key: str) -> None:
        self.session_manager.reset(session_key)

    def observation_command(self, session_key: str, command: str) -> str:
        store = self.observation_store
        if store is None:
            return "Observation memory is disabled."
        parts = command.strip().split()
        action = parts[0].lower()
        try:
            if action == "/om:status":
                status = store.status(session_key)
                jobs = status.get("jobs", {})
                reflection_failure = status.get("reflection_failure")
                reflection_line = (
                    f"\nlast reflection error: {reflection_failure.get('last_error')} "
                    f"(attempt {reflection_failure.get('attempts')}/3)"
                    if reflection_failure
                    else ""
                )
                return (
                    "Observation memory status\n"
                    f"source events: {status.get('source_events', 0)}\n"
                    f"jobs: pending={jobs.get('pending', 0)}, processing={jobs.get('processing', 0)}, "
                    f"completed={jobs.get('completed', 0)}, failed={jobs.get('failed', 0)}\n"
                    f"active observations: {status.get('active_observations', 0)}\n"
                    f"active reflections: {status.get('active_reflections', 0)}\n"
                    f"safe watermark: {status.get('safe_watermark') or '-'}"
                    f"{reflection_line}"
                )
            if action == "/om:view":
                include_archived = len(parts) == 2 and parts[1].lower() == "full"
                if len(parts) > 2 or (len(parts) == 2 and not include_archived):
                    return "Usage: /om:view [full]"
                return store.view_text(session_key, include_archived=include_archived)
            if action == "/om:recall":
                if len(parts) != 2:
                    return "Usage: /om:recall <observation-or-reflection-id>"
                result = store.recall(parts[1], session_id=session_key)
                if result is None or result.get("status") != "found":
                    return "Memory not found or not visible in this session."
                import json

                return json.dumps(result, ensure_ascii=False, indent=2)
            if action == "/om:promote":
                if len(parts) != 2:
                    return "Usage: /om:promote <reflection-id>"
                visible = store.recall(parts[1], session_id=session_key)
                if (
                    not visible
                    or visible.get("status") != "found"
                    or visible.get("kind") != "reflection"
                ):
                    return "Reflection not found or not visible in this session."
                store.promote_reflection(parts[1])
                return f"Promoted reflection {parts[1]} to workspace memory."
        except (KeyError, ValueError) as exc:
            return f"Error: {exc}"
        return "Unknown observation-memory command."

    def _enqueue_observation_turn(
        self,
        session_key: str,
        turn_id: str | None,
        events: list[Message],
        metadata: dict[str, Any] | None,
    ) -> None:
        if self.observation_store is None or turn_id is None:
            return
        enriched = []
        for event in events:
            item = dict(event)
            item["request_id"] = str((metadata or {}).get("request_id") or "")
            item["trace_id"] = str((metadata or {}).get("trace_id") or "")
            enriched.append(item)
        try:
            self.observation_store.enqueue_turn(session_key, turn_id, enriched)
        except Exception:
            logger.exception("Failed to enqueue observation turn %s", turn_id)
        else:
            worker = self.observation_worker
            if worker is not None:
                worker.wake()

    def _observation_memory_text(self, session_key: str) -> str:
        store = self.observation_store
        if store is None:
            return ""
        try:
            observations = store.active_observations(session_key)
            reflections = store.active_reflections(session_key)
            workspace_reader = getattr(store, "workspace_reflections", None)
            workspace = workspace_reader() if callable(workspace_reader) else []
        except Exception:
            logger.exception("Failed to render observation memory for %s", session_key)
            return ""

        priority = {
            "constraint": 0,
            "decision": 1,
            "open_issue": 2,
            "preference": 3,
            "progress": 4,
            "result": 5,
            "operation": 6,
        }
        observations = sorted(
            observations,
            key=lambda item: (
                priority.get(str(item.get("kind") or ""), 99),
                -float(item.get("importance") or 0),
                str(item.get("created_at") or ""),
            ),
        )
        reflection_by_id = {item["id"]: item for item in [*workspace, *reflections]}
        ordered_reflections = sorted(
            reflection_by_id.values(),
            key=lambda item: (
                priority.get(str(item.get("kind") or ""), 99),
                str(item.get("created_at") or ""),
            ),
        )
        lines = [
            f"- [reflection:{item['id']}] {item['content']}"
            for item in ordered_reflections
        ]
        lines.extend(
            f"- [{item.get('kind') or 'fact'}:{item['id']}] {item['content']}"
            for item in observations
        )
        if not lines:
            return ""
        estimator = TokenEstimator(self.config.model or self.provider.model)
        selected: list[str] = []
        for line in lines:
            candidate = "\n".join([*selected, line])
            if estimator.estimate_text(candidate) > self.config.observation_memory_max_tokens:
                continue
            selected.append(line)
        return "\n".join(selected)

    def _observation_compaction_safe(self, session: Session) -> bool:
        store = self.observation_store
        if store is None:
            return True
        if not session.messages:
            return True
        if any(not isinstance(message.get("turn_id"), str) for message in session.messages):
            return False
        turn_ids = {
            str(message["turn_id"])
            for message in session.messages
            if isinstance(message.get("turn_id"), str)
        }
        checker = getattr(store, "turns_are_safe", None)
        return bool(checker(session.key, turn_ids)) if callable(checker) else False

    def _fast_observation_summary(
        self, session_key: str, messages: list[Message]
    ) -> str | None:
        store = self.observation_store
        if store is None:
            return None
        if any(not isinstance(message.get("turn_id"), str) for message in messages):
            return None
        turn_ids = {
            str(message["turn_id"])
            for message in messages
            if isinstance(message.get("turn_id"), str)
        }
        if not turn_ids:
            return None
        checker = getattr(store, "turns_are_safe", None)
        if not callable(checker) or not checker(session_key, turn_ids):
            return None
        rendered = self._observation_memory_text(session_key)
        return rendered or "Observation processing found no durable facts in the covered turns."

    def _subagent_registry(self) -> ToolRegistry:
        if self.tool_registry is None:
            return ToolRegistry()
        parent = self.tool_registry
        constructor_values = {
            "security_store": _registry_security_value(parent, "security_store"),
            "policy_gate": _registry_security_value(parent, "policy_gate"),
            "tool_timeout_seconds": _registry_security_value(parent, "tool_timeout_seconds"),
            "max_result_chars": _registry_security_value(parent, "max_result_chars"),
        }
        try:
            sub = ToolRegistry(**constructor_values)
        except TypeError:
            sub = ToolRegistry()
        for name in self.tool_registry.tool_names:
            if name in {"spawn", "ask_user"}:
                continue
            tool = self.tool_registry.get(name)
            if tool is not None:
                sub.register(tool)
        # Security/approval state belongs to the parent workspace and must be
        # shared by the child registry.  Do not share an allow-all decision:
        # the child receives its own subject in _spawn_subagent below, so each
        # side-effecting call is evaluated independently.
        _copy_registry_security_components(self.tool_registry, sub)
        return sub

    async def _spawn_subagent(self, prompt: str, name: str | None = None) -> str:
        sub_prompt = (prompt or "").strip()
        if not sub_prompt:
            return "Error: prompt is required"
        registry = self._subagent_registry()
        parent_context = get_current_tool_context()
        model = self.config.model or self.provider.model
        messages: list[Message] = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": sub_prompt},
        ]
        with self.observability.span(
            "subagent.run",
            "agent",
            attributes={"name": name or "subtask", "prompt_chars": len(sub_prompt)},
        ) as span:
            result = await self.runner.run(
                AgentRunSpec(
                    messages=messages,
                    model=model,
                    max_iterations=min(self.config.max_turns, 4),
                    tools=registry,
                    max_tool_result_chars=self.config.max_tool_result_chars,
                    tool_context=_make_runtime_context(
                        {
                            "session_key": f"subagent:{name or 'subtask'}",
                            "channel": "subagent",
                            "workspace": self.session_manager.workspace,
                            "tool_names": sorted(registry.tool_names),
                            "ask": getattr(parent_context, "ask", None),
                            "subject": f"subagent:{name or 'subtask'}",
                            "security_subject": f"subagent:{name or 'subtask'}",
                            "allowed_tools": sorted(registry.tool_names),
                            "resource_scopes": _context_values(parent_context, "resource_scopes"),
                            "approval": getattr(parent_context, "approval", None),
                            "approval_callback": getattr(parent_context, "approval_callback", None),
                            "approval_id": getattr(parent_context, "approval_id", None),
                        }
                    ),
                )
            )
            if result.error:
                span.set_error(result.error, error_type="SubagentRunError")
        return result.content

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

    def _messages_for_run(
        self,
        session: Session,
        user_text: str,
        memory_text: str = "",
        skills_text: str = "",
        session_memory_text: str = "",
        user_memory_text: str | None = None,
        soul_text: str | None = None,
    ) -> list[Message]:
        return self.context_builder.build_messages(
            self.config,
            session.messages,
            user_text,
            context_summary=session.metadata.get(CONTEXT_SUMMARY_METADATA_KEY),
            memory_text=memory_text,
            user_text=(
                self.memory_store.read_user() if user_memory_text is None else user_memory_text
            ),
            soul_text=self.memory_store.read_soul() if soul_text is None else soul_text,
            skills_text=skills_text,
            session_memory_text=session_memory_text,
        )

    def _tool_runtime_context(
        self,
        session_key: str,
        channel: str,
        chat_id: str | None,
        metadata: dict[str, Any] | None,
        ask_callback: AskCallback | None = None,
        *,
        base_context: ToolRuntimeContext | None = None,
    ) -> ToolRuntimeContext:
        if chat_id is None:
            chat_id = session_key.split(":", 1)[1] if ":" in session_key else session_key
        metadata_values = dict(metadata or {})
        values: dict[str, Any] = {
            "session_key": session_key,
            "channel": channel,
            "chat_id": chat_id,
            "metadata": metadata_values,
            "workspace": self.session_manager.workspace,
            "tool_names": sorted(self.tool_registry.tool_names) if self.tool_registry is not None else [],
            "spawn": self._spawn_subagent,
            "ask": ask_callback,
            "subject": metadata_values.get("subject") or f"{channel}:{session_key}",
            "security_subject": metadata_values.get("security_subject")
            or metadata_values.get("subject")
            or f"{channel}:{session_key}",
            "allowed_tools": metadata_values.get("allowed_tools"),
            "resource_scopes": metadata_values.get("resource_scopes"),
            "approval": ask_callback,
            "approval_callback": ask_callback,
            "approval_id": metadata_values.get("approval_id"),
            "tool_call_id": metadata_values.get("tool_call_id", ""),
        }
        if channel == "cron":
            # A pre-scope job (including a legacy job with no scope fields) is
            # never allowed to inherit the full interactive registry.
            if not isinstance(values["allowed_tools"], list):
                values["allowed_tools"] = list(_CRON_LOCAL_READ_TOOLS)
            if not isinstance(values["resource_scopes"], (dict, list, tuple, set)):
                values["resource_scopes"] = ["workspace"]
            values["subject"] = metadata_values.get("subject") or "internal:cron"
            values["security_subject"] = metadata_values.get("security_subject") or "internal:cron"

        if isinstance(values.get("allowed_tools"), list):
            values["tool_names"] = list(values["allowed_tools"])

        if base_context is not None:
            base_values = {
                field: getattr(base_context, field)
                for field in _runtime_context_fields()
                if hasattr(base_context, field)
            }
            base_values.update({key: value for key, value in values.items() if value is not None})
            # A host-supplied background context is authoritative for policy
            # fields.  Keep the generated session/channel defaults, but do
            # not replace an explicit subject, approval callback, or scope.
            for field in (
                "subject",
                "security_subject",
                "approval",
                "approval_callback",
                "approval_id",
                "allowed_tools",
                "resource_scopes",
                "spawn",
                "ask",
            ):
                if field in base_values and getattr(base_context, field, None) is not None:
                    base_values[field] = getattr(base_context, field)
            if getattr(base_context, "channel", "") == "cron":
                base_values["spawn"] = None
                base_values["ask"] = None
            values = base_values
        return _make_runtime_context(values)

    def _persist_turn(
        self, session: Session, assistant_messages: list[Message], *, turn_id: str | None = None
    ) -> list[Message]:
        persisted: list[Message] = []
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
            if turn_id is not None:
                fields.update({"turn_id": turn_id, "event_id": uuid.uuid4().hex})
            session.add_message(role, content, **fields)
            persisted.append(dict(session.messages[-1]))
        return persisted

    def _mark_pending_user_turn(
        self, session: Session, user_text: str, *, turn_id: str | None = None
    ) -> Message:
        fields = {}
        if turn_id is not None:
            fields = {"turn_id": turn_id, "event_id": uuid.uuid4().hex}
        session.add_message("user", user_text, **fields)
        session.metadata[self._PENDING_USER_TURN_KEY] = True
        self.session_manager.save(session)
        return dict(session.messages[-1])

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


def _runtime_context_fields() -> tuple[str, ...]:
    fields = getattr(ToolRuntimeContext, "__dataclass_fields__", None)
    if isinstance(fields, dict):
        return tuple(fields)
    # ToolRuntimeContext is a dataclass in the current runtime.  Keep a
    # conservative fallback for adapters that provide a compatible class.
    return (
        "session_key",
        "channel",
        "chat_id",
        "metadata",
        "workspace",
        "tool_names",
        "spawn",
        "ask",
    )


def _make_runtime_context(values: dict[str, Any]) -> ToolRuntimeContext:
    fields = set(_runtime_context_fields())
    filtered = {key: value for key, value in values.items() if key in fields}
    return ToolRuntimeContext(**filtered)


def _context_value(context: Any, field: str) -> Any:
    value = getattr(context, field, None)
    return value


def _context_values(context: Any, field: str) -> Any:
    value = _context_value(context, field)
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    if isinstance(value, dict):
        return dict(value)
    return None


def _registry_security_value(registry: Any, field: str) -> Any:
    if registry is None:
        return None
    value = getattr(registry, field, None)
    if value is not None:
        return value
    return getattr(registry, f"_{field}", None)


def _copy_registry_security_components(parent: Any, child: Any) -> None:
    for field in (
        "security",
        "security_store",
        "policy_gate",
        "approval",
        "approval_store",
        "approval_manager",
        "ask",
        "subject",
    ):
        value = _registry_security_value(parent, field)
        if value is None:
            continue
        try:
            setattr(child, field, value)
        except (AttributeError, TypeError):
            # A future registry may expose immutable security components or
            # constructor-only slots; its own scoped factory remains in charge.
            continue
