from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from myclaw.agent.ask import AskCoordinator
from myclaw.agent.loop import AgentLoop
from myclaw.bus import InboundMessage, MessageBus, OutboundMessage
from myclaw.observability import SpanHandle
from myclaw.runtime import RequestStore
from myclaw.tools.base import ToolRuntimeContext


logger = logging.getLogger(__name__)

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


@dataclass(slots=True)
class _SessionDispatchState:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    ref_count: int = 0


@dataclass(frozen=True, slots=True)
class DispatcherLimits:
    max_concurrent_requests: int = 4
    max_pending_requests: int = 64
    max_session_pending_requests: int = 3
    queue_wait_timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        if self.max_concurrent_requests < 1:
            raise ValueError("max_concurrent_requests must be positive")
        if self.max_pending_requests < 1:
            raise ValueError("max_pending_requests must be positive")
        if self.max_session_pending_requests < 1:
            raise ValueError("max_session_pending_requests must be positive")
        if not math.isfinite(self.queue_wait_timeout_seconds) or self.queue_wait_timeout_seconds <= 0:
            raise ValueError("queue_wait_timeout_seconds must be positive")


@dataclass(frozen=True, slots=True)
class SubmissionResult:
    accepted: bool
    reason: str | None = None
    retry_after_seconds: int | None = None


class _NoopObservability:
    config = type("Config", (), {"enabled": False})()

    def trace(self, *args, **kwargs):
        return contextlib.nullcontext(SpanHandle())

    def span(self, *args, **kwargs):
        return contextlib.nullcontext(SpanHandle())

    def record_completed_span(self, *args, **kwargs) -> None:
        return None


class AgentDispatcher:
    """Continuously bridge inbound bus messages to outbound agent responses."""

    _CONTROL_COMMANDS = {"/clear", "/status", "/stop", "/plans", "/exit-plan"}
    _AUTO_COMPACT_IDLE_TICK_SECONDS = 1.0

    def __init__(
        self,
        bus: MessageBus,
        loop: AgentLoop,
        *,
        limits: DispatcherLimits | None = None,
        request_store: RequestStore | None = None,
    ) -> None:
        self.bus = bus
        self.loop = loop
        self.limits = limits or DispatcherLimits()
        self.observability = getattr(loop, "observability", _NoopObservability())
        self.ask = AskCoordinator(bus)
        self._session_states: dict[str, _SessionDispatchState] = {}
        self._active_tasks: set[asyncio.Task[None]] = set()
        self._active_session_tasks: dict[str, asyncio.Task[None]] = {}
        self._user_stop_sessions: set[str] = set()
        self._execution_slots = asyncio.Semaphore(self.limits.max_concurrent_requests)
        self._pending_agent_requests = 0
        if request_store is not None:
            self.request_store = request_store
        else:
            self.request_store = getattr(loop, "request_store", None)
            if self.request_store is None:
                workspace = getattr(getattr(loop, "session_manager", None), "workspace", None)
                self.request_store = RequestStore(workspace) if workspace is not None else None

    async def submit(self, msg: InboundMessage) -> SubmissionResult:
        """Admit a user message without allowing unbounded waiting tasks."""
        command = self._control_command(msg.content)
        if command is not None:
            if self._is_mode_turn(command):
                if self._session_is_busy(msg.session_key):
                    await self._publish_rejection(msg, "mode_switch_busy")
                    return SubmissionResult(False, reason="mode_switch_busy")
                result = self._reserve_agent_request(msg)
                if not result.accepted:
                    await self._publish_rejection(msg, result.reason or "service_overloaded")
                    return result
                state = msg._dispatch_state
                assert state is not None
                if not self._persist_queued(msg, mode="mode"):
                    self._release_reserved_request(msg.session_key, state)
                    msg._dispatch_state = None
                    await self._publish_rejection(msg, "request_persistence_failed")
                    return SubmissionResult(False, reason="request_persistence_failed")
                try:
                    self._schedule_task(self._process_mode_message(msg, state))
                except Exception as exc:
                    self._mark_request_failed(msg, str(exc))
                    self._release_reserved_request(msg.session_key, state)
                    msg._dispatch_state = None
                    await self._publish_rejection(msg, "service_overloaded")
                    return SubmissionResult(False, reason="service_overloaded", retry_after_seconds=1)
                return result
            self._schedule_task(self._process_control_message(msg, command))
            return SubmissionResult(True)
        if self.ask.submit_answer(msg.session_key, msg.content):
            return SubmissionResult(True)
        result = self._reserve_agent_request(msg)
        if not result.accepted:
            logger.warning(
                "Agent request rejected",
                extra={
                    "reason": result.reason,
                    "pending_requests": self._pending_agent_requests,
                    "session_pending_requests": self._session_pending_count(msg.session_key),
                },
            )
            return result
        if not self._persist_queued(msg, mode="normal"):
            self._release_reserved_request(msg.session_key, msg._dispatch_state)
            msg._dispatch_state = None
            await self._publish_rejection(msg, "request_persistence_failed")
            return SubmissionResult(False, reason="request_persistence_failed")
        if self.bus.try_publish_inbound(msg):
            logger.info(
                "Agent request accepted",
                extra={
                    "pending_requests": self._pending_agent_requests,
                    "session_pending_requests": self._session_pending_count(msg.session_key),
                },
            )
            return result
        self._release_reserved_request(msg.session_key, msg._dispatch_state)
        self._mark_request_failed(msg, "inbound queue is full")
        msg._dispatch_state = None
        return SubmissionResult(False, reason="service_overloaded", retry_after_seconds=1)

    async def run(self) -> None:
        observation_worker = getattr(self.loop, "observation_worker", None)
        if observation_worker is not None:
            self._schedule_background(observation_worker.run())
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(
                        self.bus.consume_inbound(),
                        timeout=self._AUTO_COMPACT_IDLE_TICK_SECONDS,
                    )
                except asyncio.TimeoutError:
                    self._check_auto_compact()
                    self._check_cron()
                    self._check_dream()
                    continue
                self._schedule_task(self._process_message(msg))
        except asyncio.CancelledError:
            await self._cancel_active_tasks()
            raise

    async def _process_message(self, msg: InboundMessage) -> None:
        command = self._control_command(msg.content)
        if command is not None:
            if self._is_mode_turn(command):
                if self._session_is_busy(msg.session_key):
                    await self._publish_rejection(msg, "mode_switch_busy")
                    return
                admission = self._reserve_agent_request(msg)
                if not admission.accepted:
                    await self._publish_rejection(msg, admission.reason or "service_overloaded")
                    return
                state = msg._dispatch_state
                assert state is not None
                if not self._persist_queued(msg, mode="mode"):
                    self._release_reserved_request(msg.session_key, state)
                    msg._dispatch_state = None
                    await self._publish_rejection(msg, "request_persistence_failed")
                    return
                await self._process_mode_message(msg, state)
                return
            await self._process_control_message(msg, command)
            return

        # A session waiting on ask_user consumes the next message as the answer.
        # This must run before acquiring the per-session lock, which the blocked
        # turn still holds while awaiting the answer.
        if self.ask.submit_answer(msg.session_key, msg.content):
            return

        state = getattr(msg, "_dispatch_state", None)
        if state is None:
            admission = self._reserve_agent_request(msg)
            if not admission.accepted:
                await self._publish_rejection(msg, admission.reason or "service_overloaded")
                return
            state = msg._dispatch_state
            if not self._persist_queued(msg, mode="normal"):
                self._release_reserved_request(msg.session_key, state)
                msg._dispatch_state = None
                await self._publish_rejection(msg, "request_persistence_failed")
                return
        await self._process_agent_message(msg, state)

    async def _process_mode_message(self, msg: InboundMessage, state: _SessionDispatchState) -> None:
        """Transition mode, then run the resulting prompt under normal admission."""
        command = self._control_command(msg.content) or msg.content
        delegated = False
        try:
            if not self._mark_request_running(msg):
                await self._publish_rejection(msg, "request_persistence_failed")
                return
            outcome = self.loop.plan_command(msg.session_key, command)
            if outcome.run_prompt is None:
                self._mark_request_completed(msg)
                await self._publish_control(msg, outcome.content, msg.metadata)
                return
            await self._publish_control(msg, outcome.content, msg.metadata, terminal=False)
            delegated = True
            await self._process_agent_message(msg, state, run_content=outcome.run_prompt)
        except asyncio.CancelledError:
            self._mark_request_interrupted(msg, self._cancel_reason(msg.session_key))
            raise
        except Exception as exc:
            logger.exception("Plan command failed for %s", msg.session_key)
            self._mark_request_failed(msg, str(exc))
            await self._publish_control(msg, f"Error: {exc}", msg.metadata)
        finally:
            if not delegated:
                self._release_reserved_request(msg.session_key, state)
                msg._dispatch_state = None

    def _check_auto_compact(self) -> None:
        auto_compact = getattr(self.loop, "auto_compact", None)
        if auto_compact is None:
            return
        auto_compact.check_expired(
            self._schedule_background,
            active_session_keys=self._session_states.keys(),
        )

    def _check_cron(self) -> None:
        cron_store = getattr(self.loop, "cron_store", None)
        if cron_store is None:
            return
        for job in cron_store.claim_due():
            self._schedule_background(self._run_cron_job(job))

    def _check_dream(self) -> None:
        dream = getattr(self.loop, "dream", None)
        if dream is None or not dream.enabled:
            return
        if dream.should_run_now() and not dream.running:
            self._schedule_background(dream.run_once())

    async def _run_cron_job(self, job: dict[str, Any]) -> None:
        job_id = str(job.get("id") or "job")
        job_name = str(job.get("name") or job_id)
        metadata = {"cron_job_id": job_id, "cron_job_name": job_name}
        allowed_tools = _scope_values(job.get("allowed_tools"))
        resource_scopes = _scope_values(job.get("resource_scopes"))
        complete_scope = isinstance(allowed_tools, list) and isinstance(resource_scopes, (list, dict))
        # Keep the old metadata shape for legacy callers while preserving the
        # explicit scope when a job has one.  The context below always applies
        # the local-read downgrade when either field is absent.
        if complete_scope:
            metadata["allowed_tools"] = allowed_tools
            metadata["resource_scopes"] = resource_scopes
        trace_id = uuid.uuid4().hex if self.observability.config.enabled else ""
        if trace_id:
            metadata["trace_id"] = trace_id
        with self.observability.trace(
            "cron.job", "cron", request_id=job_id,
            session_key=str(job.get("session_key") or f"cron:{job_id}"),
            channel="cron", model=self._model(),
            trace_id=trace_id or None, attributes={"cron_job_id": job_id, "cron_job_name": job_name},
        ) as trace:
            try:
                run_kwargs: dict[str, Any] = {
                    "session_key": str(job.get("session_key") or f"cron:{job_id}"),
                    "channel": "cron",
                    "chat_id": job_id,
                    "metadata": metadata,
                }
                if _accepts_tool_context(self.loop.run):
                    run_kwargs["tool_context"] = _cron_runtime_context(
                        self.loop,
                        job,
                        job_id=job_id,
                        session_key=run_kwargs["session_key"],
                        metadata=metadata,
                    )
                result = await self.loop.run(str(job.get("prompt") or ""), **run_kwargs)
                content = result.content
                if getattr(result, "error", None):
                    trace.set_error(result.error, error_type="CronAgentError")
            except Exception as exc:
                trace.set_error(exc)
                logger.exception("Cron job failed: %s", job_id)
                content = f"Error: {exc}"
            with self.observability.span("outbound.publish", "queue"):
                await self.bus.publish_outbound(
                    OutboundMessage(
                        channel="cron",
                        chat_id=job_id,
                        content=content,
                        metadata=metadata,
                        event_type="cron",
                    )
                )

    def _schedule_background(self, coro) -> None:
        self._schedule_task(coro)

    def _schedule_task(self, coro) -> None:
        try:
            task = asyncio.create_task(coro)
        except Exception:
            close = getattr(coro, "close", None)
            if callable(close):
                close()
            raise
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)

    async def _process_agent_message(
        self,
        msg: InboundMessage,
        state: _SessionDispatchState,
        *,
        run_content: str | None = None,
    ) -> None:
        metadata = dict(msg.metadata)
        trace_id = str(metadata.get("trace_id") or "")
        if not trace_id and self.observability.config.enabled:
            trace_id = uuid.uuid4().hex
            metadata["trace_id"] = trace_id
        try:
            with self.observability.trace(
                "agent.request", "conversation",
                request_id=str(metadata.get("request_id") or ""),
                session_key=msg.session_key,
                channel=msg.channel,
                model=self._model(),
                attributes={"input_chars": len(msg.content)},
                started_at=msg.timestamp,
                trace_id=trace_id or None,
            ) as trace:
                with self.observability.span("dispatcher.process", "agent"):
                    acquired_session = False
                    acquired_slot = False
                    try:
                        deadline = getattr(msg, "_dispatch_deadline", time.monotonic() + self.limits.queue_wait_timeout_seconds)
                        async with asyncio.timeout(max(0.0, deadline - time.monotonic())):
                            await state.lock.acquire()
                            acquired_session = True
                            await self._execution_slots.acquire()
                            acquired_slot = True
                    except TimeoutError:
                        if acquired_slot:
                            self._execution_slots.release()
                        if acquired_session:
                            state.lock.release()
                        self._mark_request_failed(msg, "Request timed out while waiting in the queue.")
                        trace.set_error("Request timed out while waiting in the queue.", error_type="QueueTimeout")
                        await self._publish_rejection(msg, "queue_timeout")
                        return
                    try:
                        self.observability.record_completed_span(
                            "queue.wait", "queue", started_at=msg.timestamp, ended_at=datetime.now(UTC),
                            attributes={"inbound_queue_size": self.bus.inbound_size},
                        )
                        current_task = asyncio.current_task()
                        if current_task is not None:
                            self._active_session_tasks[msg.session_key] = current_task
                        try:
                            run_kwargs = {
                                "session_key": msg.session_key,
                                "channel": msg.channel,
                                "chat_id": msg.chat_id,
                                "metadata": metadata,
                                "progress_callback": lambda payload: self._publish_progress(msg, payload, metadata),
                                "ask_callback": lambda question, choices: self.ask.ask(
                                    msg.session_key, question, choices
                                ),
                            }
                            if msg.channel == "gateway" or (msg.channel == "cli" and msg.metadata.get("stream") is True):
                                run_kwargs["stream_callback"] = lambda delta: self._publish_message_delta(msg, delta, metadata)
                            if not self._mark_request_running(msg):
                                await self._publish_rejection(msg, "request_persistence_failed")
                                return
                            result = await self.loop.run(run_content or msg.content, **run_kwargs)
                            content = result.content
                            if getattr(result, "error", None):
                                trace.set_error(result.error, error_type="AgentRunError")
                                self._mark_request_failed(msg, str(result.error))
                            else:
                                self._mark_request_completed(msg)
                            trace.set_attribute("stop_reason", getattr(result, "stop_reason", "completed"))
                        except asyncio.CancelledError:
                            self._mark_request_interrupted(msg, self._cancel_reason(msg.session_key))
                            trace.set_status("cancelled")
                            raise
                        except Exception as exc:
                            trace.set_error(exc)
                            logger.exception("Agent request failed for %s", msg.session_key)
                            self._mark_request_failed(msg, str(exc))
                            content = f"Error: {exc}"
                        finally:
                            if self._active_session_tasks.get(msg.session_key) is current_task:
                                del self._active_session_tasks[msg.session_key]
                        with self.observability.span("outbound.publish", "queue"):
                            await self.bus.publish_outbound(
                                OutboundMessage(
                                    channel=msg.channel,
                                    chat_id=msg.chat_id,
                                    content=content,
                                    metadata=metadata,
                                )
                            )
                        logger.info("Agent request completed")
                    finally:
                        if acquired_slot:
                            self._execution_slots.release()
                        if acquired_session:
                            state.lock.release()
        except asyncio.CancelledError:
            self._mark_request_interrupted(msg, self._cancel_reason(msg.session_key))
            raise
        finally:
            self._release_reserved_request(msg.session_key, state)

    @classmethod
    def _control_command(cls, content: str) -> str | None:
        command = content.strip()
        normalized = command.lower()
        if normalized in cls._CONTROL_COMMANDS:
            return normalized
        if normalized == "/execute":
            return normalized
        if normalized == "/plan" or normalized.startswith("/plan "):
            return command
        if normalized.startswith("/om:"):
            return command
        return None

    @staticmethod
    def _is_mode_turn(command: str) -> bool:
        normalized = command.strip().lower()
        return (
            normalized in {"/execute", "/exit-plan", "/plan"}
            or normalized.startswith("/plan ")
        )

    def _session_is_busy(self, session_key: str) -> bool:
        state = self._session_states.get(session_key)
        return state is not None and state.ref_count > 0

    async def _process_control_message(self, msg: InboundMessage, command: str) -> None:
        metadata = dict(msg.metadata)
        trace_id = str(metadata.get("trace_id") or "")
        if not trace_id and self.observability.config.enabled:
            trace_id = uuid.uuid4().hex
            metadata["trace_id"] = trace_id
        with self.observability.trace(
            "control.command", "control", request_id=str(metadata.get("request_id") or ""),
            session_key=msg.session_key, channel=msg.channel, trace_id=trace_id or None,
            attributes={"command": command},
        ):
            if command.startswith("/om:"):
                handler = getattr(self.loop, "observation_command", None)
                content = (
                    handler(msg.session_key, command)
                    if callable(handler)
                    else "Observation memory is unavailable."
                )
            elif command == "/plans":
                handler = getattr(self.loop, "plan_command", None)
                outcome = handler(msg.session_key, command) if callable(handler) else None
                content = getattr(outcome, "content", "Plan mode is unavailable.")
            elif command == "/status":
                content = self._session_status(msg.session_key)
            elif command == "/stop":
                content = await self._stop_session(msg.session_key)
            else:
                content = self._clear_session(msg.session_key)
            await self._publish_control(msg, content, metadata)

    def _session_status(self, session_key: str) -> str:
        active_task = self._active_session_tasks.get(session_key)
        running = active_task is not None and not active_task.done()
        state = self._session_states.get(session_key)
        queued = 0
        if state is not None:
            queued = max(0, state.ref_count - (1 if running else 0))
        if running and queued:
            base = f"Status: running with {queued} queued."
        elif running:
            base = "Status: running."
        elif queued:
            base = f"Status: {queued} queued."
        else:
            base = "Status: idle."
        store = self.request_store
        if store is None:
            return base
        try:
            interrupted = store.list_recent_interrupted(session_key, limit=5)
        except Exception:
            logger.exception("Failed to read interrupted request history for %s", session_key)
            return base
        if not interrupted:
            return base
        lines = [base, "Interrupted requests (latest 5):"]
        for record in interrupted:
            summary = " ".join(record.content.split())
            if len(summary) > 100:
                summary = summary[:97].rstrip() + "..."
            reason = record.error or "unknown reason"
            lines.append(
                f"- id={record.id} at={record.updated_at} reason={reason} input={summary!r}"
            )
        lines.append("重新提交会创建新任务")
        return "\n".join(lines)

    async def _stop_session(self, session_key: str) -> str:
        task = self._active_session_tasks.get(session_key)
        if task is None or task.done():
            self._active_session_tasks.pop(session_key, None)
            return "No active turn to stop."

        self._user_stop_sessions.add(session_key)
        try:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        finally:
            self._user_stop_sessions.discard(session_key)
        return "Stopped current turn."

    def _cancel_reason(self, session_key: str) -> str:
        if session_key in self._user_stop_sessions:
            return "Stopped by user."
        return "Runtime cancelled."

    def _clear_session(self, session_key: str) -> str:
        task = self._active_session_tasks.get(session_key)
        if task is not None and not task.done():
            return "Cannot clear the current session while a turn is running. Use /stop first."
        session_manager = getattr(self.loop, "session_manager", None)
        existing = session_manager.get_or_create(session_key) if session_manager is not None else None
        plan_metadata = {}
        if existing is not None:
            plan_metadata = {
                key: existing.metadata[key]
                for key in ("agent_mode", "current_plan_id", "project_id")
                if key in existing.metadata
            }
        self.loop.reset_session(session_key)
        if plan_metadata and session_manager is not None:
            restored = session_manager.get_or_create(session_key)
            restored.metadata.update(plan_metadata)
            session_manager.save(restored)
        return "Cleared current session."

    async def _publish_control(
        self, msg: InboundMessage, content: str, base_metadata: dict | None = None,
        *, terminal: bool = True,
    ) -> None:
        await self.bus.publish_outbound(
            OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=content,
                metadata=dict(base_metadata or msg.metadata),
                terminal=terminal,
                event_type="control",
            )
        )

    async def _publish_progress(self, msg: InboundMessage, payload: dict, base_metadata: dict | None = None) -> None:
        event = payload.get("event")
        tool_name = str(payload.get("tool_name") or "tool")
        index = payload.get("index")
        total = payload.get("total")
        action = "Finished" if event == "tool_completed" else "Running"
        metadata = dict(base_metadata or msg.metadata)
        metadata["session_key"] = msg.session_key
        metadata["progress"] = dict(payload)
        await self.bus.publish_outbound(
            OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"{action} tool {tool_name} ({index}/{total})",
                metadata=metadata,
                terminal=False,
                event_type="tool_progress",
            )
        )

    async def _publish_message_delta(
        self, msg: InboundMessage, delta: str, base_metadata: dict | None = None
    ) -> None:
        metadata = dict(base_metadata or msg.metadata)
        metadata["session_key"] = msg.session_key
        await self.bus.publish_outbound(
            OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=delta,
                metadata=metadata,
                terminal=False,
                event_type="message_delta",
            )
        )

    def _retain_session_state(self, session_key: str) -> _SessionDispatchState:
        state = self._session_states.get(session_key)
        if state is None:
            state = _SessionDispatchState()
            self._session_states[session_key] = state
        state.ref_count += 1
        return state

    def _reserve_agent_request(self, msg: InboundMessage) -> SubmissionResult:
        if self._pending_agent_requests >= self.limits.max_pending_requests:
            return SubmissionResult(False, reason="service_overloaded", retry_after_seconds=1)
        state = self._session_states.get(msg.session_key)
        if state is not None and state.ref_count >= 1 + self.limits.max_session_pending_requests:
            return SubmissionResult(False, reason="session_queue_full", retry_after_seconds=5)
        state = self._retain_session_state(msg.session_key)
        self._pending_agent_requests += 1
        msg._dispatch_state = state
        msg._dispatch_deadline = time.monotonic() + self.limits.queue_wait_timeout_seconds
        return SubmissionResult(True)

    def _model(self) -> str:
        config = getattr(self.loop, "config", None)
        configured = getattr(config, "model", "")
        provider = getattr(self.loop, "provider", None)
        return configured or getattr(provider, "model", "")

    def _request_id(self, msg: InboundMessage) -> str:
        request_id = msg.metadata.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            request_id = getattr(msg, "request_id", None)
        if not isinstance(request_id, str) or not request_id:
            request_id = uuid.uuid4().hex
        # Keep persistence, observability and outbound events on the same ID.
        msg.request_id = request_id
        msg.metadata["request_id"] = request_id
        return request_id

    def _persist_queued(self, msg: InboundMessage, *, mode: str) -> bool:
        store = self.request_store
        if store is None:
            return True
        request_id = self._request_id(msg)
        try:
            record = store.create_queued(
                request_id,
                session_key=msg.session_key,
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=msg.content,
                mode=mode,
                metadata=dict(msg.metadata),
            )
        except Exception:
            logger.exception("Failed to persist queued request %s", request_id)
            # If an adapter inserted the row and failed afterwards, preserve
            # the evidence as failed.  A normal INSERT failure simply has no
            # row and this best-effort update is harmless.
            self._mark_request_failed(msg, "request persistence failed")
            return False
        msg._request_persisted = True
        msg._request_internal_id = record.id
        return True

    def _mark_request_running(self, msg: InboundMessage) -> bool:
        store = self.request_store
        if store is None or not getattr(msg, "_request_persisted", False):
            return True
        internal_id = getattr(msg, "_request_internal_id", None)
        if not isinstance(internal_id, int):
            return True
        try:
            store.mark_running(internal_id)
            return True
        except Exception:
            logger.exception("Failed to persist running request %s", internal_id)
            self._mark_request_failed(msg, "request persistence failed")
            return False

    def _mark_request_completed(self, msg: InboundMessage) -> None:
        self._transition_request(msg, "completed")

    def _mark_request_failed(self, msg: InboundMessage, error: str | None = None) -> None:
        self._transition_request(msg, "failed", error)

    def _mark_request_interrupted(self, msg: InboundMessage, error: str | None = None) -> None:
        self._transition_request(msg, "interrupted", error)

    def _transition_request(
        self, msg: InboundMessage, status: str, error: str | None = None
    ) -> None:
        store = self.request_store
        if store is None or not getattr(msg, "_request_persisted", False):
            return
        internal_id = getattr(msg, "_request_internal_id", None)
        if not isinstance(internal_id, int):
            return
        try:
            if status in {"failed", "interrupted"}:
                getattr(store, f"mark_{status}")(internal_id, error)
            else:
                getattr(store, f"mark_{status}")(internal_id)
        except Exception:
            logger.exception("Failed to persist %s request %s", status, internal_id)

    def _release_reserved_request(self, session_key: str, state: _SessionDispatchState | None) -> None:
        if state is None:
            return
        self._pending_agent_requests -= 1
        state.ref_count -= 1
        if state.ref_count == 0 and self._session_states.get(session_key) is state:
            del self._session_states[session_key]

    def _session_pending_count(self, session_key: str) -> int:
        state = self._session_states.get(session_key)
        return state.ref_count if state is not None else 0

    async def _publish_rejection(self, msg: InboundMessage, reason: str) -> None:
        content = {
            "service_overloaded": "Error: Service is busy. Please retry later.",
            "session_queue_full": "Error: This session already has too many queued requests.",
            "queue_timeout": "Error: Request timed out while waiting in the queue.",
            "mode_switch_busy": "Cannot switch plan mode while this session has a running or queued turn; stop it or wait.",
            "request_persistence_failed": "Error: Request could not be durably accepted. Please retry later.",
        }[reason]
        logger.warning(
            "dispatcher_rejected reason=%s pending=%s session_key=%s",
            reason,
            self._pending_agent_requests,
            msg.session_key,
        )
        await self.bus.publish_outbound(
            OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=content,
                metadata=dict(msg.metadata),
                event_type="error",
            )
        )

    async def _cancel_active_tasks(self) -> None:
        tasks = list(self._active_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._active_tasks.difference_update(tasks)
        self._active_session_tasks.clear()


def _scope_values(value: Any) -> list[str] | dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items() if str(key).strip()}
    return None


def _accepts_tool_context(run_callable: Any) -> bool:
    try:
        parameters = inspect.signature(run_callable).parameters.values()
    except (TypeError, ValueError):
        return True
    return any(
        parameter.name == "tool_context" or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _runtime_context_fields() -> set[str]:
    fields = getattr(ToolRuntimeContext, "__dataclass_fields__", None)
    if isinstance(fields, dict):
        return set(fields)
    return {
        "session_key",
        "channel",
        "chat_id",
        "metadata",
        "workspace",
        "tool_names",
        "spawn",
        "ask",
    }


def _cron_runtime_context(
    loop: Any,
    job: dict[str, Any],
    *,
    job_id: str,
    session_key: str,
    metadata: dict[str, Any],
) -> ToolRuntimeContext:
    raw_allowed_tools = _scope_values(job.get("allowed_tools"))
    raw_resource_scopes = _scope_values(job.get("resource_scopes"))
    complete_scope = isinstance(raw_allowed_tools, list) and isinstance(raw_resource_scopes, (list, dict))
    allowed_tools = raw_allowed_tools if complete_scope else list(_CRON_LOCAL_READ_TOOLS)
    resource_scopes = raw_resource_scopes if complete_scope else ["workspace"]
    session_manager = getattr(loop, "session_manager", None)
    workspace = getattr(session_manager, "workspace", None)
    if workspace is None:
        workspace = getattr(getattr(loop, "cron_store", None), "workspace", None)
    values: dict[str, Any] = {
        "session_key": session_key,
        "channel": "cron",
        "chat_id": job_id,
        "metadata": dict(metadata),
        "workspace": workspace,
        "tool_names": list(allowed_tools),
        "allowed_tools": list(allowed_tools),
        "resource_scopes": dict(resource_scopes) if isinstance(resource_scopes, dict) else list(resource_scopes),
        "subject": "internal:cron",
        "security_subject": "internal:cron",
        # Cron runs are non-interactive and cannot recursively spawn agents.
        "spawn": None,
        "ask": None,
    }
    if complete_scope:
        values["approval"] = _approve_scoped_cron_call
        values["approval_callback"] = _approve_scoped_cron_call
    fields = _runtime_context_fields()
    return ToolRuntimeContext(**{key: value for key, value in values.items() if key in fields})


def _approve_scoped_cron_call(_question: str, _choices: list[str] | None = None) -> str:
    """Approve only after PolicyGate has enforced the persisted Cron scope."""

    return "allow"
