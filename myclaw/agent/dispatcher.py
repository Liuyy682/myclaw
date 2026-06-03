from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from myclaw.agent.ask import AskCoordinator
from myclaw.agent.loop import AgentLoop
from myclaw.bus import InboundMessage, MessageBus, OutboundMessage


@dataclass(slots=True)
class _SessionDispatchState:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    ref_count: int = 0


class AgentDispatcher:
    """Continuously bridge inbound bus messages to outbound agent responses."""

    _CONTROL_COMMANDS = {"/clear", "/status", "/stop"}
    _AUTO_COMPACT_IDLE_TICK_SECONDS = 1.0

    def __init__(self, bus: MessageBus, loop: AgentLoop) -> None:
        self.bus = bus
        self.loop = loop
        self.ask = AskCoordinator(bus)
        self._session_states: dict[str, _SessionDispatchState] = {}
        self._active_tasks: set[asyncio.Task[None]] = set()
        self._active_session_tasks: dict[str, asyncio.Task[None]] = {}

    async def run(self) -> None:
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
                    continue
                task = asyncio.create_task(self._process_message(msg))
                self._active_tasks.add(task)
                task.add_done_callback(self._active_tasks.discard)
        except asyncio.CancelledError:
            await self._cancel_active_tasks()
            raise

    async def _process_message(self, msg: InboundMessage) -> None:
        command = self._control_command(msg.content)
        if command is not None:
            await self._process_control_message(msg, command)
            return

        # A session waiting on ask_user consumes the next message as the answer.
        # This must run before acquiring the per-session lock, which the blocked
        # turn still holds while awaiting the answer.
        if self.ask.submit_answer(msg.session_key, msg.content):
            return

        await self._process_agent_message(msg)

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

    async def _run_cron_job(self, job: dict[str, Any]) -> None:
        job_id = str(job.get("id") or "job")
        job_name = str(job.get("name") or job_id)
        metadata = {"cron_job_id": job_id, "cron_job_name": job_name}
        try:
            result = await self.loop.run(
                str(job.get("prompt") or ""),
                session_key=str(job.get("session_key") or f"cron:{job_id}"),
                channel="cron",
                chat_id=job_id,
                metadata=metadata,
            )
            content = result.content
        except Exception as exc:
            content = f"Error: {exc}"
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
        task = asyncio.create_task(coro)
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)

    async def _process_agent_message(self, msg: InboundMessage) -> None:
        state = self._retain_session_state(msg.session_key)
        try:
            async with state.lock:
                current_task = asyncio.current_task()
                if current_task is not None:
                    self._active_session_tasks[msg.session_key] = current_task
                try:
                    run_kwargs = {
                        "session_key": msg.session_key,
                        "channel": msg.channel,
                        "chat_id": msg.chat_id,
                        "metadata": dict(msg.metadata),
                        "progress_callback": lambda payload: self._publish_progress(msg, payload),
                        "ask_callback": lambda question, choices: self.ask.ask(
                            msg.session_key, question, choices
                        ),
                    }
                    if msg.channel == "gateway" or (msg.channel == "cli" and msg.metadata.get("stream") is True):
                        run_kwargs["stream_callback"] = lambda delta: self._publish_message_delta(msg, delta)
                    result = await self.loop.run(msg.content, **run_kwargs)
                    content = result.content
                except Exception as exc:
                    content = f"Error: {exc}"
                finally:
                    if self._active_session_tasks.get(msg.session_key) is current_task:
                        del self._active_session_tasks[msg.session_key]
                await self.bus.publish_outbound(
                    OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=content,
                        metadata=dict(msg.metadata),
                    )
                )
        finally:
            self._release_session_state(msg.session_key, state)

    @classmethod
    def _control_command(cls, content: str) -> str | None:
        command = content.strip().lower()
        if command in cls._CONTROL_COMMANDS:
            return command
        return None

    async def _process_control_message(self, msg: InboundMessage, command: str) -> None:
        if command == "/status":
            content = self._session_status(msg.session_key)
        elif command == "/stop":
            content = await self._stop_session(msg.session_key)
        else:
            content = self._clear_session(msg.session_key)
        await self._publish_control(msg, content)

    def _session_status(self, session_key: str) -> str:
        active_task = self._active_session_tasks.get(session_key)
        running = active_task is not None and not active_task.done()
        state = self._session_states.get(session_key)
        queued = 0
        if state is not None:
            queued = max(0, state.ref_count - (1 if running else 0))
        if running and queued:
            return f"Status: running with {queued} queued."
        if running:
            return "Status: running."
        if queued:
            return f"Status: {queued} queued."
        return "Status: idle."

    async def _stop_session(self, session_key: str) -> str:
        task = self._active_session_tasks.get(session_key)
        if task is None or task.done():
            self._active_session_tasks.pop(session_key, None)
            return "No active turn to stop."

        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return "Stopped current turn."

    def _clear_session(self, session_key: str) -> str:
        task = self._active_session_tasks.get(session_key)
        if task is not None and not task.done():
            return "Cannot clear the current session while a turn is running. Use /stop first."
        self.loop.reset_session(session_key)
        return "Cleared current session."

    async def _publish_control(self, msg: InboundMessage, content: str) -> None:
        await self.bus.publish_outbound(
            OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=content,
                metadata=dict(msg.metadata),
                event_type="control",
            )
        )

    async def _publish_progress(self, msg: InboundMessage, payload: dict) -> None:
        event = payload.get("event")
        tool_name = str(payload.get("tool_name") or "tool")
        index = payload.get("index")
        total = payload.get("total")
        action = "Finished" if event == "tool_completed" else "Running"
        metadata = dict(msg.metadata)
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

    async def _publish_message_delta(self, msg: InboundMessage, delta: str) -> None:
        metadata = dict(msg.metadata)
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

    def _release_session_state(self, session_key: str, state: _SessionDispatchState) -> None:
        state.ref_count -= 1
        if state.ref_count == 0 and self._session_states.get(session_key) is state:
            del self._session_states[session_key]

    async def _cancel_active_tasks(self) -> None:
        tasks = list(self._active_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._active_tasks.difference_update(tasks)
        self._active_session_tasks.clear()
