from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from myclaw.agent.loop import AgentLoop
from myclaw.bus import InboundMessage, MessageBus, OutboundMessage


@dataclass(slots=True)
class _SessionDispatchState:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    ref_count: int = 0


class AgentDispatcher:
    """Continuously bridge inbound bus messages to outbound agent responses."""

    _CONTROL_COMMANDS = {"/new", "/status", "/stop"}

    def __init__(self, bus: MessageBus, loop: AgentLoop) -> None:
        self.bus = bus
        self.loop = loop
        self._session_states: dict[str, _SessionDispatchState] = {}
        self._active_tasks: set[asyncio.Task[None]] = set()
        self._active_session_tasks: dict[str, asyncio.Task[None]] = {}

    async def run(self) -> None:
        try:
            while True:
                msg = await self.bus.consume_inbound()
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

        await self._process_agent_message(msg)

    async def _process_agent_message(self, msg: InboundMessage) -> None:
        state = self._retain_session_state(msg.session_key)
        try:
            async with state.lock:
                current_task = asyncio.current_task()
                if current_task is not None:
                    self._active_session_tasks[msg.session_key] = current_task
                try:
                    result = await self.loop.run(
                        msg.content,
                        session_key=msg.session_key,
                        progress_callback=lambda payload: self._publish_progress(msg, payload),
                    )
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
            content = self._new_session(msg.session_key)
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

    def _new_session(self, session_key: str) -> str:
        task = self._active_session_tasks.get(session_key)
        if task is not None and not task.done():
            return "Cannot start a new session while a turn is running. Use /stop first."
        self.loop.reset_session(session_key)
        return "Started a new session."

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
