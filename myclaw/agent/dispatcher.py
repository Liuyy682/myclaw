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

    def __init__(self, bus: MessageBus, loop: AgentLoop) -> None:
        self.bus = bus
        self.loop = loop
        self._session_states: dict[str, _SessionDispatchState] = {}
        self._active_tasks: set[asyncio.Task[None]] = set()

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
        state = self._retain_session_state(msg.session_key)
        try:
            async with state.lock:
                try:
                    result = await self.loop.run(msg.content, session_key=msg.session_key)
                    content = result.content
                except Exception as exc:
                    content = f"Error: {exc}"
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
