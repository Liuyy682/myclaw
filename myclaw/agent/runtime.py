from __future__ import annotations

import asyncio
import contextlib
from types import TracebackType

from myclaw.agent.dispatcher import AgentDispatcher


class DispatcherRuntime:
    """Own the process-level dispatcher task for an input surface."""

    def __init__(self, dispatcher: AgentDispatcher) -> None:
        self.dispatcher = dispatcher
        self._task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> DispatcherRuntime:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.stop()

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self.dispatcher.run())
        await asyncio.sleep(0)

    async def stop(self) -> None:
        if self._task is None:
            return
        task = self._task
        self._task = None
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
