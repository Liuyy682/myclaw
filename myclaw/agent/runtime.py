from __future__ import annotations

import asyncio
import contextlib
import logging
from types import TracebackType

from myclaw.agent.dispatcher import AgentDispatcher

logger = logging.getLogger(__name__)


class DispatcherRuntime:
    """Own the process-level dispatcher task for an input surface."""

    def __init__(self, dispatcher: AgentDispatcher, *, enable_mcp: bool = True) -> None:
        self.dispatcher = dispatcher
        self.enable_mcp = enable_mcp
        self._task: asyncio.Task[None] | None = None
        self._mcp_manager = None

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
        await self._start_mcp()
        self._task = asyncio.create_task(self.dispatcher.run())
        await asyncio.sleep(0)

    async def stop(self) -> None:
        if self._task is None:
            await self._stop_mcp()
            return
        task = self._task
        self._task = None
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await self._stop_mcp()

    async def _start_mcp(self) -> None:
        if not self.enable_mcp or self._mcp_manager is not None:
            return
        loop = getattr(self.dispatcher, "loop", None)
        registry = getattr(loop, "tool_registry", None)
        workspace = getattr(getattr(loop, "session_manager", None), "workspace", None)
        if registry is None or workspace is None:
            return
        try:
            from myclaw.mcp import McpManager, load_mcp_configs
        except ImportError:
            return
        configs = load_mcp_configs(workspace)
        if not configs:
            return
        manager = McpManager()
        try:
            await manager.connect(configs)
            manager.register_into(registry)
        except Exception:
            logger.exception("MCP startup failed")
            await manager.aclose()
            return
        self._mcp_manager = manager

    async def _stop_mcp(self) -> None:
        manager = self._mcp_manager
        self._mcp_manager = None
        if manager is not None:
            with contextlib.suppress(Exception):
                await manager.aclose()
