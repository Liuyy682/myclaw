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
        loop = getattr(self.dispatcher, "loop", None)
        observability = getattr(loop, "observability", None)
        if observability is not None:
            observability.start()
        registry = getattr(loop, "tool_registry", None)
        security_store = getattr(registry, "security_store", None)
        recover = getattr(security_store, "recover_incomplete_operations", None)
        if callable(recover):
            recovered = recover()
            if recovered:
                logger.warning("Marked %d incomplete tool operation(s) unknown", recovered)
        await self._start_mcp()
        self._task = asyncio.create_task(self.dispatcher.run())
        await asyncio.sleep(0)

    async def stop(self) -> None:
        if self._task is None:
            await self._stop_mcp()
            loop = getattr(self.dispatcher, "loop", None)
            observability = getattr(loop, "observability", None)
            if observability is not None:
                observability.stop()
            return
        task = self._task
        self._task = None
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await self._stop_mcp()
        loop = getattr(self.dispatcher, "loop", None)
        observability = getattr(loop, "observability", None)
        if observability is not None:
            observability.stop()

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

        try:
            from myclaw.tools.security import SecurityStore
        except ImportError:
            # Configured MCP is an external side-effect boundary.  If the
            # security implementation is unavailable, fail closed instead
            # of starting an unapproved stdio process.
            logger.error("MCP startup rejected: SecurityStore is unavailable")
            return

        try:
            db_path = workspace / "security" / "tool_security.db"
            try:
                security_store = SecurityStore(path=db_path)
            except TypeError:
                # Compatibility for a minimal adapter whose only constructor
                # argument is the database path.
                security_store = SecurityStore(db_path)
        except Exception:
            logger.exception("MCP startup rejected: could not open the security store")
            return

        approved_configs = []
        for config in configs:
            config_hash = config.config_hash
            try:
                status = security_store.mcp_status(config.name, config_hash)
            except Exception:
                logger.exception("MCP server %s rejected: approval lookup failed", config.name)
                continue
            if not _mcp_approval_is_current(status):
                logger.warning(
                    "MCP server %s rejected before startup: not approved or configuration changed "
                    "(config=%s)",
                    config.name,
                    config_hash[:12],
                )
                continue
            approved_configs.append(config)

        if not approved_configs:
            return

        try:
            # ToolRegistry instances built before MCP startup have no store or
            # policy gate yet.  Bind both so every later MCP call is checked
            # by the registry rather than relying on startup approval alone.
            setattr(registry, "security_store", security_store)
            policy_gate = getattr(registry, "policy_gate", None)
            if policy_gate is None:
                from myclaw.tools.security import PolicyGate

                setattr(registry, "policy_gate", PolicyGate(security_store))
            elif getattr(policy_gate, "store", None) is None:
                setattr(policy_gate, "store", security_store)
        except Exception:
            logger.exception("MCP startup rejected: registry cannot install per-call approval")
            return

        manager = McpManager()
        try:
            await manager.connect(approved_configs)
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


def _mcp_approval_is_current(status: object) -> bool:
    """Normalize SecurityStore's approval result without exposing secrets."""

    if isinstance(status, bool):
        return status
    if isinstance(status, dict):
        return bool(status.get("approved", status.get("status") == "approved"))
    approved = getattr(status, "approved", None)
    return bool(approved if approved is not None else getattr(status, "status", "") == "approved")
