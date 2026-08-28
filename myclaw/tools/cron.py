from __future__ import annotations

from datetime import datetime
from typing import Any

from myclaw.cron import CronStore
from myclaw.tools.base import Tool, get_current_tool_context
from myclaw.tools.models import CronInput


class CronTool(Tool):
    read_only = False
    exclusive = False
    effect = "cron"
    input_model = CronInput

    def __init__(self, store: CronStore) -> None:
        self.store = store

    @property
    def name(self) -> str:
        return "cron"

    @property
    def description(self) -> str:
        return "Schedule a prompt with every_seconds or an ISO at timestamp. Full cron expressions are unsupported."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "prompt": {"type": "string"},
                "every_seconds": {"type": "integer"},
                "at": {"type": "string", "description": "ISO timestamp"},
                "cron": {"type": "string", "description": "Unsupported full cron expression"},
                "session_key": {"type": "string"},
                "allowed_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tools permitted when the scheduled job runs",
                },
                "resource_scopes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Workspace/resource scopes permitted when the job runs",
                },
            },
            "required": ["name", "prompt"],
        }

    async def execute(
        self,
        name: str | None = None,
        prompt: str | None = None,
        every_seconds: int | None = None,
        at: str | None = None,
        cron: str | None = None,
        session_key: str | None = None,
        allowed_tools: list[str] | None = None,
        resource_scopes: dict[str, Any] | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | str:
        if cron:
            return "Error: cron expressions are not supported; use every_seconds or at"
        try:
            interval = int(every_seconds) if every_seconds is not None else None
        except (TypeError, ValueError):
            return "Error: every_seconds must be an integer"
        at_value = None
        if at is not None:
            try:
                at_value = datetime.fromisoformat(at)
            except ValueError:
                return "Error: at must be an ISO timestamp"
        if interval is None and at_value is None:
            return "Error: every_seconds or at is required"
        context = get_current_tool_context()
        # When a caller omits the fields, preserve the current approved scope
        # if the host supplied one.  A missing scope remains distinguishable
        # in storage so legacy jobs can be downgraded by the dispatcher.
        if allowed_tools is None:
            allowed_tools = _context_scope_value(context, "allowed_tools")
        if resource_scopes is None:
            resource_scopes = _context_scope_value(context, "resource_scopes")
        try:
            return self.store.create(
                name="" if name is None else name,
                prompt="" if prompt is None else prompt,
                every_seconds=interval,
                at=at_value,
                session_key=session_key or context.session_key or None,
                allowed_tools=allowed_tools,
                resource_scopes=resource_scopes,
            )
        except ValueError as exc:
            return f"Error: {exc}"


def _context_scope_value(context: Any, field: str) -> dict[str, Any] | list[str] | None:
    value = getattr(context, field, None)
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    if isinstance(value, dict):
        return dict(value)
    return None
