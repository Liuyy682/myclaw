from __future__ import annotations

from typing import Any

from myclaw.tasks import TaskStore
from myclaw.tools.base import Tool, get_current_tool_context
from myclaw.tools.models import (
    TaskCreateInput,
    TaskGetInput,
    TaskListInput,
    TaskProgressInput,
    TaskUpdateInput,
)


_MODES = {"normal", "plan", "execute"}


def _mode_and_plan() -> tuple[str, str | None, str, str | None]:
    """Read run scope; plan identity never comes from mutation arguments."""

    context = get_current_tool_context()
    metadata = context.metadata or {}
    mode = str(metadata.get("agent_mode") or "normal").strip().lower()
    if mode not in _MODES:
        mode = "normal"
    plan_id = metadata.get("current_plan_id")
    project_id = metadata.get("project_id")
    return (
        mode,
        (str(plan_id) if plan_id else None),
        context.session_key,
        (str(project_id) if project_id else None),
    )


def _unknown_arguments(kwargs: dict[str, Any]) -> str | None:
    if kwargs:
        return f"Error: unsupported arguments: {', '.join(sorted(kwargs))}"
    return None


class _TaskTool(Tool):
    read_only = False
    exclusive = False
    effect = "local_write"

    def __init__(self, plan_store: Any) -> None:
        self.plan_store = plan_store

    @staticmethod
    def _mode_error(name: str, expected: str) -> str | None:
        mode, _, _, _ = _mode_and_plan()
        if mode != expected:
            return f"Error: {name} is available only in {expected.title()} mode"
        return None

    @staticmethod
    def _plan_id_error(plan_id: str | None, name: str) -> str | None:
        if not plan_id:
            return f"Error: current_plan_id is required for {name}"
        return None

    def _project_id_error(self, project_id: str | None, *, required: bool = False) -> str | None:
        store_project_id = getattr(self.plan_store, "project_id", None)
        if required and not project_id:
            return "Error: project_id is required"
        if project_id and store_project_id and project_id != str(store_project_id):
            return "Error: project scope does not match this tool store"
        return None

    @staticmethod
    def _format_error(exc: Exception) -> str:
        return f"Error: {exc}"


class TaskCreateTool(_TaskTool):
    input_model = TaskCreateInput

    @property
    def name(self) -> str:
        return "task_create"

    @property
    def description(self) -> str:
        return "Create a task in the current project plan draft."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "approach": {"type": "string"},
                "acceptance_criteria": {"type": "string"},
                "depends_on": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        title: str | None = None,
        description: str = "",
        approach: str = "",
        acceptance_criteria: str = "",
        depends_on: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | str:
        if (error := _unknown_arguments(kwargs)) is not None:
            return error
        if (error := self._mode_error(self.name, "plan")) is not None:
            return error
        _, plan_id, session_key, project_id = _mode_and_plan()
        if (error := self._project_id_error(project_id, required=True)) is not None:
            return error
        if (error := self._plan_id_error(plan_id, self.name)) is not None:
            return error
        if not title:
            return "Error: title is required"
        try:
            return self.plan_store.create_task(
                plan_id,
                session_key,
                title=title,
                description=description,
                approach=approach,
                acceptance_criteria=acceptance_criteria,
                depends_on=depends_on,
            )
        except (KeyError, PermissionError, ValueError) as exc:
            return self._format_error(exc)


class TaskUpdateTool(_TaskTool):
    exclusive = True
    input_model = TaskUpdateInput

    @property
    def name(self) -> str:
        return "task_update"

    @property
    def description(self) -> str:
        return "Edit a task in the current project plan draft."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "title": {"type": "string"},
                "description": {"type": "string"},
                "approach": {"type": "string"},
                "acceptance_criteria": {"type": "string"},
                "depends_on": {"type": "array", "items": {"type": "string"}},
                "cancel": {"type": "boolean"},
            },
            "required": ["id"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        id: str | None = None,
        title: str | None = None,
        description: str | None = None,
        approach: str | None = None,
        acceptance_criteria: str | None = None,
        depends_on: list[str] | None = None,
        cancel: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | str:
        if (error := _unknown_arguments(kwargs)) is not None:
            return error
        if (error := self._mode_error(self.name, "plan")) is not None:
            return error
        _, plan_id, session_key, project_id = _mode_and_plan()
        if (error := self._project_id_error(project_id, required=True)) is not None:
            return error
        if (error := self._plan_id_error(plan_id, self.name)) is not None:
            return error
        if not id:
            return "Error: id is required"
        try:
            return self.plan_store.edit_task(
                plan_id,
                session_key,
                id,
                title=title,
                description=description,
                approach=approach,
                acceptance_criteria=acceptance_criteria,
                depends_on=depends_on,
                cancel=cancel,
            )
        except (KeyError, PermissionError, ValueError) as exc:
            return self._format_error(exc)


class TaskProgressTool(_TaskTool):
    exclusive = True
    input_model = TaskProgressInput

    @property
    def name(self) -> str:
        return "task_progress"

    @property
    def description(self) -> str:
        return "Report execution status and progress for a task."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "status": {"type": "string"},
                "progress": {"type": "string"},
            },
            "required": ["id"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        id: str | None = None,
        status: str | None = None,
        progress: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | str:
        if (error := _unknown_arguments(kwargs)) is not None:
            return error
        if (error := self._mode_error(self.name, "execute")) is not None:
            return error
        _, plan_id, session_key, project_id = _mode_and_plan()
        if (error := self._project_id_error(project_id, required=True)) is not None:
            return error
        if (error := self._plan_id_error(plan_id, self.name)) is not None:
            return error
        if not id:
            return "Error: id is required"
        try:
            return self.plan_store.update_progress(
                plan_id,
                session_key,
                id,
                status=status,
                progress=progress,
            )
        except (KeyError, PermissionError, ValueError) as exc:
            return self._format_error(exc)


class TaskListTool(_TaskTool):
    read_only = True
    effect = "local_read"
    input_model = TaskListInput

    def __init__(self, plan_store: Any, legacy_store: TaskStore | None = None) -> None:
        super().__init__(plan_store)
        self.legacy_store = legacy_store

    @property
    def name(self) -> str:
        return "task_list"

    @property
    def description(self) -> str:
        return "List tasks in a project plan, or read legacy tasks explicitly."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "plan_id": {"type": "string"},
                "status": {"type": "string"},
                "legacy": {"type": "boolean"},
            },
            "additionalProperties": False,
        }

    async def execute(
        self,
        plan_id: str | None = None,
        status: str | None = None,
        legacy: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | str:
        if (error := _unknown_arguments(kwargs)) is not None:
            return error
        # Reads are available in every runtime mode; only mutations are
        # restricted to Plan or Execute mode.
        if legacy:
            if self.legacy_store is None:
                return "Error: legacy task store is unavailable"
            try:
                return {"tasks": self.legacy_store.list(status=status)}
            except (KeyError, PermissionError, ValueError) as exc:
                return self._format_error(exc)

        _, current_plan_id, _, project_id = _mode_and_plan()
        if (error := self._project_id_error(project_id)) is not None:
            return error
        # Reads may inspect any plan in this project.  Omitting plan_id keeps
        # the active plan as the convenient default for plan/execute turns.
        effective_plan_id = plan_id if plan_id is not None else current_plan_id
        try:
            return {"tasks": self.plan_store.list_tasks(plan_id=effective_plan_id, status=status)}
        except (KeyError, PermissionError, ValueError) as exc:
            return self._format_error(exc)


class TaskGetTool(_TaskTool):
    read_only = True
    effect = "local_read"
    input_model = TaskGetInput

    def __init__(self, plan_store: Any, legacy_store: TaskStore | None = None) -> None:
        super().__init__(plan_store)
        self.legacy_store = legacy_store

    @property
    def name(self) -> str:
        return "task_get"

    @property
    def description(self) -> str:
        return "Get a task by id, or read a legacy task explicitly."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"id": {"type": "string"}, "legacy": {"type": "boolean"}},
            "required": ["id"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        id: str | None = None,
        legacy: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | str:
        if (error := _unknown_arguments(kwargs)) is not None:
            return error
        # Reads are available in every runtime mode; only mutations are
        # restricted to Plan or Execute mode.
        if not id:
            return "Error: id is required"
        if legacy:
            if self.legacy_store is None:
                return "Error: legacy task store is unavailable"
            try:
                return self.legacy_store.get(id)
            except KeyError:
                return f"Error: task not found: {id}"

        try:
            task = self.plan_store.get_task(id)
        except KeyError:
            return f"Error: task not found: {id}"
        except (PermissionError, ValueError) as exc:
            return self._format_error(exc)
        _, _, _, project_id = _mode_and_plan()
        if (error := self._project_id_error(project_id)) is not None:
            return error
        return task


__all__ = [
    "TaskCreateTool",
    "TaskGetTool",
    "TaskListTool",
    "TaskProgressTool",
    "TaskUpdateTool",
]
