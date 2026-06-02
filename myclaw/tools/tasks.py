from __future__ import annotations

from typing import Any

from myclaw.tasks import TaskStore
from myclaw.tools.base import Tool


class _TaskTool(Tool):
    read_only = False
    exclusive = False

    def __init__(self, store: TaskStore) -> None:
        self.store = store


class TaskCreateTool(_TaskTool):
    @property
    def name(self) -> str:
        return "task_create"

    @property
    def description(self) -> str:
        return "Create a persistent task in the workspace task store."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "status": {"type": "string"},
            },
            "required": ["title"],
        }

    async def execute(
        self,
        title: str | None = None,
        description: str = "",
        status: str = "open",
        **kwargs: Any,
    ) -> dict[str, Any] | str:
        try:
            return self.store.create(title="" if title is None else title, description=description, status=status)
        except ValueError as exc:
            return f"Error: {exc}"


class TaskListTool(_TaskTool):
    read_only = True

    @property
    def name(self) -> str:
        return "task_list"

    @property
    def description(self) -> str:
        return "List persistent tasks, optionally filtered by status."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
            },
        }

    async def execute(self, status: str | None = None, **kwargs: Any) -> dict[str, Any] | str:
        try:
            return {"tasks": self.store.list(status=status)}
        except ValueError as exc:
            return f"Error: {exc}"


class TaskGetTool(_TaskTool):
    read_only = True

    @property
    def name(self) -> str:
        return "task_get"

    @property
    def description(self) -> str:
        return "Get one persistent task by id."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        }

    async def execute(self, id: str | None = None, **kwargs: Any) -> dict[str, Any] | str:
        if not id:
            return "Error: id is required"
        try:
            return self.store.get(id)
        except KeyError:
            return f"Error: task not found: {id}"


class TaskUpdateTool(_TaskTool):
    @property
    def name(self) -> str:
        return "task_update"

    @property
    def description(self) -> str:
        return "Update title, description, status, or metadata for a persistent task."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "title": {"type": "string"},
                "description": {"type": "string"},
                "status": {"type": "string"},
                "metadata": {"type": "object"},
            },
            "required": ["id"],
        }

    async def execute(
        self,
        id: str | None = None,
        title: str | None = None,
        description: str | None = None,
        status: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | str:
        if not id:
            return "Error: id is required"
        try:
            return self.store.update(
                id,
                title=title,
                description=description,
                status=status,
                metadata=metadata,
            )
        except KeyError:
            return f"Error: task not found: {id}"
        except ValueError as exc:
            return f"Error: {exc}"
