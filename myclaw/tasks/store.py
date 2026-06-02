from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


class TaskStore:
    VALID_STATUSES = frozenset({"open", "in_progress", "done", "blocked", "cancelled"})

    def __init__(self, workspace: Path | str) -> None:
        self.workspace = Path(workspace).expanduser()
        self.tasks_dir = self.workspace / "tasks"
        self.path = self.tasks_dir / "tasks.json"
        self.tasks_dir.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        *,
        title: str,
        description: str = "",
        status: str = "open",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._validate_status(status)
        now = datetime.now().isoformat()
        task = {
            "id": uuid.uuid4().hex[:12],
            "title": title.strip(),
            "description": description,
            "status": status,
            "metadata": dict(metadata or {}),
            "created_at": now,
            "updated_at": now,
        }
        if not task["title"]:
            raise ValueError("title is required")
        tasks = self.list()
        tasks.append(task)
        self._save(tasks)
        return task

    def list(self, *, status: str | None = None) -> list[dict[str, Any]]:
        tasks = self._load()
        if status is None:
            return tasks
        self._validate_status(status)
        return [task for task in tasks if task.get("status") == status]

    def get(self, task_id: str) -> dict[str, Any]:
        task = self._find(task_id)
        if task is None:
            raise KeyError(task_id)
        return task

    def update(
        self,
        task_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
        status: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        tasks = self._load()
        for index, task in enumerate(tasks):
            if task.get("id") != task_id:
                continue
            updated = dict(task)
            if title is not None:
                if not title.strip():
                    raise ValueError("title must be non-empty")
                updated["title"] = title.strip()
            if description is not None:
                updated["description"] = description
            if status is not None:
                self._validate_status(status)
                updated["status"] = status
            if metadata is not None:
                updated["metadata"] = dict(metadata)
            updated["updated_at"] = datetime.now().isoformat()
            tasks[index] = updated
            self._save(tasks)
            return updated
        raise KeyError(task_id)

    def _find(self, task_id: str) -> dict[str, Any] | None:
        for task in self._load():
            if task.get("id") == task_id:
                return task
        return None

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []
        return [task for task in data if isinstance(task, dict)]

    def _save(self, tasks: list[dict[str, Any]]) -> None:
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".json.tmp")
        try:
            tmp_path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp_path, self.path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    @classmethod
    def _validate_status(cls, status: str) -> None:
        if status not in cls.VALID_STATUSES:
            raise ValueError(f"status must be one of: {', '.join(sorted(cls.VALID_STATUSES))}")
