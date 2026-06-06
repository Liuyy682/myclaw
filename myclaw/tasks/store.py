from __future__ import annotations

import fcntl
import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


class TaskStore:
    """Persistent task graph with a one-way state machine, dependency links,
    and multi-instance-safe writes.

    Status flows one direction only::

        pending --> in_progress --> completed
                 \\            \\--> blocked --> in_progress
                  \\--> cancelled        \\--> cancelled

    ``completed`` and ``cancelled`` are terminal. Tasks may declare
    ``depends_on`` edges; the graph is kept acyclic and a task cannot advance to
    ``in_progress``/``completed`` while any dependency is unfinished.

    Every mutation takes an exclusive ``flock`` over ``tasks.json.lock`` and runs
    a fresh load -> mutate -> atomic-write cycle, so concurrent processes cannot
    lose each other's updates.
    """

    VALID_STATUSES = frozenset({"pending", "in_progress", "completed", "blocked", "cancelled"})

    # Allowed forward transitions. same -> same is always permitted (idempotent).
    _TRANSITIONS: dict[str, set[str]] = {
        "pending": {"in_progress", "cancelled"},
        "in_progress": {"completed", "blocked", "cancelled"},
        "blocked": {"in_progress", "cancelled"},
        "completed": set(),
        "cancelled": set(),
    }

    # Legacy status values from earlier versions, mapped on read.
    _LEGACY_STATUS = {"open": "pending", "done": "completed"}

    def __init__(self, workspace: Path | str) -> None:
        self.workspace = Path(workspace).expanduser()
        self.tasks_dir = self.workspace / "tasks"
        self.path = self.tasks_dir / "tasks.json"
        self.lock_path = self.tasks_dir / "tasks.json.lock"
        self.tasks_dir.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        *,
        title: str,
        description: str = "",
        status: str = "pending",
        depends_on: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._validate_status(status)
        clean_title = title.strip()
        if not clean_title:
            raise ValueError("title is required")
        now = datetime.now().isoformat()
        with self._locked():
            tasks = self._load()
            deps = self._normalize_deps(depends_on)
            self._check_deps_exist(deps, tasks)
            task = {
                "id": uuid.uuid4().hex[:12],
                "title": clean_title,
                "description": description,
                "status": status,
                "depends_on": deps,
                "metadata": dict(metadata or {}),
                "created_at": now,
                "updated_at": now,
            }
            # A brand-new task has no inbound edges, so it cannot close a cycle,
            # but validate anyway to stay correct if that ever changes.
            self._check_acyclic(task["id"], deps, tasks)
            if status in ("in_progress", "completed"):
                self._check_deps_satisfied(deps, tasks)
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
        task = self._find(self._load(), task_id)
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
        depends_on: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._locked():
            tasks = self._load()
            index = self._index_of(tasks, task_id)
            if index is None:
                raise KeyError(task_id)
            updated = dict(tasks[index])
            if title is not None:
                if not title.strip():
                    raise ValueError("title must be non-empty")
                updated["title"] = title.strip()
            if description is not None:
                updated["description"] = description
            if depends_on is not None:
                deps = self._normalize_deps(depends_on)
                self._check_deps_exist(deps, tasks, ignore_id=task_id)
                self._check_acyclic(task_id, deps, tasks)
                updated["depends_on"] = deps
            if status is not None:
                self._validate_status(status)
                self._check_transition(updated.get("status", "pending"), status)
                if status in ("in_progress", "completed"):
                    self._check_deps_satisfied(updated.get("depends_on", []), tasks, ignore_id=task_id)
                updated["status"] = status
            updated["updated_at"] = datetime.now().isoformat()
            tasks[index] = updated
            self._save(tasks)
            return updated

    # --- dependency + state-machine validation ------------------------------

    @staticmethod
    def _normalize_deps(depends_on: list[str] | None) -> list[str]:
        if depends_on is None:
            return []
        if not isinstance(depends_on, (list, tuple)):
            raise ValueError("depends_on must be a list of task ids")
        seen: list[str] = []
        for raw in depends_on:
            dep = str(raw).strip()
            if dep and dep not in seen:
                seen.append(dep)
        return seen

    @staticmethod
    def _check_deps_exist(deps: list[str], tasks: list[dict[str, Any]], *, ignore_id: str | None = None) -> None:
        known = {task.get("id") for task in tasks}
        for dep in deps:
            if dep == ignore_id:
                raise ValueError("a task cannot depend on itself")
            if dep not in known:
                raise ValueError(f"dependency not found: {dep}")

    def _check_acyclic(self, task_id: str, deps: list[str], tasks: list[dict[str, Any]]) -> None:
        """Reject deps that would introduce a cycle, given the proposed edges."""
        edges = {task["id"]: list(task.get("depends_on", [])) for task in tasks if "id" in task}
        edges[task_id] = list(deps)

        visiting: set[str] = set()
        done: set[str] = set()

        def visit(node: str, stack: tuple[str, ...]) -> None:
            if node in done:
                return
            if node in visiting:
                chain = " -> ".join(stack + (node,))
                raise ValueError(f"dependency cycle detected: {chain}")
            visiting.add(node)
            for nxt in edges.get(node, []):
                visit(nxt, stack + (node,))
            visiting.discard(node)
            done.add(node)

        visit(task_id, ())

    def _check_deps_satisfied(self, deps: list[str], tasks: list[dict[str, Any]], *, ignore_id: str | None = None) -> None:
        by_id = {task.get("id"): task for task in tasks}
        for dep in deps:
            if dep == ignore_id:
                continue
            target = by_id.get(dep)
            if target is None:
                raise ValueError(f"dependency not found: {dep}")
            if target.get("status") != "completed":
                raise ValueError(f"dependency not completed: {dep}")

    @classmethod
    def _check_transition(cls, current: str, new: str) -> None:
        current = cls._LEGACY_STATUS.get(current, current)
        if current == new:
            return
        allowed = cls._TRANSITIONS.get(current, set())
        if new not in allowed:
            allowed_text = ", ".join(sorted(allowed)) or "(none — terminal state)"
            raise ValueError(f"illegal status transition: {current} -> {new}; allowed: {allowed_text}")

    @classmethod
    def _validate_status(cls, status: str) -> None:
        if status not in cls.VALID_STATUSES:
            raise ValueError(f"status must be one of: {', '.join(sorted(cls.VALID_STATUSES))}")

    # --- persistence ---------------------------------------------------------

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("w", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    @staticmethod
    def _find(tasks: list[dict[str, Any]], task_id: str) -> dict[str, Any] | None:
        for task in tasks:
            if task.get("id") == task_id:
                return task
        return None

    @staticmethod
    def _index_of(tasks: list[dict[str, Any]], task_id: str) -> int | None:
        for index, task in enumerate(tasks):
            if task.get("id") == task_id:
                return index
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
        tasks: list[dict[str, Any]] = []
        for task in data:
            if not isinstance(task, dict):
                continue
            normalized = dict(task)
            current = normalized.get("status")
            if current in self._LEGACY_STATUS:
                normalized["status"] = self._LEGACY_STATUS[current]
            normalized.setdefault("depends_on", [])
            tasks.append(normalized)
        return tasks

    def _save(self, tasks: list[dict[str, Any]]) -> None:
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".json.tmp")
        try:
            tmp_path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp_path, self.path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
