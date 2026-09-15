"""Persistent, project-scoped plan and task state.

Plans and their tasks deliberately live in one JSON document.  A project has
one document and one lock, which makes ownership checks and graph validation a
single read-modify-write operation across multiple ``ProjectPlanStore``
instances.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from myclaw.tasks.store import TaskStore


class ProjectPlanStore:
    """Store approved plans and their task graphs for one project."""

    def __init__(self, state_workspace: Path | str, project_root: Path | str) -> None:
        self.state_workspace = Path(state_workspace).expanduser().resolve()
        self.project_root = Path(project_root).expanduser().resolve()
        self._project_id = str(self.project_root)
        digest = hashlib.sha256(self._project_id.encode("utf-8")).hexdigest()
        self.projects_dir = self.state_workspace / "tasks" / "projects"
        self.path = self.projects_dir / f"{digest}.json"
        self.lock_path = self.projects_dir / f"{digest}.json.lock"
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    @property
    def project_id(self) -> str:
        return self._project_id

    # --- plans -------------------------------------------------------------

    def create_plan(self, goal: str, session_key: str) -> dict[str, Any]:
        clean_goal = self._required_text(goal, "goal")
        owner = self._required_text(session_key, "session_key")
        with self._locked():
            state = self._load()
            if self._owned_plan(state, owner) is not None:
                raise ValueError("session already owns a plan; release it before creating another")
            now = datetime.now().isoformat()
            plan = {
                "id": uuid.uuid4().hex[:12],
                "project_id": self.project_id,
                "goal": clean_goal,
                "owner_session": owner,
                "pending_confirmation": True,
                "mode": "plan",
                "created_at": now,
                "updated_at": now,
            }
            state["plans"].append(plan)
            self._save(state)
            return plan

    def list_plans(self) -> list[dict[str, Any]]:
        with self._locked():
            return list(self._load()["plans"])

    def get_plan(self, plan_id: str) -> dict[str, Any]:
        with self._locked():
            plan = self._find_plan(self._load(), plan_id)
            if plan is None:
                raise KeyError(plan_id)
            return plan

    def enter_plan(self, plan_id: str, session_key: str) -> dict[str, Any]:
        owner = self._required_text(session_key, "session_key")
        with self._locked():
            state = self._load()
            plan = self._require_plan(state, plan_id)
            current_owner = plan.get("owner_session")
            if current_owner not in (None, owner):
                raise ValueError("plan is owned by another session")
            if current_owner is None:
                other = self._owned_plan(state, owner)
                if other is not None and other.get("id") != plan_id:
                    raise ValueError("session already owns another plan; release it before entering")
            plan["owner_session"] = owner
            plan["pending_confirmation"] = True
            plan["mode"] = "plan"
            self._touch(plan)
            self._save(state)
            return plan

    def confirm_plan(self, plan_id: str, session_key: str) -> dict[str, Any]:
        owner = self._required_text(session_key, "session_key")
        with self._locked():
            state = self._load()
            plan = self._require_plan(state, plan_id)
            self._require_owner(plan, owner)
            if plan.get("mode") != "plan":
                raise ValueError("plan must be in plan mode")
            if not self._required_text(plan.get("goal", ""), "goal"):
                raise ValueError("plan goal is required")
            tasks = self._tasks_for(state, plan_id)
            if not tasks:
                raise ValueError("plan must contain at least one task")
            # A plan can only be executed with a valid pending graph.  This
            # also catches malformed state written by an older implementation.
            self._validate_plan_tasks(tasks, plan_id)
            plan["pending_confirmation"] = False
            plan["mode"] = "execute"
            self._touch(plan)
            self._save(state)
            return plan

    def release_plan(self, plan_id: str, session_key: str) -> dict[str, Any]:
        owner = self._required_text(session_key, "session_key")
        with self._locked():
            state = self._load()
            plan = self._require_plan(state, plan_id)
            self._require_owner(plan, owner)
            plan["owner_session"] = None
            plan["mode"] = "normal"
            plan["pending_confirmation"] = True
            self._touch(plan)
            self._save(state)
            return plan

    def owned_plan(self, session_key: str) -> dict[str, Any] | None:
        owner = self._required_text(session_key, "session_key")
        with self._locked():
            return self._owned_plan(self._load(), owner)

    # --- tasks -------------------------------------------------------------

    def create_task(
        self,
        plan_id: str,
        session_key: str,
        *,
        title: str,
        description: str = "",
        approach: str = "",
        acceptance_criteria: str = "",
        depends_on: list[str] | None = None,
    ) -> dict[str, Any]:
        owner = self._required_text(session_key, "session_key")
        clean_title = self._required_text(title, "title")
        with self._locked():
            state = self._load()
            plan = self._require_editable_plan(state, plan_id, owner)
            tasks = state["tasks"]
            deps = self._normalize_deps(depends_on)
            self._check_plan_dependencies(deps, plan_id, tasks)
            task_id = uuid.uuid4().hex[:12]
            # Reuse TaskStore's graph validator for consistency with legacy
            # tasks, while the project store owns persistence and plan scope.
            TaskStore._check_acyclic(self, task_id, deps, tasks)
            now = datetime.now().isoformat()
            task = {
                "id": task_id,
                "plan_id": plan_id,
                "title": clean_title,
                "description": description,
                "approach": approach,
                "acceptance_criteria": acceptance_criteria,
                "status": "pending",
                "progress": "",
                "depends_on": deps,
                "created_at": now,
                "updated_at": now,
            }
            tasks.append(task)
            self._touch(plan)
            self._save(state)
            return task

    def edit_task(
        self,
        plan_id: str,
        session_key: str,
        task_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
        approach: str | None = None,
        acceptance_criteria: str | None = None,
        depends_on: list[str] | None = None,
        cancel: bool = False,
    ) -> dict[str, Any]:
        owner = self._required_text(session_key, "session_key")
        with self._locked():
            state = self._load()
            plan = self._require_editable_plan(state, plan_id, owner)
            task = self._require_task(state, task_id, plan_id)
            if task.get("status") in {"completed", "cancelled"}:
                raise ValueError("completed or cancelled tasks cannot be edited")
            if title is not None:
                task["title"] = self._required_text(title, "title")
            if description is not None:
                task["description"] = description
            if approach is not None:
                task["approach"] = approach
            if acceptance_criteria is not None:
                task["acceptance_criteria"] = acceptance_criteria
            if depends_on is not None:
                deps = self._normalize_deps(depends_on)
                self._check_plan_dependencies(deps, plan_id, state["tasks"], ignore_id=task_id)
                TaskStore._check_acyclic(self, task_id, deps, state["tasks"])
                task["depends_on"] = deps
            if cancel:
                dependents = [
                    candidate
                    for candidate in state["tasks"]
                    if task_id in candidate.get("depends_on", [])
                    and candidate.get("status") != "cancelled"
                ]
                if dependents:
                    raise ValueError("cannot cancel while non-cancelled dependents remain")
                TaskStore._check_transition(task.get("status", "pending"), "cancelled")
                task["status"] = "cancelled"
            task["updated_at"] = datetime.now().isoformat()
            self._touch(plan)
            self._save(state)
            return task

    def update_progress(
        self,
        plan_id: str,
        session_key: str,
        task_id: str,
        *,
        status: str | None = None,
        progress: Any = None,
    ) -> dict[str, Any]:
        owner = self._required_text(session_key, "session_key")
        with self._locked():
            state = self._load()
            plan = self._require_execute_plan(state, plan_id, owner)
            task = self._require_task(state, task_id, plan_id)
            current_status = task.get("status", "pending")
            if status == "cancelled":
                raise ValueError("cancelled status requires plan mode cancellation")
            if current_status in {"completed", "cancelled"}:
                requested_status = current_status if status is None else status
                if requested_status != current_status:
                    TaskStore._check_transition(current_status, requested_status)
                if progress is None or progress == task.get("progress"):
                    return task
                raise ValueError("terminal task progress cannot be changed")
            if progress is not None:
                task["progress"] = progress
            new_status = status if status is not None else task.get("status", "pending")
            TaskStore._validate_status(new_status)
            TaskStore._check_transition(task.get("status", "pending"), new_status)
            if new_status in {"in_progress", "completed"}:
                TaskStore._check_deps_satisfied(self, task.get("depends_on", []), state["tasks"], ignore_id=task_id)
            if new_status == "completed" and not self._nonempty(task.get("progress")):
                raise ValueError("completed task requires non-empty progress")
            task["status"] = new_status
            task["updated_at"] = datetime.now().isoformat()
            self._touch(plan)
            self._save(state)
            return task

    def list_tasks(self, plan_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        with self._locked():
            tasks = self._load()["tasks"]
            if plan_id is not None:
                tasks = [task for task in tasks if task.get("plan_id") == plan_id]
            if status is not None:
                TaskStore._validate_status(status)
                tasks = [task for task in tasks if task.get("status") == status]
            return tasks

    def get_task(self, task_id: str) -> dict[str, Any]:
        with self._locked():
            task = self._find_task(self._load()["tasks"], task_id)
            if task is None:
                raise KeyError(task_id)
            return task

    # --- state and validation ---------------------------------------------

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _load(self) -> dict[str, list[dict[str, Any]]]:
        if not self.path.exists():
            return {"plans": [], "tasks": []}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("project plan state is unreadable") from exc
        if not isinstance(data, dict):
            raise ValueError("project plan state must be a JSON object")
        if "plans" not in data or "tasks" not in data:
            raise ValueError("project plan state must contain plans and tasks")
        plans = data["plans"]
        tasks = data["tasks"]
        if not isinstance(plans, list) or not isinstance(tasks, list):
            raise ValueError("project plan state must contain plans and tasks lists")
        if any(not isinstance(item, dict) for item in plans + tasks):
            raise ValueError("project plan state contains an invalid record")
        return {
            "plans": [dict(item) for item in plans],
            "tasks": [dict(item) for item in tasks],
        }

    def _save(self, state: dict[str, list[dict[str, Any]]]) -> None:
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, self.path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    @staticmethod
    def _required_text(value: Any, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} is required")
        return value.strip()

    @staticmethod
    def _nonempty(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (dict, list, tuple, set)):
            return bool(value)
        return True

    @staticmethod
    def _normalize_deps(depends_on: list[str] | None) -> list[str]:
        return TaskStore._normalize_deps(depends_on)

    @staticmethod
    def _find_plan(state: dict[str, list[dict[str, Any]]], plan_id: str) -> dict[str, Any] | None:
        return next((plan for plan in state["plans"] if plan.get("id") == plan_id), None)

    @staticmethod
    def _find_task(tasks: list[dict[str, Any]], task_id: str) -> dict[str, Any] | None:
        return next((task for task in tasks if task.get("id") == task_id), None)

    @classmethod
    def _owned_plan(cls, state: dict[str, list[dict[str, Any]]], session_key: str) -> dict[str, Any] | None:
        return next((plan for plan in state["plans"] if plan.get("owner_session") == session_key), None)

    def _require_plan(self, state: dict[str, list[dict[str, Any]]], plan_id: str) -> dict[str, Any]:
        plan = self._find_plan(state, plan_id)
        if plan is None:
            raise KeyError(plan_id)
        return plan

    @staticmethod
    def _require_owner(plan: dict[str, Any], session_key: str) -> None:
        if plan.get("owner_session") != session_key:
            raise ValueError("session does not own plan")

    def _require_editable_plan(
        self, state: dict[str, list[dict[str, Any]]], plan_id: str, session_key: str
    ) -> dict[str, Any]:
        plan = self._require_plan(state, plan_id)
        self._require_owner(plan, session_key)
        if plan.get("mode") != "plan" or not plan.get("pending_confirmation", False):
            raise ValueError("plan is not editable in its current mode")
        return plan

    def _require_execute_plan(
        self, state: dict[str, list[dict[str, Any]]], plan_id: str, session_key: str
    ) -> dict[str, Any]:
        plan = self._require_plan(state, plan_id)
        self._require_owner(plan, session_key)
        if plan.get("mode") != "execute" or plan.get("pending_confirmation", True):
            raise ValueError("plan is not confirmed for execution")
        return plan

    @staticmethod
    def _tasks_for(state: dict[str, list[dict[str, Any]]], plan_id: str) -> list[dict[str, Any]]:
        return [task for task in state["tasks"] if task.get("plan_id") == plan_id]

    def _require_task(
        self, state: dict[str, list[dict[str, Any]]], task_id: str, plan_id: str
    ) -> dict[str, Any]:
        task = self._find_task(state["tasks"], task_id)
        if task is None:
            raise KeyError(task_id)
        if task.get("plan_id") != plan_id:
            raise ValueError("task does not belong to plan")
        return task

    def _check_plan_dependencies(
        self,
        deps: list[str],
        plan_id: str,
        tasks: list[dict[str, Any]],
        *,
        ignore_id: str | None = None,
    ) -> None:
        known = {task.get("id"): task for task in tasks}
        for dep in deps:
            if dep == ignore_id:
                raise ValueError("a task cannot depend on itself")
            target = known.get(dep)
            if target is None:
                raise ValueError(f"dependency not found: {dep}")
            if target.get("plan_id") != plan_id:
                raise ValueError("dependency must belong to the same plan")

    def _validate_plan_tasks(self, tasks: list[dict[str, Any]], plan_id: str) -> None:
        for task in tasks:
            if task.get("plan_id") != plan_id:
                raise ValueError("task belongs to another plan")
            TaskStore._validate_status(task.get("status", "pending"))
            deps = self._normalize_deps(task.get("depends_on", []))
            self._check_plan_dependencies(deps, plan_id, tasks, ignore_id=task.get("id"))
            TaskStore._check_acyclic(self, task.get("id", ""), deps, tasks)
            if task.get("status") in {"in_progress", "completed"}:
                TaskStore._check_deps_satisfied(self, deps, tasks, ignore_id=task.get("id"))

    @staticmethod
    def _touch(plan: dict[str, Any]) -> None:
        plan["updated_at"] = datetime.now().isoformat()


__all__ = ["ProjectPlanStore"]
