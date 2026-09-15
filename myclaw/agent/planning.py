from __future__ import annotations

"""Data shared by the explicit plan/execute control path."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlanCommand:
    """A command response and an optional prompt for a normal agent turn."""

    content: str
    run_prompt: str | None = None


PLAN_TOOLS = frozenset({
    "read_file",
    "list_dir",
    "glob",
    "grep",
    "ask_user",
    "task_create",
    "task_update",
    "task_get",
    "task_list",
})

EXECUTE_GRAPH_WRITE_TOOLS = frozenset({"task_create", "task_update"})
