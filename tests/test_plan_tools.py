import asyncio

from pydantic import ValidationError

from myclaw.tools.base import ToolRuntimeContext, tool_context
from myclaw.tools.models import TaskCreateInput, TaskUpdateInput
from myclaw.tools.tasks import TaskCreateTool, TaskProgressTool, TaskUpdateTool


class _PlanStore:
    def __init__(self):
        self.calls = []

    def create_task(self, plan_id, session_key, **kwargs):
        self.calls.append(("create", plan_id, session_key, kwargs))
        return {"id": "task-1", **kwargs}

    def edit_task(self, plan_id, session_key, task_id, **kwargs):
        self.calls.append(("edit", plan_id, session_key, task_id, kwargs))
        return {"id": task_id, **kwargs}

    def update_progress(self, plan_id, session_key, task_id, **kwargs):
        self.calls.append(("progress", plan_id, session_key, task_id, kwargs))
        return {"id": task_id, **kwargs}


class _ScopedPlanStore(_PlanStore):
    project_id = "/expected/project"


def _context(mode):
    return ToolRuntimeContext(
        session_key="session-1",
        metadata={"agent_mode": mode, "current_plan_id": "plan-1", "project_id": "project-1"},
    )


def test_plan_input_models_reject_status_and_metadata():
    try:
        TaskCreateInput.model_validate({"title": "x", "status": "completed"})
    except ValidationError:
        pass
    else:
        raise AssertionError("task_create must reject status")

    try:
        TaskUpdateInput.model_validate({"id": "x", "metadata": {"unsafe": True}})
    except ValidationError:
        pass
    else:
        raise AssertionError("task_update must reject metadata")


def test_plan_create_uses_runtime_plan_and_session():
    store = _PlanStore()
    tool = TaskCreateTool(store)
    with tool_context(_context("Plan")):
        result = asyncio.run(
            tool.execute(
                title="Implement plan",
                description="details",
                approach="small patch",
                acceptance_criteria="tests pass",
            )
        )

    assert result["id"] == "task-1"
    assert store.calls == [
        (
            "create",
            "plan-1",
            "session-1",
            {
                "title": "Implement plan",
                "description": "details",
                "approach": "small patch",
                "acceptance_criteria": "tests pass",
                "depends_on": None,
            },
        )
    ]


def test_mutation_tools_reject_wrong_modes_before_store_call():
    store = _PlanStore()
    create = TaskCreateTool(store)
    update = TaskUpdateTool(store)
    progress = TaskProgressTool(store)

    with tool_context(_context("normal")):
        assert "only in Plan mode" in asyncio.run(create.execute(title="x"))
        assert "only in Plan mode" in asyncio.run(update.execute(id="task-1", title="x"))
    with tool_context(_context("Plan")):
        assert "only in Execute mode" in asyncio.run(progress.execute(id="task-1", progress="halfway"))
    assert store.calls == []


def test_execute_progress_uses_runtime_plan_and_session():
    store = _PlanStore()
    tool = TaskProgressTool(store)
    with tool_context(_context("execute")):
        result = asyncio.run(tool.execute(id="task-1", status="in_progress", progress="halfway"))

    assert result == {"id": "task-1", "status": "in_progress", "progress": "halfway"}
    assert store.calls == [("progress", "plan-1", "session-1", "task-1", {"status": "in_progress", "progress": "halfway"})]


def test_mutation_rejects_a_different_project_scope():
    store = _ScopedPlanStore()
    tool = TaskCreateTool(store)
    context = ToolRuntimeContext(
        session_key="session-1",
        metadata={"agent_mode": "plan", "current_plan_id": "plan-1", "project_id": "/other/project"},
    )
    with tool_context(context):
        result = asyncio.run(tool.execute(title="x"))

    assert "project scope" in result
    assert store.calls == []
