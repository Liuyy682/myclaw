import hashlib
import json
import multiprocessing

import pytest

from myclaw.tasks import ProjectPlanStore, TaskStore


def make_store(tmp_path, project="app"):
    root = tmp_path / project
    root.mkdir(exist_ok=True)
    return ProjectPlanStore(tmp_path / "state", root)


def test_plan_lifecycle_and_single_document(tmp_path):
    store = make_store(tmp_path)
    plan = store.create_plan("ship the app", "s1")
    assert plan["project_id"] == store.project_id
    assert store.owned_plan("s1")["id"] == plan["id"]
    task = store.create_task(plan["id"], "s1", title="build", acceptance_criteria="tests pass")
    assert task["status"] == "pending"
    with pytest.raises(ValueError, match="already owns"):
        store.create_plan("second", "s1")
    assert store.confirm_plan(plan["id"], "s1")["mode"] == "execute"
    with pytest.raises(ValueError, match="editable"):
        store.edit_task(plan["id"], "s1", task["id"], title="changed")
    store.release_plan(plan["id"], "s1")
    assert store.owned_plan("s1") is None
    assert store.get_plan(plan["id"])["mode"] == "normal"
    expected = hashlib.sha256(store.project_id.encode()).hexdigest()
    data = json.loads((tmp_path / "state" / "tasks" / "projects" / f"{expected}.json").read_text())
    assert set(data) == {"plans", "tasks"}


def test_entry_and_confirmation_ownership(tmp_path):
    store = make_store(tmp_path)
    plan = store.create_plan("goal", "s1")
    with pytest.raises(ValueError, match="another session"):
        store.enter_plan(plan["id"], "s2")
    assert store.enter_plan(plan["id"], "s1")["pending_confirmation"]
    with pytest.raises(ValueError, match="own"):
        store.confirm_plan(plan["id"], "s2")
    with pytest.raises(ValueError, match="contain"):
        store.confirm_plan(plan["id"], "s1")


def test_dependencies_modes_progress_and_terminal_rules(tmp_path):
    store = make_store(tmp_path)
    plan = store.create_plan("goal", "s1")
    first = store.create_task(plan["id"], "s1", title="first")
    second = store.create_task(plan["id"], "s1", title="second", depends_on=[first["id"]])
    with pytest.raises(ValueError, match="cancel"):
        store.edit_task(plan["id"], "s1", first["id"], cancel=True)
    store.edit_task(plan["id"], "s1", second["id"], depends_on=[])
    store.edit_task(plan["id"], "s1", first["id"], cancel=True)
    with pytest.raises(ValueError, match="execution"):
        store.update_progress(plan["id"], "s1", second["id"], status="in_progress")
    store.confirm_plan(plan["id"], "s1")
    store.update_progress(plan["id"], "s1", second["id"], status="in_progress")
    with pytest.raises(ValueError, match="progress"):
        store.update_progress(plan["id"], "s1", second["id"], status="completed")
    updated = store.update_progress(plan["id"], "s1", second["id"], status="completed", progress="done")
    assert updated["status"] == "completed"
    store.release_plan(plan["id"], "s1")
    store.enter_plan(plan["id"], "s1")
    with pytest.raises(ValueError, match="completed"):
        store.edit_task(plan["id"], "s1", second["id"], title="again")


def test_taskstore_metadata_update_is_persisted(tmp_path):
    store = TaskStore(tmp_path)
    task = store.create(title="x", metadata={"a": 1})
    store.update(task["id"], metadata={"a": 2})
    assert TaskStore(tmp_path).get(task["id"])["metadata"] == {"a": 2}


def test_replanning_rejects_completed_nodes_and_unmet_active_dependencies(tmp_path):
    store = make_store(tmp_path)
    plan = store.create_plan("goal", "s1")
    completed = store.create_task(plan["id"], "s1", title="completed")
    active = store.create_task(plan["id"], "s1", title="active")
    dependency = store.create_task(plan["id"], "s1", title="dependency")
    store.confirm_plan(plan["id"], "s1")
    store.update_progress(plan["id"], "s1", completed["id"], status="in_progress")
    store.update_progress(plan["id"], "s1", completed["id"], status="completed", progress="done")
    store.update_progress(plan["id"], "s1", active["id"], status="in_progress")
    store.release_plan(plan["id"], "s1")
    store.enter_plan(plan["id"], "s1")
    with pytest.raises(ValueError, match="completed"):
        store.edit_task(plan["id"], "s1", completed["id"], title="still editable?")
    # An active task may be re-pointed while planning, but it cannot be
    # reconfirmed until that new dependency is satisfied.
    store.edit_task(plan["id"], "s1", active["id"], depends_on=[dependency["id"]])
    with pytest.raises(ValueError, match="dependency not completed"):
        store.confirm_plan(plan["id"], "s1")


def test_corrupt_state_fails_closed_without_overwriting_file(tmp_path):
    store = make_store(tmp_path)
    store.path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable"):
        store.list_plans()
    assert store.path.read_text(encoding="utf-8") == "{not json"


def test_blocked_tasks_may_be_reconfirmed_with_unmet_dependencies(tmp_path):
    store = make_store(tmp_path)
    plan = store.create_plan("goal", "s1")
    blocked = store.create_task(plan["id"], "s1", title="blocked")
    dependency = store.create_task(plan["id"], "s1", title="dependency")
    store.confirm_plan(plan["id"], "s1")
    store.update_progress(plan["id"], "s1", blocked["id"], status="in_progress")
    store.update_progress(plan["id"], "s1", blocked["id"], status="blocked", progress="waiting")
    store.release_plan(plan["id"], "s1")
    store.enter_plan(plan["id"], "s1")
    store.edit_task(plan["id"], "s1", blocked["id"], depends_on=[dependency["id"]])
    assert store.confirm_plan(plan["id"], "s1")["mode"] == "execute"


def test_execute_cannot_cancel_or_rewrite_terminal_progress(tmp_path):
    store = make_store(tmp_path)
    plan = store.create_plan("goal", "s1")
    task = store.create_task(plan["id"], "s1", title="work")
    store.confirm_plan(plan["id"], "s1")
    store.update_progress(plan["id"], "s1", task["id"], status="in_progress")
    with pytest.raises(ValueError, match="plan mode"):
        store.update_progress(plan["id"], "s1", task["id"], status="cancelled")
    store.update_progress(plan["id"], "s1", task["id"], status="completed", progress="done")
    with pytest.raises(ValueError, match="terminal"):
        store.update_progress(plan["id"], "s1", task["id"], progress="changed")
    assert store.update_progress(plan["id"], "s1", task["id"], progress="done")["progress"] == "done"


def _concurrent_plan_worker(state, root, session):
    ProjectPlanStore(state, root).create_plan("goal", session)


def test_competing_instances_do_not_lose_plan_writes(tmp_path):
    root = tmp_path / "app"
    root.mkdir()
    state = tmp_path / "state"
    ctx = multiprocessing.get_context("spawn")
    workers = [ctx.Process(target=_concurrent_plan_worker, args=(str(state), str(root), f"s{i}")) for i in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30)
        assert worker.exitcode == 0
    assert len(ProjectPlanStore(state, root).list_plans()) == 2


def _claim_plan(state, root, plan_id, session, barrier, results):
    store = ProjectPlanStore(state, root)
    barrier.wait(timeout=10)
    try:
        store.enter_plan(plan_id, session)
        results.put((session, True))
    except ValueError:
        results.put((session, False))


def test_only_one_process_can_claim_released_plan(tmp_path):
    store = make_store(tmp_path)
    plan = store.create_plan("shared", "original")
    store.release_plan(plan["id"], "original")
    ctx = multiprocessing.get_context("spawn")
    barrier, results = ctx.Barrier(2), ctx.Queue()
    workers = [ctx.Process(target=_claim_plan, args=(
        str(store.state_workspace), str(store.project_root), plan["id"], f"claim-{i}", barrier, results,
    )) for i in range(2)]
    for worker in workers:
        worker.start()
    outcomes = [results.get(timeout=15) for _ in workers]
    for worker in workers:
        worker.join(timeout=15)
        assert worker.exitcode == 0
    winners = [session for session, success in outcomes if success]
    assert len(winners) == 1
    assert store.get_plan(plan["id"])["owner_session"] == winners[0]
    results.close()


def test_project_identity_and_plan_mutation_scope(tmp_path):
    first = make_store(tmp_path, "first")
    alias = tmp_path / "alias"
    alias.symlink_to(first.project_root, target_is_directory=True)
    same_project = ProjectPlanStore(first.state_workspace, alias)
    different = make_store(tmp_path, "second")
    plan = first.create_plan("shared", "one")
    task = first.create_task(plan["id"], "one", title="first task")
    assert same_project.list_plans()[0]["id"] == plan["id"]
    assert different.list_plans() == []
    with pytest.raises(KeyError):
        different.get_task(task["id"])
    other_plan = first.create_plan("other", "two")
    with pytest.raises(ValueError, match="same plan"):
        first.create_task(other_plan["id"], "two", title="cross dependency", depends_on=[task["id"]])
    with pytest.raises(ValueError, match="belong"):
        first.edit_task(other_plan["id"], "two", task["id"], title="cross edit")
    with pytest.raises(ValueError, match="own"):
        first.edit_task(plan["id"], "two", task["id"], title="wrong session")
    first.release_plan(other_plan["id"], "two")
    with pytest.raises(ValueError, match="already owns"):
        first.enter_plan(other_plan["id"], "one")
