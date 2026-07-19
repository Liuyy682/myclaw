import multiprocessing

import pytest

from myclaw.tasks import TaskStore


def test_state_machine_allows_forward_and_rejects_illegal_transitions(tmp_path):
    store = TaskStore(tmp_path)
    task = store.create(title="ship")

    assert task["status"] == "pending"
    assert store.update(task["id"], status="in_progress")["status"] == "in_progress"
    assert store.update(task["id"], status="completed")["status"] == "completed"

    # completed is terminal: cannot reopen.
    with pytest.raises(ValueError, match="illegal status transition"):
        store.update(task["id"], status="in_progress")


def test_state_machine_rejects_skipping_straight_to_completed(tmp_path):
    store = TaskStore(tmp_path)
    task = store.create(title="skip")

    with pytest.raises(ValueError, match="illegal status transition"):
        store.update(task["id"], status="completed")


def test_same_status_update_is_idempotent(tmp_path):
    store = TaskStore(tmp_path)
    task = store.create(title="idem", status="in_progress")  # in_progress with no deps is fine
    assert store.update(task["id"], status="in_progress")["status"] == "in_progress"


def test_legacy_status_values_are_mapped_on_read(tmp_path):
    store = TaskStore(tmp_path)
    store.tasks_dir.mkdir(parents=True, exist_ok=True)
    store.path.write_text(
        '[{"id": "abc", "title": "old", "status": "open"},'
        ' {"id": "def", "title": "older", "status": "done"}]',
        encoding="utf-8",
    )

    assert store.get("abc")["status"] == "pending"
    assert store.get("def")["status"] == "completed"
    # A legacy "open" task can still advance through the modern machine.
    assert store.update("abc", status="in_progress")["status"] == "in_progress"


def test_dependency_must_exist(tmp_path):
    store = TaskStore(tmp_path)
    with pytest.raises(ValueError, match="dependency not found"):
        store.create(title="needs", depends_on=["missing"])


def test_dependency_cycle_is_rejected(tmp_path):
    store = TaskStore(tmp_path)
    a = store.create(title="a")
    b = store.create(title="b", depends_on=[a["id"]])

    # a -> b would close the cycle a -> b -> a.
    with pytest.raises(ValueError, match="dependency cycle detected"):
        store.update(a["id"], depends_on=[b["id"]])


def test_cannot_start_until_dependencies_completed(tmp_path):
    store = TaskStore(tmp_path)
    dep = store.create(title="dep")
    task = store.create(title="main", depends_on=[dep["id"]])

    with pytest.raises(ValueError, match="dependency not completed"):
        store.update(task["id"], status="in_progress")

    store.update(dep["id"], status="in_progress")
    store.update(dep["id"], status="completed")

    # Now the dependency is satisfied, the task may start.
    assert store.update(task["id"], status="in_progress")["status"] == "in_progress"


def _spawn_worker(workspace: str, count: int) -> None:
    store = TaskStore(workspace)
    for index in range(count):
        store.create(title=f"task-{index}")


def test_file_lock_prevents_lost_updates_across_processes(tmp_path):
    per_worker = 25
    ctx = multiprocessing.get_context("spawn")
    workers = [
        ctx.Process(target=_spawn_worker, args=(str(tmp_path), per_worker)),
        ctx.Process(target=_spawn_worker, args=(str(tmp_path), per_worker)),
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30)
        assert worker.exitcode == 0

    tasks = TaskStore(tmp_path).list()
    # Without the exclusive lock the two read-modify-write cycles would clobber
    # each other and we would see fewer than 2 * per_worker tasks.
    assert len(tasks) == 2 * per_worker
    assert len({task["id"] for task in tasks}) == 2 * per_worker
