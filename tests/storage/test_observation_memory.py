import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from myclaw.memory import ObservationMemoryStore
from myclaw.memory.observation import LeaseError


def test_schema_uses_workspace_memory_wal_and_stable_ids(tmp_path):
    store = ObservationMemoryStore(tmp_path)
    first = store.enqueue_turn("session-a", "turn-1", [{"role": "user", "content": "hello"}])
    second = ObservationMemoryStore(tmp_path).enqueue_turn(
        "session-a", "turn-1", [{"role": "user", "content": "hello"}]
    )

    assert store.path == tmp_path / "memory" / "observation_memory.db"
    assert store.path.exists()
    assert first["id"] == second["id"]
    assert isinstance(first["id"], str)
    assert len(first["input_hash"]) == 64
    assert first["worker_version"] == "observation-v1"
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert {"source_events", "observation_jobs", "observations", "reflections"} <= tables


def test_schema_upgrades_existing_jobs_with_processing_identity(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    path = memory_dir / "observation_memory.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE observation_jobs (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                event_ids_json TEXT NOT NULL DEFAULT '[]',
                first_sequence INTEGER,
                last_sequence INTEGER,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                lease_owner TEXT,
                lease_until TEXT,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT,
                UNIQUE(session_id, idempotency_key)
            )
            """
        )

    store = ObservationMemoryStore(tmp_path)

    with sqlite3.connect(store.path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(observation_jobs)")
        }
        indexes = {
            row[1] for row in connection.execute("PRAGMA index_list(observation_jobs)")
        }
    assert {"input_hash", "worker_version"} <= columns
    assert "idx_observation_jobs_turn_input_worker" in indexes


def test_enqueue_turn_is_atomic_and_does_not_mutate_immutable_events(tmp_path):
    store = ObservationMemoryStore(tmp_path)
    events = [
        {"id": "event-1", "role": "user", "content": "first"},
        {"id": "event-2", "role": "assistant", "content": "second"},
    ]
    job = store.enqueue_turn("s", "t", events)
    duplicate = store.enqueue_turn("s", "t", list(events))

    assert duplicate["id"] == job["id"]
    assert duplicate["event_ids"] == ["event-1", "event-2"]
    with sqlite3.connect(store.path) as connection:
        rows = connection.execute(
            "SELECT id, event_index, content FROM source_events ORDER BY event_index"
        ).fetchall()
    assert rows == [("event-1", 0, "first"), ("event-2", 1, "second")]

    with pytest.raises(ValueError, match="different events"):
        store.enqueue_turn("s", "t", [{"id": "event-1", "content": "changed"}, events[1]])


def test_claim_lease_and_expiry_reclaim_limit(tmp_path):
    store = ObservationMemoryStore(tmp_path, lease_seconds=10)
    job = store.enqueue_turn("s", "t", [{"id": "e", "content": "fact"}])
    start = datetime(2026, 1, 1, tzinfo=UTC)

    first = store.claim_pending(worker_id="worker-1", now=start)
    assert first["id"] == job["id"]
    assert first["status"] == "processing"
    assert first["attempts"] == 1
    assert first["lease_until"] == "2026-01-01T00:00:10+00:00"
    assert store.claim_pending(worker_id="worker-2", now=start) is None

    second = store.claim_pending(worker_id="worker-2", now=start + timedelta(seconds=10))
    assert second["attempts"] == 2
    assert second["lease_owner"] == "worker-2"
    assert store.fail(second, "temporary", worker_id="worker-2", now=start + timedelta(seconds=11))["status"] == "pending"

    third = store.claim_pending(worker_id="worker-3", now=start + timedelta(seconds=12))
    assert third["attempts"] == 3
    failed = store.fail(third, "permanent", worker_id="worker-3", now=start + timedelta(seconds=13))
    assert failed["status"] == "failed"
    assert store.claim_pending(worker_id="worker-4", now=start + timedelta(seconds=100)) is None


def test_stale_worker_cannot_complete_or_fail_a_reclaimed_job(tmp_path):
    store = ObservationMemoryStore(tmp_path, lease_seconds=5)
    job = store.enqueue_turn("s", "t", [{"id": "e", "content": "fact"}])
    start = datetime(2026, 1, 1, tzinfo=UTC)
    store.claim_pending(worker_id="old", now=start)
    current = store.claim_pending(worker_id="new", now=start + timedelta(seconds=5))

    with pytest.raises(LeaseError):
        store.complete(job["id"], ["fact"], worker_id="old", now=start + timedelta(seconds=6))
    with pytest.raises(LeaseError):
        store.fail(job["id"], "late", worker_id="old", now=start + timedelta(seconds=6))
    assert current["lease_owner"] == "new"


def test_complete_and_no_facts_advance_only_safe_watermark(tmp_path):
    store = ObservationMemoryStore(tmp_path)
    first = store.enqueue_turn("s", "t-1", [{"id": "e-1", "content": "one"}])
    second = store.enqueue_turn("s", "t-2", [{"id": "e-2", "content": "two"}])
    third = store.enqueue_turn("s", "t-3", [{"id": "e-3", "content": "three"}])
    now = datetime(2026, 1, 1, tzinfo=UTC)

    store.claim_pending(worker_id="w", now=now)
    store.complete(first, [{"content": "one is durable", "relevance": "high"}], worker_id="w", now=now)
    assert store.safe_watermark("s") == "e-1"

    store.claim_pending(worker_id="w", now=now)
    store.fail(second, "retry", worker_id="w", now=now)
    assert store.safe_watermark("s") == "e-1"

    store.claim_pending(worker_id="w", now=now)
    store.no_facts(second, worker_id="w", now=now)
    assert store.safe_watermark_info("s") == {"session_id": "s", "event_id": "e-2", "sequence": 1}

    store.claim_pending(worker_id="w", now=now)
    store.complete(third, [], worker_id="w", now=now)
    assert store.safe_watermark("s") == "e-3"
    assert store.active_observations("s")[0]["content"] == "one is durable"


def test_complete_is_idempotent_and_writes_stable_observation(tmp_path):
    store = ObservationMemoryStore(tmp_path)
    store.enqueue_turn("s", "t", [{"id": "e", "content": "source"}])
    now = datetime(2026, 1, 1, tzinfo=UTC)
    claimed = store.claim_pending(worker_id="w", now=now)
    result = store.complete(
        claimed,
        [{"content": "User prefers concise answers.", "source_event_ids": ["e"]}],
        worker_id="w",
        now=now,
    )
    repeated = store.complete(claimed, [{"content": "ignored"}], worker_id="w", now=now)

    assert result["status"] == repeated["status"] == "completed"
    observations = store.list_observations("s")
    assert len(observations) == 1
    assert observations[0]["source_event_ids"] == ["e"]
    assert observations[0]["id"] == store.list_observations("s")[0]["id"]


def test_reflection_atomically_records_support_and_archives_observations(tmp_path):
    store = ObservationMemoryStore(tmp_path)
    first = store.write_observation("s", "fact one", ["e-1"])
    second = store.write_observation("s", "fact two", ["e-2"])

    reflection = store.record_reflection(
        "s", "User prefers source-backed facts.", [second["id"], first["id"]]
    )
    assert reflection["supporting_observation_ids"] == [second["id"], first["id"]]
    assert store.active_observations("s") == []
    assert store.active_reflections("s")[0]["content"] == "User prefers source-backed facts."
    assert {item["status"] for item in store.list_observations("s", active_only=False)} == {"archived"}

    repeated = store.record_reflection(
        "s", "User prefers source-backed facts.", [second["id"], first["id"]]
    )
    assert repeated["id"] == reflection["id"]
    assert len(store.list_reflections("s", active_only=False)) == 1


def test_reflection_rolls_back_when_support_is_missing(tmp_path):
    store = ObservationMemoryStore(tmp_path)
    with pytest.raises(KeyError, match="supporting observation not found"):
        store.record_reflection("s", "broken", ["missing"])
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM reflections").fetchone()[0] == 0


def test_reflection_supersedes_only_existing_reflection_in_same_session(tmp_path):
    store = ObservationMemoryStore(tmp_path)
    old_support = store.write_observation("s", "old fact", ["e-1"])
    old = store.record_reflection("s", "old conclusion", [old_support["id"]])
    new_support = store.write_observation("s", "corrected fact", ["e-2"])

    replacement = store.record_reflection(
        "s",
        "corrected conclusion",
        [new_support["id"]],
        supersedes=[old["id"]],
    )

    assert replacement["supersedes"] == [old["id"]]
    assert store.active_reflections("s") == [replacement]
    stored_old = next(
        item
        for item in store.list_reflections("s", active_only=False)
        if item["id"] == old["id"]
    )
    assert stored_old["status"] == "superseded"

    foreign_support = store.write_observation("other", "foreign", ["e-3"])
    foreign = store.record_reflection("other", "foreign conclusion", [foreign_support["id"]])
    another_support = store.write_observation("s", "another fact", ["e-4"])
    with pytest.raises(ValueError, match="another session"):
        store.record_reflection(
            "s",
            "invalid replacement",
            [another_support["id"]],
            supersedes=[foreign["id"]],
        )


def test_promote_is_idempotent_and_recall_resolves_evidence(tmp_path):
    store = ObservationMemoryStore(tmp_path)
    store.enqueue_turn("s", "t", [{"id": "source-1", "role": "user", "content": "raw"}])
    observation = store.write_observation("s", "durable fact", ["source-1"])
    reflection = store.record_reflection("s", "durable conclusion", [observation["id"]])

    promoted = store.promote_reflection(reflection["id"])
    store.promote_reflection(reflection["id"])
    assert promoted["promoted_at"] is not None
    assert promoted["scope"] == "workspace"
    assert not (tmp_path / "memory" / "MEMORY.md").exists()
    assert store.workspace_reflections()[0]["content"] == "durable conclusion"

    recalled = store.recall(reflection["id"])
    assert recalled["status"] == "found"
    assert recalled["kind"] == "reflection"
    assert recalled["supporting_observations"][0]["content"] == "durable fact"
    assert recalled["source_entries"][0]["id"] == "source-1"
    assert recalled["partial"] is False


def test_recall_reports_missing_source_and_view_status(tmp_path):
    store = ObservationMemoryStore(tmp_path)
    observation = store.write_observation("s", "orphan fact", ["missing-source"])
    recalled = store.recall(observation["id"])

    assert recalled["partial"] is True
    assert recalled["missing_source_event_ids"] == ["missing-source"]
    assert store.recall("does-not-exist")["status"] == "not_found"
    view = store.view("s")
    assert view["observations"][0]["content"] == "orphan fact"
    assert "orphan fact" in view["text"]
    assert store.status("s")["active_observations"] == 1


def test_turn_safety_and_workspace_promotion_are_queryable(tmp_path):
    store = ObservationMemoryStore(tmp_path)
    first = store.enqueue_turn("s", "safe", [{"id": "e-safe", "content": "safe"}])
    second = store.enqueue_turn("s", "unsafe", [{"id": "e-unsafe", "content": "unsafe"}])
    now = datetime(2026, 1, 1, tzinfo=UTC)

    assert store.turns_are_safe("s", {"safe"}) is False
    store.claim_pending(worker_id="w", now=now)
    store.no_facts(first, worker_id="w", now=now)
    assert store.turns_are_safe("s", {"safe"}) is True
    assert store.turns_are_safe("s", {"safe", "missing"}) is False
    assert store.turns_are_safe("s", {"unsafe"}) is False

    # Keep the second job pending: it must not affect the completed turn's
    # safety result.
    assert second["status"] == "pending"
