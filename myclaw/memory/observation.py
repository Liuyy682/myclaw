"""SQLite-backed observational memory.

The module intentionally keeps the worker-facing API synchronous.  A worker
can enqueue a turn, claim a leased job, and then finish it without sharing a
SQLite connection with another worker.  Source events are append-only; the
other records are small read models that can be rebuilt or inspected with
ordinary SQLite tooling.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


class ObservationMemoryError(RuntimeError):
    """Base exception for worker/storage protocol errors."""


class LeaseError(ObservationMemoryError):
    """Raised when a worker tries to use a lease it does not own."""


class ObservationMemoryStore:
    """Persist observations, reflections, and their source-event jobs.

    ``workspace/memory/observation_memory.db`` is the durable source for this
    store.  Every method opens a short-lived connection, which keeps the API
    safe for synchronous workers and makes separate processes participate in
    SQLite's WAL locking naturally.

    Job state is one of ``pending``, ``processing``, ``completed``,
    ``no_facts``, or ``failed``.  ``attempts`` counts claims, and no more than
    :attr:`MAX_ATTEMPTS` claims are allowed for one job.  A successful
    completion or an intentional ``no_facts`` result advances the session's
    safe watermark; a failed or unfinished job does not.
    """

    MAX_ATTEMPTS = 3
    DEFAULT_LEASE_SECONDS = 300
    TERMINAL_JOB_STATUSES = frozenset({"completed", "no_facts", "failed"})
    SAFE_JOB_STATUSES = frozenset({"completed", "no_facts"})

    def __init__(
        self,
        workspace: Path | str,
        *,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")

        self.workspace = Path(workspace).expanduser().resolve()
        self.memory_dir = self.workspace / "memory"
        self.path = self.memory_dir / "observation_memory.db"
        # ``db_path`` is a convenient explicit spelling for callers that need
        # to inspect the file, while ``path`` matches the other stores.
        self.db_path = self.path
        self.lease_seconds = int(lease_seconds)
        self.max_attempts = int(max_attempts)
        self.initialize()

    # ------------------------------------------------------------------ setup

    def initialize(self) -> None:
        """Create the schema and configure WAL mode if necessary."""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS observation_jobs (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    worker_version TEXT NOT NULL,
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
                );

                CREATE TABLE IF NOT EXISTS source_events (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    event_index INTEGER NOT NULL,
                    sequence INTEGER NOT NULL,
                    role TEXT,
                    content TEXT,
                    content_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(session_id, turn_id, event_index),
                    UNIQUE(session_id, sequence),
                    FOREIGN KEY(job_id) REFERENCES observation_jobs(id)
                );

                CREATE TABLE IF NOT EXISTS observations (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_event_ids_json TEXT NOT NULL DEFAULT '[]',
                    kind TEXT,
                    importance REAL,
                    confidence REAL,
                    relevance TEXT NOT NULL DEFAULT 'medium',
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    archived_at TEXT,
                    reflected_at TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS reflections (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    supporting_observation_ids_json TEXT NOT NULL DEFAULT '[]',
                    kind TEXT,
                    supersedes_json TEXT NOT NULL DEFAULT '[]',
                    scope TEXT NOT NULL DEFAULT 'session',
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    archived_at TEXT,
                    promoted_at TEXT,
                    promoted_path TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS reflection_failures (
                    session_id TEXT PRIMARY KEY,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_source_events_session_sequence
                    ON source_events(session_id, sequence);
                CREATE INDEX IF NOT EXISTS idx_source_events_job
                    ON source_events(job_id, event_index);
                CREATE INDEX IF NOT EXISTS idx_observation_jobs_session_status
                    ON observation_jobs(session_id, status, created_at);
                CREATE INDEX IF NOT EXISTS idx_observations_session_status
                    ON observations(session_id, status, created_at);
                CREATE INDEX IF NOT EXISTS idx_reflections_session_status
                    ON reflections(session_id, status, created_at);

                CREATE TRIGGER IF NOT EXISTS source_events_immutable_update
                BEFORE UPDATE ON source_events
                BEGIN
                    SELECT RAISE(ABORT, 'source_events are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS source_events_immutable_delete
                BEFORE DELETE ON source_events
                BEGIN
                    SELECT RAISE(ABORT, 'source_events are immutable');
                END;

                PRAGMA user_version = 2;
                """
            )
            job_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(observation_jobs)")
            }
            if "input_hash" not in job_columns:
                connection.execute(
                    "ALTER TABLE observation_jobs "
                    "ADD COLUMN input_hash TEXT NOT NULL DEFAULT ''"
                )
            if "worker_version" not in job_columns:
                connection.execute(
                    "ALTER TABLE observation_jobs "
                    "ADD COLUMN worker_version TEXT NOT NULL DEFAULT 'observation-v1'"
                )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS "
                "idx_observation_jobs_turn_input_worker "
                "ON observation_jobs(session_id, turn_id, input_hash, worker_version)"
            )
            connection.execute("PRAGMA user_version = 2")

    def _connect(self) -> sqlite3.Connection:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
        except BaseException:
            connection.rollback()
            raise
        else:
            connection.commit()
        finally:
            connection.close()

    # ------------------------------------------------------------- enqueueing

    def enqueue_turn(
        self,
        session_id: str,
        turn_id: str,
        events: Iterable[Any] | None = None,
        *,
        source_events: Iterable[Any] | None = None,
        idempotency_key: str | None = None,
        worker_version: str = "observation-v1",
    ) -> dict[str, Any]:
        """Atomically append a turn's immutable events and enqueue one job.

        Events may be mappings, strings, dataclass-like objects, or JSON-safe
        values.  A mapping's ``id``/``event_id``/``source_event_id`` is kept as
        its stable source ID; otherwise the ID is derived from the canonical
        event payload, turn, and event index.  Repeating the same turn (or the
        same explicit idempotency key) returns the original job and does not
        append or mutate anything.
        """
        session_id = _required_text(session_id, "session_id")
        turn_id = _required_text(turn_id, "turn_id")
        if events is not None and source_events is not None:
            raise ValueError("pass events or source_events, not both")
        raw_events = source_events if source_events is not None else events
        if isinstance(raw_events, Mapping) or isinstance(raw_events, (str, bytes)):
            raw_events = [raw_events]
        normalized_events = self._normalize_events(raw_events or (), session_id, turn_id)
        dedupe_key = _required_text(idempotency_key or turn_id, "idempotency_key")
        worker_version = _required_text(worker_version, "worker_version")
        input_hash = hashlib.sha256(
            _json(
                [
                    {
                        "id": event["id"],
                        "role": event["role"],
                        "content_hash": event["content_hash"],
                        "payload": json.loads(event["payload_json"]),
                    }
                    for event in normalized_events
                ]
            ).encode("utf-8")
        ).hexdigest()
        job_id = stable_id("job", session_id, dedupe_key)

        now = _utc_now()
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM observation_jobs WHERE session_id = ? AND idempotency_key = ?",
                (session_id, dedupe_key),
            ).fetchone()
            if existing is not None:
                if existing["turn_id"] != turn_id:
                    raise ValueError("idempotency key already used by another turn")
                if existing["input_hash"] not in {"", input_hash}:
                    raise ValueError("idempotency key already used with different events")
                if existing["worker_version"] != worker_version:
                    raise ValueError("idempotency key already used by another worker version")
                self._verify_existing_enqueue(connection, existing, normalized_events)
                return self._job_row(existing, connection=connection)

            # The job is inserted first because source_events carries a
            # foreign key to it.  The transaction means a failed event insert
            # cannot leave a visible empty job behind.
            connection.execute(
                """
                INSERT INTO observation_jobs (
                    id, session_id, turn_id, idempotency_key, input_hash,
                    worker_version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    session_id,
                    turn_id,
                    dedupe_key,
                    input_hash,
                    worker_version,
                    now,
                    now,
                ),
            )

            next_sequence = self._next_sequence(connection, session_id)
            event_ids: list[str] = []
            sequences: list[int] = []
            for event in normalized_events:
                sequence = next_sequence
                next_sequence += 1
                connection.execute(
                    """
                    INSERT INTO source_events (
                        id, session_id, turn_id, job_id, event_index, sequence,
                        role, content, content_hash, payload_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event["id"],
                        session_id,
                        turn_id,
                        job_id,
                        event["event_index"],
                        sequence,
                        event["role"],
                        event["content"],
                        event["content_hash"],
                        event["payload_json"],
                        now,
                    ),
                )
                event_ids.append(event["id"])
                sequences.append(sequence)

            connection.execute(
                """
                UPDATE observation_jobs
                SET event_ids_json = ?, first_sequence = ?, last_sequence = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    _json(event_ids),
                    sequences[0] if sequences else None,
                    sequences[-1] if sequences else None,
                    now,
                    job_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM observation_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            assert row is not None
            return self._job_row(row, connection=connection)

    def _normalize_events(
        self, events: Iterable[Any], session_id: str, turn_id: str
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for index, raw in enumerate(events):
            payload = _as_payload(raw)
            explicit_id = _first_text(payload, "id", "event_id", "source_event_id")
            event_id = explicit_id or stable_id("source-event", session_id, turn_id, index, payload)
            role = _optional_text(payload.get("role"))
            content_value = payload.get("content")
            if content_value is None:
                content_value = payload.get("text")
            content = None if content_value is None else str(content_value)
            payload_json = _json(payload)
            normalized.append(
                {
                    "id": event_id,
                    "event_index": index,
                    "role": role,
                    "content": content,
                    "content_hash": hashlib.sha256(
                        (content if content is not None else payload_json).encode("utf-8")
                    ).hexdigest(),
                    "payload_json": payload_json,
                }
            )
        return normalized

    def _verify_existing_enqueue(
        self,
        connection: sqlite3.Connection,
        job: sqlite3.Row,
        events: list[dict[str, Any]],
    ) -> None:
        existing_rows = connection.execute(
            "SELECT id, event_index, role, content, payload_json FROM source_events "
            "WHERE job_id = ? ORDER BY event_index",
            (job["id"],),
        ).fetchall()
        if len(existing_rows) != len(events):
            raise ValueError("idempotency key already used with different events")
        for existing, candidate in zip(existing_rows, events):
            if (
                existing["id"] != candidate["id"]
                or existing["event_index"] != candidate["event_index"]
                or existing["role"] != candidate["role"]
                or existing["content"] != candidate["content"]
                or existing["payload_json"] != candidate["payload_json"]
            ):
                raise ValueError("idempotency key already used with different events")

    @staticmethod
    def _next_sequence(connection: sqlite3.Connection, session_id: str) -> int:
        row = connection.execute(
            "SELECT COALESCE(MAX(sequence), -1) + 1 FROM source_events WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row[0]) if row is not None else 0

    # --------------------------------------------------------------- job lease

    def claim_pending(
        self,
        *,
        worker_id: str | None = None,
        session_id: str | None = None,
        now: datetime | str | None = None,
        lease_seconds: int | None = None,
    ) -> dict[str, Any] | None:
        """Claim the oldest pending or expired processing job.

        Claiming is itself the attempt boundary.  Once ``max_attempts`` has
        been claimed, an expired job is terminally marked ``failed`` and will
        not be handed out again.
        """
        owner = _required_text(worker_id, "worker_id") if worker_id is not None else uuid.uuid4().hex
        duration = self.lease_seconds if lease_seconds is None else int(lease_seconds)
        if duration <= 0:
            raise ValueError("lease_seconds must be positive")
        now_text = _isoformat(now)
        with self._transaction() as connection:
            clauses = [
                "(status = 'pending' OR (status = 'processing' AND lease_until IS NOT NULL AND lease_until <= ?))"
            ]
            params: list[Any] = [now_text]
            if session_id is not None:
                clauses.append("session_id = ?")
                params.append(_required_text(session_id, "session_id"))
            rows = connection.execute(
                "SELECT * FROM observation_jobs WHERE "
                + " AND ".join(clauses)
                + " ORDER BY created_at, id",
                params,
            ).fetchall()

            for candidate in rows:
                attempts = int(candidate["attempts"])
                if attempts >= self.max_attempts:
                    connection.execute(
                        """
                        UPDATE observation_jobs
                        SET status = 'failed', last_error = COALESCE(last_error, ?),
                            lease_owner = NULL, lease_until = NULL, updated_at = ?, completed_at = ?
                        WHERE id = ? AND status = 'processing'
                        """,
                        ("maximum attempts exceeded", now_text, now_text, candidate["id"]),
                    )
                    continue

                lease_until = _isoformat(_parse_datetime(now_text) + timedelta(seconds=duration))
                connection.execute(
                    """
                    UPDATE observation_jobs
                    SET status = 'processing', attempts = attempts + 1,
                        lease_owner = ?, lease_until = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (owner, lease_until, now_text, candidate["id"]),
                )
                claimed = connection.execute(
                    "SELECT * FROM observation_jobs WHERE id = ?", (candidate["id"],)
                ).fetchone()
                assert claimed is not None
                return self._job_row(claimed, connection=connection)
            return None

    def claim_observation_job(self) -> dict[str, Any] | None:
        """Worker-facing claim entry point using a fresh lease owner."""
        return self.claim_pending()

    def _get_job(self, connection: sqlite3.Connection, job_id: str | Mapping[str, Any]) -> sqlite3.Row:
        value = _record_id(job_id, "job_id", "id")
        row = connection.execute("SELECT * FROM observation_jobs WHERE id = ?", (value,)).fetchone()
        if row is None:
            raise KeyError(value)
        return row

    def complete(
        self,
        job_id: str | Mapping[str, Any],
        observations: Iterable[Any] | Mapping[str, Any] | None = None,
        *,
        worker_id: str | None = None,
        now: datetime | str | None = None,
    ) -> dict[str, Any]:
        """Complete a claimed job and atomically persist its observations."""
        with self._transaction() as connection:
            job = self._get_job(connection, job_id)
            if job["status"] in {"completed", "no_facts"}:
                return self._job_row(job, connection=connection)
            self._assert_claim_owned(job, worker_id, now)
            items = _observation_items(observations)
            if not items:
                return self._finish_job(connection, job["id"], "no_facts", now=now)

            event_ids = _json_list(job["event_ids_json"])
            for item in items:
                self._insert_observation(
                    connection,
                    session_id=job["session_id"],
                    item=item,
                    default_source_event_ids=event_ids,
                    now=now,
                )
            return self._finish_job(connection, job["id"], "completed", now=now)

    def commit_observation_result(
        self,
        job_id: str | Mapping[str, Any],
        observations: Iterable[Any] | Mapping[str, Any] | None,
        *,
        no_facts: bool = False,
    ) -> dict[str, Any]:
        """Commit a worker result and finish its job in one transaction."""
        with self._transaction() as connection:
            job = self._get_job(connection, job_id)
            session_id = str(job["session_id"])
            if job["status"] in {"completed", "no_facts"}:
                return self._job_row(job, connection=connection)
            self._assert_claim_owned(job, None, None)
            items = [] if no_facts else _observation_items(observations)
            event_ids = _json_list(job["event_ids_json"])
            for item in items:
                self._insert_observation(
                    connection,
                    session_id=session_id,
                    item=item,
                    default_source_event_ids=event_ids,
                    now=None,
                )
            if items:
                connection.execute(
                    "DELETE FROM reflection_failures WHERE session_id = ?", (session_id,)
                )
            return self._finish_job(
                connection,
                job["id"],
                "no_facts" if not items else "completed",
                now=None,
            )

    def no_facts(
        self,
        job_id: str | Mapping[str, Any],
        *,
        worker_id: str | None = None,
        now: datetime | str | None = None,
    ) -> dict[str, Any]:
        """Mark a claimed job as intentionally having no durable facts."""
        with self._transaction() as connection:
            job = self._get_job(connection, job_id)
            if job["status"] == "no_facts":
                return self._job_row(job, connection=connection)
            if job["status"] == "completed":
                return self._job_row(job, connection=connection)
            self._assert_claim_owned(job, worker_id, now)
            return self._finish_job(connection, job["id"], "no_facts", now=now)

    def fail(
        self,
        job_id: str | Mapping[str, Any],
        error: str | BaseException = "worker failed",
        *,
        worker_id: str | None = None,
        now: datetime | str | None = None,
        retryable: bool = True,
    ) -> dict[str, Any]:
        """Release a failed claim for retry, or terminally fail on attempt 3."""
        error_text = str(error).strip() or "worker failed"
        with self._transaction() as connection:
            job = self._get_job(connection, job_id)
            if job["status"] in self.TERMINAL_JOB_STATUSES:
                return self._job_row(job, connection=connection)
            self._assert_claim_owned(job, worker_id, now, allow_expired=True)
            now_text = _isoformat(now)
            attempts = int(job["attempts"])
            if not retryable or attempts >= self.max_attempts:
                status = "failed"
                completed_at = now_text
            else:
                status = "pending"
                completed_at = None
            connection.execute(
                """
                UPDATE observation_jobs
                SET status = ?, last_error = ?, lease_owner = NULL, lease_until = NULL,
                    updated_at = ?, completed_at = ?
                WHERE id = ?
                """,
                (status, error_text, now_text, completed_at, job["id"]),
            )
            row = connection.execute(
                "SELECT * FROM observation_jobs WHERE id = ?", (job["id"],)
            ).fetchone()
            assert row is not None
            return self._job_row(row, connection=connection)

    def fail_observation_job(
        self,
        job_id: str | Mapping[str, Any],
        error: str | BaseException,
        *,
        retryable: bool = True,
    ) -> dict[str, Any]:
        """Worker-facing failure entry point."""
        return self.fail(job_id, error, retryable=retryable)

    def _assert_claim_owned(
        self,
        job: sqlite3.Row,
        worker_id: str | None,
        now: datetime | str | None,
        *,
        allow_expired: bool = False,
    ) -> None:
        if job["status"] != "processing":
            raise ObservationMemoryError(f"job {job['id']} is not processing")
        if worker_id is not None and job["lease_owner"] != worker_id:
            raise LeaseError(f"job {job['id']} is leased by another worker")
        if worker_id is not None and not allow_expired and _lease_expired(job["lease_until"], now):
            raise LeaseError(f"job {job['id']} lease expired")

    def _finish_job(
        self,
        connection: sqlite3.Connection,
        job_id: str,
        status: str,
        *,
        now: datetime | str | None,
    ) -> dict[str, Any]:
        now_text = _isoformat(now)
        connection.execute(
            """
            UPDATE observation_jobs
            SET status = ?, lease_owner = NULL, lease_until = NULL,
                updated_at = ?, completed_at = ?, last_error = NULL
            WHERE id = ?
            """,
            (status, now_text, now_text, job_id),
        )
        row = connection.execute("SELECT * FROM observation_jobs WHERE id = ?", (job_id,)).fetchone()
        assert row is not None
        return self._job_row(row, connection=connection)

    # ------------------------------------------------------------- observations

    def write_observation(
        self,
        session_id: str,
        content: str,
        source_event_ids: Iterable[str] | None = None,
        *,
        observation_id: str | None = None,
        relevance: str = "medium",
        kind: str | None = None,
        importance: float | None = None,
        confidence: float | None = None,
        timestamp: datetime | str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Insert one stable observation, returning the existing row on retry."""
        session_id = _required_text(session_id, "session_id")
        item = {
            "content": content,
            "source_event_ids": list(source_event_ids or ()),
            "observation_id": observation_id,
            "relevance": relevance,
            "kind": kind,
            "importance": importance,
            "confidence": confidence,
            "timestamp": timestamp,
            "metadata": metadata,
        }
        with self._transaction() as connection:
            observation_id_value = self._insert_observation(
                connection,
                session_id=session_id,
                item=item,
                default_source_event_ids=[],
                now=timestamp,
            )
            row = connection.execute(
                "SELECT * FROM observations WHERE id = ?", (observation_id_value,)
            ).fetchone()
            assert row is not None
            return _observation_row(row)

    def _insert_observation(
        self,
        connection: sqlite3.Connection,
        *,
        session_id: str,
        item: Mapping[str, Any] | Any,
        default_source_event_ids: list[str],
        now: datetime | str | None,
    ) -> str:
        value = _as_payload(item)
        content = _required_text(value.get("content") or value.get("text"), "content")
        source_ids = _string_list(
            value.get("source_event_ids")
            or value.get("source_ids")
            or value.get("sourceEntryIds")
            or default_source_event_ids
        )
        relevance = _required_text(value.get("relevance", "medium"), "relevance")
        kind = _optional_text(value.get("kind"))
        importance = _optional_score(value.get("importance"))
        confidence = _optional_score(value.get("confidence"))
        explicit_id = _first_text(value, "observation_id", "id")
        observation_id = explicit_id or stable_id("observation", session_id, content, source_ids)
        created_at = _isoformat(value.get("timestamp") or now)
        metadata = value.get("metadata")
        if not isinstance(metadata, Mapping):
            metadata = {}
        payload = (
            observation_id,
            session_id,
            content,
            _json(source_ids),
            kind,
            importance,
            confidence,
            relevance,
            created_at,
            _json(dict(metadata)),
        )
        connection.execute(
            """
            INSERT INTO observations (
                id, session_id, content, source_event_ids_json, kind,
                importance, confidence, relevance, created_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO NOTHING
            """,
            payload,
        )
        existing = connection.execute(
            "SELECT * FROM observations WHERE id = ?", (observation_id,)
        ).fetchone()
        assert existing is not None
        if (
            existing["session_id"] != session_id
            or existing["content"] != content
            or existing["source_event_ids_json"] != _json(source_ids)
            or existing["relevance"] != relevance
        ):
            raise ValueError(f"observation id already exists with different content: {observation_id}")
        return observation_id

    def list_observations(
        self,
        session_id: str | None = None,
        *,
        active_only: bool = True,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(_required_text(session_id, "session_id"))
        if active_only:
            clauses.append("status = 'active'")
        query = "SELECT * FROM observations"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at, id"
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be non-negative")
            query += " LIMIT ?"
            params.append(int(limit))
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [_observation_row(row) for row in rows]

    def active_observations(self, session_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        return self.list_observations(session_id, active_only=True, limit=limit)

    def list_unreflected_observations(
        self,
        session_id: str,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return active observations not yet cited by a reflection."""
        session_id = _required_text(session_id, "session_id")
        observations = self.list_observations(session_id, active_only=True, limit=None)
        reflected_ids: set[str] = set()
        for reflection in self.list_reflections(session_id, active_only=False):
            reflected_ids.update(reflection["supporting_observation_ids"])
        result = [item for item in observations if item["id"] not in reflected_ids]
        return result if limit is None else result[: max(0, int(limit))]

    def next_reflection_session(self, batch_size: int) -> str | None:
        """Return one session with a retryable full reflection batch."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT observations.session_id
                FROM observations
                LEFT JOIN reflection_failures
                    ON reflection_failures.session_id = observations.session_id
                WHERE observations.status = 'active'
                  AND COALESCE(reflection_failures.attempts, 0) < 3
                GROUP BY observations.session_id
                HAVING COUNT(*) >= ?
                ORDER BY MIN(observations.created_at), observations.session_id
                LIMIT 1
                """,
                (int(batch_size),),
            ).fetchone()
        return str(row["session_id"]) if row is not None else None

    # -------------------------------------------------------------- reflections

    def record_reflection(
        self,
        session_id: str,
        content: str,
        supporting_observation_ids: Iterable[str],
        *,
        reflection_id: str | None = None,
        archive_observations: bool = True,
        timestamp: datetime | str | None = None,
        metadata: Mapping[str, Any] | None = None,
        kind: str | None = None,
        supersedes: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Atomically insert a reflection, its support IDs, and archival."""
        session_id = _required_text(session_id, "session_id")
        content = _required_text(content, "content")
        support_ids = _string_list(supporting_observation_ids)
        if not support_ids:
            raise ValueError("supporting_observation_ids must not be empty")
        explicit_id = _required_text(reflection_id, "reflection_id") if reflection_id is not None else None
        reflection_id_value = explicit_id or stable_id("reflection", session_id, content, support_ids)
        metadata_value = dict(metadata) if isinstance(metadata, Mapping) else {}

        with self._transaction() as connection:
            row = self._insert_reflection(
                connection,
                session_id=session_id,
                content=content,
                support_ids=support_ids,
                reflection_id=reflection_id_value,
                archive_observations=archive_observations,
                timestamp=timestamp,
                metadata=metadata_value,
                kind=kind,
                supersedes=supersedes,
            )
            return _reflection_row(row)

    def _insert_reflection(
        self,
        connection: sqlite3.Connection,
        *,
        session_id: str,
        content: str,
        support_ids: list[str],
        reflection_id: str,
        archive_observations: bool,
        timestamp: datetime | str | None,
        metadata: Mapping[str, Any] | None = None,
        kind: str | None = None,
        supersedes: Iterable[str] | None = None,
    ) -> sqlite3.Row:
        placeholders = ",".join("?" for _ in support_ids)
        support_rows = connection.execute(
            f"SELECT id, session_id FROM observations WHERE id IN ({placeholders})",
            support_ids,
        ).fetchall()
        by_id = {row["id"]: row for row in support_rows}
        missing = [observation_id for observation_id in support_ids if observation_id not in by_id]
        wrong_session = [
            observation_id
            for observation_id in support_ids
            if observation_id in by_id and by_id[observation_id]["session_id"] != session_id
        ]
        if missing:
            raise KeyError(f"supporting observation not found: {missing[0]}")
        if wrong_session:
            raise ValueError(f"supporting observation belongs to another session: {wrong_session[0]}")

        supersedes_ids = _string_list(supersedes)
        if reflection_id in supersedes_ids:
            raise ValueError("a reflection cannot supersede itself")
        if supersedes_ids:
            supersedes_placeholders = ",".join("?" for _ in supersedes_ids)
            superseded_rows = connection.execute(
                f"SELECT id, session_id FROM reflections "
                f"WHERE id IN ({supersedes_placeholders})",
                supersedes_ids,
            ).fetchall()
            superseded_by_id = {row["id"]: row for row in superseded_rows}
            missing_superseded = [
                item for item in supersedes_ids if item not in superseded_by_id
            ]
            wrong_superseded_session = [
                item
                for item in supersedes_ids
                if item in superseded_by_id
                and superseded_by_id[item]["session_id"] != session_id
            ]
            if missing_superseded:
                raise KeyError(f"superseded reflection not found: {missing_superseded[0]}")
            if wrong_superseded_session:
                raise ValueError(
                    "superseded reflection belongs to another session: "
                    f"{wrong_superseded_session[0]}"
                )
        connection.execute(
            """
            INSERT INTO reflections (
                id, session_id, content, supporting_observation_ids_json,
                kind, supersedes_json, scope, created_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, 'session', ?, ?)
            ON CONFLICT(id) DO NOTHING
            """,
            (
                reflection_id,
                session_id,
                content,
                _json(support_ids),
                _optional_text(kind),
                _json(supersedes_ids),
                _isoformat(timestamp),
                _json(dict(metadata or {})),
            ),
        )
        existing = connection.execute(
            "SELECT * FROM reflections WHERE id = ?", (reflection_id,)
        ).fetchone()
        assert existing is not None
        if (
            existing["session_id"] != session_id
            or existing["content"] != content
            or existing["supporting_observation_ids_json"] != _json(support_ids)
            or existing["kind"] != _optional_text(kind)
            or existing["supersedes_json"] != _json(supersedes_ids)
        ):
            raise ValueError(f"reflection id already exists with different content: {reflection_id}")

        if supersedes_ids:
            now_text = _isoformat(timestamp)
            connection.execute(
                f"UPDATE reflections SET status = 'superseded', "
                f"archived_at = COALESCE(archived_at, ?) "
                f"WHERE session_id = ? AND id IN ({supersedes_placeholders})",
                [now_text, session_id, *supersedes_ids],
            )
        if archive_observations:
            now_text = _isoformat(timestamp)
            connection.execute(
                f"UPDATE observations SET status = 'archived', archived_at = COALESCE(archived_at, ?), "
                f"reflected_at = COALESCE(reflected_at, ?) WHERE session_id = ? AND id IN ({placeholders})",
                [now_text, now_text, session_id, *support_ids],
            )
        return existing

    def list_reflections(
        self,
        session_id: str | None = None,
        *,
        active_only: bool = True,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(_required_text(session_id, "session_id"))
        if active_only:
            clauses.append("status = 'active'")
        query = "SELECT * FROM reflections"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at, id"
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be non-negative")
            query += " LIMIT ?"
            params.append(int(limit))
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [_reflection_row(row) for row in rows]

    def active_reflections(self, session_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        return self.list_reflections(session_id, active_only=True, limit=limit)

    def workspace_reflections(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Return active reflections promoted to workspace scope."""
        clauses = ["scope = 'workspace'", "status = 'active'"]
        params: list[Any] = []
        query = "SELECT * FROM reflections WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at, id"
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be non-negative")
            query += " LIMIT ?"
            params.append(int(limit))
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [_reflection_row(row) for row in rows]

    def commit_reflection_result(
        self,
        session_id: str,
        reflections: Iterable[Any] | Mapping[str, Any],
        *,
        observation_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Atomically persist a worker reflection batch and archive support."""
        session_id = _required_text(session_id, "session_id")
        items = _reflection_items(reflections)
        fallback_support = _string_list(observation_ids)
        records: list[dict[str, Any]] = []
        with self._transaction() as connection:
            for item in items:
                value = _as_payload(item)
                content = _required_text(value.get("content"), "content")
                support_ids = _string_list(
                    value.get("supports")
                    or value.get("supporting_observation_ids")
                    or fallback_support
                )
                if not support_ids:
                    raise ValueError("reflection supports must not be empty")
                reflection_id = _first_text(value, "reflection_id", "id")
                if reflection_id is None:
                    reflection_id = stable_id("reflection", session_id, content, support_ids)
                row = self._insert_reflection(
                    connection,
                    session_id=session_id,
                    content=content,
                    support_ids=support_ids,
                    reflection_id=reflection_id,
                    archive_observations=True,
                    timestamp=value.get("timestamp"),
                    metadata=value.get("metadata"),
                    kind=value.get("kind"),
                    supersedes=value.get("supersedes"),
                )
                records.append(_reflection_row(row))
            if records:
                connection.execute(
                    "DELETE FROM reflection_failures WHERE session_id = ?", (session_id,)
                )
        return {
            "status": "completed" if records else "no_facts",
            "session_id": session_id,
            "reflections": records,
            "observation_ids": fallback_support,
        }

    def fail_reflection(
        self, session_id: str, error: str, *, retryable: bool = True
    ) -> dict[str, Any]:
        """Record a failed reflection attempt without consuming observations."""
        session_id = _required_text(session_id, "session_id")
        error = _required_text(error, "error")
        now = _utc_now()
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO reflection_failures (session_id, attempts, last_error, updated_at)
                VALUES (?, 1, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    attempts = reflection_failures.attempts + 1,
                    last_error = excluded.last_error,
                    updated_at = excluded.updated_at
                """,
                (session_id, error, now),
            )
            row = connection.execute(
                "SELECT * FROM reflection_failures WHERE session_id = ?", (session_id,)
            ).fetchone()
        assert row is not None
        return dict(row)

    def archive_observations(
        self,
        session_id: str,
        observation_ids: Iterable[str],
        *,
        archived_at: datetime | str | None = None,
    ) -> int:
        ids = _string_list(observation_ids)
        if not ids:
            return 0
        session_id = _required_text(session_id, "session_id")
        placeholders = ",".join("?" for _ in ids)
        with self._transaction() as connection:
            result = connection.execute(
                f"UPDATE observations SET status = 'archived', archived_at = COALESCE(archived_at, ?) "
                f"WHERE session_id = ? AND id IN ({placeholders})",
                [_isoformat(archived_at), session_id, *ids],
            )
            return int(result.rowcount)

    def turns_are_safe(self, session_id: str, turn_ids: Iterable[str]) -> bool:
        """Return whether every requested turn has only safe terminal jobs."""
        session_id = _required_text(session_id, "session_id")
        ids = _string_list(turn_ids)
        if not ids:
            return True
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT turn_id, status FROM observation_jobs WHERE session_id = ? AND turn_id IN ({placeholders})",
                [session_id, *ids],
            ).fetchall()
        statuses: dict[str, list[str]] = {turn_id: [] for turn_id in ids}
        for row in rows:
            statuses.setdefault(row["turn_id"], []).append(row["status"])
        return all(
            values and all(status in self.SAFE_JOB_STATUSES for status in values)
            for values in statuses.values()
        )

    # -------------------------------------------------------------- workspace

    def promote_reflection(
        self,
        reflection_id: str | Mapping[str, Any],
        *,
        path: Path | str | None = None,
    ) -> dict[str, Any]:
        """Promote a session reflection to workspace scope in the DB."""
        reflection_key = _record_id(reflection_id, "reflection_id", "id")
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM reflections WHERE id = ?", (reflection_key,)
            ).fetchone()
            if row is None:
                raise KeyError(reflection_key)
            now_text = row["promoted_at"] or _utc_now()
            if row["scope"] != "workspace" or row["promoted_at"] is None:
                connection.execute(
                    "UPDATE reflections SET scope = 'workspace', promoted_at = ?, promoted_path = NULL WHERE id = ?",
                    (now_text, reflection_key),
                )
            updated = connection.execute(
                "SELECT * FROM reflections WHERE id = ?", (reflection_key,)
            ).fetchone()
            assert updated is not None
            return _reflection_row(updated)

    # ------------------------------------------------------------------ reads

    def status(self, session_id: str | None = None) -> dict[str, Any]:
        """Return compact counts and lease/watermark state."""
        session = _required_text(session_id, "session_id") if session_id is not None else None
        where = " WHERE session_id = ?" if session is not None else ""
        params: tuple[Any, ...] = (session,) if session is not None else ()
        with self._connect() as connection:
            event_count = int(
                connection.execute(f"SELECT COUNT(*) FROM source_events{where}", params).fetchone()[0]
            )
            job_rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM observation_jobs" + where + " GROUP BY status",
                params,
            ).fetchall()
            observation_rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM observations" + where + " GROUP BY status",
                params,
            ).fetchall()
            reflection_rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM reflections" + where + " GROUP BY status",
                params,
            ).fetchall()
            processing = int(
                connection.execute(
                    "SELECT COUNT(*) FROM observation_jobs" + where + (" AND " if where else " WHERE ") + "status = 'processing'",
                    params,
                ).fetchone()[0]
            )
            reflection_failure = None
            if session is not None:
                row = connection.execute(
                    "SELECT attempts, last_error, updated_at FROM reflection_failures WHERE session_id = ?",
                    (session,),
                ).fetchone()
                reflection_failure = dict(row) if row is not None else None

        job_counts = {str(row["status"]): int(row["count"]) for row in job_rows}
        observation_counts = {str(row["status"]): int(row["count"]) for row in observation_rows}
        reflection_counts = {str(row["status"]): int(row["count"]) for row in reflection_rows}
        result: dict[str, Any] = {
            "database": str(self.path),
            "journal_mode": "wal",
            "session_id": session,
            "source_events": event_count,
            "jobs": job_counts,
            "job_counts": job_counts,
            "observations": observation_counts,
            "reflection_counts": reflection_counts,
            "reflections": reflection_counts,
            "processing": processing,
            "active_observations": observation_counts.get("active", 0),
            "active_reflections": reflection_counts.get("active", 0),
            "safe_watermark": self.safe_watermark(session) if session is not None else None,
            "reflection_failure": reflection_failure,
        }
        return result

    def view(
        self,
        session_id: str | None = None,
        *,
        include_archived: bool = False,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Return active memory plus a deterministic human-readable rendering."""
        observations = self.list_observations(
            session_id, active_only=not include_archived, limit=limit
        )
        reflections = self.list_reflections(
            session_id, active_only=not include_archived, limit=limit
        )
        if session_id is not None:
            workspace = self.workspace_reflections(limit=limit)
            by_id = {item["id"]: item for item in reflections}
            by_id.update({item["id"]: item for item in workspace})
            reflections = list(by_id.values())
        lines = ["Reflections:"]
        lines.extend(f"[{item['id']}] {item['content']}" for item in reflections)
        lines.append("Observations:")
        lines.extend(
            f"[{item['id']}] {item['content']}" for item in observations
        )
        return {
            "session_id": session_id,
            "reflections": reflections,
            "observations": observations,
            "text": "\n".join(lines),
            "safe_watermark": self.safe_watermark(session_id) if session_id is not None else None,
        }

    def view_text(
        self,
        session_id: str | None = None,
        *,
        include_archived: bool = False,
        limit: int | None = None,
    ) -> str:
        return str(
            self.view(
                session_id,
                include_archived=include_archived,
                limit=limit,
            )["text"]
        )

    def recall(
        self,
        memory_id: str,
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Recall an observation/reflection and resolve available evidence."""
        memory_id = _required_text(memory_id, "memory_id")
        session = _required_text(session_id, "session_id") if session_id is not None else None
        observation_clause = " AND session_id = ?" if session is not None else ""
        observation_params: tuple[Any, ...] = (
            (memory_id, session) if session is not None else (memory_id,)
        )
        reflection_clause = " AND (session_id = ? OR scope = 'workspace')" if session is not None else ""
        reflection_params: tuple[Any, ...] = (
            (memory_id, session) if session is not None else (memory_id,)
        )
        with self._connect() as connection:
            observation_rows = connection.execute(
                "SELECT * FROM observations WHERE id = ?" + observation_clause,
                observation_params,
            ).fetchall()
            reflection_rows = connection.execute(
                "SELECT * FROM reflections WHERE id = ?" + reflection_clause,
                reflection_params,
            ).fetchall()
            observations = [_observation_row(row) for row in observation_rows]
            reflections = [_reflection_row(row) for row in reflection_rows]

            supporting_observation_ids: list[str] = []
            for reflection in reflections:
                for observation_id in reflection["supporting_observation_ids"]:
                    if observation_id not in supporting_observation_ids:
                        supporting_observation_ids.append(observation_id)
            supporting_observations: list[dict[str, Any]] = []
            missing_supporting_observation_ids: list[str] = []
            if supporting_observation_ids:
                placeholders = ",".join("?" for _ in supporting_observation_ids)
                support_rows = connection.execute(
                    f"SELECT * FROM observations WHERE id IN ({placeholders})",
                    supporting_observation_ids,
                ).fetchall()
                support_by_id = {row["id"]: _observation_row(row) for row in support_rows}
                for observation_id in supporting_observation_ids:
                    if observation_id in support_by_id:
                        supporting_observations.append(support_by_id[observation_id])
                    else:
                        missing_supporting_observation_ids.append(observation_id)

            # A reflection's source evidence lives on its supporting
            # observations, so resolve sources from both a directly recalled
            # observation and any support rows.
            source_entries: list[dict[str, Any]] = []
            missing_source_event_ids: list[str] = []
            all_source_ids: list[str] = []
            for observation in [*observations, *supporting_observations]:
                for source_id in observation["source_event_ids"]:
                    if source_id not in all_source_ids:
                        all_source_ids.append(source_id)
            if all_source_ids:
                placeholders = ",".join("?" for _ in all_source_ids)
                source_rows = connection.execute(
                    f"SELECT * FROM source_events WHERE id IN ({placeholders})",
                    all_source_ids,
                ).fetchall()
                source_by_id = {row["id"]: _source_event_row(row) for row in source_rows}
                for source_id in all_source_ids:
                    if source_id in source_by_id:
                        source_entries.append(source_by_id[source_id])
                    else:
                        missing_source_event_ids.append(source_id)

        if not observations and not reflections:
            return {
                "status": "not_found",
                "memory_id": memory_id,
                "id": memory_id,
                "kind": None,
                "observations": [],
                "reflections": [],
                "source_entries": [],
                "supporting_observations": [],
                "missing_source_event_ids": [],
                "missing_supporting_observation_ids": [],
                "partial": False,
                "collision": False,
            }

        kind = "mixed" if observations and reflections else "observation" if observations else "reflection"
        content = (observations[0] if observations else reflections[0])["content"]
        return {
            "status": "found",
            "memory_id": memory_id,
            "id": memory_id,
            "kind": kind,
            "content": content,
            "observation": observations[0] if len(observations) == 1 else None,
            "reflection": reflections[0] if len(reflections) == 1 else None,
            "observations": observations,
            "reflections": reflections,
            "source_entries": source_entries,
            "supporting_observations": supporting_observations,
            "missing_source_event_ids": missing_source_event_ids,
            "missing_supporting_observation_ids": missing_supporting_observation_ids,
            "partial": bool(missing_source_event_ids or missing_supporting_observation_ids),
            "collision": len(observations) + len(reflections) > 1,
        }

    # ------------------------------------------------------------- watermarks

    def safe_watermark(self, session_id: str) -> str | None:
        """Return the last contiguous source-event ID with safe job coverage."""
        session_id = _required_text(session_id, "session_id")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT source_events.id, observation_jobs.status
                FROM source_events
                JOIN observation_jobs ON observation_jobs.id = source_events.job_id
                WHERE source_events.session_id = ?
                ORDER BY source_events.sequence
                """,
                (session_id,),
            ).fetchall()
        watermark: str | None = None
        for row in rows:
            if row["status"] not in self.SAFE_JOB_STATUSES:
                break
            watermark = str(row["id"])
        return watermark

    def safe_watermark_info(self, session_id: str) -> dict[str, Any]:
        session_id = _required_text(session_id, "session_id")
        watermark = self.safe_watermark(session_id)
        with self._connect() as connection:
            row = (
                connection.execute(
                    "SELECT sequence FROM source_events WHERE session_id = ? AND id = ?",
                    (session_id, watermark),
                ).fetchone()
                if watermark is not None
                else None
            )
        return {
            "session_id": session_id,
            "event_id": watermark,
            "sequence": int(row["sequence"]) if row is not None else None,
        }

    # ---------------------------------------------------------- row adapters

    @staticmethod
    def _job_row(row: sqlite3.Row, *, connection: sqlite3.Connection | None = None) -> dict[str, Any]:
        event_ids = _json_list(row["event_ids_json"])
        events = []
        if connection is not None:
            events = [
                _source_event_row(event)
                for event in connection.execute(
                    "SELECT * FROM source_events WHERE job_id = ? ORDER BY event_index",
                    (row["id"],),
                ).fetchall()
            ]
        return {
            "id": row["id"],
            "job_id": row["id"],
            "session_id": row["session_id"],
            "turn_id": row["turn_id"],
            "idempotency_key": row["idempotency_key"],
            "input_hash": row["input_hash"],
            "worker_version": row["worker_version"],
            "event_ids": event_ids,
            "source_event_ids": event_ids,
            "events": events,
            "source_events": events,
            "first_sequence": row["first_sequence"],
            "last_sequence": row["last_sequence"],
            "status": row["status"],
            "state": row["status"],
            "attempts": int(row["attempts"]),
            "attempt_count": int(row["attempts"]),
            "lease_owner": row["lease_owner"],
            "lease_until": row["lease_until"],
            "last_error": row["last_error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "completed_at": row["completed_at"],
        }


def stable_id(namespace: str, *parts: Any) -> str:
    """Derive a stable lowercase string ID from canonical JSON parts."""
    payload = _json([namespace, *parts]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _as_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        return {"content": value}
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {"content": value}


def _observation_items(value: Iterable[Any] | Mapping[str, Any] | None) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        for key in ("observations", "facts", "items"):
            if key in value:
                nested = value[key]
                if nested is None:
                    return []
                if isinstance(nested, (str, bytes, Mapping)):
                    return [nested]
                return list(nested)
        return [value]
    if isinstance(value, (str, bytes)):
        return [value]
    return list(value)


def _reflection_items(value: Iterable[Any] | Mapping[str, Any]) -> list[Any]:
    if isinstance(value, Mapping):
        nested = value.get("reflections", value.get("items"))
        if nested is None:
            return [value]
        if isinstance(nested, (str, bytes, Mapping)):
            return [nested]
        return list(nested)
    if isinstance(value, (str, bytes)):
        return [value]
    return list(value)


def _record_id(value: str | Mapping[str, Any], *keys: str) -> str:
    if isinstance(value, Mapping):
        for key in keys:
            if key in value:
                return _required_text(value[key], key)
        raise ValueError(f"record must contain one of: {', '.join(keys)}")
    return _required_text(value, keys[0])


def _required_text(value: Any, name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"{name} is required")
    return text


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_score(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("score must be a finite number from 0 through 1")
    score = float(value)
    if score < 0 or score > 1:
        raise ValueError("score must be a finite number from 0 through 1")
    return score


def _first_text(mapping: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            text = str(mapping[key]).strip()
            if text:
                return text
    return None


def _string_list(values: Iterable[Any] | Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        values = [values]
    result: list[str] = []
    for value in values:
        text = _required_text(value, "id")
        if text not in result:
            result.append(text)
    return result


def _json_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    return _string_list(decoded) if isinstance(decoded, list) else []


def _isoformat(value: datetime | str | None) -> str:
    if value is None:
        return _utc_now()
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat()


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _lease_expired(lease_until: str | None, now: datetime | str | None) -> bool:
    return lease_until is not None and _parse_datetime(lease_until) <= _parse_datetime(_isoformat(now))


def _source_event_row(row: sqlite3.Row) -> dict[str, Any]:
    try:
        payload = json.loads(row["payload_json"])
    except (TypeError, json.JSONDecodeError):
        payload = {}
    return {
        "id": row["id"],
        "event_id": row["id"],
        "session_id": row["session_id"],
        "turn_id": row["turn_id"],
        "job_id": row["job_id"],
        "event_index": int(row["event_index"]),
        "sequence": int(row["sequence"]),
        "role": row["role"],
        "content": row["content"],
        "content_hash": row["content_hash"],
        "payload": payload,
        "created_at": row["created_at"],
    }


def _observation_row(row: sqlite3.Row) -> dict[str, Any]:
    try:
        metadata = json.loads(row["metadata_json"])
    except (TypeError, json.JSONDecodeError):
        metadata = {}
    return {
        "id": row["id"],
        "observation_id": row["id"],
        "session_id": row["session_id"],
        "content": row["content"],
        "source_event_ids": _json_list(row["source_event_ids_json"]),
        "kind": row["kind"],
        "importance": row["importance"],
        "confidence": row["confidence"],
        "relevance": row["relevance"],
        "status": row["status"],
        "active": row["status"] == "active",
        "created_at": row["created_at"],
        "archived_at": row["archived_at"],
        "reflected_at": row["reflected_at"],
        "metadata": metadata if isinstance(metadata, dict) else {},
    }


def _reflection_row(row: sqlite3.Row) -> dict[str, Any]:
    try:
        metadata = json.loads(row["metadata_json"])
    except (TypeError, json.JSONDecodeError):
        metadata = {}
    support_ids = _json_list(row["supporting_observation_ids_json"])
    return {
        "id": row["id"],
        "reflection_id": row["id"],
        "session_id": row["session_id"],
        "content": row["content"],
        "supporting_observation_ids": support_ids,
        "support_ids": support_ids,
        "kind": row["kind"],
        "supersedes": _json_list(row["supersedes_json"]),
        "scope": row["scope"],
        "status": row["status"],
        "active": row["status"] == "active",
        "created_at": row["created_at"],
        "archived_at": row["archived_at"],
        "promoted_at": row["promoted_at"],
        "promoted_path": row["promoted_path"],
        "metadata": metadata if isinstance(metadata, dict) else {},
    }


__all__ = [
    "LeaseError",
    "ObservationMemoryError",
    "ObservationMemoryStore",
    "stable_id",
]
