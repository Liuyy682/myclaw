from __future__ import annotations

"""Durable records for user requests accepted by the dispatcher.

This is intentionally a small, synchronous SQLite store.  Request admission
is synchronous from the dispatcher's point of view, so a request is never
placed on the in-memory bus before its ``queued`` row is durable.
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY,
    request_id TEXT NOT NULL,
    session_key TEXT NOT NULL,
    channel TEXT NOT NULL,
    chat_id TEXT NOT NULL,
    content TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'normal',
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'completed', 'failed', 'interrupted')),
    error TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""


@dataclass(frozen=True, slots=True)
class RequestRecord:
    id: int
    request_id: str
    session_key: str
    channel: str
    chat_id: str
    content: str
    mode: str
    status: str
    error: str | None
    metadata: dict[str, Any]
    created_at: str
    updated_at: str

    @property
    def reason(self) -> str | None:
        """Human-readable failure/interruption reason, if one was recorded."""

        return self.error

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "request_id": self.request_id,
            "session_key": self.session_key,
            "channel": self.channel,
            "chat_id": self.chat_id,
            "content": self.content,
            "mode": self.mode,
            "status": self.status,
            "error": self.error,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class RequestStore:
    """Persist the minimal lifecycle of accepted user requests.

    ``workspace/runtime/requests.db`` is the only database path and the
    database has one application table, ``requests``.  A new store instance
    exposes an explicit incomplete-row reconciliation method.  The dispatcher
    invokes it at runtime startup, after the store has been opened.
    """

    def __init__(self, workspace: Path | str) -> None:
        self.workspace = Path(workspace).expanduser()
        self.runtime_dir = self.workspace / "runtime"
        self.path = self.runtime_dir / "requests.db"
        self.db_path = self.path
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(_TABLE_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    def create_queued(
        self,
        request_id: str,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        content: str,
        mode: str = "normal",
        metadata: dict[str, Any] | None = None,
    ) -> RequestRecord:
        now = self._now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO requests
                    (request_id, session_key, channel, chat_id, content, mode,
                     status, metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?)
                """,
                (
                    request_id,
                    session_key,
                    channel,
                    chat_id,
                    content,
                    mode,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            internal_id = int(cursor.lastrowid)
        record = self._get_by_id(internal_id)
        if record is None:  # pragma: no cover - defensive against a broken adapter
            raise RuntimeError(f"request {request_id} was not persisted")
        return record

    # ``create`` is a convenient public alias for focused callers/tests.
    create = create_queued

    def _transition(self, internal_id: int, status: str, error: str | None = None) -> RequestRecord:
        now = self._now()
        with self._connect() as connection:
            current = connection.execute(
                "SELECT status FROM requests WHERE id = ?", (internal_id,)
            ).fetchone()
            if current is None:
                raise KeyError(f"unknown request: {internal_id}")
            if current["status"] in {"completed", "failed", "interrupted"}:
                record = self.get(internal_id)
                assert record is not None
                return record
            cursor = connection.execute(
                "UPDATE requests SET status = ?, error = ?, updated_at = ? WHERE id = ?",
                (status, error, now, internal_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown request: {internal_id}")
        record = self.get(internal_id)
        if record is None:  # pragma: no cover - defensive against a broken adapter
            raise KeyError(f"unknown request: {internal_id}")
        return record

    def mark_running(self, internal_id: int) -> RequestRecord:
        return self._transition(internal_id, "running")

    def mark_completed(self, internal_id: int) -> RequestRecord:
        return self._transition(internal_id, "completed")

    def mark_failed(self, internal_id: int, error: str | None = None) -> RequestRecord:
        return self._transition(internal_id, "failed", error)

    def mark_interrupted(self, internal_id: int, error: str | None = None) -> RequestRecord:
        return self._transition(internal_id, "interrupted", error)

    def mark_incomplete_interrupted(self) -> int:
        """Mark rows left queued/running by a prior process as interrupted."""

        now = self._now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE requests
                   SET status = 'interrupted',
                       error = COALESCE(error, 'Request interrupted before the runtime started.'),
                       updated_at = ?
                 WHERE status IN ('queued', 'running')
                """,
                (now,),
            )
            return cursor.rowcount

    # Compatibility-oriented name for callers that describe this as startup
    # recovery rather than incomplete-row reconciliation.
    recover_incomplete = mark_incomplete_interrupted
    recover_on_startup = mark_incomplete_interrupted

    def get(self, request_id: str | int) -> RequestRecord | None:
        with self._connect() as connection:
            if isinstance(request_id, int):
                row = connection.execute("SELECT * FROM requests WHERE id = ?", (request_id,)).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM requests WHERE request_id = ? ORDER BY id DESC LIMIT 1", (request_id,)
                ).fetchone()
        return self._record(row) if row is not None else None

    def _get_by_id(self, internal_id: int) -> RequestRecord | None:
        return self.get(internal_id)

    def list_recent(
        self,
        session_key: str,
        *,
        limit: int = 5,
        status: str | None = None,
    ) -> list[RequestRecord]:
        if limit < 1:
            return []
        query = "SELECT * FROM requests WHERE session_key = ?"
        parameters: list[Any] = [session_key]
        if status is not None:
            query += " AND status = ?"
            parameters.append(status)
        query += " ORDER BY updated_at DESC, rowid DESC LIMIT ?"
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._record(row) for row in rows]

    # Useful explicit spelling for the /status consumer.
    def list_recent_interrupted(self, session_key: str, limit: int = 5) -> list[RequestRecord]:
        return self.list_recent(session_key, limit=limit, status="interrupted")

    @staticmethod
    def _record(row: sqlite3.Row) -> RequestRecord:
        try:
            metadata = json.loads(row["metadata_json"])
        except (TypeError, json.JSONDecodeError):
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        return RequestRecord(
            id=int(row["id"]),
            request_id=str(row["request_id"]),
            session_key=str(row["session_key"]),
            channel=str(row["channel"]),
            chat_id=str(row["chat_id"]),
            content=str(row["content"]),
            mode=str(row["mode"]),
            status=str(row["status"]),
            error=row["error"],
            metadata=metadata,
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )
