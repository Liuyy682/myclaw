from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from myclaw.observability.types import ObservationEvent


class ObservabilityStore:
    """SQLite persistence and read models for local observability data."""

    def __init__(self, workspace: Path | str) -> None:
        self.directory = Path(workspace).expanduser() / "observability"
        self.path = self.directory / "observability.db"

    def initialize(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    root_span_id TEXT NOT NULL,
                    request_id TEXT NOT NULL DEFAULT '',
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    duration_ms REAL,
                    session_key TEXT NOT NULL DEFAULT '',
                    channel TEXT NOT NULL DEFAULT '',
                    model TEXT NOT NULL DEFAULT '',
                    error_type TEXT,
                    error_message TEXT,
                    attributes_json TEXT NOT NULL DEFAULT '{}',
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER
                );
                CREATE TABLE IF NOT EXISTS spans (
                    span_id TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL,
                    parent_span_id TEXT,
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT NOT NULL,
                    duration_ms REAL NOT NULL,
                    error_type TEXT,
                    error_message TEXT,
                    attributes_json TEXT NOT NULL DEFAULT '{}',
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    FOREIGN KEY(trace_id) REFERENCES traces(trace_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    component TEXT NOT NULL,
                    message TEXT NOT NULL,
                    trace_id TEXT NOT NULL DEFAULT '',
                    span_id TEXT NOT NULL DEFAULT '',
                    request_id TEXT NOT NULL DEFAULT '',
                    session_key TEXT NOT NULL DEFAULT '',
                    error_type TEXT,
                    error_message TEXT,
                    attributes_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_traces_started ON traces(started_at DESC);
                CREATE INDEX IF NOT EXISTS idx_traces_status ON traces(status, started_at DESC);
                CREATE INDEX IF NOT EXISTS idx_traces_kind ON traces(kind, started_at DESC);
                CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id, started_at);
                CREATE INDEX IF NOT EXISTS idx_spans_kind ON spans(kind, started_at DESC);
                CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_logs_trace ON logs(trace_id, timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_logs_level ON logs(level, timestamp DESC);
                PRAGMA user_version = 1;
                """
            )
            now = _utc_now()
            connection.execute(
                """
                UPDATE traces
                SET status = 'abandoned', ended_at = ?,
                    duration_ms = MAX(0, (julianday(?) - julianday(started_at)) * 86400000.0)
                WHERE status = 'running'
                """,
                (now, now),
            )

    def write_batch(self, events: Iterable[ObservationEvent]) -> None:
        with self._connect() as connection:
            for event in events:
                payload = event.payload
                if event.operation == "trace_start":
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO traces (
                            trace_id, root_span_id, request_id, name, kind, status,
                            started_at, session_key, channel, model, attributes_json
                        ) VALUES (?, ?, ?, ?, ?, 'running', ?, ?, ?, ?, ?)
                        """,
                        (
                            payload["trace_id"], payload["root_span_id"], payload.get("request_id", ""),
                            payload["name"], payload["kind"], payload["started_at"],
                            payload.get("session_key", ""), payload.get("channel", ""),
                            payload.get("model", ""), _json(payload.get("attributes", {})),
                        ),
                    )
                elif event.operation == "trace_end":
                    connection.execute(
                        """
                        UPDATE traces SET status = ?, ended_at = ?, duration_ms = ?,
                            error_type = ?, error_message = ?, attributes_json = ?
                        WHERE trace_id = ?
                        """,
                        (
                            payload["status"], payload["ended_at"], payload["duration_ms"],
                            payload.get("error_type"), payload.get("error_message"),
                            _json(payload.get("attributes", {})), payload["trace_id"],
                        ),
                    )
                elif event.operation == "span":
                    inserted = connection.execute(
                        """
                        INSERT OR IGNORE INTO spans (
                            span_id, trace_id, parent_span_id, name, kind, status,
                            started_at, ended_at, duration_ms, error_type, error_message,
                            attributes_json, prompt_tokens, completion_tokens, total_tokens
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            payload["span_id"], payload["trace_id"], payload.get("parent_span_id"),
                            payload["name"], payload["kind"], payload["status"],
                            payload["started_at"], payload["ended_at"], payload["duration_ms"],
                            payload.get("error_type"), payload.get("error_message"),
                            _json(payload.get("attributes", {})), payload.get("prompt_tokens"),
                            payload.get("completion_tokens"), payload.get("total_tokens"),
                        ),
                    )
                    if inserted.rowcount:
                        self._add_trace_usage(connection, payload)
                elif event.operation == "log":
                    connection.execute(
                        """
                        INSERT INTO logs (
                            timestamp, level, component, message, trace_id, span_id,
                            request_id, session_key, error_type, error_message, attributes_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            payload["timestamp"], payload["level"], payload["component"],
                            payload["message"], payload.get("trace_id", ""),
                            payload.get("span_id", ""), payload.get("request_id", ""),
                            payload.get("session_key", ""), payload.get("error_type"),
                            payload.get("error_message"), _json(payload.get("attributes", {})),
                        ),
                    )

    @staticmethod
    def _add_trace_usage(connection: sqlite3.Connection, payload: dict[str, Any]) -> None:
        values = (
            payload.get("prompt_tokens"),
            payload.get("completion_tokens"),
            payload.get("total_tokens"),
        )
        if all(value is None for value in values):
            return
        connection.execute(
            """
            UPDATE traces SET
                prompt_tokens = COALESCE(prompt_tokens, 0) + COALESCE(?, 0),
                completion_tokens = COALESCE(completion_tokens, 0) + COALESCE(?, 0),
                total_tokens = COALESCE(total_tokens, 0) + COALESCE(?, 0)
            WHERE trace_id = ?
            """,
            (*values, payload["trace_id"]),
        )

    def summary(self, since: str) -> dict[str, Any]:
        with self._connect() as connection:
            traces = connection.execute(
                "SELECT * FROM traces WHERE started_at >= ? ORDER BY started_at",
                (since,),
            ).fetchall()
            spans = connection.execute(
                "SELECT kind, status, duration_ms FROM spans WHERE started_at >= ?",
                (since,),
            ).fetchall()
        durations = [float(row["duration_ms"]) for row in traces if row["duration_ms"] is not None]
        queue_durations = [float(row["duration_ms"]) for row in spans if row["kind"] == "queue"]
        completed = [row for row in traces if row["status"] not in {"running", "abandoned"}]
        successful = sum(row["status"] == "ok" for row in completed)
        series: dict[str, dict[str, Any]] = {}
        for row in traces:
            bucket = str(row["started_at"])[:13] + ":00:00Z"
            item = series.setdefault(bucket, {"bucket": bucket, "requests": 0, "errors": 0})
            item["requests"] += 1
            item["errors"] += int(row["status"] == "error")
        return {
            "requests": len(traces),
            "running": sum(row["status"] == "running" for row in traces),
            "errors": sum(row["status"] == "error" for row in traces),
            "success_rate": round(successful / len(completed), 4) if completed else None,
            "duration_ms": {"p50": _percentile(durations, 50), "p95": _percentile(durations, 95)},
            "queue_wait_ms": {"p95": _percentile(queue_durations, 95)},
            "llm_calls": sum(row["kind"] == "llm" for row in spans),
            "tool_calls": sum(row["kind"] == "tool" for row in spans),
            "tool_errors": sum(row["kind"] == "tool" and row["status"] == "error" for row in spans),
            "tokens": {
                "prompt": sum(row["prompt_tokens"] or 0 for row in traces),
                "completion": sum(row["completion_tokens"] or 0 for row in traces),
                "total": sum(row["total_tokens"] or 0 for row in traces),
                "available": any(row["total_tokens"] is not None for row in traces),
            },
            "series": list(series.values()),
        }

    def list_traces(
        self,
        since: str,
        *,
        status: str | None = None,
        kind: str | None = None,
        session_key: str | None = None,
        before: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        clauses = ["started_at >= ?"]
        params: list[Any] = [since]
        for column, value in (("status", status), ("kind", kind), ("session_key", session_key)):
            if value:
                clauses.append(f"{column} = ?")
                params.append(value)
        if before:
            clauses.append("started_at < ?")
            params.append(before)
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM traces WHERE {' AND '.join(clauses)} "
                "ORDER BY started_at DESC, trace_id DESC LIMIT ?",
                params,
            ).fetchall()
        return [_trace_row(row) for row in rows]

    def trace_detail(self, trace_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            trace = connection.execute("SELECT * FROM traces WHERE trace_id = ?", (trace_id,)).fetchone()
            if trace is None:
                return None
            spans = connection.execute(
                "SELECT * FROM spans WHERE trace_id = ? ORDER BY started_at, span_id", (trace_id,)
            ).fetchall()
            logs = connection.execute(
                "SELECT * FROM logs WHERE trace_id = ? ORDER BY timestamp, id", (trace_id,)
            ).fetchall()
        return {
            "trace": _trace_row(trace),
            "spans": [_span_row(row) for row in spans],
            "logs": [_log_row(row) for row in logs],
        }

    def list_logs(
        self,
        since: str,
        *,
        level: str | None = None,
        component: str | None = None,
        trace_id: str | None = None,
        query: str | None = None,
        before: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        clauses = ["timestamp >= ?"]
        params: list[Any] = [since]
        for column, value in (("level", level), ("component", component), ("trace_id", trace_id)):
            if value:
                clauses.append(f"{column} = ?")
                params.append(value)
        if query:
            clauses.append("message LIKE ? ESCAPE '\\'")
            params.append(f"%{_escape_like(query)}%")
        if before:
            clauses.append("timestamp < ?")
            params.append(before)
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM logs WHERE {' AND '.join(clauses)} "
                "ORDER BY timestamp DESC, id DESC LIMIT ?",
                params,
            ).fetchall()
        return [_log_row(row) for row in rows]

    def cleanup(self, *, retention_days: int, max_bytes: int) -> None:
        cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()
        with self._connect() as connection:
            connection.execute("DELETE FROM logs WHERE timestamp < ?", (cutoff,))
            connection.execute("DELETE FROM traces WHERE started_at < ? AND status != 'running'", (cutoff,))
            for _ in range(1000):
                if self._effective_bytes(connection) <= max_bytes:
                    break
                ids = [
                    row[0]
                    for row in connection.execute(
                        "SELECT trace_id FROM traces WHERE status != 'running' ORDER BY started_at LIMIT 100"
                    ).fetchall()
                ]
                if ids:
                    placeholders = ",".join("?" for _ in ids)
                    connection.execute(f"DELETE FROM traces WHERE trace_id IN ({placeholders})", ids)
                    continue
                deleted = connection.execute(
                    "DELETE FROM logs WHERE id IN (SELECT id FROM logs ORDER BY timestamp LIMIT 1000)"
                ).rowcount
                if not deleted:
                    break
            connection.commit()
            connection.execute("PRAGMA wal_checkpoint(PASSIVE)")

    @staticmethod
    def _effective_bytes(connection: sqlite3.Connection) -> int:
        page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
        page_count = int(connection.execute("PRAGMA page_count").fetchone()[0])
        freelist = int(connection.execute("PRAGMA freelist_count").fetchone()[0])
        return max(0, page_count - freelist) * page_size

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _decode_json(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _trace_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "trace_id": row["trace_id"], "root_span_id": row["root_span_id"],
        "request_id": row["request_id"], "name": row["name"], "kind": row["kind"],
        "status": row["status"], "started_at": row["started_at"], "ended_at": row["ended_at"],
        "duration_ms": row["duration_ms"], "session_key": row["session_key"],
        "channel": row["channel"], "model": row["model"], "error_type": row["error_type"],
        "error_message": row["error_message"], "attributes": _decode_json(row["attributes_json"]),
        "prompt_tokens": row["prompt_tokens"], "completion_tokens": row["completion_tokens"],
        "total_tokens": row["total_tokens"],
    }


def _span_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "span_id": row["span_id"], "trace_id": row["trace_id"],
        "parent_span_id": row["parent_span_id"], "name": row["name"], "kind": row["kind"],
        "status": row["status"], "started_at": row["started_at"], "ended_at": row["ended_at"],
        "duration_ms": row["duration_ms"], "error_type": row["error_type"],
        "error_message": row["error_message"], "attributes": _decode_json(row["attributes_json"]),
        "prompt_tokens": row["prompt_tokens"], "completion_tokens": row["completion_tokens"],
        "total_tokens": row["total_tokens"],
    }


def _log_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"], "timestamp": row["timestamp"], "level": row["level"],
        "component": row["component"], "message": row["message"], "trace_id": row["trace_id"],
        "span_id": row["span_id"], "request_id": row["request_id"],
        "session_key": row["session_key"], "error_type": row["error_type"],
        "error_message": row["error_message"], "attributes": _decode_json(row["attributes_json"]),
    }


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((percentile / 100) * (len(ordered) - 1))))
    return round(ordered[index], 3)


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
