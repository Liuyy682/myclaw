from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import re
import sys
import threading
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from myclaw.observability.store import ObservabilityStore
from myclaw.observability.types import ObservationEvent, TRACE_STATUSES, TraceContext
from myclaw.config import (
    DEFAULT_LOG_FORMAT,
    DEFAULT_LOG_LEVEL,
    DEFAULT_OBSERVABILITY_ENABLED,
    DEFAULT_OBSERVABILITY_MAX_MB,
    DEFAULT_OBSERVABILITY_RETENTION_DAYS,
    LOG_FORMAT_ENV_VAR,
    LOG_LEVEL_ENV_VAR,
    OBSERVABILITY_ENABLED_ENV_VAR,
    OBSERVABILITY_MAX_MB_ENV_VAR,
    OBSERVABILITY_RETENTION_DAYS_ENV_VAR,
)


_CURRENT_CONTEXT: ContextVar[TraceContext | None] = ContextVar("myclaw_trace_context", default=None)
_CURRENT_RUNTIME: ContextVar[Any] = ContextVar("myclaw_observability_runtime", default=None)
_SECRET_PATTERN = re.compile(
    r"(?i)(authorization\s*[:=]\s*bearer\s+|api[_-]?key\s*[:=]\s*|secret\s*[:=]\s*|token\s*[:=]\s*)[^\s,;]+"
)


@dataclass(slots=True)
class ObservabilityConfig:
    enabled: bool = DEFAULT_OBSERVABILITY_ENABLED
    log_level: str = DEFAULT_LOG_LEVEL
    log_format: str = DEFAULT_LOG_FORMAT
    retention_days: int = DEFAULT_OBSERVABILITY_RETENTION_DAYS
    max_bytes: int = DEFAULT_OBSERVABILITY_MAX_MB * 1024 * 1024

    @classmethod
    def from_env(cls) -> ObservabilityConfig:
        return cls(
            enabled=_env_bool(OBSERVABILITY_ENABLED_ENV_VAR, DEFAULT_OBSERVABILITY_ENABLED),
            log_level=os.environ.get(LOG_LEVEL_ENV_VAR, DEFAULT_LOG_LEVEL).upper(),
            log_format=os.environ.get(LOG_FORMAT_ENV_VAR, DEFAULT_LOG_FORMAT).lower(),
            retention_days=max(1, _env_int(
                OBSERVABILITY_RETENTION_DAYS_ENV_VAR, DEFAULT_OBSERVABILITY_RETENTION_DAYS
            )),
            max_bytes=max(1, _env_int(OBSERVABILITY_MAX_MB_ENV_VAR, DEFAULT_OBSERVABILITY_MAX_MB)) * 1024 * 1024,
        )


@dataclass(slots=True)
class SpanHandle:
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "running"
    error_type: str | None = None
    error_message: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = _safe_value(value)

    def set_status(self, status: str) -> None:
        if status in TRACE_STATUSES:
            self.status = status

    def set_error(self, error: BaseException | str, *, error_type: str | None = None) -> None:
        self.status = "error"
        self.error_type = error_type or (type(error).__name__ if isinstance(error, BaseException) else "Error")
        self.error_message = _redact(str(error))

    def set_usage(self, prompt: int | None, completion: int | None, total: int | None) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = total


@dataclass(slots=True)
class _FlushRequest:
    completed: threading.Event = field(default_factory=threading.Event)


class _ObservationLogHandler(logging.Handler):
    def __init__(self, runtime: ObservabilityRuntime) -> None:
        super().__init__(level=getattr(logging, runtime.config.log_level, logging.INFO))
        self.runtime = runtime

    def emit(self, record: logging.LogRecord) -> None:
        if record.name != "myclaw" and not record.name.startswith("myclaw."):
            return
        try:
            context = current_trace_context()
            error_type = None
            error_message = None
            if record.exc_info and record.exc_info[1] is not None:
                error_type = type(record.exc_info[1]).__name__
                error_message = _redact(str(record.exc_info[1]))
            self.runtime.emit(
                ObservationEvent(
                    "log",
                    {
                        "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
                        "level": record.levelname,
                        "component": record.name,
                        "message": _redact(record.getMessage()),
                        "trace_id": getattr(record, "trace_id", "") or (context.trace_id if context else ""),
                        "span_id": getattr(record, "span_id", "") or (context.span_id if context else ""),
                        "request_id": getattr(record, "request_id", "") or (context.request_id if context else ""),
                        "session_key": getattr(record, "session_key", "") or (context.session_key if context else ""),
                        "error_type": error_type,
                        "error_message": error_message,
                    },
                )
            )
        except Exception:
            self.handleError(record)


class _JsonConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
                "level": record.levelname,
                "component": record.name,
                "message": _redact(record.getMessage()),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )


class _MyClawLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.name == "myclaw" or record.name.startswith("myclaw.")


class ObservabilityRuntime:
    """Process-local tracer, structured log sink, and query facade."""

    def __init__(self, workspace: Path | str, config: ObservabilityConfig | None = None) -> None:
        self.config = config or ObservabilityConfig.from_env()
        self.store = ObservabilityStore(workspace)
        self._queue: queue.Queue[ObservationEvent | _FlushRequest | object] = queue.Queue()
        self._stop_token = object()
        self._thread: threading.Thread | None = None
        self._started = False
        self._log_handler: logging.Handler | None = None
        self._console_handler: logging.Handler | None = None
        self._last_cleanup = 0.0
        self._last_warning = 0.0

    def start(self) -> None:
        if not self.config.enabled or self._started:
            return
        try:
            self.store.initialize()
            self.store.cleanup(
                retention_days=self.config.retention_days,
                max_bytes=self.config.max_bytes,
            )
        except Exception as exc:
            self._fallback(f"observability startup failed: {exc}")
            return
        self._started = True
        self._last_cleanup = time.monotonic()
        self._thread = threading.Thread(target=self._writer_loop, name="myclaw-observability", daemon=True)
        self._thread.start()
        self._install_logging()

    def stop(self) -> None:
        if not self._started:
            return
        self._remove_logging()
        self._queue.put(self._stop_token)
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._thread = None
        self._started = False

    def flush(self, timeout: float = 2.0) -> None:
        if not self._started:
            return
        request = _FlushRequest()
        self._queue.put(request)
        request.completed.wait(timeout)

    def emit(self, event: ObservationEvent) -> None:
        if not self.config.enabled:
            return
        if not self._started:
            self.start()
        if self._started:
            self._queue.put(event)

    @contextlib.contextmanager
    def trace(
        self,
        name: str,
        kind: str,
        *,
        request_id: str = "",
        session_key: str = "",
        channel: str = "",
        model: str = "",
        attributes: dict[str, Any] | None = None,
        started_at: datetime | None = None,
        trace_id: str | None = None,
    ) -> Iterator[SpanHandle]:
        if not self.config.enabled:
            yield SpanHandle()
            return
        self.start()
        trace_id = trace_id or uuid.uuid4().hex
        root_span_id = _span_id()
        started = started_at or datetime.now(UTC)
        payload = {
            "trace_id": trace_id,
            "root_span_id": root_span_id,
            "request_id": request_id,
            "name": name,
            "kind": kind,
            "started_at": _iso(started),
            "session_key": session_key,
            "channel": channel,
            "model": model,
            "attributes": _safe_attributes(attributes),
        }
        self.emit(ObservationEvent("trace_start", payload))
        handle = SpanHandle(attributes=dict(payload["attributes"]))
        token = _CURRENT_CONTEXT.set(TraceContext(trace_id, root_span_id, request_id, session_key))
        runtime_token = _CURRENT_RUNTIME.set(self)
        try:
            yield handle
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                handle.set_status("cancelled")
            elif exc.__class__.__name__ == "CancelledError":
                handle.set_status("cancelled")
                handle.error_type = "CancelledError"
            else:
                handle.set_error(exc)
            raise
        finally:
            _CURRENT_CONTEXT.reset(token)
            _CURRENT_RUNTIME.reset(runtime_token)
            ended = datetime.now(UTC)
            if handle.status == "running":
                handle.status = "ok"
            self.emit(
                ObservationEvent(
                    "trace_end",
                    {
                        "trace_id": trace_id,
                        "status": handle.status,
                        "ended_at": _iso(ended),
                        "duration_ms": _duration_ms(started, ended),
                        "error_type": handle.error_type,
                        "error_message": handle.error_message,
                        "attributes": _safe_attributes(handle.attributes),
                    },
                )
            )

    @contextlib.contextmanager
    def span(
        self,
        name: str,
        kind: str,
        *,
        attributes: dict[str, Any] | None = None,
        started_at: datetime | None = None,
    ) -> Iterator[SpanHandle]:
        context = current_trace_context()
        if not self.config.enabled or context is None:
            yield SpanHandle()
            return
        span_id = _span_id()
        started = started_at or datetime.now(UTC)
        handle = SpanHandle(attributes=_safe_attributes(attributes))
        token = _CURRENT_CONTEXT.set(
            TraceContext(context.trace_id, span_id, context.request_id, context.session_key)
        )
        try:
            yield handle
        except BaseException as exc:
            if exc.__class__.__name__ == "CancelledError":
                handle.set_status("cancelled")
                handle.error_type = "CancelledError"
            else:
                handle.set_error(exc)
            raise
        finally:
            _CURRENT_CONTEXT.reset(token)
            ended = datetime.now(UTC)
            if handle.status == "running":
                handle.status = "ok"
            self.emit(
                ObservationEvent(
                    "span",
                    {
                        "span_id": span_id,
                        "trace_id": context.trace_id,
                        "parent_span_id": context.span_id,
                        "name": name,
                        "kind": kind,
                        "status": handle.status,
                        "started_at": _iso(started),
                        "ended_at": _iso(ended),
                        "duration_ms": _duration_ms(started, ended),
                        "error_type": handle.error_type,
                        "error_message": handle.error_message,
                        "attributes": _safe_attributes(handle.attributes),
                        "prompt_tokens": handle.prompt_tokens,
                        "completion_tokens": handle.completion_tokens,
                        "total_tokens": handle.total_tokens,
                    },
                )
            )

    def record_completed_span(
        self,
        name: str,
        kind: str,
        *,
        started_at: datetime,
        ended_at: datetime,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        context = current_trace_context()
        if context is None:
            return
        self.emit(
            ObservationEvent(
                "span",
                {
                    "span_id": _span_id(), "trace_id": context.trace_id,
                    "parent_span_id": context.span_id, "name": name, "kind": kind,
                    "status": "ok", "started_at": _iso(started_at), "ended_at": _iso(ended_at),
                    "duration_ms": _duration_ms(started_at, ended_at),
                    "attributes": _safe_attributes(attributes),
                },
            )
        )

    def summary(self, since: str) -> dict[str, Any]:
        self.flush()
        return self.store.summary(since)

    def list_traces(self, since: str, **filters: Any) -> list[dict[str, Any]]:
        self.flush()
        return self.store.list_traces(since, **filters)

    def trace_detail(self, trace_id: str) -> dict[str, Any] | None:
        self.flush()
        return self.store.trace_detail(trace_id)

    def list_logs(self, since: str, **filters: Any) -> list[dict[str, Any]]:
        self.flush()
        return self.store.list_logs(since, **filters)

    def _writer_loop(self) -> None:
        pending: list[ObservationEvent] = []
        stopping = False
        while not stopping:
            try:
                item = self._queue.get(timeout=0.25)
            except queue.Empty:
                item = None
            if item is self._stop_token:
                stopping = True
                self._queue.task_done()
            elif isinstance(item, ObservationEvent):
                pending.append(item)
                self._queue.task_done()
            flush_request = item if isinstance(item, _FlushRequest) else None
            if pending and (stopping or item is None or flush_request is not None or len(pending) >= 100):
                try:
                    self.store.write_batch(pending)
                except Exception as exc:
                    self._fallback(f"observability write failed: {exc}")
                pending.clear()
            if flush_request is not None:
                flush_request.completed.set()
                self._queue.task_done()
            if time.monotonic() - self._last_cleanup >= 3600:
                try:
                    self.store.cleanup(
                        retention_days=self.config.retention_days,
                        max_bytes=self.config.max_bytes,
                    )
                except Exception as exc:
                    self._fallback(f"observability cleanup failed: {exc}")
                self._last_cleanup = time.monotonic()

    def _install_logging(self) -> None:
        root = logging.getLogger()
        root.setLevel(getattr(logging, self.config.log_level, logging.INFO))
        self._log_handler = _ObservationLogHandler(self)
        self._log_handler.addFilter(_MyClawLogFilter())
        root.addHandler(self._log_handler)
        self._console_handler = logging.StreamHandler()
        self._console_handler.addFilter(_MyClawLogFilter())
        if self.config.log_format == "json":
            self._console_handler.setFormatter(_JsonConsoleFormatter())
        else:
            self._console_handler.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s"
            ))
        root.addHandler(self._console_handler)

    def _remove_logging(self) -> None:
        root = logging.getLogger()
        for handler in (self._log_handler, self._console_handler):
            if handler is not None:
                root.removeHandler(handler)
                handler.close()
        self._log_handler = None
        self._console_handler = None

    def _fallback(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_warning < 60:
            return
        self._last_warning = now
        print(f"MyClaw observability warning: {message}", file=sys.stderr, flush=True)


def current_trace_context() -> TraceContext | None:
    return _CURRENT_CONTEXT.get()


def current_observability() -> ObservabilityRuntime | None:
    return _CURRENT_RUNTIME.get()


def _safe_attributes(value: dict[str, Any] | None) -> dict[str, Any]:
    if not value:
        return {}
    return {str(key): _safe_value(item) for key, item in value.items()}


def _safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _redact(value)[:1000]
    if isinstance(value, (list, tuple, set)):
        return [_safe_value(item) for item in list(value)[:50]]
    return str(value)[:1000]


def _redact(value: str) -> str:
    return _SECRET_PATTERN.sub(lambda match: f"{match.group(1)}[REDACTED]", value)[:2000]


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _duration_ms(started: datetime, ended: datetime) -> float:
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    if ended.tzinfo is None:
        ended = ended.replace(tzinfo=UTC)
    return round(max(0.0, (ended - started).total_seconds() * 1000), 3)


def _span_id() -> str:
    return uuid.uuid4().hex[:16]


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default
