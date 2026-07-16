from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


TRACE_STATUSES = frozenset({"running", "ok", "error", "cancelled", "abandoned"})


@dataclass(slots=True)
class TraceContext:
    trace_id: str
    span_id: str
    request_id: str = ""
    session_key: str = ""


@dataclass(slots=True)
class ObservationEvent:
    operation: str
    payload: dict[str, Any] = field(default_factory=dict)

