"""No-tool background extraction of observations and reflections.

The storage layer owns leases and transactions.  This module owns only the
provider boundary and validates the complete model response before asking the
store to commit it.  It intentionally assumes a small concrete
``ObservationMemoryStore`` API so the interactive agent does not share any of
the worker's implementation details.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar


logger = logging.getLogger(__name__)

DEFAULT_REFLECTION_BATCH_SIZE = 5
OBSERVATION_KINDS = frozenset(
    {
        "operation",
        "result",
        "decision",
        "constraint",
        "progress",
        "preference",
        "open_issue",
    }
)
_OBSERVATION_FIELDS = frozenset(
    {"id", "content", "kind", "importance", "confidence", "source_event_ids", "timestamp"}
)
_REFLECTION_FIELDS = frozenset({"id", "content", "kind", "supports", "supersedes"})
_MISSING = object()
T = TypeVar("T")


class ObservationOutputError(ValueError):
    """The provider returned an invalid observation document."""


class ReflectionOutputError(ValueError):
    """The provider returned an invalid reflection document."""


@dataclass(frozen=True, slots=True)
class Observation:
    content: str
    kind: str
    importance: float
    confidence: float
    source_event_ids: tuple[str, ...]
    id: str
    timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "content": self.content,
            "kind": self.kind,
            "importance": self.importance,
            "confidence": self.confidence,
            "source_event_ids": list(self.source_event_ids),
        }
        if self.timestamp is not None:
            result["timestamp"] = self.timestamp
        return result


@dataclass(frozen=True, slots=True)
class Reflection:
    content: str
    kind: str
    supports: tuple[str, ...]
    supersedes: tuple[str, ...]
    id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "kind": self.kind,
            "supports": list(self.supports),
            "supersedes": list(self.supersedes),
        }


class ObservationMemoryStore(Protocol):
    """Concrete store boundary used by :class:`ObservationMemoryWorker`.

    All methods are synchronous; the worker awaits only the provider.  The two
    commit methods must be atomic.  ``commit_observation_result(...,
    no_facts=True)`` is the durable marker for an intentionally empty result.
    """

    def claim_observation_job(self) -> Mapping[str, Any] | None: ...

    def commit_observation_result(
        self,
        job_id: str,
        observations: list[dict[str, Any]],
        *,
        no_facts: bool = False,
    ) -> Any: ...

    def fail_observation_job(self, job_id: str, error: str, *, retryable: bool = True) -> Any: ...

    def list_unreflected_observations(self, session_id: str) -> list[Mapping[str, Any]]: ...

    def next_reflection_session(self, batch_size: int) -> str | None: ...

    def commit_reflection_result(
        self, session_id: str, reflections: list[dict[str, Any]]
    ) -> Any: ...

    def fail_reflection(self, session_id: str, error: str, *, retryable: bool = True) -> Any: ...


@dataclass(frozen=True, slots=True)
class ProcessResult:
    """Outcome retained in ``worker.last_result`` and returned by a step."""

    outcome: str
    job_id: str | None = None
    session_id: str | None = None
    observation_count: int = 0
    reflection_count: int = 0
    error: str | None = None

    def __bool__(self) -> bool:
        return self.outcome in {"observed", "no_facts", "reflected"}


_OBSERVATION_SYSTEM = """Extract durable observations from the supplied source-event job.
Return only strict JSON, without markdown or commentary:
{"observations": [{"content": "...", "kind": "operation|result|decision|constraint|progress|preference|open_issue", "importance": 0.0, "confidence": 0.0, "source_event_ids": ["event-id"]}]}
importance and confidence must be finite numbers in [0, 1]. Every source event
id must be copied exactly from this job. An empty array means no_facts.
"""

_REFLECTION_SYSTEM = """Distill the supplied active observations into durable reflections.
Return only strict JSON, without markdown or commentary:
{"reflections": [{"content": "single-line durable fact", "kind": "category", "supports": ["observation-id"], "supersedes": []}]}
Every reflection must contain all four fields. supports may contain only ids
shown in the active observations; supersedes may be empty. An empty array means
no_facts.
"""


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-standard JSON constant: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_strict_json(raw: str | bytes | bytearray) -> Any:
    """Parse one complete JSON document, rejecting extensions and duplicates."""

    if isinstance(raw, (bytes, bytearray)):
        raw = bytes(raw).decode("utf-8")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("provider output must be non-empty JSON text")
    try:
        return json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON output: {exc}") from exc


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _score(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be a finite number in [0, 1]")
    return result


def _ids(value: Any, field: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{field} must be a non-empty id array")
    result: list[str] = []
    for raw_id in value:
        if isinstance(raw_id, bool) or not isinstance(raw_id, (str, int)):
            raise ValueError(f"{field} must contain string ids")
        item = str(raw_id).strip()
        if not item:
            raise ValueError(f"{field} must contain non-empty ids")
        if item in result:
            raise ValueError(f"{field} contains duplicate id: {item}")
        result.append(item)
    return tuple(result)


def _array(payload: Any, field: str) -> list[Any]:
    # A raw array is still strict JSON and is convenient for simple providers;
    # the object form is the documented contract.
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, Mapping) or set(payload) != {field}:
        raise ValueError(f"top-level JSON must contain only {field}")
    value = payload[field]
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return value


def _stable_id(prefix: str, *parts: Any) -> str:
    encoded = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{prefix}_{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:20]}"


def parse_observations(
    raw: str | bytes | bytearray,
    allowed_source_event_ids: Sequence[str],
    *,
    session_id: str = "",
) -> list[Observation]:
    """Validate all observations, including source provenance."""

    try:
        items = _array(parse_strict_json(raw), "observations")
        allowed = {str(value) for value in allowed_source_event_ids}
        result: list[Observation] = []
        seen: set[str] = set()
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise ValueError(f"observations[{index}] must be an object")
            unknown = sorted(set(item) - _OBSERVATION_FIELDS)
            if unknown:
                raise ValueError(f"observation contains unsupported fields: {', '.join(unknown)}")
            content = _text(item.get("content"), "content")
            kind = _text(item.get("kind"), "kind")
            if kind not in OBSERVATION_KINDS:
                raise ValueError(f"unsupported observation kind: {kind}")
            importance = _score(item.get("importance"), "importance")
            confidence = _score(item.get("confidence"), "confidence")
            source_ids = _ids(item.get("source_event_ids"), "source_event_ids")
            missing = sorted(set(source_ids) - allowed)
            if missing:
                raise ValueError("source_event_ids must come from job: " + ", ".join(missing))
            observation_id = item.get("id")
            if observation_id is None:
                observation_id = _stable_id("obs", session_id, index, content, source_ids)
            else:
                observation_id = _text(observation_id, "id")
            if observation_id in seen:
                raise ValueError(f"duplicate observation id: {observation_id}")
            seen.add(observation_id)
            timestamp = item.get("timestamp")
            if timestamp is not None:
                timestamp = _text(timestamp, "timestamp")
            result.append(
                Observation(
                    content=content,
                    kind=kind,
                    importance=importance,
                    confidence=confidence,
                    source_event_ids=source_ids,
                    id=observation_id,
                    timestamp=timestamp,
                )
            )
        return result
    except (TypeError, ValueError, KeyError) as exc:
        raise ObservationOutputError(str(exc)) from exc


def parse_reflections(
    raw: str | bytes | bytearray,
    allowed_observation_ids: Sequence[str],
    *,
    session_id: str = "",
) -> list[Reflection]:
    """Validate all reflections and restrict supports to existing ids."""

    try:
        items = _array(parse_strict_json(raw), "reflections")
        allowed = {str(value) for value in allowed_observation_ids}
        result: list[Reflection] = []
        seen: set[str] = set()
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise ValueError(f"reflections[{index}] must be an object")
            unknown = sorted(set(item) - _REFLECTION_FIELDS)
            if unknown:
                raise ValueError(f"reflection contains unsupported fields: {', '.join(unknown)}")
            missing = {"content", "kind", "supports", "supersedes"} - set(item)
            if missing:
                raise ValueError("reflection is missing: " + ", ".join(sorted(missing)))
            content = _text(item.get("content"), "content")
            if "\n" in content or "\r" in content:
                raise ValueError("reflection content must be a single line")
            kind = _text(item.get("kind"), "kind")
            supports = _ids(item.get("supports"), "supports")
            invalid = sorted(set(supports) - allowed)
            if invalid:
                raise ValueError("reflection supports unknown observations: " + ", ".join(invalid))
            supersedes = _ids(item.get("supersedes"), "supersedes", allow_empty=True)
            reflection_id = item.get("id")
            if reflection_id is None:
                reflection_id = _stable_id(
                    "ref", session_id, index, content, kind, supports, supersedes
                )
            else:
                reflection_id = _text(reflection_id, "id")
            if reflection_id in seen:
                raise ValueError(f"duplicate reflection id: {reflection_id}")
            seen.add(reflection_id)
            result.append(
                Reflection(
                    content=content,
                    kind=kind,
                    supports=supports,
                    supersedes=supersedes,
                    id=reflection_id,
                )
            )
        return result
    except (TypeError, ValueError, KeyError) as exc:
        raise ReflectionOutputError(str(exc)) from exc


async def _await(value: T | Awaitable[T]) -> T:
    if hasattr(value, "__await__"):
        return await value
    return value


def _response_text(response: Any) -> str | bytes | bytearray:
    if isinstance(response, (str, bytes, bytearray)):
        return response
    content = getattr(response, "content", _MISSING)
    if isinstance(content, (str, bytes, bytearray)):
        return content
    raise ValueError("provider response must contain JSON text")


def _job_value(job: Mapping[str, Any], *keys: str, default: Any = _MISSING) -> Any:
    for key in keys:
        if key in job:
            return job[key]
    if default is _MISSING:
        raise KeyError(keys[0])
    return default


def _job_parts(job: Mapping[str, Any]) -> tuple[str, str, list[Any], list[str]]:
    job_id = _text(_job_value(job, "job_id", "id"), "job_id")
    session_id = _text(_job_value(job, "session_id", "session_key"), "session_id")
    events_value = _job_value(job, "events", "source_events", default=[])
    if isinstance(events_value, Mapping):
        events = [
            ({"id": key, "content": event} if not isinstance(event, Mapping) else {"id": key, **event})
            for key, event in events_value.items()
        ]
    elif isinstance(events_value, Sequence) and not isinstance(events_value, (str, bytes, bytearray)):
        events = list(events_value)
    elif events_value in (None, ""):
        events = []
    else:
        events = [events_value]
    event_ids: list[str] = []
    for event in events:
        if isinstance(event, Mapping):
            event_id = _job_value(event, "id", "event_id", "source_event_id", default=None)
        else:
            event_id = getattr(event, "id", getattr(event, "event_id", None))
        if event_id is not None:
            event_ids.append(str(event_id))
    if not event_ids:
        declared = _job_value(job, "event_ids", "source_event_ids", default=[])
        if isinstance(declared, Sequence) and not isinstance(declared, (str, bytes, bytearray)):
            event_ids = [str(item) for item in declared if str(item).strip()]
    return job_id, session_id, events, list(dict.fromkeys(event_ids))


def _prompt_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


class ObservationMemoryWorker:
    """Process one observation job and, when due, one reflection batch."""

    def __init__(
        self,
        provider: Any,
        store: ObservationMemoryStore,
        *,
        batch_size: int = DEFAULT_REFLECTION_BATCH_SIZE,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.provider = provider
        self.store = store
        self.batch_size = batch_size
        self._running = False
        self.last_result = ProcessResult("idle")

    async def process_once(self) -> ProcessResult:
        """Claim one job and contain every failure at the background boundary."""

        if self._running:
            return ProcessResult("busy")
        self._running = True
        try:
            try:
                raw_job = self.store.claim_observation_job()
            except Exception as exc:
                logger.exception("observation job claim failed")
                return self._remember(ProcessResult("failed", error=str(exc)))
            if raw_job is None:
                session_id = self.store.next_reflection_session(self.batch_size)
                if session_id is None:
                    return self._remember(ProcessResult("idle"))
                reflection_count = await self._reflect_if_due(session_id)
                return self._remember(
                    ProcessResult(
                        "reflected" if reflection_count else "failed",
                        session_id=session_id,
                        reflection_count=reflection_count,
                    )
                )
            try:
                job_id, session_id, events, event_ids = _job_parts(raw_job)
            except Exception as exc:
                job_id = str(raw_job.get("job_id", raw_job.get("id", "unknown")))
                self._fail_observation(job_id, f"invalid observation job: {exc}")
                return self._remember(ProcessResult("failed", job_id=job_id, error=str(exc)))

            try:
                response = await self._complete(self._observation_messages(job_id, session_id, events, event_ids))
                observations = parse_observations(
                    _response_text(response), event_ids, session_id=session_id
                )
            except Exception as exc:
                self._fail_observation(job_id, str(exc))
                return self._remember(
                    ProcessResult("failed", job_id=job_id, session_id=session_id, error=str(exc))
                )

            try:
                self.store.commit_observation_result(
                    job_id,
                    [item.to_dict() for item in observations],
                    no_facts=not observations,
                )
            except Exception as exc:
                self._fail_observation(job_id, f"store commit failed: {exc}")
                return self._remember(
                    ProcessResult("failed", job_id=job_id, session_id=session_id, error=str(exc))
                )

            reflection_count = await self._reflect_if_due(session_id)
            outcome = "no_facts" if not observations else "observed"
            return self._remember(
                ProcessResult(
                    outcome,
                    job_id=job_id,
                    session_id=session_id,
                    observation_count=len(observations),
                    reflection_count=reflection_count,
                )
            )
        except Exception as exc:
            logger.exception("background observation processing failed")
            return self._remember(ProcessResult("failed", error=str(exc)))
        finally:
            self._running = False

    async def drain(self, *, max_jobs: int | None = None) -> int:
        if max_jobs is not None and (
            isinstance(max_jobs, bool) or not isinstance(max_jobs, int) or max_jobs < 1
        ):
            raise ValueError("max_jobs must be a positive integer when provided")
        count = 0
        while max_jobs is None or count < max_jobs:
            result = await self.process_once()
            if not result:
                break
            count += 1
        return count

    async def _complete(self, messages: list[dict[str, str]]) -> Any:
        complete = getattr(self.provider, "complete", None)
        if not callable(complete):
            raise AttributeError("provider must expose complete(messages)")
        # No tools/model kwargs: the extractor is a pure provider call.
        return await _await(complete(messages))

    @staticmethod
    def _observation_messages(
        job_id: str, session_id: str, events: Sequence[Any], event_ids: Sequence[str]
    ) -> list[dict[str, str]]:
        rendered = []
        for index, event in enumerate(events):
            if isinstance(event, Mapping):
                event_id = _job_value(event, "id", "event_id", "source_event_id", default=None)
            else:
                event_id = getattr(event, "id", getattr(event, "event_id", None))
            rendered.append(
                {
                    "id": str(event_id or (event_ids[index] if index < len(event_ids) else "")),
                    "event": event,
                }
            )
        payload = {"job_id": job_id, "session_id": session_id, "events": rendered}
        return [
            {"role": "system", "content": _OBSERVATION_SYSTEM},
            {"role": "user", "content": "Extract observations from this job:\n" + _prompt_json(payload)},
        ]

    async def _reflect_if_due(self, session_id: str) -> int:
        active = self.store.list_unreflected_observations(session_id)
        if len(active) < self.batch_size:
            return 0
        batch = active[: self.batch_size]
        ids = [str(_job_value(item, "id", "observation_id")) for item in batch]
        try:
            response = await self._complete(self._reflection_messages(session_id, batch))
            reflections = parse_reflections(_response_text(response), ids, session_id=session_id)
        except Exception as exc:
            self._fail_reflection(session_id, str(exc))
            return 0
        if not reflections:
            self._fail_reflection(session_id, "reflection returned no durable facts")
            return 0
        try:
            # One atomic store operation marks all support ids as reflected.
            self.store.commit_reflection_result(
                session_id, [item.to_dict() for item in reflections]
            )
        except Exception as exc:
            self._fail_reflection(session_id, f"store commit failed: {exc}")
            return 0
        return len(reflections)

    @staticmethod
    def _reflection_messages(
        session_id: str, observations: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, str]]:
        payload = {
            "session_id": session_id,
            "active_unreflected_observations": [dict(item) for item in observations],
        }
        return [
            {"role": "system", "content": _REFLECTION_SYSTEM},
            {"role": "user", "content": "Create reflections:\n" + _prompt_json(payload)},
        ]

    def _fail_observation(self, job_id: str, error: str) -> None:
        try:
            self.store.fail_observation_job(job_id, error, retryable=True)
        except Exception:
            logger.exception("could not record observation failure for %s", job_id)

    def _fail_reflection(self, session_id: str, error: str) -> None:
        try:
            self.store.fail_reflection(session_id, error, retryable=True)
        except Exception:
            logger.exception("could not record reflection failure for %s", session_id)

    def _remember(self, result: ProcessResult) -> ProcessResult:
        self.last_result = result
        return result


__all__ = [
    "DEFAULT_REFLECTION_BATCH_SIZE",
    "OBSERVATION_KINDS",
    "Observation",
    "Reflection",
    "ObservationMemoryStore",
    "ObservationOutputError",
    "ReflectionOutputError",
    "parse_strict_json",
    "parse_observations",
    "parse_reflections",
    "ProcessResult",
    "ObservationMemoryWorker",
]
