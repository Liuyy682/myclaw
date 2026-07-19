from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from myclaw.agent import AgentConfig
from myclaw.providers import FakeProvider
from myclaw.agent.autocompact import AutoCompactManager
from myclaw.agent.context import ContextBudgetManager, ContextBuilder
from myclaw.agent.dream import DreamManager
from myclaw.memory import MemoryStore
from myclaw.observability import ObservabilityConfig, ObservabilityRuntime, ObservedProvider
from myclaw.observability.types import ObservationEvent
from myclaw.session import SessionManager


def _runtime(tmp_path) -> ObservabilityRuntime:
    return ObservabilityRuntime(
        tmp_path,
        ObservabilityConfig(enabled=True, log_level="INFO", retention_days=7, max_bytes=100 * 1024 * 1024),
    )


def test_trace_spans_logs_and_summary_are_correlated(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.start()
    with runtime.trace(
        "conversation", "conversation", request_id="req-1", session_key="gateway:direct",
        channel="gateway", model="fake", trace_id="a" * 32,
    ):
        with runtime.span("llm.complete", "llm", attributes={"model": "fake"}) as span:
            span.set_usage(12, 3, 15)
        with runtime.span("tool.execute", "tool", attributes={"tool_name": "read_file"}):
            logging.getLogger("myclaw.test").info("tool completed")
    runtime.flush()

    detail = runtime.trace_detail("a" * 32)
    summary = runtime.summary((datetime.now(UTC) - timedelta(hours=1)).isoformat())
    runtime.stop()

    assert detail is not None
    assert detail["trace"]["status"] == "ok"
    assert [span["name"] for span in detail["spans"]] == ["llm.complete", "tool.execute"]
    assert detail["logs"][0]["trace_id"] == "a" * 32
    assert summary["requests"] == 1
    assert summary["llm_calls"] == 1
    assert summary["tool_calls"] == 1
    assert summary["tokens"] == {"prompt": 12, "completion": 3, "total": 15, "available": True}


def test_async_trace_contexts_do_not_leak(tmp_path):
    runtime = _runtime(tmp_path)

    async def run(trace_id):
        with runtime.trace("conversation", "conversation", trace_id=trace_id):
            await asyncio.sleep(0)
            with runtime.span("agent.turn", "agent"):
                await asyncio.sleep(0)

    async def scenario():
        await asyncio.gather(run("1" * 32), run("2" * 32))

    asyncio.run(scenario())
    runtime.flush()
    first = runtime.trace_detail("1" * 32)
    second = runtime.trace_detail("2" * 32)
    runtime.stop()

    assert first is not None and second is not None
    assert {span["trace_id"] for span in first["spans"]} == {"1" * 32}
    assert {span["trace_id"] for span in second["spans"]} == {"2" * 32}


def test_running_trace_from_previous_process_is_marked_abandoned(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.store.initialize()
    with runtime.store._connect() as connection:
        connection.execute(
            """
            INSERT INTO traces (
                trace_id, root_span_id, name, kind, status, started_at
            ) VALUES (?, ?, 'conversation', 'conversation', 'running', ?)
            """,
            ("b" * 32, "c" * 16, (datetime.now(UTC) - timedelta(minutes=1)).isoformat()),
        )

    restarted = _runtime(tmp_path)
    restarted.start()
    detail = restarted.trace_detail("b" * 32)
    restarted.stop()
    assert detail is not None
    assert detail["trace"]["status"] == "abandoned"


def test_disabled_runtime_is_a_noop(tmp_path):
    runtime = ObservabilityRuntime(tmp_path, ObservabilityConfig(enabled=False))
    with runtime.trace("conversation", "conversation"):
        with runtime.span("llm.complete", "llm"):
            pass
    assert not runtime.store.path.exists()


def test_writer_failure_never_interrupts_traced_work(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path)
    runtime.start()

    def fail_write(_events):
        raise OSError("disk full")

    monkeypatch.setattr(runtime.store, "write_batch", fail_write)

    with runtime.trace("agent.request", "conversation"):
        with runtime.span("tool.execute", "tool"):
            result = "business result"

    runtime.flush()
    runtime.stop()
    assert result == "business result"


def test_retention_removes_old_completed_data_but_keeps_running_trace(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.store.initialize()
    old = (datetime.now(UTC) - timedelta(days=10)).isoformat()
    runtime.store.write_batch([
        ObservationEvent("trace_start", {
            "trace_id": "e" * 32, "root_span_id": "1" * 16, "name": "old", "kind": "conversation",
            "started_at": old, "attributes": {},
        }),
        ObservationEvent("trace_end", {
            "trace_id": "e" * 32, "status": "ok", "ended_at": old, "duration_ms": 1, "attributes": {},
        }),
        ObservationEvent("trace_start", {
            "trace_id": "f" * 32, "root_span_id": "2" * 16, "name": "active", "kind": "conversation",
            "started_at": old, "attributes": {},
        }),
    ])

    runtime.store.cleanup(retention_days=7, max_bytes=100 * 1024 * 1024)

    assert runtime.store.trace_detail("e" * 32) is None
    assert runtime.store.trace_detail("f" * 32) is not None


def test_dream_and_auto_compact_create_background_root_traces(tmp_path):
    runtime = _runtime(tmp_path)
    manager = SessionManager(tmp_path)
    memory = MemoryStore(tmp_path)
    provider = ObservedProvider(FakeProvider(), runtime)
    config = AgentConfig(
        model="fake",
        dream_interval_minutes=1,
        idle_compact_after_minutes=1,
        auto_compact_recent_messages=2,
    )
    dream = DreamManager(
        manager, provider, memory, config, model="fake", observability=runtime
    )
    memory.append_history("durable history entry")

    session = manager.get_or_create("cli:background")
    for index in range(4):
        session.add_message("user", f"question {index}")
        session.add_message("assistant", f"answer {index}")
    manager.save(session)
    budget = ContextBudgetManager(provider, ContextBuilder())
    compact = AutoCompactManager(
        manager, budget, memory, config, model="fake", observability=runtime
    )

    assert asyncio.run(dream.run_once()) is True
    assert asyncio.run(compact.compact_session("cli:background")) is True
    runtime.flush()
    traces = runtime.list_traces(
        (datetime.now(UTC) - timedelta(hours=1)).isoformat(), limit=20
    )
    runtime.stop()

    assert {trace["kind"] for trace in traces} >= {"dream", "autocompact"}
