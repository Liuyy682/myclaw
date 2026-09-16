import asyncio
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from myclaw.agent import AgentDispatcher, DispatcherLimits, DispatcherRuntime
from myclaw.agent.planning import PlanCommand
from myclaw.bus import InboundMessage, MessageBus
from myclaw.runtime import RequestStore


def test_request_store_uses_one_table_and_internal_ids(tmp_path):
    store = RequestStore(tmp_path)
    first = store.create_queued(
        "same-external-id",
        session_key="gateway:a",
        channel="gateway",
        chat_id="a",
        content="first",
    )
    second = store.create_queued(
        "same-external-id",
        session_key="gateway:a",
        channel="gateway",
        chat_id="a",
        content="second",
    )

    assert first.id != second.id
    assert store.get(first.id).status == "queued"
    store.mark_running(first.id)
    assert store.mark_completed(first.id).status == "completed"
    tables = sqlite3.connect(store.path).execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    assert tables == [("requests",)]


def test_request_store_startup_recovery_marks_only_incomplete_rows(tmp_path):
    original = RequestStore(tmp_path)
    queued = original.create_queued(
        "queued", session_key="cli:a", channel="cli", chat_id="a", content="queued"
    )
    running = original.create_queued(
        "running", session_key="cli:a", channel="cli", chat_id="a", content="running"
    )
    original.mark_running(running.id)
    done = original.create_queued(
        "done", session_key="cli:a", channel="cli", chat_id="a", content="done"
    )
    original.mark_completed(done.id)

    recovered = RequestStore(tmp_path)
    assert recovered.get(queued.id).status == "queued"
    assert recovered.get(running.id).status == "running"
    assert recovered.mark_incomplete_interrupted() == 2
    assert recovered.get(queued.id).status == "interrupted"
    assert recovered.get(running.id).status == "interrupted"
    assert recovered.get(done.id).status == "completed"


def test_request_store_keeps_the_first_terminal_state(tmp_path):
    store = RequestStore(tmp_path)
    record = store.create_queued(
        "terminal", session_key="cli:a", channel="cli", chat_id="a", content="done"
    )
    store.mark_running(record.id)
    store.mark_completed(record.id)

    store.mark_interrupted(record.id, "late cancellation")
    store.mark_failed(record.id, "late publish failure")

    persisted = store.get(record.id)
    assert persisted.status == "completed"
    assert persisted.reason is None


class _TinyLoop:
    def __init__(self, workspace: Path):
        self.session_manager = SimpleNamespace(workspace=workspace)
        self.config = SimpleNamespace(model="tiny")
        self.provider = SimpleNamespace(model="tiny")
        self.calls = 0

    async def run(self, text, **kwargs):
        self.calls += 1
        return SimpleNamespace(content=f"reply: {text}", error=None, stop_reason="completed")

    def plan_command(self, _session_key, command):
        return PlanCommand(f"handled {command}")


class _ResultLoop(_TinyLoop):
    def __init__(self, workspace: Path, *, error=None, exception=None):
        super().__init__(workspace)
        self.error = error
        self.exception = exception

    async def run(self, text, **kwargs):
        self.calls += 1
        if self.exception is not None:
            raise self.exception
        return SimpleNamespace(content=f"reply: {text}", error=self.error, stop_reason="completed")


class _BlockingLoop(_TinyLoop):
    def __init__(self, workspace: Path):
        super().__init__(workspace)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def run(self, text, **kwargs):
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return SimpleNamespace(content=f"reply: {text}", error=None, stop_reason="completed")


def test_dispatcher_persists_queued_then_completes_and_runtime_recovers(tmp_path):
    bus = MessageBus()
    loop = _TinyLoop(tmp_path)
    store = RequestStore(tmp_path)
    dispatcher = AgentDispatcher(bus, loop, request_store=store)

    async def scenario():
        async with DispatcherRuntime(dispatcher, enable_mcp=False):
            admission = await dispatcher.submit(InboundMessage(
                channel="cli", sender_id="u", chat_id="a", content="hello",
                metadata={"request_id": "cli-1"},
            ))
            outbound = await bus.consume_outbound()
            return admission, outbound

    admission, outbound = asyncio.run(scenario())
    assert admission.accepted
    assert outbound.content == "reply: hello"
    record = store.get("cli-1")
    assert record is not None
    assert record.status == "completed"
    assert loop.calls == 1


def test_dispatcher_request_persistence_failure_releases_capacity_and_skips_agent(tmp_path):
    class FailingStore:
        def create_queued(self, *args, **kwargs):
            raise OSError("disk full")

    bus = MessageBus()
    loop = _TinyLoop(tmp_path)
    dispatcher = AgentDispatcher(bus, loop, request_store=FailingStore())

    async def scenario():
        admission = await dispatcher.submit(
            InboundMessage(channel="gateway", sender_id="u", chat_id="a", content="hello")
        )
        outbound = await bus.consume_outbound()
        return admission, outbound

    admission, outbound = asyncio.run(scenario())
    assert not admission.accepted
    assert admission.reason == "request_persistence_failed"
    assert "durably accepted" in outbound.content
    assert dispatcher._pending_agent_requests == 0
    assert loop.calls == 0


def test_persistence_failure_releases_capacity_for_a_later_request(tmp_path):
    class FailOnceStore(RequestStore):
        def __init__(self, workspace):
            super().__init__(workspace)
            self.failed = False

        def create_queued(self, *args, **kwargs):
            if not self.failed:
                self.failed = True
                raise OSError("disk temporarily unavailable")
            return super().create_queued(*args, **kwargs)

    bus = MessageBus()
    loop = _TinyLoop(tmp_path)
    store = FailOnceStore(tmp_path)
    dispatcher = AgentDispatcher(bus, loop, request_store=store)

    async def scenario():
        first = await dispatcher.submit(InboundMessage(
            channel="cli", sender_id="u", chat_id="a", content="first"
        ))
        await bus.consume_outbound()
        async with DispatcherRuntime(dispatcher, enable_mcp=False):
            second = await dispatcher.submit(InboundMessage(
                channel="cli", sender_id="u", chat_id="a", content="second",
                metadata={"request_id": "second"},
            ))
            await bus.consume_outbound()
        return first, second

    first, second = asyncio.run(scenario())
    assert not first.accepted and first.reason == "request_persistence_failed"
    assert second.accepted and store.get("second").status == "completed"


def test_agent_error_and_exception_are_failed(tmp_path):
    async def run_case(loop, store, request_id):
        bus = MessageBus()
        dispatcher = AgentDispatcher(bus, loop, request_store=store)
        async with DispatcherRuntime(dispatcher, enable_mcp=False):
            admission = await dispatcher.submit(InboundMessage(
                channel="cli", sender_id="u", chat_id="a", content="hello",
                metadata={"request_id": request_id},
            ))
            outbound = await bus.consume_outbound()
        return admission, outbound, store.get(request_id)

    error_store = RequestStore(tmp_path / "agent-error")
    error_result = asyncio.run(run_case(
        _ResultLoop(tmp_path / "agent-error", error="agent returned error"),
        error_store,
        "agent-error",
    ))
    exception_store = RequestStore(tmp_path / "exception")
    exception_result = asyncio.run(run_case(
        _ResultLoop(tmp_path / "exception", exception=RuntimeError("boom")),
        exception_store,
        "exception",
    ))

    assert error_result[0].accepted and error_result[2].status == "failed"
    assert error_result[2].reason == "agent returned error"
    assert exception_result[0].accepted and exception_result[2].status == "failed"
    assert exception_result[2].reason == "boom"
    assert exception_result[1].content == "Error: boom"


def test_queue_timeout_and_enqueue_failure_are_failed(tmp_path):
    async def timeout_scenario():
        bus = MessageBus()
        loop = _BlockingLoop(tmp_path / "timeout")
        store = RequestStore(tmp_path / "timeout")
        dispatcher = AgentDispatcher(
            bus,
            loop,
            request_store=store,
            limits=DispatcherLimits(queue_wait_timeout_seconds=0.02),
        )
        async with DispatcherRuntime(dispatcher, enable_mcp=False):
            await dispatcher.submit(InboundMessage(
                channel="cli", sender_id="u", chat_id="a", content="first",
                metadata={"request_id": "first"},
            ))
            await loop.started.wait()
            await dispatcher.submit(InboundMessage(
                channel="cli", sender_id="u", chat_id="a", content="second",
                metadata={"request_id": "second"},
            ))
            outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=1)
            loop.release.set()
        return outbound, store.get("second")

    timeout_outbound, timeout_record = asyncio.run(timeout_scenario())
    assert timeout_outbound.event_type == "error"
    assert timeout_record.status == "failed"
    assert "timed out" in timeout_record.reason

    full_bus = MessageBus(inbound_maxsize=1)
    full_bus.try_publish_inbound(InboundMessage(
        channel="cli", sender_id="u", chat_id="occupied", content="occupied"
    ))
    store = RequestStore(tmp_path / "enqueue")
    dispatcher = AgentDispatcher(
        full_bus, _TinyLoop(tmp_path / "enqueue"), request_store=store
    )
    admission = asyncio.run(dispatcher.submit(InboundMessage(
        channel="cli", sender_id="u", chat_id="a", content="new",
        metadata={"request_id": "enqueue-failed"},
    )))
    assert not admission.accepted
    assert store.get("enqueue-failed").status == "failed"


def test_user_stop_and_runtime_cancel_are_interrupted(tmp_path):
    async def user_stop_scenario():
        bus = MessageBus()
        loop = _BlockingLoop(tmp_path / "stop")
        store = RequestStore(tmp_path / "stop")
        dispatcher = AgentDispatcher(bus, loop, request_store=store)
        async with DispatcherRuntime(dispatcher, enable_mcp=False):
            await dispatcher.submit(InboundMessage(
                channel="cli", sender_id="u", chat_id="a", content="block",
                metadata={"request_id": "stopped"},
            ))
            await loop.started.wait()
            await dispatcher.submit(InboundMessage(
                channel="cli", sender_id="u", chat_id="a", content="/stop"
            ))
            await bus.consume_outbound()
        return store.get("stopped")

    stopped = asyncio.run(user_stop_scenario())
    assert stopped.status == "interrupted"
    assert stopped.reason == "Stopped by user."

    async def runtime_cancel_scenario():
        bus = MessageBus()
        loop = _BlockingLoop(tmp_path / "cancel")
        store = RequestStore(tmp_path / "cancel")
        dispatcher = AgentDispatcher(bus, loop, request_store=store)
        runtime = DispatcherRuntime(dispatcher, enable_mcp=False)
        await runtime.start()
        await dispatcher.submit(InboundMessage(
            channel="cli", sender_id="u", chat_id="a", content="block",
            metadata={"request_id": "cancelled"},
        ))
        await loop.started.wait()
        await runtime.stop()
        return store.get("cancelled")

    cancelled = asyncio.run(runtime_cancel_scenario())
    assert cancelled.status == "interrupted"
    assert cancelled.reason == "Runtime cancelled."


def test_mode_request_is_recorded_but_control_query_is_not(tmp_path):
    bus = MessageBus()
    loop = _TinyLoop(tmp_path)
    store = RequestStore(tmp_path)
    dispatcher = AgentDispatcher(bus, loop, request_store=store)

    async def scenario():
        mode = await dispatcher.submit(InboundMessage(
            channel="cli", sender_id="u", chat_id="a", content="/exit-plan",
            metadata={"request_id": "mode-1"},
        ))
        mode_out = await bus.consume_outbound()
        control = await dispatcher.submit(InboundMessage(
            channel="cli", sender_id="u", chat_id="a", content="/status",
            metadata={"request_id": "control-1"},
        ))
        control_out = await bus.consume_outbound()
        await asyncio.sleep(0)
        return mode, mode_out, control, control_out

    mode, mode_out, control, control_out = asyncio.run(scenario())
    assert mode.accepted and mode_out.event_type == "control"
    assert control.accepted and control_out.content.startswith("Status: idle.")
    records = store.list_recent("cli:a", limit=10)
    assert [(record.request_id, record.status) for record in records] == [("mode-1", "completed")]


def test_execute_persists_raw_command_before_prompt_conversion(tmp_path):
    class ExecuteLoop(_TinyLoop):
        def __init__(self, workspace):
            super().__init__(workspace)
            self.seen_text = None

        def plan_command(self, _session_key, command):
            assert command == "/execute"
            return PlanCommand("Executing.", run_prompt="converted execution prompt")

        async def run(self, text, **kwargs):
            self.calls += 1
            self.seen_text = text
            return SimpleNamespace(content="done", error=None, stop_reason="completed")

    bus = MessageBus()
    loop = ExecuteLoop(tmp_path)
    store = RequestStore(tmp_path)
    dispatcher = AgentDispatcher(bus, loop, request_store=store)

    async def scenario():
        async with DispatcherRuntime(dispatcher, enable_mcp=False):
            admission = await dispatcher.submit(InboundMessage(
                channel="cli", sender_id="u", chat_id="a", content="/execute",
                metadata={"request_id": "execute-1"},
            ))
            await bus.consume_outbound()  # mode transition notice
            await bus.consume_outbound()  # agent result
        return admission

    assert asyncio.run(scenario()).accepted
    record = store.get("execute-1")
    assert record.content == "/execute"
    assert record.status == "completed"
    assert loop.seen_text == "converted execution prompt"


def test_mode_schedule_failure_marks_saved_request_failed(tmp_path):
    bus = MessageBus()
    store = RequestStore(tmp_path)
    dispatcher = AgentDispatcher(bus, _TinyLoop(tmp_path), request_store=store)

    def fail_schedule(coro):
        coro.close()
        raise RuntimeError("scheduler unavailable")

    dispatcher._schedule_task = fail_schedule

    async def scenario():
        admission = await dispatcher.submit(InboundMessage(
            channel="cli", sender_id="u", chat_id="a", content="/exit-plan",
            metadata={"request_id": "mode-schedule-failed"},
        ))
        outbound = await bus.consume_outbound()
        return admission, outbound

    admission, outbound = asyncio.run(scenario())
    assert not admission.accepted
    assert outbound.event_type == "error"
    record = store.get("mode-schedule-failed")
    assert record.status == "failed"
    assert record.reason == "scheduler unavailable"


def test_follow_up_answer_is_not_recorded(tmp_path):
    bus = MessageBus()
    store = RequestStore(tmp_path)
    dispatcher = AgentDispatcher(bus, _TinyLoop(tmp_path), request_store=store)

    async def scenario():
        pending = asyncio.create_task(dispatcher.ask.ask("cli:a", "question?"))
        question = await bus.consume_outbound()
        admission = await dispatcher.submit(InboundMessage(
            channel="cli", sender_id="u", chat_id="a", content="answer",
            metadata={"request_id": "answer-id"},
        ))
        return question, admission, await pending

    question, admission, answer = asyncio.run(scenario())
    assert question.event_type == "ask"
    assert admission.accepted and answer == "answer"
    assert store.get("answer-id") is None


def test_running_persistence_failure_never_starts_agent(tmp_path):
    class RunningFailureStore(RequestStore):
        def mark_running(self, internal_id):
            raise OSError("database unavailable")

    bus = MessageBus()
    loop = _TinyLoop(tmp_path)
    store = RunningFailureStore(tmp_path)
    dispatcher = AgentDispatcher(bus, loop, request_store=store)

    async def scenario():
        task = asyncio.create_task(dispatcher.run())
        admission = await dispatcher.submit(InboundMessage(
            channel="cli", sender_id="u", chat_id="a", content="hello",
            metadata={"request_id": "running-failure"},
        ))
        await asyncio.sleep(0)
        outbound = await bus.consume_outbound()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return admission, outbound

    admission, outbound = asyncio.run(scenario())
    assert admission.accepted
    assert outbound.event_type == "error"
    assert "durably accepted" in outbound.content
    assert loop.calls == 0
    assert store.get("running-failure").status == "failed"


def test_status_lists_only_current_session_interrupted_records(tmp_path):
    store = RequestStore(tmp_path)
    for index in range(6):
        record = store.create_queued(
            f"a-{index}", session_key="cli:a", channel="cli", chat_id="a",
            content="x" * 120,
        )
        store.mark_interrupted(record.id, "Stopped by user.")
    other = store.create_queued(
        "other", session_key="cli:b", channel="cli", chat_id="b", content="other"
    )
    store.mark_interrupted(other.id, "Runtime cancelled.")
    dispatcher = AgentDispatcher(MessageBus(), _TinyLoop(tmp_path), request_store=store)

    status = dispatcher._session_status("cli:a")
    assert "other" not in status
    assert status.count("id=") == 5
    assert "reason=Stopped by user." in status
    assert "重新提交会创建新任务" in status
    input_line = next(line for line in status.splitlines() if "input=" in line)
    assert len(input_line.split("input=", 1)[1].strip("'")) <= 100


def test_separate_process_orphan_is_interrupted_without_agent_call(tmp_path):
    script = """
from pathlib import Path
from myclaw.runtime import RequestStore
import sys
RequestStore(Path(sys.argv[1])).create_queued(
    'orphan', session_key='cli:a', channel='cli', chat_id='a', content='orphan'
)
"""
    subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=True)
    store = RequestStore(tmp_path)
    loop = _TinyLoop(tmp_path)
    dispatcher = AgentDispatcher(MessageBus(), loop, request_store=store)

    async def start_and_stop():
        async with DispatcherRuntime(dispatcher, enable_mcp=False):
            await asyncio.sleep(0)

    asyncio.run(start_and_stop())
    record = store.get("orphan")
    assert record.status == "interrupted"
    assert loop.calls == 0
