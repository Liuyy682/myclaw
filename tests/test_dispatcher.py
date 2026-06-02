import asyncio
import contextlib

from myclaw import AgentConfig, AgentDispatcher, AgentLoop, FakeProvider, FunctionTool, ToolCallRequest, ToolRegistry
from myclaw.bus import InboundMessage, MessageBus
from myclaw.providers import LLMResponse
from myclaw.session import SessionManager


async def _stop(task):
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def test_dispatcher_run_processes_inbound_and_publishes_outbound(tmp_path):
    bus = MessageBus()
    loop = AgentLoop(
        FakeProvider(prefix="Echo"),
        AgentConfig(system_prompt=""),
        session_manager=SessionManager(tmp_path),
    )
    dispatcher = AgentDispatcher(bus, loop)

    async def scenario():
        task = asyncio.create_task(dispatcher.run())
        await bus.publish_inbound(
            InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="hello")
        )
        outbound = await bus.consume_outbound()
        await _stop(task)
        return outbound

    outbound = asyncio.run(scenario())

    assert outbound.channel == "cli"
    assert outbound.chat_id == "direct"
    assert outbound.content == "Echo: hello"


class CapturingLoop:
    def __init__(self):
        self.calls = []

    async def run(self, text, *, session_key, progress_callback=None, **kwargs):
        self.calls.append((text, session_key, kwargs))
        return type("Result", (), {"content": f"{session_key}: {text}"})()


class RecordingAutoCompact:
    def __init__(self):
        self.calls = []
        self.called = asyncio.Event()

    def check_expired(self, schedule_background, active_session_keys=()):
        self.calls.append(set(active_session_keys))
        self.called.set()


def test_dispatcher_idle_tick_checks_auto_compact():
    bus = MessageBus()
    loop = CapturingLoop()
    loop.auto_compact = RecordingAutoCompact()
    dispatcher = AgentDispatcher(bus, loop)
    dispatcher._AUTO_COMPACT_IDLE_TICK_SECONDS = 0.01

    async def scenario():
        task = asyncio.create_task(dispatcher.run())
        await asyncio.wait_for(loop.auto_compact.called.wait(), timeout=0.5)
        await _stop(task)
        return loop.auto_compact.calls

    calls = asyncio.run(scenario())

    assert calls == [set()]


def test_dispatcher_idle_tick_skips_active_or_queued_sessions():
    bus = MessageBus()
    loop = BlockingLoop()
    loop.auto_compact = RecordingAutoCompact()
    dispatcher = AgentDispatcher(bus, loop)
    dispatcher._AUTO_COMPACT_IDLE_TICK_SECONDS = 0.01

    async def scenario():
        task = asyncio.create_task(dispatcher.run())
        await bus.publish_inbound(
            InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="one")
        )
        await loop.started.wait()
        await bus.publish_inbound(
            InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="two")
        )
        await asyncio.wait_for(loop.auto_compact.called.wait(), timeout=0.5)
        loop.release.set()
        await bus.consume_outbound()
        await bus.consume_outbound()
        await _stop(task)
        return loop.auto_compact.calls

    calls = asyncio.run(scenario())

    assert calls[-1] == {"cli:direct"}


class StreamingLoop:
    def __init__(self):
        self.calls = []
        self.saw_stream_callback = None

    async def run(self, text, *, session_key, progress_callback=None, stream_callback=None, **kwargs):
        self.calls.append((text, session_key))
        self.saw_stream_callback = stream_callback is not None
        if stream_callback is not None:
            await stream_callback("hel")
            await stream_callback("lo")
        return type("Result", (), {"content": "hello"})()


def test_dispatcher_run_passes_message_session_key_to_loop():
    bus = MessageBus()
    loop = CapturingLoop()
    dispatcher = AgentDispatcher(bus, loop)

    async def scenario():
        task = asyncio.create_task(dispatcher.run())
        await bus.publish_inbound(
            InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="hello")
        )
        outbound = await bus.consume_outbound()
        await _stop(task)
        return outbound

    outbound = asyncio.run(scenario())

    assert loop.calls == [("hello", "cli:direct", {"channel": "cli", "chat_id": "direct", "metadata": {}})]
    assert outbound.content == "cli:direct: hello"


def test_dispatcher_run_respects_session_key_override():
    bus = MessageBus()
    loop = CapturingLoop()
    dispatcher = AgentDispatcher(bus, loop)

    async def scenario():
        task = asyncio.create_task(dispatcher.run())
        await bus.publish_inbound(
            InboundMessage(
                channel="cli",
                sender_id="user",
                chat_id="direct",
                content="hello",
                session_key_override="shared:session",
            )
        )
        outbound = await bus.consume_outbound()
        await _stop(task)
        return outbound

    outbound = asyncio.run(scenario())

    assert loop.calls == [("hello", "shared:session", {"channel": "cli", "chat_id": "direct", "metadata": {}})]
    assert outbound.content == "shared:session: hello"


def test_dispatcher_publishes_stream_deltas_for_gateway_channel():
    bus = MessageBus()
    loop = StreamingLoop()
    dispatcher = AgentDispatcher(bus, loop)

    async def scenario():
        task = asyncio.create_task(dispatcher.run())
        await bus.publish_inbound(
            InboundMessage(
                channel="gateway",
                sender_id="user",
                chat_id="direct",
                content="hello",
                metadata={"request_id": "req-1"},
            )
        )
        first = await bus.consume_outbound()
        second = await bus.consume_outbound()
        final = await bus.consume_outbound()
        await _stop(task)
        return first, second, final

    first, second, final = asyncio.run(scenario())

    assert loop.saw_stream_callback is True
    assert first.event_type == "message_delta"
    assert first.terminal is False
    assert first.content == "hel"
    assert first.metadata == {"request_id": "req-1", "session_key": "gateway:direct"}
    assert second.event_type == "message_delta"
    assert second.terminal is False
    assert second.content == "lo"
    assert final.event_type == "message"
    assert final.terminal is True
    assert final.content == "hello"


def test_dispatcher_does_not_stream_cli_channel():
    bus = MessageBus()
    loop = StreamingLoop()
    dispatcher = AgentDispatcher(bus, loop)

    async def scenario():
        task = asyncio.create_task(dispatcher.run())
        await bus.publish_inbound(
            InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="hello")
        )
        outbound = await bus.consume_outbound()
        await _stop(task)
        return outbound

    outbound = asyncio.run(scenario())

    assert loop.saw_stream_callback is False
    assert outbound.event_type == "message"
    assert outbound.terminal is True
    assert outbound.content == "hello"


def test_dispatcher_streams_cli_channel_when_requested_by_metadata():
    bus = MessageBus()
    loop = StreamingLoop()
    dispatcher = AgentDispatcher(bus, loop)

    async def scenario():
        task = asyncio.create_task(dispatcher.run())
        await bus.publish_inbound(
            InboundMessage(
                channel="cli",
                sender_id="user",
                chat_id="direct",
                content="hello",
                metadata={"stream": True},
            )
        )
        first = await bus.consume_outbound()
        second = await bus.consume_outbound()
        final = await bus.consume_outbound()
        await _stop(task)
        return first, second, final

    first, second, final = asyncio.run(scenario())

    assert loop.saw_stream_callback is True
    assert first.event_type == "message_delta"
    assert first.terminal is False
    assert first.content == "hel"
    assert first.metadata == {"stream": True, "session_key": "cli:direct"}
    assert second.event_type == "message_delta"
    assert second.terminal is False
    assert second.content == "lo"
    assert final.event_type == "message"
    assert final.terminal is True
    assert final.content == "hello"


class BlockingLoop:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = []
        self.running_sessions = set()
        self.overlaps = []

    async def run(self, text, *, session_key, progress_callback=None, **kwargs):
        if session_key in self.running_sessions:
            self.overlaps.append((text, session_key))
        self.running_sessions.add(session_key)
        try:
            self.calls.append((text, session_key))
            if text == "one":
                self.started.set()
                await self.release.wait()
            return type("Result", (), {"content": f"done: {text}"})()
        finally:
            self.running_sessions.remove(session_key)


def test_dispatcher_run_serializes_messages_with_same_session_key():
    bus = MessageBus()
    loop = BlockingLoop()
    dispatcher = AgentDispatcher(bus, loop)

    async def scenario():
        task = asyncio.create_task(dispatcher.run())
        await bus.publish_inbound(
            InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="one")
        )
        await loop.started.wait()

        await bus.publish_inbound(
            InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="two")
        )
        await asyncio.sleep(0)
        calls_while_blocked = list(loop.calls)

        loop.release.set()
        first = await bus.consume_outbound()
        second = await bus.consume_outbound()
        await _stop(task)
        return calls_while_blocked, first, second, list(loop.calls), list(loop.overlaps)

    calls_while_blocked, first, second, calls, overlaps = asyncio.run(scenario())

    assert calls_while_blocked == [("one", "cli:direct")]
    assert calls == [("one", "cli:direct"), ("two", "cli:direct")]
    assert overlaps == []
    assert first.content == "done: one"
    assert second.content == "done: two"


class CrossSessionBlockingLoop:
    def __init__(self):
        self.calls = []
        self.started_sessions = set()
        self.both_started = asyncio.Event()
        self.release = asyncio.Event()

    async def run(self, text, *, session_key, progress_callback=None, **kwargs):
        self.calls.append((text, session_key))
        self.started_sessions.add(session_key)
        if len(self.started_sessions) == 2:
            self.both_started.set()
        await self.release.wait()
        return type("Result", (), {"content": f"{session_key}: {text}"})()


def test_dispatcher_run_allows_different_sessions_to_run_concurrently():
    bus = MessageBus()
    loop = CrossSessionBlockingLoop()
    dispatcher = AgentDispatcher(bus, loop)

    async def scenario():
        task = asyncio.create_task(dispatcher.run())
        await bus.publish_inbound(
            InboundMessage(channel="cli", sender_id="user", chat_id="one", content="first")
        )
        await bus.publish_inbound(
            InboundMessage(channel="cli", sender_id="user", chat_id="two", content="second")
        )

        await asyncio.wait_for(loop.both_started.wait(), timeout=0.5)
        calls_while_blocked = list(loop.calls)

        loop.release.set()
        first = await bus.consume_outbound()
        second = await bus.consume_outbound()
        await _stop(task)
        return calls_while_blocked, {first.content, second.content}

    calls_while_blocked, contents = asyncio.run(scenario())

    assert calls_while_blocked == [("first", "cli:one"), ("second", "cli:two")]
    assert contents == {"cli:one: first", "cli:two: second"}


class BrokenLoop:
    def __init__(self):
        self.calls = 0

    async def run(self, text, *, session_key, progress_callback=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("loop unavailable")
        return type("Result", (), {"content": f"recovered: {text}"})()


def test_dispatcher_run_turns_loop_errors_into_outbound_message_and_continues():
    bus = MessageBus()
    dispatcher = AgentDispatcher(bus, BrokenLoop())

    async def scenario():
        task = asyncio.create_task(dispatcher.run())
        await bus.publish_inbound(
            InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="hello")
        )
        await bus.publish_inbound(
            InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="again")
        )
        first = await bus.consume_outbound()
        second = await bus.consume_outbound()
        await _stop(task)
        return first, second

    first, second = asyncio.run(scenario())

    assert first.channel == "cli"
    assert first.chat_id == "direct"
    assert first.content == "Error: loop unavailable"
    assert second.content == "recovered: again"


def test_dispatcher_status_reports_running_and_queued_messages():
    bus = MessageBus()
    loop = BlockingLoop()
    dispatcher = AgentDispatcher(bus, loop)

    async def scenario():
        task = asyncio.create_task(dispatcher.run())
        await bus.publish_inbound(
            InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="one")
        )
        await loop.started.wait()
        await bus.publish_inbound(
            InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="two")
        )
        await asyncio.sleep(0)
        await bus.publish_inbound(
            InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="/status")
        )
        status = await bus.consume_outbound()
        loop.release.set()
        first = await bus.consume_outbound()
        second = await bus.consume_outbound()
        await _stop(task)
        return status, first, second

    status, first, second = asyncio.run(scenario())

    assert status.content == "Status: running with 1 queued."
    assert status.event_type == "control"
    assert status.terminal is True
    assert first.content == "done: one"
    assert second.content == "done: two"


def test_dispatcher_stop_cancels_active_session_task_without_final_message():
    bus = MessageBus()
    loop = CancellableLoop()
    dispatcher = AgentDispatcher(bus, loop)

    async def scenario():
        task = asyncio.create_task(dispatcher.run())
        await bus.publish_inbound(
            InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="hello")
        )
        await loop.started.wait()
        await bus.publish_inbound(
            InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="/stop")
        )
        outbound = await bus.consume_outbound()
        active = dict(dispatcher._active_session_tasks)
        states = dict(dispatcher._session_states)
        await _stop(task)
        return outbound, loop.cancelled.is_set(), active, states, bus.outbound_size

    outbound, worker_cancelled, active, states, outbound_size = asyncio.run(scenario())

    assert outbound.content == "Stopped current turn."
    assert outbound.event_type == "control"
    assert outbound.terminal is True
    assert worker_cancelled is True
    assert active == {}
    assert states == {}
    assert outbound_size == 0


def test_dispatcher_clear_resets_idle_session_history(tmp_path):
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("cli:direct")
    session.add_message("user", "old")
    session.metadata["pending_user_turn"] = True
    session.metadata["title"] = "Old title"
    manager.save(session)
    bus = MessageBus()
    loop = AgentLoop(
        FakeProvider(prefix="Echo"),
        AgentConfig(system_prompt=""),
        session_manager=manager,
    )
    dispatcher = AgentDispatcher(bus, loop)

    async def scenario():
        task = asyncio.create_task(dispatcher.run())
        await bus.publish_inbound(
            InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="/clear")
        )
        outbound = await bus.consume_outbound()
        await _stop(task)
        return outbound

    outbound = asyncio.run(scenario())
    reloaded = SessionManager(tmp_path).get_or_create("cli:direct")

    assert outbound.content == "Cleared current session."
    assert outbound.event_type == "control"
    assert outbound.terminal is True
    assert reloaded.key == "cli:direct"
    assert reloaded.messages == []
    assert reloaded.metadata == {}


def test_dispatcher_clear_refuses_to_reset_active_session():
    bus = MessageBus()
    loop = BlockingLoop()
    dispatcher = AgentDispatcher(bus, loop)

    async def scenario():
        task = asyncio.create_task(dispatcher.run())
        await bus.publish_inbound(
            InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="one")
        )
        await loop.started.wait()
        await bus.publish_inbound(
            InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="/clear")
        )
        outbound = await bus.consume_outbound()
        loop.release.set()
        final = await bus.consume_outbound()
        await _stop(task)
        return outbound, final

    outbound, final = asyncio.run(scenario())

    assert outbound.content == "Cannot clear the current session while a turn is running. Use /stop first."
    assert outbound.event_type == "control"
    assert final.content == "done: one"


def test_dispatcher_treats_new_as_regular_user_message():
    bus = MessageBus()
    loop = CapturingLoop()
    dispatcher = AgentDispatcher(bus, loop)

    async def scenario():
        task = asyncio.create_task(dispatcher.run())
        await bus.publish_inbound(
            InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="/new")
        )
        outbound = await bus.consume_outbound()
        await _stop(task)
        return outbound

    outbound = asyncio.run(scenario())

    assert loop.calls == [("/new", "cli:direct", {"channel": "cli", "chat_id": "direct", "metadata": {}})]
    assert outbound.content == "cli:direct: /new"


class OneToolProvider:
    model = "tools"

    def __init__(self):
        self.calls = 0

    async def complete(self, messages, *, tools=None):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content="",
                final=False,
                stop_reason="tool_calls",
                tool_calls=[ToolCallRequest(id="call_add", name="add", arguments={"a": 2, "b": 3})],
            )
        return LLMResponse(content="done", final=True)


def test_dispatcher_forwards_tool_progress_as_non_terminal_outbound_messages(tmp_path):
    registry = ToolRegistry()
    registry.register(FunctionTool("add", "Add", {"type": "object"}, lambda a, b: a + b))
    bus = MessageBus()
    loop = AgentLoop(
        OneToolProvider(),
        AgentConfig(system_prompt=""),
        session_manager=SessionManager(tmp_path),
        tool_registry=registry,
    )
    dispatcher = AgentDispatcher(bus, loop)

    async def scenario():
        task = asyncio.create_task(dispatcher.run())
        await bus.publish_inbound(
            InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="use tool")
        )
        started = await bus.consume_outbound()
        completed = await bus.consume_outbound()
        final = await bus.consume_outbound()
        await _stop(task)
        return started, completed, final

    started, completed, final = asyncio.run(scenario())

    assert started.terminal is False
    assert started.event_type == "tool_progress"
    assert started.content == "Running tool add (1/1)"
    assert started.metadata["progress"]["event"] == "tool_started"
    assert completed.terminal is False
    assert completed.event_type == "tool_progress"
    assert completed.content == "Finished tool add (1/1)"
    assert completed.metadata["progress"]["event"] == "tool_completed"
    assert final.terminal is True
    assert final.content == "done"


def test_dispatcher_run_can_be_cancelled_while_waiting_for_inbound(tmp_path):
    bus = MessageBus()
    loop = AgentLoop(
        FakeProvider(prefix="Echo"),
        AgentConfig(system_prompt=""),
        session_manager=SessionManager(tmp_path),
    )
    dispatcher = AgentDispatcher(bus, loop)

    async def scenario():
        task = asyncio.create_task(dispatcher.run())
        await asyncio.sleep(0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return task.cancelled()

    assert asyncio.run(scenario()) is True


class CancellableLoop:
    def __init__(self):
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def run(self, text, *, session_key, progress_callback=None, **kwargs):
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


def test_dispatcher_run_cancels_active_workers_on_shutdown():
    bus = MessageBus()
    loop = CancellableLoop()
    dispatcher = AgentDispatcher(bus, loop)

    async def scenario():
        task = asyncio.create_task(dispatcher.run())
        await bus.publish_inbound(
            InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="hello")
        )
        await loop.started.wait()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return (
            task.cancelled(),
            loop.cancelled.is_set(),
            set(dispatcher._active_tasks),
            dict(dispatcher._session_states),
        )

    cancelled, worker_cancelled, active_tasks, session_states = asyncio.run(scenario())

    assert cancelled is True
    assert worker_cancelled is True
    assert active_tasks == set()
    assert session_states == {}
