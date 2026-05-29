import asyncio
import contextlib

from myclaw import AgentConfig, AgentDispatcher, AgentLoop, FakeProvider
from myclaw.bus import InboundMessage, MessageBus
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

    async def run(self, text, *, session_key):
        self.calls.append((text, session_key))
        return type("Result", (), {"content": f"{session_key}: {text}"})()


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

    assert loop.calls == [("hello", "cli:direct")]
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

    assert loop.calls == [("hello", "shared:session")]
    assert outbound.content == "shared:session: hello"


class BlockingLoop:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = []

    async def run(self, text, *, session_key):
        self.calls.append((text, session_key))
        self.started.set()
        await self.release.wait()
        return type("Result", (), {"content": f"done: {text}"})()


def test_dispatcher_run_queues_new_inbound_while_current_message_is_running():
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
        queued_size = bus.inbound_size

        loop.release.set()
        first = await bus.consume_outbound()
        second = await bus.consume_outbound()
        await _stop(task)
        return queued_size, first, second, list(loop.calls)

    queued_size, first, second, calls = asyncio.run(scenario())

    assert queued_size == 1
    assert calls == [("one", "cli:direct"), ("two", "cli:direct")]
    assert first.content == "done: one"
    assert second.content == "done: two"


class BrokenLoop:
    def __init__(self):
        self.calls = 0

    async def run(self, text, *, session_key):
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
