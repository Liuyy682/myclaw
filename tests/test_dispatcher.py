import asyncio

from myclaw import AgentConfig, AgentDispatcher, AgentLoop, FakeProvider
from myclaw.bus import InboundMessage, MessageBus


def test_dispatcher_processes_inbound_and_publishes_outbound():
    bus = MessageBus()
    loop = AgentLoop(FakeProvider(prefix="Echo"), AgentConfig(system_prompt=""))
    dispatcher = AgentDispatcher(bus, loop)

    asyncio.run(bus.publish_inbound(
        InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="hello")
    ))
    asyncio.run(dispatcher.process_next())
    outbound = asyncio.run(bus.consume_outbound())

    assert outbound.channel == "cli"
    assert outbound.chat_id == "direct"
    assert outbound.content == "Echo: hello"


class BrokenLoop:
    async def process(self, text):
        raise RuntimeError("loop unavailable")


def test_dispatcher_turns_loop_errors_into_outbound_message():
    bus = MessageBus()
    dispatcher = AgentDispatcher(bus, BrokenLoop())

    asyncio.run(bus.publish_inbound(
        InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="hello")
    ))
    asyncio.run(dispatcher.process_next())
    outbound = asyncio.run(bus.consume_outbound())

    assert outbound.channel == "cli"
    assert outbound.chat_id == "direct"
    assert outbound.content == "Error: loop unavailable"
