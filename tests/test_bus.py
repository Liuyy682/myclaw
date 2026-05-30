import asyncio

from myclaw.bus import InboundMessage, MessageBus, OutboundMessage


def test_bus_publish_and_consume_inbound_in_order():
    bus = MessageBus()
    first = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="one")
    second = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="two")

    asyncio.run(bus.publish_inbound(first))
    asyncio.run(bus.publish_inbound(second))

    assert bus.inbound_size == 2
    assert asyncio.run(bus.consume_inbound()) == first
    assert asyncio.run(bus.consume_inbound()) == second
    assert bus.inbound_size == 0


def test_bus_publish_and_consume_outbound_in_order():
    bus = MessageBus()
    first = OutboundMessage(channel="cli", chat_id="direct", content="one")
    second = OutboundMessage(channel="cli", chat_id="direct", content="two")

    asyncio.run(bus.publish_outbound(first))
    asyncio.run(bus.publish_outbound(second))

    assert bus.outbound_size == 2
    assert asyncio.run(bus.consume_outbound()) == first
    assert asyncio.run(bus.consume_outbound()) == second
    assert bus.outbound_size == 0


def test_outbound_message_marks_terminal_messages_by_default():
    msg = OutboundMessage(channel="cli", chat_id="direct", content="done")

    assert msg.terminal is True
    assert msg.event_type == "message"


def test_outbound_message_can_represent_non_terminal_progress():
    msg = OutboundMessage(
        channel="cli",
        chat_id="direct",
        content="Running tool add (1/1)",
        terminal=False,
        event_type="tool_progress",
        metadata={"progress": {"event": "tool_started", "tool_name": "add"}},
    )

    assert msg.terminal is False
    assert msg.event_type == "tool_progress"
    assert msg.metadata["progress"]["event"] == "tool_started"


def test_inbound_message_session_key_uses_default_or_override():
    msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="hello")
    override = InboundMessage(
        channel="cli",
        sender_id="user",
        chat_id="direct",
        content="hello",
        session_key_override="custom:session",
    )

    assert msg.session_key == "cli:direct"
    assert override.session_key == "custom:session"
