import asyncio

import pytest

from myclaw import AgentConfig, AgentLoop, FakeProvider
from myclaw.session import SessionManager


def test_process_appends_user_and_assistant_messages_in_order():
    loop = AgentLoop(FakeProvider(prefix="Echo"), AgentConfig(system_prompt="You are helpful."))

    result = asyncio.run(loop.process("hello"))

    assert result.content == "Echo: hello"
    assert result.model == "fake"
    assert result.messages == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Echo: hello"},
    ]


def test_process_reuses_history_between_turns():
    loop = AgentLoop(FakeProvider(prefix="Echo"), AgentConfig(system_prompt="You are helpful."))

    asyncio.run(loop.process("first"))
    result = asyncio.run(loop.process("second"))

    assert result.content == "Echo: second"
    assert [message["content"] for message in result.messages] == [
        "You are helpful.",
        "first",
        "Echo: first",
        "second",
        "Echo: second",
    ]


def test_process_persists_user_and_assistant_messages_to_session(tmp_path):
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("cli:direct")
    loop = AgentLoop(
        FakeProvider(prefix="Echo"),
        AgentConfig(system_prompt=""),
        session=session,
        session_manager=manager,
    )

    asyncio.run(loop.process("hello"))

    reloaded = SessionManager(tmp_path).get_or_create("cli:direct")
    assert reloaded.messages == [
        {"role": "user", "content": "hello", "timestamp": reloaded.messages[0]["timestamp"]},
        {"role": "assistant", "content": "Echo: hello", "timestamp": reloaded.messages[1]["timestamp"]},
    ]


def test_process_rejects_blank_input():
    loop = AgentLoop(FakeProvider())

    with pytest.raises(ValueError, match="empty"):
        asyncio.run(loop.process("   "))


class FailingProvider:
    model = "broken"

    async def complete(self, messages):
        raise RuntimeError("provider unavailable")


def test_provider_error_returns_clear_message_and_keeps_user_turn():
    loop = AgentLoop(FailingProvider())

    result = asyncio.run(loop.process("please answer"))

    assert result.model == "broken"
    assert result.content == "Error: provider unavailable"
    assert result.messages == [
        {"role": "system", "content": AgentConfig().system_prompt},
        {"role": "user", "content": "please answer"},
        {"role": "assistant", "content": "Error: provider unavailable"},
    ]
