import asyncio

import pytest

from myclaw import Agent, AgentConfig, FakeProvider


def test_run_appends_user_and_assistant_messages_in_order():
    agent = Agent(FakeProvider(prefix="Echo"), AgentConfig(system_prompt="You are helpful."))

    result = asyncio.run(agent.run("hello"))

    assert result.content == "Echo: hello"
    assert result.model == "fake"
    assert result.messages == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Echo: hello"},
    ]


def test_run_reuses_history_between_turns():
    agent = Agent(FakeProvider(prefix="Echo"), AgentConfig(system_prompt="You are helpful."))

    asyncio.run(agent.run("first"))
    result = asyncio.run(agent.run("second"))

    assert result.content == "Echo: second"
    assert [message["content"] for message in result.messages] == [
        "You are helpful.",
        "first",
        "Echo: first",
        "second",
        "Echo: second",
    ]


def test_run_rejects_blank_input():
    agent = Agent(FakeProvider())

    with pytest.raises(ValueError, match="empty"):
        asyncio.run(agent.run("   "))


class FailingProvider:
    model = "broken"

    async def complete(self, messages):
        raise RuntimeError("provider unavailable")


def test_provider_error_returns_clear_message_and_keeps_user_turn():
    agent = Agent(FailingProvider())

    result = asyncio.run(agent.run("please answer"))

    assert result.model == "broken"
    assert result.content == "Error: provider unavailable"
    assert result.messages == [
        {"role": "system", "content": AgentConfig().system_prompt},
        {"role": "user", "content": "please answer"},
        {"role": "assistant", "content": "Error: provider unavailable"},
    ]
