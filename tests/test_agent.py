import asyncio

import pytest

from myclaw import AgentConfig, AgentLoop, FakeProvider
from myclaw.providers import LLMResponse
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


class CapturingProvider:
    model = "capture"

    def __init__(self):
        self.calls = []

    async def complete(self, messages):
        self.calls.append([dict(message) for message in messages])
        last_user = next(message["content"] for message in reversed(messages) if message["role"] == "user")
        return f"Echo: {last_user}"


def test_process_sends_previous_turn_history_to_provider():
    provider = CapturingProvider()
    loop = AgentLoop(provider, AgentConfig(system_prompt="You are helpful."))

    asyncio.run(loop.process("first"))
    asyncio.run(loop.process("second"))

    assert provider.calls[1] == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "Echo: first"},
        {"role": "user", "content": "second"},
    ]


def test_process_loads_persisted_history_for_new_loop(tmp_path):
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("cli:direct")
    session.add_message("user", "persisted")
    session.add_message("assistant", "Echo: persisted")
    manager.save(session)

    provider = CapturingProvider()
    reloaded_session = SessionManager(tmp_path).get_or_create("cli:direct")
    loop = AgentLoop(
        provider,
        AgentConfig(system_prompt="", history=[
            {"role": message["role"], "content": message["content"]}
            for message in reloaded_session.messages
        ]),
        session=reloaded_session,
        session_manager=manager,
    )

    asyncio.run(loop.process("next"))

    assert provider.calls[0] == [
        {"role": "user", "content": "persisted"},
        {"role": "assistant", "content": "Echo: persisted"},
        {"role": "user", "content": "next"},
    ]


class MultiAssistantProvider:
    model = "multi"

    def __init__(self):
        self.calls = 0

    async def complete(self, messages):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(content="draft", final=False, stop_reason="continue")
        return LLMResponse(content="final", final=True)


def test_process_persists_all_assistant_messages_from_internal_iterations(tmp_path):
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("cli:direct")
    loop = AgentLoop(
        MultiAssistantProvider(),
        AgentConfig(system_prompt="", max_turns=2),
        session=session,
        session_manager=manager,
    )

    result = asyncio.run(loop.process("work"))

    assert [message["content"] for message in result.messages] == ["work", "draft", "final"]
    reloaded = SessionManager(tmp_path).get_or_create("cli:direct")
    assert [message["content"] for message in reloaded.messages] == ["work", "draft", "final"]


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
