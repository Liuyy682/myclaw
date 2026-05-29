import asyncio

import pytest

from myclaw import AgentConfig, AgentLoop, FakeProvider
from myclaw.providers import LLMResponse
from myclaw.session import SessionManager

SESSION_KEY = "cli:direct"


def test_run_appends_user_and_assistant_messages_in_order(tmp_path):
    loop = AgentLoop(
        FakeProvider(prefix="Echo"),
        AgentConfig(system_prompt="You are helpful."),
        session_manager=SessionManager(tmp_path),
    )

    result = asyncio.run(loop.run("hello", session_key=SESSION_KEY))

    assert result.content == "Echo: hello"
    assert result.model == "fake"
    assert result.messages == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Echo: hello"},
    ]


class CapturingProvider:
    model = "capture"

    def __init__(self):
        self.calls = []

    async def complete(self, messages):
        self.calls.append([dict(message) for message in messages])
        last_user = next(message["content"] for message in reversed(messages) if message["role"] == "user")
        return f"Echo: {last_user}"


def test_run_reuses_history_for_same_session_key(tmp_path):
    provider = CapturingProvider()
    loop = AgentLoop(
        provider,
        AgentConfig(system_prompt="You are helpful."),
        session_manager=SessionManager(tmp_path),
    )

    asyncio.run(loop.run("first", session_key=SESSION_KEY))
    result = asyncio.run(loop.run("second", session_key=SESSION_KEY))

    assert result.content == "Echo: second"
    assert [message["content"] for message in result.messages] == [
        "You are helpful.",
        "first",
        "Echo: first",
        "second",
        "Echo: second",
    ]
    assert provider.calls[1] == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "Echo: first"},
        {"role": "user", "content": "second"},
    ]


def test_run_isolates_history_by_session_key(tmp_path):
    provider = CapturingProvider()
    loop = AgentLoop(
        provider,
        AgentConfig(system_prompt=""),
        session_manager=SessionManager(tmp_path),
    )

    asyncio.run(loop.run("first", session_key="cli:one"))
    asyncio.run(loop.run("second", session_key="cli:two"))

    assert provider.calls[1] == [
        {"role": "user", "content": "second"},
    ]


def test_run_loads_persisted_history_for_new_loop(tmp_path):
    manager = SessionManager(tmp_path)
    session = manager.get_or_create(SESSION_KEY)
    session.add_message("user", "persisted")
    session.add_message("assistant", "Echo: persisted")
    manager.save(session)

    provider = CapturingProvider()
    loop = AgentLoop(
        provider,
        AgentConfig(system_prompt=""),
        session_manager=SessionManager(tmp_path),
    )

    asyncio.run(loop.run("next", session_key=SESSION_KEY))

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


def test_run_persists_all_assistant_messages_from_internal_iterations(tmp_path):
    manager = SessionManager(tmp_path)
    loop = AgentLoop(
        MultiAssistantProvider(),
        AgentConfig(system_prompt="", max_turns=2),
        session_manager=manager,
    )

    result = asyncio.run(loop.run("work", session_key=SESSION_KEY))

    assert [message["content"] for message in result.messages] == ["work", "draft", "final"]
    reloaded = SessionManager(tmp_path).get_or_create(SESSION_KEY)
    assert [message["content"] for message in reloaded.messages] == ["work", "draft", "final"]


def test_run_persists_user_and_assistant_messages_to_session(tmp_path):
    manager = SessionManager(tmp_path)
    loop = AgentLoop(
        FakeProvider(prefix="Echo"),
        AgentConfig(system_prompt=""),
        session_manager=manager,
    )

    asyncio.run(loop.run("hello", session_key=SESSION_KEY))

    reloaded = SessionManager(tmp_path).get_or_create(SESSION_KEY)
    assert reloaded.messages == [
        {"role": "user", "content": "hello", "timestamp": reloaded.messages[0]["timestamp"]},
        {"role": "assistant", "content": "Echo: hello", "timestamp": reloaded.messages[1]["timestamp"]},
    ]


def test_run_rejects_blank_input(tmp_path):
    loop = AgentLoop(FakeProvider(), session_manager=SessionManager(tmp_path))

    with pytest.raises(ValueError, match="empty"):
        asyncio.run(loop.run("   ", session_key=SESSION_KEY))


class FailingProvider:
    model = "broken"

    async def complete(self, messages):
        raise RuntimeError("provider unavailable")


def test_provider_error_returns_clear_message_and_keeps_user_turn(tmp_path):
    manager = SessionManager(tmp_path)
    loop = AgentLoop(FailingProvider(), session_manager=manager)

    result = asyncio.run(loop.run("please answer", session_key=SESSION_KEY))

    assert result.model == "broken"
    assert result.content == "Error: provider unavailable"
    assert result.messages == [
        {"role": "system", "content": AgentConfig().system_prompt},
        {"role": "user", "content": "please answer"},
        {"role": "assistant", "content": "Error: provider unavailable"},
    ]
    reloaded = SessionManager(tmp_path).get_or_create(SESSION_KEY)
    assert [message["content"] for message in reloaded.messages] == [
        "please answer",
        "Error: provider unavailable",
    ]
