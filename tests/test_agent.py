import asyncio

import pytest

from myclaw import (
    AgentConfig,
    AgentLoop,
    FakeProvider,
    FunctionTool,
    ToolCallRequest,
    ToolRegistry,
    build_default_tool_registry,
)
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

    async def complete(self, messages, *, tools=None):
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

    async def complete(self, messages, *, tools=None):
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

    async def complete(self, messages, *, tools=None):
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


class ToolLoopProvider:
    model = "tools"

    def __init__(self):
        self.calls = []

    async def complete(self, messages, *, tools=None):
        self.calls.append([dict(message) for message in messages])
        if len(self.calls) == 1:
            return LLMResponse(
                content="",
                final=False,
                stop_reason="tool_calls",
                tool_calls=[ToolCallRequest(id="call_add", name="add", arguments={"a": 2, "b": 3})],
            )
        return LLMResponse(content="sum is 5", final=True)


def test_run_executes_tool_loop_and_persists_only_final_assistant_message(tmp_path):
    manager = SessionManager(tmp_path)
    registry = ToolRegistry()
    registry.register(FunctionTool("add", "Add", {"type": "object"}, lambda a, b: a + b))
    provider = ToolLoopProvider()
    loop = AgentLoop(
        provider,
        AgentConfig(system_prompt=""),
        session_manager=manager,
        tool_registry=registry,
    )

    result = asyncio.run(loop.run("what is 2 + 3?", session_key=SESSION_KEY))

    assert result.content == "sum is 5"
    assert result.messages == [
        {"role": "user", "content": "what is 2 + 3?"},
        {"role": "assistant", "content": "sum is 5"},
    ]
    assert provider.calls[1] == [
        {"role": "user", "content": "what is 2 + 3?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_add",
                    "type": "function",
                    "function": {
                        "name": "add",
                        "arguments": '{"a": 2, "b": 3}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_add", "name": "add", "content": "5"},
    ]
    reloaded = SessionManager(tmp_path).get_or_create(SESSION_KEY)
    assert [message["content"] for message in reloaded.messages] == [
        "what is 2 + 3?",
        "sum is 5",
    ]


class BuiltinReadFileProvider:
    model = "tools"

    def __init__(self):
        self.calls = []

    async def complete(self, messages, *, tools=None):
        self.calls.append([dict(message) for message in messages])
        if len(self.calls) == 1:
            return LLMResponse(
                content="",
                final=False,
                stop_reason="tool_calls",
                tool_calls=[ToolCallRequest(id="call_read", name="read_file", arguments={"path": "note.txt"})],
            )
        assert messages[-1]["role"] == "tool"
        assert messages[-1]["content"] == "1|hello from file"
        return LLMResponse(content="read complete", final=True)


def test_run_executes_builtin_read_file_tool(tmp_path):
    (tmp_path / "note.txt").write_text("hello from file\n", encoding="utf-8")
    manager = SessionManager(tmp_path / "sessions")
    provider = BuiltinReadFileProvider()
    loop = AgentLoop(
        provider,
        AgentConfig(system_prompt=""),
        session_manager=manager,
        tool_registry=build_default_tool_registry(tmp_path),
    )

    result = asyncio.run(loop.run("read note", session_key=SESSION_KEY))

    assert result.content == "read complete"
    reloaded = SessionManager(tmp_path / "sessions").get_or_create(SESSION_KEY)
    assert [message["content"] for message in reloaded.messages] == ["read note", "read complete"]
