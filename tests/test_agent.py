import asyncio
import json
from datetime import datetime, timedelta

import pytest

from myclaw import (
    AgentConfig,
    AgentLoop,
    AgentRunResult,
    FakeProvider,
    FunctionTool,
    ToolCallRequest,
    ToolRegistry,
    build_default_tool_registry,
)
from myclaw.providers import LLMResponse
from myclaw.session import SessionManager
from myclaw.tools.base import get_current_tool_context

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


class TitleProvider:
    model = "title"

    def __init__(self):
        self.calls = []

    async def complete(self, messages, *, tools=None):
        self.calls.append({"messages": [dict(message) for message in messages], "tools": tools})
        if len(self.calls) == 1:
            return "assistant reply"
        return "Project Planning"


def test_run_generates_session_title_once_after_first_turn(tmp_path):
    provider = TitleProvider()
    loop = AgentLoop(
        provider,
        AgentConfig(system_prompt="", auto_title=True),
        session_manager=SessionManager(tmp_path),
    )

    asyncio.run(loop.run("plan the launch", session_key=SESSION_KEY))

    reloaded = SessionManager(tmp_path).get_or_create(SESSION_KEY)
    assert reloaded.metadata["title"] == "Project Planning"
    assert [message["content"] for message in reloaded.messages] == ["plan the launch", "assistant reply"]
    assert len(provider.calls) == 2
    assert provider.calls[1]["tools"] is None


def test_run_does_not_refresh_existing_session_title(tmp_path):
    manager = SessionManager(tmp_path)
    session = manager.get_or_create(SESSION_KEY)
    session.metadata["title"] = "Existing Title"
    manager.save(session)
    provider = TitleProvider()
    loop = AgentLoop(
        provider,
        AgentConfig(system_prompt="", auto_title=True),
        session_manager=manager,
    )

    asyncio.run(loop.run("new topic", session_key=SESSION_KEY))

    reloaded = SessionManager(tmp_path).get_or_create(SESSION_KEY)
    assert reloaded.metadata["title"] == "Existing Title"
    assert len(provider.calls) == 1


class FailingTitleProvider:
    model = "title"

    def __init__(self):
        self.calls = 0

    async def complete(self, messages, *, tools=None):
        self.calls += 1
        if self.calls == 1:
            return "assistant reply"
        raise RuntimeError("title unavailable")


def test_run_uses_first_user_message_when_title_generation_fails(tmp_path):
    loop = AgentLoop(
        FailingTitleProvider(),
        AgentConfig(system_prompt="", auto_title=True),
        session_manager=SessionManager(tmp_path),
    )

    asyncio.run(loop.run("explain multi session design", session_key=SESSION_KEY))

    reloaded = SessionManager(tmp_path).get_or_create(SESSION_KEY)
    assert reloaded.metadata["title"] == "explain multi session design"


def test_run_uses_first_user_message_for_fake_provider_title(tmp_path):
    loop = AgentLoop(
        FakeProvider(prefix="Echo"),
        AgentConfig(system_prompt="", auto_title=True),
        session_manager=SessionManager(tmp_path),
    )

    asyncio.run(loop.run("hello offline mode", session_key=SESSION_KEY))

    reloaded = SessionManager(tmp_path).get_or_create(SESSION_KEY)
    assert reloaded.metadata["title"] == "hello offline mode"


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


class CancellingRunner:
    async def run(self, spec):
        raise asyncio.CancelledError()


def test_run_restores_pending_user_turn_after_interruption(tmp_path):
    manager = SessionManager(tmp_path)
    interrupted_loop = AgentLoop(
        FakeProvider(),
        AgentConfig(system_prompt=""),
        session_manager=manager,
    )
    interrupted_loop.runner = CancellingRunner()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(interrupted_loop.run("interrupted", session_key=SESSION_KEY))

    interrupted = SessionManager(tmp_path).get_or_create(SESSION_KEY)
    assert [message["content"] for message in interrupted.messages] == ["interrupted"]
    assert interrupted.metadata["pending_user_turn"] is True
    assert "runtime_checkpoint" not in interrupted.metadata

    provider = CapturingProvider()
    recovery_loop = AgentLoop(
        provider,
        AgentConfig(system_prompt=""),
        session_manager=SessionManager(tmp_path),
    )

    asyncio.run(recovery_loop.run("next", session_key=SESSION_KEY))

    assert provider.calls[0] == [
        {"role": "user", "content": "interrupted"},
        {"role": "assistant", "content": "Error: Task interrupted before a response was generated."},
        {"role": "user", "content": "next"},
    ]
    recovered = SessionManager(tmp_path).get_or_create(SESSION_KEY)
    assert [message["content"] for message in recovered.messages] == [
        "interrupted",
        "Error: Task interrupted before a response was generated.",
        "next",
        "Echo: next",
    ]
    assert recovered.metadata == {}


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


def test_run_executes_tool_loop_and_persists_complete_tool_turn(tmp_path):
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
    assert reloaded.messages == [
        {
            "role": "user",
            "content": "what is 2 + 3?",
            "timestamp": reloaded.messages[0]["timestamp"],
        },
        {
            "role": "assistant",
            "content": "",
            "timestamp": reloaded.messages[1]["timestamp"],
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
        {
            "role": "tool",
            "content": "5",
            "timestamp": reloaded.messages[2]["timestamp"],
            "tool_call_id": "call_add",
            "name": "add",
        },
        {
            "role": "assistant",
            "content": "sum is 5",
            "timestamp": reloaded.messages[3]["timestamp"],
        },
    ]
    assert reloaded.metadata == {}


def test_run_writes_full_transcript_including_tool_calls(tmp_path):
    manager = SessionManager(tmp_path)
    registry = ToolRegistry()
    registry.register(FunctionTool("add", "Add", {"type": "object"}, lambda a, b: a + b))
    loop = AgentLoop(
        ToolLoopProvider(),
        AgentConfig(system_prompt=""),
        session_manager=manager,
        tool_registry=registry,
    )

    asyncio.run(loop.run("what is 2 + 3?", session_key=SESSION_KEY))

    path = tmp_path / "transcripts" / f"{SessionManager.safe_key(SESSION_KEY)}.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert all(record["session_key"] == SESSION_KEY for record in records)
    assert all(record["logged_at"] for record in records)
    messages = [record["message"] for record in records]

    # User message, then the verbatim assistant tool call, tool result, and final reply.
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "what is 2 + 3?"

    tool_call_message = messages[1]
    assert tool_call_message["role"] == "assistant"
    assert tool_call_message["tool_calls"][0]["function"]["name"] == "add"
    assert tool_call_message["tool_calls"][0]["function"]["arguments"] == '{"a": 2, "b": 3}'

    tool_result = messages[2]
    assert tool_result["role"] == "tool"
    assert tool_result["tool_call_id"] == "call_add"
    assert tool_result["name"] == "add"
    assert tool_result["content"] == "5"

    assert messages[3]["role"] == "assistant"
    assert messages[3]["content"] == "sum is 5"


class SpecCapturingRunner:
    def __init__(self):
        self.spec = None

    async def run(self, spec):
        self.spec = spec
        return AgentRunResult(content="ok", messages=[{"role": "assistant", "content": "ok"}])


def test_run_passes_max_tool_result_chars_to_runner_spec(tmp_path):
    loop = AgentLoop(
        FakeProvider(),
        AgentConfig(system_prompt="", max_tool_result_chars=12),
        session_manager=SessionManager(tmp_path),
    )
    runner = SpecCapturingRunner()
    loop.runner = runner

    asyncio.run(loop.run("hello", session_key=SESSION_KEY))

    assert runner.spec.max_tool_result_chars == 12


class ContextToolProvider:
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
                tool_calls=[ToolCallRequest(id="call_context", name="context", arguments={})],
            )
        return LLMResponse(content=messages[-1]["content"], final=True)


def test_run_passes_runtime_context_to_tool_calls(tmp_path):
    async def context_tool():
        context = get_current_tool_context()
        return {
            "session_key": context.session_key,
            "channel": context.channel,
            "chat_id": context.chat_id,
            "metadata": context.metadata,
        }

    registry = ToolRegistry()
    registry.register(FunctionTool("context", "Context", {"type": "object"}, context_tool))
    loop = AgentLoop(
        ContextToolProvider(),
        AgentConfig(system_prompt=""),
        session_manager=SessionManager(tmp_path),
        tool_registry=registry,
    )

    result = asyncio.run(
        loop.run(
            "read context",
            session_key="gateway:chat-1",
            channel="gateway",
            chat_id="chat-1",
            metadata={"request_id": "req-1"},
        )
    )

    assert result.content == (
        '{"session_key": "gateway:chat-1", "channel": "gateway", '
        '"chat_id": "chat-1", "metadata": {"request_id": "req-1"}}'
    )


def test_run_forwards_tool_progress_without_persisting_progress_events(tmp_path):
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
    progress = []

    async def record_progress(payload):
        progress.append(payload)

    asyncio.run(loop.run("what is 2 + 3?", session_key=SESSION_KEY, progress_callback=record_progress))

    assert [(event["event"], event["tool_name"], event["tool_call_id"]) for event in progress] == [
        ("tool_started", "add", "call_add"),
        ("tool_completed", "add", "call_add"),
    ]
    reloaded = SessionManager(tmp_path).get_or_create(SESSION_KEY)
    assert [message["role"] for message in reloaded.messages] == ["user", "assistant", "tool", "assistant"]
    assert reloaded.metadata == {}


class PartialToolProvider:
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
                tool_calls=[
                    ToolCallRequest(id="call_first", name="first", arguments={}),
                    ToolCallRequest(id="call_second", name="second", arguments={}),
                ],
            )
        return LLMResponse(content="done", final=True)


async def _cancelled_tool():
    raise asyncio.CancelledError()


def test_run_restores_runtime_checkpoint_with_pending_tool_result(tmp_path):
    registry = ToolRegistry()
    registry.register(FunctionTool("first", "First", {"type": "object"}, lambda: "first ok"))
    registry.register(FunctionTool("second", "Second", {"type": "object"}, _cancelled_tool))
    manager = SessionManager(tmp_path)
    interrupted_loop = AgentLoop(
        PartialToolProvider(),
        AgentConfig(system_prompt=""),
        session_manager=manager,
        tool_registry=registry,
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(interrupted_loop.run("use tools", session_key=SESSION_KEY))

    interrupted = SessionManager(tmp_path).get_or_create(SESSION_KEY)
    assert [message["content"] for message in interrupted.messages] == ["use tools"]
    assert interrupted.metadata["pending_user_turn"] is True
    checkpoint = interrupted.metadata["runtime_checkpoint"]
    assert checkpoint["phase"] == "tools_in_progress"
    assert [message["role"] for message in checkpoint["messages"]] == ["assistant", "tool"]
    assert checkpoint["messages"][1]["content"] == "first ok"
    assert [call["id"] for call in checkpoint["pending_tool_calls"]] == ["call_second"]

    provider = CapturingProvider()
    recovery_loop = AgentLoop(
        provider,
        AgentConfig(system_prompt=""),
        session_manager=SessionManager(tmp_path),
    )

    asyncio.run(recovery_loop.run("next", session_key=SESSION_KEY))

    assert provider.calls[0] == [
        {"role": "user", "content": "use tools"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_first",
                    "type": "function",
                    "function": {"name": "first", "arguments": "{}"},
                },
                {
                    "id": "call_second",
                    "type": "function",
                    "function": {"name": "second", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "content": "first ok", "tool_call_id": "call_first", "name": "first"},
        {
            "role": "tool",
            "content": "Error: Task interrupted before this tool finished.",
            "tool_call_id": "call_second",
            "name": "second",
        },
        {"role": "user", "content": "next"},
    ]
    recovered = SessionManager(tmp_path).get_or_create(SESSION_KEY)
    assert [message["role"] for message in recovered.messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "user",
        "assistant",
    ]
    assert recovered.metadata == {}


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
    assert [message["role"] for message in reloaded.messages] == ["user", "assistant", "tool", "assistant"]
    assert reloaded.messages[1]["tool_calls"][0]["function"]["name"] == "read_file"
    assert reloaded.messages[2]["tool_call_id"] == "call_read"
    assert reloaded.messages[2]["name"] == "read_file"
    assert reloaded.messages[2]["content"] == "1|hello from file"
    assert reloaded.messages[3]["content"] == "read complete"




class MemoryCapturingProvider:
    model = "memory"

    def __init__(self):
        self.calls = []

    async def complete(self, messages, *, tools=None):
        self.calls.append({"messages": [dict(message) for message in messages], "tools": tools})
        return "memory seen"


def test_run_injects_existing_memory_into_system_context(tmp_path):
    manager = SessionManager(tmp_path)
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "MEMORY.md").write_text(
        "# Memory\n\n- User prefers concise answers.\n",
        encoding="utf-8",
    )
    provider = MemoryCapturingProvider()
    loop = AgentLoop(
        provider,
        AgentConfig(system_prompt="Base system."),
        session_manager=manager,
    )

    asyncio.run(loop.run("hello", session_key=SESSION_KEY))

    assert provider.calls[0]["messages"][0] == {
        "role": "system",
        "content": "Base system.\n\nLong-term memory:\n# Memory\n\n- User prefers concise answers.",
    }


class RememberThenCaptureProvider:
    model = "memory"

    def __init__(self):
        self.calls = []

    async def complete(self, messages, *, tools=None):
        self.calls.append({"messages": [dict(message) for message in messages], "tools": tools})
        if len(self.calls) == 1:
            return LLMResponse(
                content="",
                final=False,
                stop_reason="tool_calls",
                tool_calls=[
                    ToolCallRequest(
                        id="call_remember",
                        name="remember",
                        arguments={"content": "User prefers concise answers."},
                    )
                ],
            )
        return LLMResponse(content="ok", final=True)


def test_remember_tool_persists_memory_for_later_turns(tmp_path):
    manager = SessionManager(tmp_path / "workspace")
    provider = RememberThenCaptureProvider()
    loop = AgentLoop(
        provider,
        AgentConfig(system_prompt="Base system."),
        session_manager=manager,
        tool_registry=build_default_tool_registry(tmp_path / "files", memory_workspace=manager.workspace),
    )

    asyncio.run(loop.run("remember my preference", session_key=SESSION_KEY))
    asyncio.run(loop.run("use my preference", session_key=SESSION_KEY))

    memory_path = manager.workspace / "memory" / "MEMORY.md"
    assert "User prefers concise answers." in memory_path.read_text(encoding="utf-8")
    assert not (manager.workspace / "memory" / "history.jsonl").exists()
    assert "Long-term memory:" not in provider.calls[1]["messages"][0]["content"]
    injected = provider.calls[2]["messages"][0]
    assert injected["role"] == "system"
    assert injected["content"].startswith("Base system.\n\nLong-term memory:\n# Memory\n\n- ")
    assert injected["content"].endswith(" User prefers concise answers.")


@pytest.mark.parametrize(
    "field",
    [
        "max_context_messages",
        "max_context_tokens",
        "context_summary_max_chars",
        "context_summary_chunk_tokens",
    ],
)
def test_agent_loop_rejects_invalid_context_budget_config(tmp_path, field):
    config = AgentConfig(system_prompt="", **{field: 0})

    with pytest.raises(ValueError, match=field):
        AgentLoop(FakeProvider(), config, session_manager=SessionManager(tmp_path))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("idle_compact_after_minutes", -1),
        ("auto_compact_recent_messages", 0),
    ],
)
def test_agent_loop_rejects_invalid_auto_compact_config(tmp_path, field, value):
    config = AgentConfig(system_prompt="", **{field: value})

    with pytest.raises(ValueError, match=field):
        AgentLoop(FakeProvider(), config, session_manager=SessionManager(tmp_path))


class BudgetSummaryProvider:
    model = "budget"

    def __init__(self):
        self.calls = []

    async def complete(self, messages, *, tools=None):
        copied = [dict(message) for message in messages]
        self.calls.append({"messages": copied, "tools": tools})
        if messages and messages[0]["role"] == "system" and "Summarize older conversation turns" in messages[0]["content"]:
            return "compressed old turns"
        last_user = next(message["content"] for message in reversed(messages) if message["role"] == "user")
        return f"final: {last_user}"


class FailingSummaryProvider:
    model = "budget"

    def __init__(self):
        self.calls = []

    async def complete(self, messages, *, tools=None):
        copied = [dict(message) for message in messages]
        self.calls.append({"messages": copied, "tools": tools})
        if messages and messages[0]["role"] == "system" and "Summarize older conversation turns" in messages[0]["content"]:
            raise RuntimeError("summary unavailable")
        return "main answer"


class SlowBudgetSummaryProvider(BudgetSummaryProvider):
    def __init__(self):
        super().__init__()
        self.summary_started = asyncio.Event()
        self.release_summary = asyncio.Event()

    async def complete(self, messages, *, tools=None):
        if messages and messages[0]["role"] == "system" and "Summarize older conversation turns" in messages[0]["content"]:
            self.summary_started.set()
            await self.release_summary.wait()
        return await super().complete(messages, tools=tools)


def _seed_plain_turns(manager, pairs):
    session = manager.get_or_create(SESSION_KEY)
    for user_text, assistant_text in pairs:
        session.add_message("user", user_text)
        session.add_message("assistant", assistant_text)
    manager.save(session)
    return session


def _set_saved_session_updated_at(workspace, key, updated_at):
    path = SessionManager(workspace)._get_session_path(key)
    lines = path.read_text(encoding="utf-8").splitlines()
    metadata = json.loads(lines[0])
    metadata["updated_at"] = updated_at.isoformat()
    lines[0] = json.dumps(metadata)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_run_summarizes_old_turns_when_message_budget_exceeded(tmp_path):
    manager = SessionManager(tmp_path)
    _seed_plain_turns(
        manager,
        [
            ("old question 1", "old answer 1"),
            ("old question 2", "old answer 2"),
            ("recent question", "recent answer"),
        ],
    )
    provider = BudgetSummaryProvider()
    loop = AgentLoop(
        provider,
        AgentConfig(
            system_prompt="",
            max_context_messages=5,
            max_context_tokens=100_000,
            context_summary_max_chars=1000,
        ),
        session_manager=manager,
    )

    result = asyncio.run(loop.run("next", session_key=SESSION_KEY))

    assert result.content == "final: next"
    assert len(provider.calls) == 2
    summary_call, main_call = provider.calls
    assert summary_call["tools"] is None
    assert "old question 1" in summary_call["messages"][1]["content"]
    assert main_call["messages"][0] == {
        "role": "system",
        "content": "Summary of earlier conversation:\ncompressed old turns",
    }
    assert {"role": "user", "content": "recent question"} in main_call["messages"]
    assert not any(message.get("content") == "old question 1" for message in main_call["messages"])

    reloaded = SessionManager(tmp_path).get_or_create(SESSION_KEY)
    summary = reloaded.metadata["context_summary"]
    assert summary["content"] == "compressed old turns"
    assert summary["covered_message_count"] == 4
    assert summary["token_estimate"] > 0
    history = [
        json.loads(line)
        for line in (tmp_path / "memory" / "history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert history == [
        {
            "id": 0,
            "timestamp": history[0]["timestamp"],
            "source": "compact",
            "content": "compressed old turns",
        }
    ]


def test_run_auto_compacts_expired_session_before_prompt(tmp_path):
    seed_manager = SessionManager(tmp_path)
    _seed_plain_turns(
        seed_manager,
        [(f"old question {index}", f"old answer {index}") for index in range(6)],
    )
    _set_saved_session_updated_at(tmp_path, SESSION_KEY, datetime.now() - timedelta(minutes=20))
    provider = BudgetSummaryProvider()
    loop = AgentLoop(
        provider,
        AgentConfig(
            system_prompt="",
            max_context_messages=100,
            max_context_tokens=100_000,
            context_summary_max_chars=1000,
            idle_compact_after_minutes=15,
            auto_compact_recent_messages=8,
        ),
        session_manager=SessionManager(tmp_path),
    )

    result = asyncio.run(loop.run("next", session_key=SESSION_KEY))

    assert result.content == "final: next"
    assert len(provider.calls) == 2
    main_messages = provider.calls[-1]["messages"]
    assert main_messages[0] == {
        "role": "system",
        "content": "Summary of earlier conversation:\ncompressed old turns",
    }
    assert not any(message.get("content") == "old question 0" for message in main_messages)
    assert {"role": "user", "content": "old question 2"} in main_messages

    reloaded = SessionManager(tmp_path).get_or_create(SESSION_KEY)
    assert [message["content"] for message in reloaded.messages[:2]] == ["old question 2", "old answer 2"]
    assert len(reloaded.messages) == 10
    assert "auto_compact_pending_summary" not in reloaded.metadata
    assert reloaded.metadata["context_summary"]["content"] == "compressed old turns"
    assert reloaded.metadata["context_summary"]["covered_message_count"] == 0

    history = [
        json.loads(line)
        for line in (tmp_path / "memory" / "history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert history[0]["source"] == "compact"
    assert history[0]["content"] == "compressed old turns"
    assert not (tmp_path / "memory" / "MEMORY.md").exists()


def test_run_waits_for_in_progress_background_auto_compact(tmp_path):
    seed_manager = SessionManager(tmp_path)
    _seed_plain_turns(
        seed_manager,
        [(f"old question {index}", f"old answer {index}") for index in range(6)],
    )
    _set_saved_session_updated_at(tmp_path, SESSION_KEY, datetime.now() - timedelta(minutes=20))
    provider = SlowBudgetSummaryProvider()
    loop = AgentLoop(
        provider,
        AgentConfig(
            system_prompt="",
            max_context_messages=100,
            max_context_tokens=100_000,
            context_summary_max_chars=1000,
            idle_compact_after_minutes=15,
            auto_compact_recent_messages=8,
        ),
        session_manager=SessionManager(tmp_path),
    )

    async def scenario():
        compact_task = asyncio.create_task(loop.auto_compact.compact_session(SESSION_KEY))
        await provider.summary_started.wait()
        run_task = asyncio.create_task(loop.run("next", session_key=SESSION_KEY))
        await asyncio.sleep(0)
        waiting_before_release = not run_task.done()
        provider.release_summary.set()
        result = await run_task
        await compact_task
        return waiting_before_release, result

    waiting_before_release, result = asyncio.run(scenario())

    assert waiting_before_release is True
    assert result.content == "final: next"
    main_messages = provider.calls[-1]["messages"]
    assert main_messages[0] == {
        "role": "system",
        "content": "Summary of earlier conversation:\ncompressed old turns",
    }
    assert not any(message.get("content") == "old question 0" for message in main_messages)


def test_run_does_not_auto_compact_when_disabled(tmp_path):
    seed_manager = SessionManager(tmp_path)
    _seed_plain_turns(seed_manager, [(f"old question {index}", f"old answer {index}") for index in range(6)])
    _set_saved_session_updated_at(tmp_path, SESSION_KEY, datetime.now() - timedelta(minutes=20))
    provider = BudgetSummaryProvider()
    loop = AgentLoop(
        provider,
        AgentConfig(system_prompt="", max_context_messages=100, max_context_tokens=100_000),
        session_manager=SessionManager(tmp_path),
    )

    asyncio.run(loop.run("next", session_key=SESSION_KEY))

    assert len(provider.calls) == 1
    assert {"role": "user", "content": "old question 0"} in provider.calls[0]["messages"]
    reloaded = SessionManager(tmp_path).get_or_create(SESSION_KEY)
    assert "context_summary" not in reloaded.metadata
    assert not (tmp_path / "memory" / "history.jsonl").exists()


def test_run_consumes_pending_auto_compact_summary_after_restart(tmp_path):
    manager = SessionManager(tmp_path)
    session = _seed_plain_turns(manager, [("recent question", "recent answer")])
    session.metadata["auto_compact_pending_summary"] = {
        "content": "archived idle summary",
        "updated_at": "2026-01-01T00:00:00",
        "token_estimate": 4,
    }
    manager.save(session)
    provider = BudgetSummaryProvider()
    loop = AgentLoop(
        provider,
        AgentConfig(system_prompt="", max_context_messages=100, max_context_tokens=100_000),
        session_manager=SessionManager(tmp_path),
    )

    asyncio.run(loop.run("next", session_key=SESSION_KEY))

    messages = provider.calls[0]["messages"]
    assert messages[0] == {
        "role": "system",
        "content": "Summary of earlier conversation:\narchived idle summary",
    }
    reloaded = SessionManager(tmp_path).get_or_create(SESSION_KEY)
    assert "auto_compact_pending_summary" not in reloaded.metadata
    assert reloaded.metadata["context_summary"]["content"] == "archived idle summary"
    assert reloaded.metadata["context_summary"]["covered_message_count"] == 0


def test_auto_compact_removes_already_covered_raw_messages_without_memory_write(tmp_path):
    seed_manager = SessionManager(tmp_path)
    session = _seed_plain_turns(
        seed_manager,
        [
            ("covered question", "covered answer"),
            ("old question", "old answer"),
            ("recent question", "recent answer"),
        ],
    )
    session.metadata["context_summary"] = {
        "content": "saved summary",
        "covered_message_count": 2,
        "updated_at": "2026-01-01T00:00:00",
        "token_estimate": 3,
    }
    seed_manager.save(session)
    _set_saved_session_updated_at(tmp_path, SESSION_KEY, datetime.now() - timedelta(minutes=20))
    provider = BudgetSummaryProvider()
    loop = AgentLoop(
        provider,
        AgentConfig(
            system_prompt="",
            max_context_messages=100,
            max_context_tokens=100_000,
            idle_compact_after_minutes=15,
            auto_compact_recent_messages=8,
        ),
        session_manager=SessionManager(tmp_path),
    )

    asyncio.run(loop.run("next", session_key=SESSION_KEY))

    assert len(provider.calls) == 1
    reloaded = SessionManager(tmp_path).get_or_create(SESSION_KEY)
    assert [message["content"] for message in reloaded.messages[:2]] == ["old question", "old answer"]
    assert reloaded.metadata["context_summary"]["content"] == "saved summary"
    assert reloaded.metadata["context_summary"]["covered_message_count"] == 0
    assert not (tmp_path / "memory" / "history.jsonl").exists()
    assert not (tmp_path / "memory" / "MEMORY.md").exists()


def test_run_summarizes_old_turns_when_token_budget_exceeded(tmp_path):
    manager = SessionManager(tmp_path)
    _seed_plain_turns(manager, [("x" * 200, "y" * 200), ("recent", "answer")])
    provider = BudgetSummaryProvider()
    loop = AgentLoop(
        provider,
        AgentConfig(system_prompt="", max_context_messages=100, max_context_tokens=40, context_summary_max_chars=1000),
        session_manager=manager,
    )

    asyncio.run(loop.run("next", session_key=SESSION_KEY))

    assert len(provider.calls) >= 2
    assert "compressed old turns" in provider.calls[-1]["messages"][0]["content"]


def test_run_reuses_persisted_context_summary_without_resummarizing(tmp_path):
    manager = SessionManager(tmp_path)
    session = _seed_plain_turns(manager, [("old", "answer"), ("recent", "answer")])
    session.metadata["context_summary"] = {
        "content": "saved summary",
        "covered_message_count": 2,
        "updated_at": "2026-01-01T00:00:00",
        "token_estimate": 3,
    }
    manager.save(session)
    provider = BudgetSummaryProvider()
    loop = AgentLoop(
        provider,
        AgentConfig(system_prompt="", max_context_messages=100, max_context_tokens=100_000),
        session_manager=manager,
    )

    asyncio.run(loop.run("next", session_key=SESSION_KEY))

    assert len(provider.calls) == 1
    messages = provider.calls[0]["messages"]
    assert messages[0] == {"role": "system", "content": "Summary of earlier conversation:\nsaved summary"}
    assert not any(message.get("content") == "old" for message in messages)
    assert {"role": "user", "content": "recent"} in messages


def test_run_uses_fallback_summary_when_summary_provider_fails(tmp_path):
    manager = SessionManager(tmp_path)
    _seed_plain_turns(manager, [("old fallback question", "old fallback answer"), ("recent", "answer")])
    provider = FailingSummaryProvider()
    loop = AgentLoop(
        provider,
        AgentConfig(system_prompt="", max_context_messages=4, max_context_tokens=100_000, context_summary_max_chars=500),
        session_manager=manager,
    )

    asyncio.run(loop.run("next", session_key=SESSION_KEY))

    assert len(provider.calls) == 2
    main_messages = provider.calls[1]["messages"]
    assert main_messages[0]["role"] == "system"
    assert "old fallback question" in main_messages[0]["content"]
    summary = SessionManager(tmp_path).get_or_create(SESSION_KEY).metadata["context_summary"]
    assert "old fallback question" in summary["content"]


def test_run_does_not_summarize_under_context_budget(tmp_path):
    manager = SessionManager(tmp_path)
    _seed_plain_turns(manager, [("first", "answer")])
    provider = BudgetSummaryProvider()
    loop = AgentLoop(
        provider,
        AgentConfig(system_prompt="", max_context_messages=100, max_context_tokens=100_000),
        session_manager=manager,
    )

    asyncio.run(loop.run("next", session_key=SESSION_KEY))

    assert len(provider.calls) == 1
    reloaded = SessionManager(tmp_path).get_or_create(SESSION_KEY)
    assert "context_summary" not in reloaded.metadata
