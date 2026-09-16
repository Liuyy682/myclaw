from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel, ConfigDict

from myclaw.agent import AgentConfig, AgentLoop, AgentRunSpec, AgentRunner
from myclaw.agent.context import TokenEstimator
from myclaw.providers import FakeProvider, LLMResponse
from myclaw.session import SessionManager
from myclaw.tools import FunctionTool, ToolCallRequest, ToolRegistry, ToolRuntimeContext, build_default_tool_registry


@pytest.fixture(autouse=True)
def _offline_token_estimator(monkeypatch):
    monkeypatch.setattr(TokenEstimator, "_load_encoding", staticmethod(lambda _model: None))


class _RequiredValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: int


class _ScriptedProvider:
    model = "eval-scripted"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def complete(self, messages, *, tools=None):
        self.calls.append([dict(message) for message in messages])
        if not self.responses:
            raise AssertionError("provider script exhausted")
        return self.responses.pop(0)


def _tool_response(call_id: str, name: str, arguments: dict) -> LLMResponse:
    return LLMResponse(
        content="",
        final=False,
        stop_reason="tool_calls",
        tool_calls=[ToolCallRequest(id=call_id, name=name, arguments=arguments)],
    )


def _run_runner(provider, registry, *, max_iterations=5):
    async def approve(_question, _choices):
        return "allow"

    return asyncio.run(AgentRunner(provider).run(AgentRunSpec(
        messages=[{"role": "user", "content": "complete the eval task"}],
        model=provider.model,
        max_iterations=max_iterations,
        tools=registry,
        tool_context=ToolRuntimeContext(session_key="eval:runner", ask=approve),
    )))


def _loop(provider, workspace, registry=None):
    return AgentLoop(
        provider,
        AgentConfig(system_prompt="", auto_title=False, max_context_messages=100, max_context_tokens=100_000),
        session_manager=SessionManager(workspace),
        tool_registry=registry,
    )


# Tool-correction closure: validation errors must be visible to the next model
# iteration, while the underlying side effect happens at most once.
def test_eval_missing_argument_can_be_corrected_without_duplicate_side_effect():
    executions = []
    registry = ToolRegistry()
    registry.register(FunctionTool(
        "record", "Record a value", {"type": "object"},
        lambda value: executions.append(value) or "recorded",
        read_only=True,
        effect="local_read",
        input_model=_RequiredValue,
    ))
    provider = _ScriptedProvider([
        _tool_response("bad", "record", {}),
        _tool_response("good", "record", {"value": 7}),
        LLMResponse(content="done"),
    ])

    result = _run_runner(provider, registry)

    assert result.content == "done"
    assert executions == [7]
    assert "Error validating record" in provider.calls[1][-1]["content"]


def test_eval_unknown_tool_can_be_replaced_by_available_tool():
    registry = ToolRegistry()
    registry.register(FunctionTool("known", "Known tool", {"type": "object"}, lambda: "ok", read_only=True, effect="local_read"))
    provider = _ScriptedProvider([
        _tool_response("missing", "unknown", {}),
        _tool_response("known", "known", {}),
        LLMResponse(content="recovered"),
    ])

    result = _run_runner(provider, registry)

    assert result.content == "recovered"
    assert "Tool 'unknown' not found" in provider.calls[1][-1]["content"]
    assert provider.calls[2][-1]["content"] == "ok"


def test_eval_execution_error_can_be_corrected_with_alternative_tool():
    executions = []
    registry = ToolRegistry()
    registry.register(FunctionTool("primary", "Primary", {"type": "object"}, lambda: 1 / 0, read_only=True, effect="local_read"))
    registry.register(FunctionTool(
        "fallback", "Fallback", {"type": "object"},
        lambda: executions.append("fallback") or "ok",
        read_only=True,
        effect="local_read",
    ))
    provider = _ScriptedProvider([
        _tool_response("primary", "primary", {}),
        _tool_response("fallback", "fallback", {}),
        LLMResponse(content="done"),
    ])

    result = _run_runner(provider, registry)

    assert result.content == "done"
    assert executions == ["fallback"]
    assert "Error executing primary" in provider.calls[1][-1]["content"]


# Context behavior is asserted at the actual provider boundary, not by testing
# the summarizer in isolation.
def test_eval_same_session_retains_prior_turn_at_provider_boundary(tmp_path):
    provider = _ScriptedProvider([LLMResponse(content="first answer"), LLMResponse(content="second answer")])
    loop = _loop(provider, tmp_path)

    asyncio.run(loop.run("first fact", session_key="eval:same"))
    asyncio.run(loop.run("recall it", session_key="eval:same"))

    assert {"role": "user", "content": "first fact"} in provider.calls[1]
    assert {"role": "assistant", "content": "first answer"} in provider.calls[1]


def test_eval_updated_fact_remains_latest_in_provider_context(tmp_path):
    provider = _ScriptedProvider([
        LLMResponse(content="noted old"),
        LLMResponse(content="noted new"),
        LLMResponse(content="September 18"),
    ])
    loop = _loop(provider, tmp_path)

    asyncio.run(loop.run("deadline is September 10", session_key="eval:update"))
    asyncio.run(loop.run("deadline changed to September 18", session_key="eval:update"))
    asyncio.run(loop.run("what is the deadline?", session_key="eval:update"))

    contents = [message["content"] for message in provider.calls[2] if message["role"] == "user"]
    assert contents[-2:] == ["deadline changed to September 18", "what is the deadline?"]


def test_eval_sessions_do_not_leak_context(tmp_path):
    provider = _ScriptedProvider([LLMResponse(content="one"), LLMResponse(content="two")])
    loop = _loop(provider, tmp_path)

    asyncio.run(loop.run("secret-one", session_key="eval:one"))
    asyncio.run(loop.run("question-two", session_key="eval:two"))

    assert not any(message.get("content") == "secret-one" for message in provider.calls[1])


class _CancelProvider:
    model = "eval-cancel"

    def __init__(self):
        self.calls = 0

    async def complete(self, messages, *, tools=None):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content="", final=False, stop_reason="tool_calls",
                tool_calls=[
                    ToolCallRequest(id="first", name="first", arguments={}),
                    ToolCallRequest(id="second", name="second", arguments={}),
                ],
            )
        return LLMResponse(content="unexpected")


async def _cancelled():
    raise asyncio.CancelledError()


def _interrupted_workspace(workspace, effects):
    registry = ToolRegistry()
    registry.register(FunctionTool("first", "First", {"type": "object"}, lambda: effects.append("first") or "ok", read_only=True, effect="local_read"))
    registry.register(FunctionTool("second", "Second", {"type": "object"}, _cancelled, read_only=True, effect="local_read"))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_loop(_CancelProvider(), workspace, registry).run("start", session_key="eval:recovery"))


def test_eval_recovery_preserves_completed_tool_without_reexecution(tmp_path):
    effects = []
    _interrupted_workspace(tmp_path, effects)
    capture = _ScriptedProvider([LLMResponse(content="recovered")])

    asyncio.run(_loop(capture, tmp_path).run("continue", session_key="eval:recovery"))

    assert effects == ["first"]
    assert any(message.get("tool_call_id") == "first" and message["content"] == "ok" for message in capture.calls[0])


def test_eval_recovery_marks_unfinished_tool_as_interrupted(tmp_path):
    _interrupted_workspace(tmp_path, [])
    capture = _ScriptedProvider([LLMResponse(content="recovered")])

    asyncio.run(_loop(capture, tmp_path).run("continue", session_key="eval:recovery"))

    assert any(
        message.get("tool_call_id") == "second"
        and "no operation ledger is configured" in message["content"]
        for message in capture.calls[0]
    )


def test_eval_recovery_clears_runtime_checkpoint_after_next_turn(tmp_path):
    _interrupted_workspace(tmp_path, [])
    asyncio.run(_loop(FakeProvider(), tmp_path).run("continue", session_key="eval:recovery"))

    session = SessionManager(tmp_path).get_or_create("eval:recovery")
    assert "runtime_checkpoint" not in session.metadata
    assert "pending_user_turn" not in session.metadata


def test_eval_unknown_forbidden_tool_has_no_side_effect():
    effects = []
    registry = ToolRegistry()
    registry.register(FunctionTool("safe", "Safe", {"type": "object"}, lambda: effects.append("safe"), read_only=True, effect="local_read"))
    provider = _ScriptedProvider([_tool_response("forbidden", "delete_everything", {}), LLMResponse(content="refused")])

    result = _run_runner(provider, registry)

    assert result.content == "refused"
    assert effects == []


def test_eval_case_workspaces_isolate_file_side_effects(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    provider = _ScriptedProvider([
        _tool_response("write", "write_file", {"path": "result.txt", "content": "first"}),
        LLMResponse(content="done"),
    ])

    _run_runner(provider, build_default_tool_registry(first_root, memory_workspace=first_root))

    assert (first_root / "result.txt").read_text(encoding="utf-8") == "first"
    assert not (second_root / "result.txt").exists()


def test_eval_case_sessions_isolate_persisted_state(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    asyncio.run(_loop(FakeProvider(), first).run("alpha", session_key="eval:case"))
    asyncio.run(_loop(FakeProvider(), second).run("beta", session_key="eval:case"))

    first_messages = SessionManager(first).get_or_create("eval:case").messages
    second_messages = SessionManager(second).get_or_create("eval:case").messages
    assert [message["content"] for message in first_messages] == ["alpha", "Echo: alpha"]
    assert [message["content"] for message in second_messages] == ["beta", "Echo: beta"]
