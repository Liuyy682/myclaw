import asyncio
import contextlib
import json

import pytest

from myclaw.agent import AgentConfig, AgentDispatcher, AgentLoop
from myclaw.agent.context import TokenEstimator
from myclaw.bus import InboundMessage, MessageBus
from myclaw.providers import LLMResponse, ToolCallRequest
from myclaw.session import SessionManager
from myclaw.tools import FunctionTool, ToolRuntimeContext, build_default_tool_registry
from myclaw.tools.base import get_current_tool_context


@pytest.fixture(autouse=True)
def _offline_token_encoding(monkeypatch):
    """Keep runtime tests deterministic when tiktoken has no local cache."""

    monkeypatch.setattr(TokenEstimator, "_load_encoding", staticmethod(lambda _model: None))


class ScriptedProvider:
    """A deterministic provider that still exercises AgentRunner and Registry."""

    model = "fake"

    def __init__(self, steps=None):
        self.steps = list(steps or [])
        self.calls = []
        self.schemas = []
        self.tool_results = []

    async def complete(self, messages, *, tools=None, **kwargs):
        self.calls.append([dict(message) for message in messages])
        self.schemas.append({entry["function"]["name"] for entry in tools or []})
        self.tool_results.extend(
            str(message["content"])
            for message in messages
            if message.get("role") == "tool"
        )
        step = self.steps.pop(0)
        if isinstance(step, str):
            return LLMResponse(content=step)
        return LLMResponse(
            content="",
            final=False,
            stop_reason="tool_calls",
            tool_calls=[
                ToolCallRequest(id=f"call-{len(self.calls)}-{index}", name=name, arguments=args)
                for index, (name, args) in enumerate(step)
            ],
        )


def _loop(tmp_path, provider, *, workspace_name="code"):
    code = tmp_path / workspace_name
    code.mkdir(exist_ok=True)
    state = tmp_path / f"{workspace_name}-state"
    registry = build_default_tool_registry(
        code,
        state_workspace=state,
        memory_workspace=state,
    )
    loop = AgentLoop(
        provider,
        AgentConfig(auto_title=False, observation_memory_enabled=False),
        session_manager=SessionManager(state),
        tool_registry=registry,
    )
    return loop, registry, code, state


def _set_plan(loop, registry, session_key, *, execute=False):
    store = registry.get("task_list").plan_store
    plan = store.create_plan("runtime plan", session_key)
    if execute:
        store.create_task(
            plan["id"],
            session_key,
            title="runtime task",
            acceptance_criteria="the runtime test passes",
        )
        plan = store.confirm_plan(plan["id"], session_key)
    loop._set_plan_metadata(session_key, plan)
    return store, plan


def test_real_runtime_schemas_and_forbidden_calls_follow_mode(tmp_path):
    cases = (
        ("normal", "normal", "task_create", "task_create"),
        ("plan", "plan", "write_file", "write_file"),
        ("execute", "execute", "task_update", "task_update"),
    )

    for index, (label, mode, forbidden, schema_name) in enumerate(cases):
        provider = ScriptedProvider([[ (forbidden, {}) ], "done"])
        loop, registry, _code, _state = _loop(tmp_path, provider, workspace_name=f"{label}-{index}")
        session_key = f"cli:{label}"
        if mode == "plan":
            _set_plan(loop, registry, session_key)
        elif mode == "execute":
            _set_plan(loop, registry, session_key, execute=True)

        result = asyncio.run(loop.run("continue", session_key=session_key))

        assert result.content == "done"
        assert schema_name not in provider.schemas[0]
        assert any("not found" in content for content in provider.tool_results)

        if mode == "normal":
            assert "read_file" in provider.schemas[0]
            assert "task_list" in provider.schemas[0]
        elif mode == "plan":
            assert {"task_create", "task_update", "task_list"} <= provider.schemas[0]
            assert "task_progress" not in provider.schemas[0]
        else:
            assert "task_progress" in provider.schemas[0]
            assert "task_create" not in provider.schemas[0]


def test_external_metadata_cannot_forge_mode_or_plan_identity(tmp_path):
    provider = ScriptedProvider([[ ("capture", {}) ], "captured"])
    loop, registry, _code, _state = _loop(tmp_path, provider)

    def capture():
        context = get_current_tool_context()
        return {
            "agent_mode": context.metadata.get("agent_mode", "normal"),
            "current_plan_id": context.metadata.get("current_plan_id"),
            "project_id": context.metadata.get("project_id"),
        }

    registry.register(
        FunctionTool(
            "capture",
            "Capture trusted runtime mode",
            {"type": "object", "properties": {}, "additionalProperties": False},
            capture,
            read_only=True,
            effect="local_read",
        )
    )
    result = asyncio.run(
        loop.run(
            "capture mode",
            session_key="cli:metadata",
            metadata={
                "agent_mode": "execute",
                "current_plan_id": "forged-plan",
                "project_id": "forged-project",
            },
        )
    )

    assert result.content == "captured"
    observed = json.loads(provider.tool_results[0])
    assert observed == {
        "agent_mode": "normal",
        "current_plan_id": None,
        "project_id": None,
    }


def test_allowed_tools_intersect_mode_base_context_and_cron_defaults(tmp_path):
    provider = ScriptedProvider()
    loop, registry, _code, _state = _loop(tmp_path, provider)

    execute_key = "cli:intersection"
    _store, _plan = _set_plan(loop, registry, execute_key, execute=True)
    base = ToolRuntimeContext(
        channel="gateway",
        allowed_tools=["read_file", "write_file", "task_create", "task_progress"],
        resource_scopes=["workspace"],
    )
    execute_context = loop._tool_runtime_context(
        execute_key,
        "gateway",
        "intersection",
        {
            "allowed_tools": ["read_file", "write_file", "task_create", "task_progress"],
            "resource_scopes": ["workspace"],
        },
        base_context=base,
    )
    assert execute_context.allowed_tools == ["read_file", "task_progress", "write_file"]
    assert execute_context.tool_names == execute_context.allowed_tools

    plan_key = "cli:plan-intersection"
    _set_plan(loop, registry, plan_key)
    plan_context = loop._tool_runtime_context(
        plan_key,
        "cli",
        "plan-intersection",
        {"allowed_tools": ["read_file", "write_file", "task_create"]},
        base_context=ToolRuntimeContext(allowed_tools=["read_file", "task_create", "write_file"]),
    )
    assert plan_context.allowed_tools == ["read_file", "task_create"]

    cron_context = loop._tool_runtime_context(
        "cron:legacy",
        "cron",
        "legacy",
        {},
    )
    assert set(cron_context.allowed_tools) <= {
        "read_file", "list_dir", "grep", "glob", "task_list", "task_get", "my", "skill_load",
    }
    assert "write_file" not in cron_context.allowed_tools
    assert cron_context.resource_scopes == ["workspace"]


def test_restart_recovers_owner_and_mode_without_replaying_pending_tool(tmp_path):
    session_key = "cli:restart"
    first_provider = ScriptedProvider(["initial"])
    first_loop, first_registry, _code, state = _loop(tmp_path, first_provider)
    store, plan = _set_plan(first_loop, first_registry, session_key, execute=True)
    task_id = store.list_tasks(plan["id"])[0]["id"]

    session_manager = SessionManager(state)
    session = session_manager.get_or_create(session_key)
    session.metadata["runtime_checkpoint"] = {
        "phase": "tools_in_progress",
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "pending-side-effect",
                        "type": "function",
                        "function": {"name": "side_effect", "arguments": "{}"},
                    }
                ],
            }
        ],
        "pending_tool_calls": [
            {
                "id": "pending-side-effect",
                "type": "function",
                "function": {"name": "side_effect", "arguments": "{}"},
            }
        ],
    }
    session.metadata["pending_user_turn"] = True
    session_manager.save(session)

    side_effects = []
    second_provider = ScriptedProvider(["recovered"])
    second_loop, second_registry, _code, _state = _loop(tmp_path, second_provider)
    second_registry.register(
        FunctionTool(
            "side_effect",
            "Side effect",
            {"type": "object", "properties": {}, "additionalProperties": False},
            lambda: side_effects.append("called"),
            effect="local_write",
        )
    )

    result = asyncio.run(second_loop.run("continue after restart", session_key=session_key))

    assert result.content == "recovered"
    assert side_effects == []
    recovered = SessionManager(state).get_or_create(session_key)
    assert recovered.metadata["agent_mode"] == "execute"
    assert recovered.metadata["current_plan_id"] == plan["id"]
    assert recovered.metadata["project_id"] == store.project_id
    assert store.get_plan(plan["id"])["owner_session"] == session_key
    assert any(
        "interrupted before this tool finished" in message["content"]
        for message in recovered.messages
        if message.get("role") == "tool"
    )
    assert task_id in {task["id"] for task in store.list_tasks(plan["id"])}


class _SlowProvider:
    model = "fake"

    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(self, messages, *, tools=None, **kwargs):
        self.started.set()
        await self.release.wait()
        return "finished"


def test_busy_session_rejects_plan_and_execute_but_allows_plans_read(tmp_path):
    async def scenario():
        provider = _SlowProvider()
        loop, _registry, _code, _state = _loop(tmp_path, provider)
        bus = MessageBus()
        dispatcher = AgentDispatcher(bus, loop)
        running = asyncio.create_task(dispatcher.run())

        try:
            first = await dispatcher.submit(
                InboundMessage(channel="cli", sender_id="user", chat_id="busy", content="long turn")
            )
            assert first.accepted
            await asyncio.wait_for(provider.started.wait(), timeout=2)

            plan = await dispatcher.submit(
                InboundMessage(channel="cli", sender_id="user", chat_id="busy", content="/plan new x")
            )
            assert not plan.accepted
            plan_error = await asyncio.wait_for(bus.consume_outbound(), timeout=2)
            assert plan_error.event_type == "error"
            assert "Cannot switch plan mode" in plan_error.content

            execute = await dispatcher.submit(
                InboundMessage(channel="cli", sender_id="user", chat_id="busy", content="/execute")
            )
            assert not execute.accepted
            execute_error = await asyncio.wait_for(bus.consume_outbound(), timeout=2)
            assert execute_error.event_type == "error"
            assert "Cannot switch plan mode" in execute_error.content

            listing = await dispatcher.submit(
                InboundMessage(channel="cli", sender_id="user", chat_id="busy", content="/plans")
            )
            assert listing.accepted
            listing_event = await asyncio.wait_for(bus.consume_outbound(), timeout=2)
            assert listing_event.event_type == "control"
        finally:
            provider.release.set()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(bus.consume_outbound(), timeout=2)
            running.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await running

    asyncio.run(scenario())
