import asyncio
import json

import pytest

from myclaw.agent import AgentConfig, AgentLoop
from myclaw.providers import LLMResponse
from myclaw.session import SessionManager
from myclaw.tools import FunctionTool, ToolRegistry
from myclaw.tools.security import SecurityStore, canonical_args_hash, operation_id


def _loop(registry):
    loop = object.__new__(AgentLoop)
    loop.tool_registry = registry
    return loop


def _pending(call_id="call-1", name="write", arguments=None):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments if arguments is not None else {"value": "7"}),
        },
    }


def _registry(tmp_path):
    store = SecurityStore(path=tmp_path / "security.db")
    calls = []

    def cast(arguments):
        calls.append(dict(arguments))
        return {"value": int(arguments["value"])}

    registry = ToolRegistry(security_store=store)

    class CastingTool:
        name = "write"
        description = "Write"
        parameters = {"type": "object"}
        read_only = False
        exclusive = False
        input_model = None
        effect = "local_write"

        def cast_params(self, arguments):
            return cast(arguments)

        def validate_params(self, arguments):
            return None

        async def execute(self, **kwargs):
            raise AssertionError("recovery executed a tool")

    registry.register(CastingTool())
    return registry, store, calls


@pytest.mark.parametrize("status", ["succeeded", "failed", "prepared", "running", "unknown"])
def test_pending_recovery_uses_normalized_operation_and_reports_status(tmp_path, status):
    registry, store, cast_calls = _registry(tmp_path)
    session_key = "cli:ledger"
    call_id = "call-status"
    normalized = {"value": 7}
    operation = operation_id(session_key, call_id, "write", canonical_args_hash(normalized))
    store.prepare_operation(
        operation_id=operation,
        session_key=session_key,
        tool_call_id=call_id,
        tool="write",
        args_hash=canonical_args_hash(normalized),
    )
    if status == "running":
        assert store.start_operation(operation)
    elif status != "prepared":
        store.finish_operation(operation, status=status)

    messages = _loop(registry)._checkpoint_messages(
        {"messages": [], "pending_tool_calls": [_pending(call_id)]},
        session_key=session_key,
    )

    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == call_id
    assert messages[0]["name"] == "write"
    assert status in messages[0]["content"]
    assert "original tool output" in messages[0]["content"]
    assert cast_calls == [{"value": "7"}]
    assert store.get_operation(operation)["status"] == status


def test_recovery_does_not_match_operation_across_session_call_tool_or_arguments(tmp_path):
    registry, store, _ = _registry(tmp_path)
    loop = _loop(registry)
    session_key = "cli:ledger"
    pending = _pending("call-exact")
    expected_hash = canonical_args_hash({"value": 7})
    for wrong in (
        operation_id("cli:other", "call-exact", "write", expected_hash),
        operation_id(session_key, "call-other", "write", expected_hash),
        operation_id(session_key, "call-exact", "other", expected_hash),
        operation_id(session_key, "call-exact", "write", canonical_args_hash({"value": 8})),
    ):
        store.prepare_operation(
            operation_id=wrong,
            session_key="different",
            tool_call_id="different",
            tool="different",
            args_hash="different",
        )

    message = loop._checkpoint_messages(
        {"pending_tool_calls": [pending]}, session_key=session_key
    )[0]
    assert "no operation ledger record exists" in message["content"]


def test_recovery_preserves_existing_results_and_deduplicates_pending_calls(tmp_path):
    registry, _store, _ = _registry(tmp_path)
    loop = _loop(registry)
    checkpoint = {
        "messages": [
            {"role": "tool", "content": "existing result", "tool_call_id": "done", "name": "write"}
        ],
        "pending_tool_calls": [
            _pending("done"),
            _pending("pending"),
            _pending("pending"),
        ],
    }
    restored = loop._checkpoint_messages(checkpoint, session_key="cli:ledger")
    assert [message["tool_call_id"] for message in restored] == ["done", "pending"]
    assert restored[0]["content"] == "existing result"


def test_recovery_failure_branches_never_execute_and_do_not_change_ledger(tmp_path, caplog):
    registry, store, _ = _registry(tmp_path)
    loop = _loop(registry)
    before = store.list_operations()

    malformed = {
        "id": "bad-json",
        "function": {"name": "write", "arguments": "not-json secret-payload"},
    }
    removed = _pending("removed", name="removed")
    normalization_failed = _pending("bad-normalization")
    normalization_failed["function"]["arguments"] = json.dumps({"value": "not-an-int"})
    messages = loop._checkpoint_messages(
        {"pending_tool_calls": [malformed, removed, normalization_failed]},
        session_key="cli:ledger",
    )

    assert all(message["role"] == "tool" for message in messages)
    assert all("unable to verify" in message["content"] for message in messages)
    assert store.list_operations() == before
    assert "secret-payload" not in caplog.text


def test_recovery_ledger_query_exception_is_non_fatal_and_redacted(tmp_path, caplog):
    registry, _store, _ = _registry(tmp_path)

    class BrokenStore:
        def get_operation(self, operation):
            raise RuntimeError("query failed: secret-argument result-body")

    registry.security_store = BrokenStore()
    caplog.set_level("WARNING")
    message = _loop(registry)._checkpoint_messages(
        {"pending_tool_calls": [_pending("query-error")]},
        session_key="cli:ledger",
    )[0]
    assert "unable to verify" in message["content"]
    assert "secret-argument" not in caplog.text
    assert "result-body" not in caplog.text
    assert "query-error" in caplog.text


def test_recovery_unknown_ledger_status_is_non_fatal_and_redacted(tmp_path, caplog):
    registry, store, _ = _registry(tmp_path)
    session_key = "cli:ledger"
    call_id = "call-unknown-status"
    args_hash = canonical_args_hash({"value": 7})
    operation = operation_id(session_key, call_id, "write", args_hash)
    store.prepare_operation(
        operation_id=operation,
        session_key=session_key,
        tool_call_id=call_id,
        tool="write",
        args_hash=args_hash,
    )
    store.finish_operation(operation, status="unexpected")
    caplog.set_level("WARNING")
    message = _loop(registry)._checkpoint_messages(
        {"pending_tool_calls": [_pending(call_id)]}, session_key=session_key
    )[0]
    assert "unable to verify" in message["content"]
    assert "unexpected" in caplog.text
    assert "value" not in caplog.text
    assert "7" not in caplog.text


def test_recovery_without_ledger_is_explicit_and_does_not_prepare_or_execute(tmp_path):
    called = []
    registry = ToolRegistry()
    registry.register(
        FunctionTool("write", "Write", {"type": "object"}, lambda: called.append(True))
    )
    message = _loop(registry)._checkpoint_messages(
        {"pending_tool_calls": [_pending()]}, session_key="cli:no-ledger"
    )[0]
    assert "no operation ledger is configured" in message["content"]
    assert "original tool output" in message["content"]
    assert called == []


def test_rebuilt_agent_recovers_from_session_file_and_sqlite_without_duplicate_marker(tmp_path):
    session_key = "cli:rebuild"
    state = tmp_path / "state"
    db_path = state / "security.db"
    first_store = SecurityStore(path=db_path)
    first_registry = ToolRegistry(security_store=first_store)

    class CastingTool:
        name = "write"
        description = "Write"
        parameters = {"type": "object"}
        read_only = False
        exclusive = False
        input_model = None
        effect = "local_write"

        def cast_params(self, arguments):
            return {"value": int(arguments["value"])}

        def validate_params(self, arguments):
            return None

        async def execute(self, **kwargs):
            raise AssertionError("recovery executed a tool")

    first_registry.register(CastingTool())
    call_id = "call-rebuild"
    args_hash = canonical_args_hash({"value": 7})
    op_id = operation_id(session_key, call_id, "write", args_hash)
    first_store.prepare_operation(
        operation_id=op_id,
        session_key=session_key,
        tool_call_id=call_id,
        tool="write",
        args_hash=args_hash,
    )
    first_store.finish_operation(op_id, status="succeeded", result_hash="hash", result_len=4)

    session_manager = SessionManager(state)
    session = session_manager.get_or_create(session_key)
    session.add_message("user", "first")
    session.metadata.update(
        {
            "pending_user_turn": True,
            "runtime_checkpoint": {
                "phase": "awaiting_tools",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [_pending(call_id)],
                    }
                ],
                "pending_tool_calls": [_pending(call_id)],
            },
        }
    )
    session_manager.save(session)
    assert list((state / "sessions").glob("*.jsonl"))
    before = first_store.list_operations()

    class Provider:
        model = "recovery"

        def __init__(self):
            self.calls = []

        async def complete(self, messages, *, tools=None):
            self.calls.append([dict(message) for message in messages])
            return LLMResponse(content="recovered")

    provider = Provider()
    rebuilt_registry = ToolRegistry(security_store=SecurityStore(path=db_path))
    rebuilt_registry.register(CastingTool())
    rebuilt = AgentLoop(
        provider,
        config=AgentConfig(auto_title=False, system_prompt=""),
        session_manager=SessionManager(state),
        tool_registry=rebuilt_registry,
    )
    result = asyncio.run(rebuilt.run("continue", session_key=session_key))

    assert result.content == "recovered"
    recovered_messages = provider.calls[0]
    markers = [
        message for message in recovered_messages
        if message.get("role") == "tool" and message.get("tool_call_id") == call_id
    ]
    assert len(markers) == 1
    assert "succeeded" in markers[0]["content"]
    assert "Do not execute this operation again" in markers[0]["content"]
    assert rebuilt_registry.security_store.list_operations() == before

    asyncio.run(rebuilt.run("next", session_key=session_key))
    saved = SessionManager(state).get_or_create(session_key)
    saved_markers = [
        message for message in saved.messages
        if message.get("role") == "tool" and message.get("tool_call_id") == call_id
    ]
    assert len(saved_markers) == 1
    assert rebuilt_registry.security_store.list_operations() == before
