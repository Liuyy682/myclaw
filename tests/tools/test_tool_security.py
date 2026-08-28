import asyncio
import sqlite3

from myclaw.providers.base import ToolCallRequest
from myclaw.mcp import McpServerConfig, mcp_config_hash
from myclaw.tools import FunctionTool, PolicyGate, SecurityStore, ToolRegistry
from myclaw.tools.base import ToolRuntimeContext
from myclaw.tools.security import canonical_args, canonical_args_hash, operation_id


def test_security_store_uses_workspace_database_and_wal(tmp_path):
    store = SecurityStore(tmp_path)

    assert store.path == tmp_path / "security" / "tool_security.db"
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert {"approvals", "operations", "audit_events", "mcp_configs"} <= tables


def test_canonical_arguments_and_operation_identity_are_stable():
    assert canonical_args({"b": {"d": 2, "c": 1}, "a": [2, 1]}) == (
        '{"a":[2,1],"b":{"c":1,"d":2}}'
    )
    assert canonical_args_hash({"a": 1, "b": 2}) == canonical_args_hash({"b": 2, "a": 1})
    assert operation_id("session", "call-1", "tool", {"x": 1}) == operation_id(
        "session", "call-1", "tool", {"x": 1}
    )
    assert operation_id("session", "call-1", "tool", {"x": 1}) != operation_id(
        "session", "call-2", "tool", {"x": 1}
    )


def test_policy_gate_is_explicit_and_fail_closed():
    local = FunctionTool("local", "local", {"type": "object"}, lambda: "ok", read_only=True, effect="local_read")
    network = FunctionTool("network", "network", {"type": "object"}, lambda: "ok", read_only=True, effect="network_read")
    unknown = FunctionTool("unknown", "unknown", {"type": "object"}, lambda: "ok", read_only=True)
    ask_user = FunctionTool("ask_user", "ask", {"type": "object"}, lambda: "ok")
    gate = PolicyGate()

    assert gate.evaluate(local, {}).action == "allow"
    assert gate.evaluate(network, {}).action == "ask"
    assert gate.evaluate(unknown, {}).action == "ask"
    assert gate.evaluate(ask_user, {}).action == "allow"
    assert asyncio.run(gate.authorize(network, {})).action == "deny"


def test_policy_gate_enforces_tool_and_resource_scope(tmp_path):
    tool = FunctionTool(
        "write_file",
        "write",
        {"type": "object"},
        lambda **_: "ok",
        effect="local_write",
    )
    gate = PolicyGate()
    context = ToolRuntimeContext(
        workspace=tmp_path,
        allowed_tools=["write_file"],
        resource_scopes=[str(tmp_path / "memory")],
    )

    assert gate.evaluate(tool, {"path": "memory/note.md"}, context).action == "ask"
    assert gate.evaluate(tool, {"path": "outside.md"}, context).action == "deny"
    context.allowed_tools = ["read_file"]
    assert gate.evaluate(tool, {"path": "memory/note.md"}, context).action == "deny"


def test_approval_is_bound_and_consumed_once(tmp_path):
    store = SecurityStore(tmp_path)
    digest = canonical_args_hash({"value": 1})
    approval_id = store.create_approval(
        security_subject="subject",
        session_key="session",
        channel="cli",
        tool_call_id="call-1",
        tool="network",
        args_hash=digest,
    )

    binding = {
        "approval_id": approval_id,
        "security_subject": "subject",
        "session_key": "session",
        "channel": "cli",
        "tool_call_id": "call-1",
        "tool": "network",
        "args_hash": digest,
    }
    assert store.consume_approval(**binding) is True
    assert store.consume_approval(**binding) is False
    assert store.consume_approval(**{**binding, "tool_call_id": "call-2"}) is False


def test_registry_envelope_read_audit_and_result_cap(tmp_path):
    calls: list[int] = []
    store = SecurityStore(tmp_path)
    registry = ToolRegistry(store, max_result_chars=5)
    registry.register(
        FunctionTool(
            "read",
            "read",
            {"type": "object"},
            lambda: calls.append(1) or "abcdef",
            read_only=True,
            effect="local_read",
        )
    )
    context = ToolRuntimeContext(session_key="session", channel="cli", subject="subject")

    async def run():
        first = await registry.execute_envelope(
            ToolCallRequest("call-1", "read", {}), context=context
        )
        second = await registry.execute_envelope(
            ToolCallRequest("call-1", "read", {}), context=context
        )
        return first, second

    first, second = asyncio.run(run())
    assert first["status"] == "ok"
    assert first["result"] == "abcde\n[tool result truncated: 1 chars omitted]"
    assert second["status"] == "ok"
    assert calls == [1, 1]
    events = store.list_audit_events()
    assert events
    assert events[0]["result_hash"] is not None
    assert "abcdef" not in str(events)


def test_registry_rejects_duplicate_side_effect_operation(tmp_path):
    calls: list[int] = []
    store = SecurityStore(tmp_path)
    registry = ToolRegistry(store)
    registry.register(
        FunctionTool(
            "write",
            "write",
            {"type": "object"},
            lambda: calls.append(1) or "ok",
            effect="local_write",
        )
    )
    context = ToolRuntimeContext(session_key="session", channel="cli", subject="subject", approval_callback=lambda *_: "yes")

    async def run():
        first = await registry.execute_envelope(ToolCallRequest("call-1", "write", {}), context=context)
        second = await registry.execute_envelope(ToolCallRequest("call-1", "write", {}), context=context)
        return first, second

    first, second = asyncio.run(run())
    assert first["status"] == "ok"
    assert second["status"] == "duplicate"
    assert calls == [1]


def test_registry_exclusive_tools_are_serialized_and_timeout_becomes_unknown(tmp_path):
    active = 0
    maximum = 0

    async def exclusive_tool():
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        try:
            await asyncio.sleep(0.02)
            return "ok"
        finally:
            active -= 1

    store = SecurityStore(tmp_path)
    registry = ToolRegistry(store, tool_timeout_seconds=0.01)
    registry.register(
        FunctionTool(
            "exclusive",
            "exclusive",
            {"type": "object"},
            exclusive_tool,
            exclusive=True,
            effect="local_write",
        )
    )

    async def run():
        # No approval callback means both calls are denied before execution;
        # use a direct allow policy gate for this concurrency/timeout check.
        registry.policy_gate = PolicyGate(approval_callback=lambda *_: "yes")
        first = asyncio.create_task(
            registry.execute_envelope(ToolCallRequest("call-1", "exclusive", {}))
        )
        second = asyncio.create_task(
            registry.execute_envelope(ToolCallRequest("call-2", "exclusive", {}))
        )
        return await asyncio.gather(first, second)

    results = asyncio.run(run())
    assert all(result["status"] == "timeout" for result in results)
    assert maximum == 1
    assert {row["status"] for row in store.list_operations()} == {"unknown"}


def test_mcp_approval_stores_hash_server_and_env_names_only(tmp_path):
    store = SecurityStore(tmp_path)
    approved = store.approve_mcp("demo", "hash-1", ["TOKEN", "ENDPOINT"])

    assert approved["approved"] is True
    assert approved["env_keys"] == ["ENDPOINT", "TOKEN"]
    assert store.mcp_status("demo", "other-hash")["status"] == "config_changed"
    revoked = store.revoke_mcp("demo")
    assert revoked["approved"] is False
    assert revoked["status"] == "revoked"
    assert "secret-value" not in str(store.mcp_status("demo"))


def test_mcp_hash_changes_with_args_or_env_and_database_redacts_values(tmp_path):
    original = McpServerConfig(
        name="demo",
        command="python",
        args=["server.py"],
        env={"TOKEN": "secret-value"},
    )
    changed_arg = McpServerConfig(
        name="demo",
        command="python",
        args=["other.py"],
        env={"TOKEN": "secret-value"},
    )
    changed_env = McpServerConfig(
        name="demo",
        command="python",
        args=["server.py"],
        env={"TOKEN": "different-secret"},
    )
    assert mcp_config_hash(original) != mcp_config_hash(changed_arg)
    assert mcp_config_hash(original) != mcp_config_hash(changed_env)

    store = SecurityStore(tmp_path)
    store.approve_mcp(original.name, original.config_hash, original.env_keys)
    assert b"secret-value" not in store.path.read_bytes()
    assert store.mcp_status(original.name, changed_env.config_hash)["status"] == "config_changed"


def test_startup_marks_prepared_and_running_operations_unknown(tmp_path):
    store = SecurityStore(tmp_path)
    operation = operation_id("session", "call", "tool", {})
    assert store.prepare_operation(
        operation_id=operation,
        session_key="session",
        tool_call_id="call",
        tool="tool",
        args_hash=canonical_args_hash({}),
    )
    store.start_operation(operation)
    reopened = SecurityStore(path=store.path)
    assert reopened.get_operation(operation)["status"] == "running"
    assert reopened.recover_incomplete_operations() == 1
    assert reopened.get_operation(operation)["status"] == "unknown"


def test_degraded_exec_result_is_explicit_in_audit(tmp_path):
    store = SecurityStore(tmp_path)
    registry = ToolRegistry(store)
    registry.register(
        FunctionTool(
            "exec",
            "exec",
            {"type": "object"},
            lambda: {"exit_code": 0, "degraded": True},
            exclusive=True,
            effect="exec",
        )
    )
    context = ToolRuntimeContext(
        session_key="session",
        channel="cli",
        subject="subject",
        approval_callback=lambda *_: "allow",
    )

    envelope = asyncio.run(
        registry.execute_envelope(ToolCallRequest("call-1", "exec", {}), context=context)
    )

    assert envelope["status"] == "ok"
    assert store.list_audit_events()[0]["error_type"] == "DegradedExecSandbox"
