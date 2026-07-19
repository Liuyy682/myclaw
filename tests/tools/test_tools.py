import asyncio

from myclaw import FunctionTool, ToolCallRequest, ToolRegistry
from myclaw.tools.base import ToolRuntimeContext, get_current_tool_context


def test_tool_call_request_serializes_to_openai_tool_call():
    request = ToolCallRequest(id="call_add", name="add", arguments={"a": 2, "b": 3})

    assert request.to_openai_tool_call() == {
        "id": "call_add",
        "type": "function",
        "function": {
            "name": "add",
            "arguments": '{"a": 2, "b": 3}',
        },
    }


def test_registry_returns_cached_definitions_in_stable_name_order():
    registry = ToolRegistry()
    registry.register(FunctionTool("write_file", "Write a file", {"type": "object"}, lambda: "ok"))
    registry.register(FunctionTool("read_file", "Read a file", {"type": "object"}, lambda: "ok"))

    first = registry.definitions()
    second = registry.definitions()

    assert first is second
    assert [definition["function"]["name"] for definition in first] == ["read_file", "write_file"]

    registry.unregister("write_file")
    updated = registry.definitions()
    assert updated is not first
    assert [definition["function"]["name"] for definition in updated] == ["read_file"]


def test_function_tool_exposes_openai_schema():
    tool = FunctionTool("ping", "Ping", {"type": "object"}, lambda: "pong")

    assert tool.to_schema() == {
        "type": "function",
        "function": {
            "name": "ping",
            "description": "Ping",
            "parameters": {"type": "object"},
        },
    }


class CastingTool:
    name = "double"
    description = "Double a positive integer"
    parameters = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"],
    }

    def cast_params(self, params):
        return {"count": int(params["count"])}

    def validate_params(self, params):
        if params["count"] < 1:
            raise ValueError("count must be positive")

    async def execute(self, count):
        return count * 2


def test_registry_casts_and_validates_tool_arguments_before_execute():
    registry = ToolRegistry()
    registry.register(CastingTool())

    cast = asyncio.run(
        registry.execute(ToolCallRequest(id="call_double", name="double", arguments={"count": "3"}))
    )
    invalid = asyncio.run(
        registry.execute(ToolCallRequest(id="call_double", name="double", arguments={"count": "0"}))
    )

    assert cast == "6"
    assert invalid == "Error validating double: count must be positive"


class ContextTool:
    name = "context"
    description = "Read runtime context"
    parameters = {"type": "object", "properties": {}}

    async def execute(self):
        context = get_current_tool_context()
        return {
            "session_key": context.session_key,
            "channel": context.channel,
            "chat_id": context.chat_id,
            "metadata": context.metadata,
        }


def test_registry_provides_runtime_context_during_tool_execution():
    registry = ToolRegistry()
    registry.register(ContextTool())
    context = ToolRuntimeContext(
        session_key="gateway:chat-1",
        channel="gateway",
        chat_id="chat-1",
        metadata={"request_id": "req-1"},
    )

    result = asyncio.run(
        registry.execute(ToolCallRequest(id="call_context", name="context", arguments={}), context=context)
    )

    assert result == (
        '{"session_key": "gateway:chat-1", "channel": "gateway", '
        '"chat_id": "chat-1", "metadata": {"request_id": "req-1"}}'
    )


def test_registry_executes_sync_and_async_function_tools():
    registry = ToolRegistry()
    registry.register(
        FunctionTool(
            "add",
            "Add two numbers",
            {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
            lambda a, b: a + b,
        )
    )

    async def shout(text):
        return text.upper()

    registry.register(
        FunctionTool(
            "shout",
            "Uppercase text",
            {"type": "object", "properties": {"text": {"type": "string"}}},
            shout,
        )
    )

    add_result = asyncio.run(
        registry.execute(ToolCallRequest(id="call_add", name="add", arguments={"a": 2, "b": 3}))
    )
    shout_result = asyncio.run(
        registry.execute(ToolCallRequest(id="call_shout", name="shout", arguments={"text": "hello"}))
    )

    assert add_result == "5"
    assert shout_result == "HELLO"


def test_registry_truncates_normalized_string_tool_results():
    registry = ToolRegistry()
    registry.register(FunctionTool("long", "Long result", {"type": "object"}, lambda: "abcdef"))

    result = asyncio.run(
        registry.execute(ToolCallRequest(id="call_long", name="long", arguments={}), max_result_chars=3)
    )

    assert result == "abc\n[tool result truncated: 3 chars omitted]"


def test_registry_truncates_normalized_json_tool_results():
    registry = ToolRegistry()
    registry.register(FunctionTool("json", "JSON result", {"type": "object"}, lambda: {"value": "abcdef"}))

    result = asyncio.run(
        registry.execute(ToolCallRequest(id="call_json", name="json", arguments={}), max_result_chars=10)
    )

    assert result == '{"value": \n[tool result truncated: 9 chars omitted]'


def test_registry_returns_readable_errors_for_unknown_tools_bad_arguments_and_exceptions():
    registry = ToolRegistry()
    registry.register(FunctionTool("boom", "Raise", {"type": "object"}, lambda: (_ for _ in ()).throw(ValueError("bad"))))

    missing = asyncio.run(
        registry.execute(ToolCallRequest(id="call_missing", name="missing", arguments={}))
    )
    bad_args = asyncio.run(
        registry.execute(ToolCallRequest(id="call_bad", name="boom", arguments=["not", "object"]))
    )
    raised = asyncio.run(
        registry.execute(ToolCallRequest(id="call_boom", name="boom", arguments={}))
    )

    assert missing == "Error: Tool 'missing' not found. Available: boom"
    assert bad_args == "Error: Tool 'boom' arguments must be a JSON object, got list"
    assert raised == "Error executing boom: bad"
