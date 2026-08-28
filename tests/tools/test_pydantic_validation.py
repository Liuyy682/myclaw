import asyncio

import pytest
from pydantic import ValidationError

from myclaw.skills import SkillCatalog
from myclaw.tools import FunctionTool, SkillLoadTool, ToolCallRequest, ToolRegistry, build_default_tool_registry
from myclaw.tools.models import (
    AskUserInput,
    CronInput,
    EditFileInput,
    ExecInput,
    GlobInput,
    GrepInput,
    ListDirInput,
    MemoryWriteInput,
    MessageInput,
    MyInput,
    NotebookEditInput,
    ReadFileInput,
    SkillLoadInput,
    SpawnInput,
    TaskCreateInput,
    TaskGetInput,
    TaskListInput,
    TaskUpdateInput,
    ToolInputModel,
    WebFetchInput,
    WebSearchInput,
    WriteFileInput,
)


_MODEL_CASES = [
    (AskUserInput, {"question": "q", "choices": ["a"]}),
    (CronInput, {"name": "n", "prompt": "p"}),
    (EditFileInput, {"path": "f", "old_text": "a", "new_text": "b"}),
    (ExecInput, {"cmd": "echo hi"}),
    (GlobInput, {"pattern": "*.py"}),
    (GrepInput, {"pattern": "foo"}),
    (ListDirInput, {}),
    (MemoryWriteInput, {"content": "remember this"}),
    (MessageInput, {"content": "hello"}),
    (MyInput, {}),
    (NotebookEditInput, {"path": "n.ipynb", "cell_index": 0, "source": "x"}),
    (ReadFileInput, {"path": "f"}),
    (SkillLoadInput, {"name": "review"}),
    (SpawnInput, {"prompt": "do it"}),
    (TaskCreateInput, {"title": "task"}),
    (TaskGetInput, {"id": "task-1"}),
    (TaskListInput, {}),
    (TaskUpdateInput, {"id": "task-1"}),
    (WebFetchInput, {"url": "https://example.com"}),
    (WebSearchInput, {"query": "example"}),
    (WriteFileInput, {"path": "f", "content": "x"}),
]


@pytest.mark.parametrize("model,valid", _MODEL_CASES)
def test_every_tool_input_model_accepts_valid_input_and_forbids_extra(model, valid):
    parsed = model.model_validate(valid)

    assert isinstance(parsed, model)
    with pytest.raises(ValidationError):
        model.model_validate({**valid, "__unexpected__": 1})


@pytest.mark.parametrize(
    "model,valid,field",
    [
        (AskUserInput, {"question": "q"}, "question"),
        (CronInput, {"name": "n", "prompt": "p"}, "every_seconds"),
        (EditFileInput, {"path": "f", "old_text": "a", "new_text": "b"}, "path"),
        (ExecInput, {"cmd": "echo hi"}, "timeout_seconds"),
        (GlobInput, {"pattern": "*.py"}, "max_matches"),
        (GrepInput, {"pattern": "foo"}, "case_sensitive"),
        (ListDirInput, {}, "recursive"),
        (MemoryWriteInput, {"content": "x"}, "content"),
        (MessageInput, {"content": "x"}, "channel"),
        (MyInput, {}, "__unexpected__"),
        (NotebookEditInput, {"path": "f", "cell_index": 0, "source": "x"}, "cell_index"),
        (ReadFileInput, {"path": "f"}, "offset"),
        (SkillLoadInput, {"name": "x"}, "name"),
        (SpawnInput, {"prompt": "x"}, "name"),
        (TaskCreateInput, {"title": "x"}, "depends_on"),
        (TaskGetInput, {"id": "x"}, "id"),
        (TaskListInput, {}, "status"),
        (TaskUpdateInput, {"id": "x"}, "metadata"),
        (WebFetchInput, {"url": "https://example.com"}, "max_chars"),
        (WebSearchInput, {"query": "x"}, "max_results"),
        (WriteFileInput, {"path": "f", "content": "x"}, "content"),
    ],
)
def test_every_tool_input_model_rejects_a_typical_wrong_field_type(model, valid, field):
    wrong_value = [] if field not in {"timeout_seconds", "max_matches", "max_entries", "offset", "max_chars", "max_results", "cell_index"} else "not-an-integer"
    if field in {"case_sensitive", "recursive"}:
        wrong_value = "not-a-boolean"
    if field in {"depends_on", "metadata"}:
        wrong_value = "not-a-container"
    if field in {"name", "path", "content", "question", "channel", "prompt", "id", "query", "url"}:
        wrong_value = []

    with pytest.raises(ValidationError):
        model.model_validate({**valid, field: wrong_value})


def test_tool_input_model_has_shared_runtime_config_and_number_to_string_coercion():
    assert ToolInputModel.model_config["extra"] == "forbid"
    assert ToolInputModel.model_config["coerce_numbers_to_str"] is True

    registry = ToolRegistry()
    registry.register(
        FunctionTool(
            "typed",
            "Typed",
            {"type": "object", "properties": {"cmd": {"type": "string"}}},
            lambda **kwargs: kwargs,
            input_model=ExecInput,
            read_only=True,
            effect="local_read",
        )
    )

    coerced = asyncio.run(
        registry.execute(ToolCallRequest(id="call-typed", name="typed", arguments={"cmd": 123}))
    )
    rejected = asyncio.run(
        registry.execute(
            ToolCallRequest(id="call-typed-extra", name="typed", arguments={"cmd": "x", "extra": 1})
        )
    )

    # Excluding unset fields means execute receives only the caller-supplied
    # field; model defaults do not leak into an existing tool call.
    assert coerced == '{"cmd": "123"}'
    assert rejected == "Error validating typed: extra: Extra inputs are not permitted"


def test_registry_runs_cast_model_dump_validate_and_execute_in_order():
    events = []

    class OrderedInput(ToolInputModel):
        value: int

    class OrderedTool:
        name = "ordered"
        description = "Check preparation order"
        read_only = True
        effect = "local_read"
        parameters = {"type": "object", "properties": {"value": {"type": "integer"}}}
        input_model = OrderedInput

        def cast_params(self, params):
            events.append("cast")
            return dict(params)

        def validate_params(self, params):
            events.append(("validate", params["value"], type(params["value"])))

        async def execute(self, value):
            events.append(("execute", value, type(value)))
            return value

    registry = ToolRegistry()
    registry.register(OrderedTool())

    result = asyncio.run(
        registry.execute(ToolCallRequest(id="call-ordered", name="ordered", arguments={"value": "7"}))
    )

    assert result == "7"
    assert events == [
        "cast",
        ("validate", 7, int),
        ("execute", 7, int),
    ]


def test_registry_redacts_pydantic_validation_details_to_one_line():
    registry = ToolRegistry()
    registry.register(
        FunctionTool(
            "typed",
            "Typed",
            {"type": "object"},
            lambda **kwargs: kwargs,
            input_model=ExecInput,
        )
    )

    result = asyncio.run(
        registry.execute(
            ToolCallRequest(
                id="call-invalid",
                name="typed",
                arguments={"cmd": "echo", "timeout_seconds": "not-an-int"},
            )
        )
    )

    assert result.startswith("Error validating typed: timeout_seconds: ")
    assert "\n" not in result
    assert all(secret not in result for secret in ("input_value", "input_type", "url"))


def test_all_default_local_tools_and_skill_load_use_input_models(tmp_path):
    registry = build_default_tool_registry(tmp_path)

    assert len(registry) == 20
    assert all(getattr(registry.get(name), "input_model", None) is not None for name in registry.tool_names)

    skill_tool = SkillLoadTool(SkillCatalog.discover(tmp_path / "skills"))
    assert skill_tool.input_model is SkillLoadInput


def test_function_tool_constructor_remains_compatible_without_input_model():
    tool = FunctionTool("ping", "Ping", {"type": "object"}, lambda: "pong")

    assert tool.input_model is None


def test_function_tool_with_input_model_keeps_hand_written_schema_unchanged():
    parameters = {
        "type": "object",
        "properties": {"cmd": {"type": "string", "description": "hand-written"}},
        "required": ["cmd"],
    }
    tool = FunctionTool("typed", "Typed", parameters, lambda **kwargs: kwargs, input_model=ExecInput)

    assert tool.to_schema()["function"]["parameters"] == parameters
    registry = ToolRegistry()
    registry.register(tool)
    assert registry.definitions()[0]["function"]["parameters"] == parameters


def test_registry_rejects_missing_required_and_non_object_without_executing_spy():
    calls = []

    def spy(**kwargs):
        calls.append(kwargs)
        return kwargs

    registry = ToolRegistry()
    registry.register(FunctionTool("typed", "Typed", {"type": "object"}, spy, input_model=ExecInput))

    missing = asyncio.run(
        registry.execute(ToolCallRequest(id="call-missing", name="typed", arguments={"timeout_seconds": 1}))
    )
    wrong_container = asyncio.run(
        registry.execute(ToolCallRequest(id="call-list", name="typed", arguments=["echo"]))
    )

    assert missing == "Error validating typed: cmd: Field required"
    assert wrong_container == "Error: Tool 'typed' arguments must be a JSON object, got list"
    assert calls == []
