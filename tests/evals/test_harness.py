import json
import asyncio

from myclaw.evals import (
    DatasetValidationError,
    EvalCase,
    main,
    load_jsonl_dataset,
    run_evaluation,
)
from myclaw.evals.fixtures import build_fixture_registry
from myclaw.evals.harness import build_summary
from myclaw.providers import LLMResponse, LLMUsage, ToolCallRequest


class ToolFixtureProvider:
    model = "fake"

    def __init__(self):
        self.calls = 0

    async def complete(self, messages, *, tools=None):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content="",
                final=False,
                stop_reason="tool_calls",
                tool_calls=[
                    ToolCallRequest(
                        id="call_write",
                        name="write_file",
                        arguments={"path": "result.txt", "content": "done"},
                    )
                ],
            )
        return LLMResponse(
            content=f"finished: {messages[-1]['content']}",
            usage=LLMUsage(prompt_tokens=10, completion_tokens=3, total_tokens=13),
        )


class PersistentToolProvider:
    model = "fake"

    def __init__(self, tool_name="task_create", arguments=None, final="persisted"):
        self.calls = 0
        self.tool_name = tool_name
        self.arguments = arguments or {}
        self.final = final

    async def complete(self, messages, *, tools=None):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content="",
                final=False,
                stop_reason="tool_calls",
                tool_calls=[ToolCallRequest(id="call_persist", name=self.tool_name, arguments=self.arguments)],
            )
        return LLMResponse(content=self.final)


class RecoveryProvider:
    model = "fake"

    def __init__(self):
        self.calls = 0

    async def complete(self, messages, *, tools=None):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content="",
                final=False,
                stop_reason="tool_calls",
                tool_calls=[ToolCallRequest(
                    id="call_cancelled",
                    name="write_file",
                    arguments={"path": "partial.txt", "content": "should-not-run"},
                )],
            )
        return LLMResponse(content="recovered on the next turn")


def _write_dataset(path, *records):
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def test_live_jsonl_schema_supports_aliases_turns_and_family(tmp_path):
    dataset = tmp_path / "live.jsonl"
    _write_dataset(
        dataset,
        {
            "id": "multi",
            "turns": ["remember this", {"role": "user", "content": "use it"}],
            "family": "context",
            "category": "multiturn",
            "expect": {"output_contains": ["use it"]},
        },
    )

    loaded = load_jsonl_dataset(dataset)
    case = loaded.cases[0]
    assert case.case_id == "multi"
    assert case.prompt == "remember this"
    assert [turn.content for turn in case.turns] == ["remember this", "use it"]
    assert case.family == "context"
    assert case.expected.final_contains == ["use it"]


def test_invalid_jsonl_reports_line_number(tmp_path):
    dataset = tmp_path / "bad.jsonl"
    dataset.write_text('{"id":"missing-prompt"}\n', encoding="utf-8")

    try:
        load_jsonl_dataset(dataset)
    except DatasetValidationError as exc:
        assert "line 1" in str(exc)
        assert "prompt or turns is required" in str(exc)
    else:
        raise AssertionError("invalid dataset unexpectedly loaded")


def test_duplicate_case_ids_are_rejected(tmp_path):
    dataset = tmp_path / "duplicate.jsonl"
    _write_dataset(
        dataset,
        {"case_id": "same", "prompt": "first"},
        {"case_id": "same", "prompt": "second"},
    )

    try:
        load_jsonl_dataset(dataset)
    except DatasetValidationError as exc:
        assert "duplicate case_id 'same'" in str(exc)
        assert "line 2" in str(exc)
    else:
        raise AssertionError("duplicate dataset unexpectedly loaded")


def test_harness_runs_restricted_tool_and_writes_artifacts(tmp_path):
    dataset = tmp_path / "live.jsonl"
    _write_dataset(
        dataset,
        {
            "case_id": "write-result",
            "prompt": "Write done to result.txt",
            "tags": ["tools"],
            "family": "tool_use",
            "setup": {"files": {"input.txt": "source"}},
            "expected": {
                "final_contains": ["finished"],
                "tool_sequence": ["write_file"],
                "file_equals": {"result.txt": "done"},
                "no_tool_errors": True,
            },
        },
    )

    result = run_evaluation(
        dataset,
        profile="fake",
        output=tmp_path / "out",
        provider_factory=lambda case: ToolFixtureProvider(),
    )

    assert result.exit_code == 0
    assert result.summary["passed_runs"] == 1
    assert result.summary["by_family"]["tool_use"]["tool_calls"] == 1
    assert result.summary["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 3,
        "total_tokens": 13,
    }
    assert result.summary["successful_duration_ms"]["p50"] is not None
    assert result.report_path.exists()
    assert result.summary_path.exists()
    run = json.loads(result.runs_path.read_text(encoding="utf-8").splitlines()[0])
    assert run["tool_trajectory"][0]["name"] == "write_file"
    assert run["final_state"]["files"]["result.txt"] == "done"
    assert "Family metrics" in result.report_path.read_text(encoding="utf-8")


def test_filter_repeat_and_cli_exit_code(tmp_path):
    dataset = tmp_path / "live.jsonl"
    _write_dataset(
        dataset,
        {"case_id": "keep", "prompt": "hello", "tags": ["smoke"], "expected": {"final_contains": ["hello"]}},
        {"case_id": "drop", "prompt": "bye", "tags": ["other"], "expected": {"final_contains": ["bye"]}},
    )
    output = tmp_path / "cli-out"
    assert main([
        "--dataset", str(dataset), "--profile", "fake", "--repeat", "3",
        "--case", "keep", "--tag", "smoke", "--output", str(output),
    ]) == 0
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["total_runs"] == 3
    assert summary["by_family"]["default"]["pass_at_1"] == 1.0
    assert summary["by_family"]["default"]["pass_at_3"] == 1.0
    assert summary["by_family"]["default"]["strict_pass_at_3"] == 1.0


def test_summary_uses_all_cases_and_durations_across_families(tmp_path):
    runs = [
        {
            "case_id": "alpha-pass",
            "family": "alpha",
            "repeat_index": 1,
            "passed": True,
            "duration_ms": 10,
            "score": {"score": 1.0},
            "tool_trajectory": [],
        },
        {
            "case_id": "beta-fail",
            "family": "beta",
            "repeat_index": 1,
            "passed": False,
            "duration_ms": 20,
            "score": {"score": 0.0},
            "tool_trajectory": [],
        },
        {
            "case_id": "beta-pass",
            "family": "beta",
            "repeat_index": 1,
            "passed": True,
            "duration_ms": 30,
            "score": {"score": 1.0},
            "tool_trajectory": [],
        },
    ]

    summary = build_summary(
        runs,
        dataset=tmp_path / "live.jsonl",
        profile="fake",
        repeat=1,
        case_filters=[],
        tag_filters=[],
    )

    assert summary["pass_at_1"] == 0.666667
    assert summary["pass_at_3"] == 0.666667
    assert summary["successful_duration_ms"]["p50"] == 20.0


def test_case_model_accepts_turns_without_prompt():
    case = EvalCase.model_validate({"case_id": "turns", "turns": ["one", "two"]})
    assert case.prompt == "one"
    assert len(case.turns) == 2
    assert case.fault.phase == "before"


def test_fixture_registry_unregisters_disabled_tools_and_keeps_persistent_tools(tmp_path):
    registry = build_fixture_registry(
        tmp_path / "fixture",
        workspace=tmp_path,
        disabled_tools=["write_file", "task_update"],
    )

    assert registry.has("read_file")
    assert registry.has("task_create")
    assert registry.has("task_list")
    assert registry.has("cron")
    assert registry.has("remember")
    assert registry.has("ask_user")
    assert registry.has("spawn")
    assert not registry.has("write_file")
    assert not registry.has("task_update")
    assert "write_file" not in registry.tool_names


def test_harness_persists_task_tool_state_in_temp_workspace(tmp_path):
    dataset = tmp_path / "live.jsonl"
    _write_dataset(
        dataset,
        {
            "case_id": "task-persist",
            "prompt": "Create a task",
            "expected": {
                "tool_sequence": ["task_create"],
                "file_exists": ["tasks/tasks.json"],
                "final_contains": ["persisted"],
            },
        },
    )
    result = run_evaluation(
        dataset,
        profile="fake",
        output=tmp_path / "out",
        provider_factory=lambda case: PersistentToolProvider(
            arguments={"title": "eval task"},
            final="persisted",
        ),
    )

    assert result.exit_code == 0
    run = result.runs[0]
    assert "tasks/tasks.json" in run["final_state"]["workspace_files"]
    assert "task_create" in run["allowed_tools"]


def test_ask_answers_are_returned_in_order(tmp_path):
    dataset = tmp_path / "live.jsonl"
    _write_dataset(
        dataset,
        {
            "case_id": "ask",
            "prompt": "Ask me",
            "ask_answers": ["yes"],
            "expected": {"tool_sequence": ["ask_user"], "final_contains": ["yes"]},
        },
    )
    result = run_evaluation(
        dataset,
        profile="fake",
        output=tmp_path / "out",
        provider_factory=lambda case: PersistentToolProvider(
            tool_name="ask_user",
            arguments={"question": "continue?"},
            final="yes",
        ),
    )

    assert result.exit_code == 0
    assert result.runs[0]["ask_events"][0]["answer"] == "yes"


def test_fault_cancellation_recovers_on_next_user_turn(tmp_path):
    dataset = tmp_path / "live.jsonl"
    _write_dataset(
        dataset,
        {
            "case_id": "fault-recovery",
            "turns": ["start the interrupted step", "continue after restart"],
            "fault": {"cancel_on_tool_call": 1, "phase": "before"},
            "expected": {"final_contains": ["recovered on the next turn"]},
        },
    )
    result = run_evaluation(
        dataset,
        profile="fake",
        output=tmp_path / "out",
        provider_factory=lambda case: RecoveryProvider(),
    )

    assert result.exit_code == 0
    run = result.runs[0]
    assert run["fault_injection"]["injected"] is True
    assert run["fault_injection"]["recovered_on_next_user_turn"] is True
    assert run["recovery_mode"] == "checkpoint_turn_recovery"
    assert any(
        "interrupted before this tool finished" in str(message.get("content", ""))
        for message in run["final_state"]["session_messages"]
    )


def test_fault_after_tool_call_runs_side_effect_once_before_cancel(tmp_path):
    registry = build_fixture_registry(
        tmp_path / "fixture",
        workspace=tmp_path,
        cancel_on_tool_call=1,
        fault_phase="after",
    )
    request = ToolCallRequest(
        id="call_after",
        name="write_file",
        arguments={"path": "after.txt", "content": "written"},
    )
    try:
        asyncio.run(registry.execute(request))
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("after fault did not cancel")
    assert (tmp_path / "fixture" / "after.txt").read_text(encoding="utf-8") == "written"
    assert registry.injected is True
