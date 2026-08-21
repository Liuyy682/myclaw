"""Pydantic models and JSONL loading for the local Agent eval harness.

The evaluator deliberately keeps the dataset format small.  A line describes
one independent case; setup is materialised in a temporary fixture directory
and ``expected`` contains only deterministic, rule-based checks.

The input normalisation accepts a few descriptive aliases (``id`` for
``case_id``, ``input`` for ``prompt``, and ``expect``/``assertions`` for
``expected``).  This makes hand-written JSONL less error prone while the
validated model remains the single shape used by the runner and scorer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class DatasetValidationError(ValueError):
    """Raised when one or more JSONL records do not match the live schema."""

    def __init__(self, message: str, *, line_errors: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.line_errors = line_errors or []


class EvalModel(BaseModel):
    """Common model policy for forward-compatible hand-written datasets."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class ToolCallExpectation(EvalModel):
    """A deterministic expectation for one tool call in the trajectory."""

    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    exact_arguments: bool = False

    @model_validator(mode="before")
    @classmethod
    def _normalise(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"name": value}
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "name" not in data:
            for alias in ("tool", "tool_name", "function"):
                if alias in data:
                    data["name"] = data[alias]
                    break
        if "arguments" not in data:
            for alias in ("args", "parameters", "arguments_subset"):
                if alias in data:
                    data["arguments"] = data[alias]
                    break
        if "exact_arguments" not in data and "exact" in data:
            data["exact_arguments"] = data["exact"]
        return data


class RuleExpectation(EvalModel):
    """Optional extensible rule form used by ``expected.rules``."""

    type: str = Field(min_length=1)
    value: Any = None
    path: str | None = None
    name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    contains: Any = None
    equals: Any = None

    @model_validator(mode="before")
    @classmethod
    def _normalise(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"type": value}
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "type" not in data:
            for alias in ("rule", "kind", "check"):
                if alias in data:
                    data["type"] = data[alias]
                    break
        if "value" not in data:
            if "expected" in data:
                data["value"] = data["expected"]
            elif "text" in data:
                data["value"] = data["text"]
        return data


class EvalExpectations(EvalModel):
    """Deterministic assertions made against one run."""

    final_contains: list[str] = Field(default_factory=list)
    final_equals: str | None = None
    final_not_contains: list[str] = Field(default_factory=list)
    final_regex: list[str] = Field(default_factory=list)
    tool_sequence: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCallExpectation] = Field(default_factory=list)
    file_contains: dict[str, str] = Field(default_factory=dict)
    file_equals: dict[str, Any] = Field(default_factory=dict)
    file_exists: list[str] = Field(default_factory=list)
    file_absent: list[str] = Field(default_factory=list)
    stop_reason: str | None = None
    min_tool_calls: int | None = Field(default=None, ge=0)
    max_tool_calls: int | None = Field(default=None, ge=0)
    no_tool_errors: bool = False
    state: dict[str, Any] = Field(default_factory=dict)
    rules: list[RuleExpectation] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalise(cls, value: Any) -> Any:
        if value is None:
            return {}
        if isinstance(value, str):
            return {"final_contains": [value]}
        if isinstance(value, list):
            # A short ``expected: ["..."]`` form means output fragments.
            return {"final_contains": value}
        if not isinstance(value, dict):
            return value

        data = dict(value)
        aliases = {
            "output_contains": "final_contains",
            "answer_contains": "final_contains",
            "contains": "final_contains",
            "output_equals": "final_equals",
            "answer_equals": "final_equals",
            "output_not_contains": "final_not_contains",
            "answer_not_contains": "final_not_contains",
            "regex": "final_regex",
            "tools": "tool_sequence",
            "tool_trace": "tool_sequence",
            "expected_tools": "tool_sequence",
            "file_contents": "file_equals",
            "files_contains": "file_contains",
            "files_exist": "file_exists",
            "files_absent": "file_absent",
            "expected_stop_reason": "stop_reason",
            "assertions": "rules",
        }
        for source, target in aliases.items():
            if target not in data and source in data:
                data[target] = data[source]

        # ``tools``/``tool_trace`` are often written as objects with argument
        # matchers.  Convert those to the explicit tool-call form.
        sequence = data.get("tool_sequence")
        if isinstance(sequence, list) and any(isinstance(item, dict) for item in sequence):
            data.pop("tool_sequence", None)
            data.setdefault("tool_calls", sequence)
        elif isinstance(sequence, str):
            data["tool_sequence"] = [sequence]

        calls = data.get("tool_calls")
        if isinstance(calls, str):
            data["tool_calls"] = [calls]

        # ``final_state: {files: ...}`` is accepted as a spelling of state.
        final_state = data.get("final_state")
        if "state" not in data and isinstance(final_state, dict):
            data["state"] = final_state
        if "state" not in data and isinstance(data.get("files"), dict):
            data["state"] = {"files": data["files"]}

        # A convenient ``state.files`` map becomes file equality checks.  A
        # nested rule can still be used when existence/contains semantics are
        # required.
        state = data.get("state")
        if isinstance(state, dict) and isinstance(state.get("files"), dict):
            files = state["files"]
            if "file_equals" not in data:
                exact: dict[str, str] = {}
                for path, expected in files.items():
                    if isinstance(expected, str):
                        exact[str(path)] = expected
                if exact:
                    data["file_equals"] = exact

        return data


class FixtureSetup(EvalModel):
    """Initial files and directories created in a case's fixture root."""

    files: dict[str, Any] = Field(default_factory=dict)
    directories: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalise(cls, value: Any) -> Any:
        if value is None:
            return {}
        if not isinstance(value, dict):
            return value
        data = dict(value)
        for alias in ("initial_files", "fixture_files"):
            if "files" not in data and alias in data:
                data["files"] = data[alias]
        if "files" not in data and "directories" not in data:
            # ``setup: {"notes/todo.txt": "..."}`` is useful in tiny smoke
            # datasets and unambiguous because known setup keys are absent.
            data = {"files": data}
        return data


class EvalTurn(EvalModel):
    """One conversational turn; strings are normalised to user turns."""

    role: str = "user"
    content: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _normalise(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"role": "user", "content": value}
        if isinstance(value, dict):
            data = dict(value)
            if "content" not in data:
                for alias in ("text", "prompt", "input"):
                    if alias in data:
                        data["content"] = data[alias]
                        break
            return data
        return value


class FaultConfig(EvalModel):
    """Optional one-shot cancellation injected around a tool call."""

    cancel_on_tool_call: int | None = Field(default=None, ge=1)
    phase: Literal["before", "after"] = "before"


class EvalCase(EvalModel):
    """One live JSONL evaluation case."""

    case_id: str = Field(min_length=1)
    prompt: str = ""
    turns: list[EvalTurn] = Field(default_factory=list)
    fault: FaultConfig = Field(default_factory=FaultConfig)
    ask_answers: list[str] = Field(default_factory=list)
    family: str = "default"
    category: str | None = None
    capability: str | None = None
    claim: str | None = None
    metric_type: str | None = None
    pytest_node_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    setup: FixtureSetup = Field(default_factory=FixtureSetup)
    expected: EvalExpectations = Field(default_factory=EvalExpectations)
    profile: str = "live"
    system_prompt: str | None = None
    max_turns: int = Field(default=4, ge=1, le=32)
    max_context_messages: int = Field(default=100, ge=2)
    max_context_tokens: int = Field(default=100_000, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    disabled_tools: list[str] = Field(default_factory=list)

    # This marker is intentionally public: callers can use it when generating
    # or documenting a dataset without importing implementation details.
    schema_name: ClassVar[str] = "myclaw.eval.live.case.v1"

    @model_validator(mode="before")
    @classmethod
    def _normalise(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)

        if "case_id" not in data:
            for alias in ("id", "name", "case"):
                candidate = data.get(alias)
                if isinstance(candidate, str) and candidate.strip():
                    data["case_id"] = candidate
                    break
        if "prompt" not in data:
            for alias in ("input", "task", "goal", "user_prompt"):
                if alias in data:
                    data["prompt"] = data[alias]
                    break
        if "prompt" not in data and isinstance(data.get("turns"), list):
            for turn in data["turns"]:
                content = turn if isinstance(turn, str) else turn.get("content") if isinstance(turn, dict) else None
                role = "user" if isinstance(turn, str) else turn.get("role", "user") if isinstance(turn, dict) else ""
                if isinstance(content, str) and content.strip() and role == "user":
                    data["prompt"] = content
                    break
        if "expected" not in data:
            for alias in ("expect", "assert", "assertions", "rules"):
                if alias in data:
                    value_to_use = data[alias]
                    if alias in {"assertions", "rules"} and isinstance(value_to_use, list):
                        value_to_use = {"rules": value_to_use}
                    data["expected"] = value_to_use
                    break
        if "setup" not in data:
            for alias in ("fixture", "initial_state", "initial_files"):
                if alias in data:
                    value_to_use = data[alias]
                    if alias == "initial_files":
                        value_to_use = {"files": value_to_use}
                    data["setup"] = value_to_use
                    break
        if "turns" not in data and isinstance(data.get("initial_messages"), list):
            data["turns"] = data["initial_messages"]
        if "ask_answers" not in data:
            for alias in ("answers", "ask_user_answers"):
                if alias in data:
                    data["ask_answers"] = data[alias]
                    break
        if "fault" not in data and (
            "cancel_on_tool_call" in data or "fault_phase" in data
        ):
            data["fault"] = {
                "cancel_on_tool_call": data.get("cancel_on_tool_call"),
                "phase": data.get("fault_phase", "before"),
            }
        elif data.get("fault") is None:
            data["fault"] = {}
        if isinstance(data.get("tags"), str):
            data["tags"] = [tag.strip() for tag in data["tags"].split(",") if tag.strip()]
        metadata = data.get("metadata")
        if isinstance(metadata, dict):
            if "family" not in data and isinstance(metadata.get("family"), str):
                data["family"] = metadata["family"]
            if "metric_type" not in data and isinstance(metadata.get("metric_type"), str):
                data["metric_type"] = metadata["metric_type"]
            if "category" not in data and isinstance(metadata.get("metric_type"), str):
                data["category"] = metadata["metric_type"]
        if "family" not in data and isinstance(data.get("category"), str):
            data["family"] = data["category"]
        for target, aliases in {
            "family": ("task_family", "group"),
            "category": ("task_category",),
            "disabled_tools": ("deny_tools", "forbidden_tools"),
        }.items():
            if target not in data:
                for alias in aliases:
                    if alias in data:
                        data[target] = data[alias]
                        break
        return data

    @model_validator(mode="after")
    def _require_prompt_or_turns(self) -> "EvalCase":
        if not self.prompt.strip() and not self.turns:
            raise ValueError("prompt or turns is required")
        return self


class LiveJSONLDataset(EvalModel):
    """In-memory representation of parsed JSONL cases."""

    schema_name: str = EvalCase.schema_name
    cases: list[EvalCase] = Field(default_factory=list)


def load_jsonl_dataset(path: str | Path) -> LiveJSONLDataset:
    """Read and validate one case per non-empty JSONL line.

    Validation errors include the source line number and Pydantic's field
    details.  The runner can therefore fail before touching a provider or
    creating a temporary workspace.
    """

    source = Path(path).expanduser()
    if not source.exists():
        raise DatasetValidationError(f"dataset does not exist: {source}")
    if not source.is_file():
        raise DatasetValidationError(f"dataset is not a file: {source}")

    cases: list[EvalCase] = []
    case_lines: dict[str, int] = {}
    errors: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append({"line": line_number, "error": f"invalid JSON: {exc.msg}"})
            continue
        if not isinstance(payload, dict):
            errors.append({"line": line_number, "error": "record must be a JSON object"})
            continue
        try:
            case = EvalCase.model_validate(payload)
            previous_line = case_lines.get(case.case_id)
            if previous_line is not None:
                errors.append({
                    "line": line_number,
                    "error": f"duplicate case_id {case.case_id!r}; first declared on line {previous_line}",
                })
                continue
            case_lines[case.case_id] = line_number
            cases.append(case)
        except ValidationError as exc:
            details = exc.errors(include_url=False)
            errors.append({"line": line_number, "errors": details})

    if errors:
        rendered = []
        for item in errors:
            line = item["line"]
            if "error" in item:
                rendered.append(f"line {line}: {item['error']}")
            else:
                rendered.append(f"line {line}: {json.dumps(item['errors'], ensure_ascii=False, default=str)}")
        raise DatasetValidationError("invalid eval dataset:\n" + "\n".join(rendered), line_errors=errors)
    if not cases:
        raise DatasetValidationError("dataset contains no cases")
    return LiveJSONLDataset(cases=cases)


def live_jsonl_schema() -> dict[str, Any]:
    """Return the JSON schema used for a single live JSONL record."""

    return EvalCase.model_json_schema()


# Public names used by a few small integrations.
LiveCase = EvalCase
EvalDataset = LiveJSONLDataset
load_dataset = load_jsonl_dataset


__all__ = [
    "DatasetValidationError",
    "EvalCase",
    "EvalDataset",
    "EvalExpectations",
    "FaultConfig",
    "EvalTurn",
    "FixtureSetup",
    "LiveJSONLDataset",
    "LiveCase",
    "RuleExpectation",
    "ToolCallExpectation",
    "live_jsonl_schema",
    "load_jsonl_dataset",
    "load_dataset",
]
