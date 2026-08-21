"""Deterministic rule scoring for Agent evaluation runs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from myclaw.evals.schema import EvalCase, EvalExpectations, RuleExpectation, ToolCallExpectation


@dataclass(slots=True)
class RuleResult:
    name: str
    passed: bool
    expected: Any = None
    actual: Any = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "expected": self.expected,
            "actual": self.actual,
            "detail": self.detail,
        }


@dataclass(slots=True)
class ScoreResult:
    passed: bool
    score: float
    passed_rules: int
    total_rules: int
    rules: list[RuleResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "passed_rules": self.passed_rules,
            "total_rules": self.total_rules,
            "rules": [rule.to_dict() for rule in self.rules],
        }


def score_case(case: EvalCase, run: dict[str, Any]) -> ScoreResult:
    """Apply all case rules to one serialisable run record.

    No model or fuzzy semantic judge is involved.  A run with a provider or
    execution error fails even if the case contains no explicit assertions.
    """

    expected = case.expected
    final = _final_text(run)
    trajectory = _trajectory(run)
    files = _files(run)
    results: list[RuleResult] = []

    if run.get("error"):
        results.append(
            RuleResult(
                "run_completed",
                False,
                expected="no error",
                actual=run.get("error"),
                detail="provider or agent execution failed",
            )
        )
    else:
        results.append(RuleResult("run_completed", True, expected="no error", actual=None))

    for index, fragment in enumerate(expected.final_contains, start=1):
        results.append(
            RuleResult(
                f"final_contains[{index}]",
                fragment in final,
                expected=fragment,
                actual=final,
                detail="fragment found" if fragment in final else "fragment not found",
            )
        )
    if expected.final_equals is not None:
        results.append(
            RuleResult(
                "final_equals",
                final == expected.final_equals,
                expected=expected.final_equals,
                actual=final,
            )
        )
    for index, fragment in enumerate(expected.final_not_contains, start=1):
        results.append(
            RuleResult(
                f"final_not_contains[{index}]",
                fragment not in final,
                expected=fragment,
                actual=final,
                detail="fragment absent" if fragment not in final else "forbidden fragment found",
            )
        )
    for index, pattern in enumerate(expected.final_regex, start=1):
        try:
            matched = re.search(pattern, final) is not None
            detail = "pattern matched" if matched else "pattern did not match"
        except re.error as exc:
            matched = False
            detail = f"invalid regex: {exc}"
        results.append(
            RuleResult(
                f"final_regex[{index}]",
                matched,
                expected=pattern,
                actual=final,
                detail=detail,
            )
        )

    tool_names = [str(item.get("name", "")) for item in trajectory if isinstance(item, dict)]
    if expected.tool_sequence:
        actual = tool_names
        passed = actual == expected.tool_sequence
        results.append(
            RuleResult(
                "tool_sequence",
                passed,
                expected=expected.tool_sequence,
                actual=actual,
                detail="tool order matched" if passed else "tool order or count differed",
            )
        )
    for index, call_expectation in enumerate(expected.tool_calls, start=1):
        match = _find_tool_call(trajectory, call_expectation, start=0)
        results.append(
            RuleResult(
                f"tool_call[{index}]",
                match is not None,
                expected=call_expectation.model_dump(exclude_none=True),
                actual=match,
                detail="matching call found" if match is not None else "matching call not found",
            )
        )
    if expected.min_tool_calls is not None:
        passed = len(trajectory) >= expected.min_tool_calls
        results.append(
            RuleResult(
                "min_tool_calls",
                passed,
                expected=expected.min_tool_calls,
                actual=len(trajectory),
            )
        )
    if expected.max_tool_calls is not None:
        passed = len(trajectory) <= expected.max_tool_calls
        results.append(
            RuleResult(
                "max_tool_calls",
                passed,
                expected=expected.max_tool_calls,
                actual=len(trajectory),
            )
        )
    if expected.no_tool_errors:
        errors = [item for item in trajectory if item.get("error") or not item.get("succeeded", True)]
        results.append(
            RuleResult(
                "no_tool_errors",
                not errors,
                expected=True,
                actual=errors,
                detail="all tool calls succeeded" if not errors else "tool error found",
            )
        )
    if expected.stop_reason is not None:
        actual_stop = run.get("stop_reason")
        results.append(
            RuleResult(
                "stop_reason",
                actual_stop == expected.stop_reason,
                expected=expected.stop_reason,
                actual=actual_stop,
            )
        )

    for path, content in expected.file_contains.items():
        actual = files.get(path)
        passed = actual is not None and content in actual
        results.append(
            RuleResult(
                f"file_contains[{path}]",
                passed,
                expected=content,
                actual=actual,
                detail="content found" if passed else "file or content missing",
            )
        )
    for path, content in expected.file_equals.items():
        actual = files.get(path)
        passed = actual == content
        results.append(
            RuleResult(
                f"file_equals[{path}]",
                passed,
                expected=content,
                actual=actual,
            )
        )
    for path in expected.file_exists:
        passed = path in files
        results.append(RuleResult(f"file_exists[{path}]", passed, expected=True, actual=passed))
    for path in expected.file_absent:
        passed = path not in files
        results.append(RuleResult(f"file_absent[{path}]", passed, expected=False, actual=path in files))

    _score_state_rules(results, expected.state, files)
    for index, rule in enumerate(expected.rules, start=1):
        results.append(_score_named_rule(f"rules[{index}]", rule, final, trajectory, files, run))

    passed_rules = sum(1 for result in results if result.passed)
    total_rules = len(results)
    score = passed_rules / total_rules if total_rules else 0.0
    return ScoreResult(
        passed=bool(results) and passed_rules == total_rules,
        score=score,
        passed_rules=passed_rules,
        total_rules=total_rules,
        rules=results,
    )


def _final_text(run: dict[str, Any]) -> str:
    value = run.get("final", run.get("content", ""))
    return value if isinstance(value, str) else str(value)


def _trajectory(run: dict[str, Any]) -> list[dict[str, Any]]:
    value = run.get("tool_trajectory", run.get("trajectory", []))
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _files(run: dict[str, Any]) -> dict[str, str]:
    state = run.get("final_state")
    if not isinstance(state, dict):
        state = {}
    value = state.get("files", run.get("files", {}))
    if not isinstance(value, dict):
        return {}
    return {str(path): content for path, content in value.items() if isinstance(content, str)}


def _find_tool_call(
    trajectory: list[dict[str, Any]], expectation: ToolCallExpectation, *, start: int
) -> dict[str, Any] | None:
    for item in trajectory[start:]:
        if item.get("name") != expectation.name:
            continue
        actual_arguments = item.get("arguments")
        if not isinstance(actual_arguments, dict):
            continue
        if expectation.exact_arguments:
            if actual_arguments != expectation.arguments:
                continue
        elif not _mapping_contains(actual_arguments, expectation.arguments):
            continue
        return item
    return None


def _mapping_contains(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(key in actual and _value_equal(actual[key], value) for key, value in expected.items())


def _value_equal(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return _mapping_contains(left, right)
    if isinstance(left, list) and isinstance(right, list):
        return left == right
    return left == right


def _score_state_rules(results: list[RuleResult], state: dict[str, Any], files: dict[str, str]) -> None:
    if not isinstance(state, dict):
        return
    expected_files = state.get("files")
    if expected_files is None:
        # A compact ``state: {"path.txt": "content"}`` spelling is also
        # useful for one-file smoke cases.
        expected_files = {
            key: value
            for key, value in state.items()
            if key not in {"session", "session_messages", "metadata"}
        }
    if isinstance(expected_files, dict):
        for path, expected in expected_files.items():
            path = str(path)
            actual = files.get(path)
            if isinstance(expected, dict):
                if "contains" in expected:
                    passed = actual is not None and str(expected["contains"]) in actual
                    detail = "content found" if passed else "file or content missing"
                elif "equals" in expected:
                    passed = actual == expected["equals"]
                    detail = "exact content matched" if passed else "exact content differed"
                elif "exists" in expected:
                    passed = (path in files) == bool(expected["exists"])
                    detail = "existence matched" if passed else "existence differed"
                else:
                    passed = actual == expected
                    detail = "exact content matched" if passed else "exact content differed"
            elif expected is None:
                passed = path not in files
                detail = "file absent" if passed else "file exists"
            else:
                passed = actual == expected
                detail = "exact content matched" if passed else "exact content differed"
            results.append(
                RuleResult(
                    f"state.files[{path}]",
                    passed,
                    expected=expected,
                    actual=actual,
                    detail=detail,
                )
            )


def _score_named_rule(
    name: str,
    rule: RuleExpectation,
    final: str,
    trajectory: list[dict[str, Any]],
    files: dict[str, str],
    run: dict[str, Any],
) -> RuleResult:
    kind = rule.type.strip().lower().replace("-", "_")
    value = rule.value
    if kind in {"final_contains", "output_contains", "answer_contains"}:
        expected = value if value is not None else rule.contains
        passed = isinstance(expected, str) and expected in final
        return RuleResult(name, passed, expected=expected, actual=final)
    if kind in {"final_equals", "output_equals", "answer_equals"}:
        expected = value if value is not None else rule.equals
        return RuleResult(name, final == expected, expected=expected, actual=final)
    if kind in {"tool_called", "tool_used"}:
        expected_name = rule.name or (value if isinstance(value, str) else "")
        passed = any(item.get("name") == expected_name for item in trajectory)
        return RuleResult(name, passed, expected=expected_name, actual=[item.get("name") for item in trajectory])
    if kind in {"tool_sequence", "tools_in_order"}:
        expected = value if isinstance(value, list) else []
        actual = [item.get("name") for item in trajectory]
        return RuleResult(name, actual == expected, expected=expected, actual=actual)
    if kind in {"file_contains", "fixture_contains"}:
        path = rule.path or ""
        expected = value if value is not None else rule.contains
        actual = files.get(path)
        return RuleResult(name, isinstance(expected, str) and actual is not None and expected in actual, expected=expected, actual=actual)
    if kind in {"file_equals", "fixture_equals"}:
        path = rule.path or ""
        expected = value if value is not None else rule.equals
        return RuleResult(name, files.get(path) == expected, expected=expected, actual=files.get(path))
    if kind in {"file_exists", "fixture_exists"}:
        path = rule.path or (value if isinstance(value, str) else "")
        return RuleResult(name, path in files, expected=True, actual=path in files)
    if kind == "file_absent":
        path = rule.path or (value if isinstance(value, str) else "")
        return RuleResult(name, path not in files, expected=False, actual=path in files)
    if kind in {"stop_reason", "run_stop_reason"}:
        return RuleResult(name, run.get("stop_reason") == value, expected=value, actual=run.get("stop_reason"))
    if kind in {"no_tool_errors", "tools_succeeded"}:
        errors = [item for item in trajectory if item.get("error") or not item.get("succeeded", True)]
        return RuleResult(name, not errors, expected=True, actual=errors)
    return RuleResult(name, False, expected=rule.model_dump(exclude_none=True), detail=f"unknown rule type: {rule.type}")


__all__ = ["RuleResult", "ScoreResult", "score_case"]
