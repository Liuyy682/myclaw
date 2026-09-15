"""End-to-end Agent evaluation harness.

Each repeat gets a fresh temporary workspace, a real MyClaw ``AgentLoop``,
the selected OpenAI-compatible provider, and a restricted fixture registry.
Results are intentionally plain JSON so they can be diffed or consumed by
CI without a database or an LLM judge.
"""

from __future__ import annotations

import argparse
import asyncio
import fnmatch
import inspect
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from myclaw.agent import AgentConfig, AgentLoop
from myclaw.agent.types import RunResult
from myclaw.config import (
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    OPENAI_API_KEY_ENV_VAR,
    OPENAI_BASE_URL_ENV_VAR,
    OPENAI_MODEL_ENV_VAR,
    load_env_file,
)
from myclaw.evals.fixtures import (
    build_fixture_registry,
    snapshot_fixture,
    snapshot_workspace,
    write_fixture_setup,
)
from myclaw.evals.schema import DatasetValidationError, EvalCase, load_jsonl_dataset
from myclaw.evals.scoring import ScoreResult, score_case
from myclaw.observability import ObservabilityConfig, ObservabilityRuntime
from myclaw.providers import FakeProvider, LLMResponse, OpenAICompatibleProvider
from myclaw.session import SessionManager


EXIT_OK = 0
EXIT_FAILED = 1
EXIT_INPUT_ERROR = 2


class EvalInputError(ValueError):
    """Raised for invalid harness options or an empty filter result."""


class ProviderConfigurationError(RuntimeError):
    """Raised when the requested live provider cannot be configured."""


@dataclass(slots=True)
class HarnessResult:
    output_dir: Path
    runs: list[dict[str, Any]]
    summary: dict[str, Any]

    @property
    def exit_code(self) -> int:
        return int(self.summary.get("exit_code", EXIT_FAILED))

    @property
    def report_path(self) -> Path:
        return self.output_dir / "report.md"

    @property
    def runs_path(self) -> Path:
        return self.output_dir / "runs.jsonl"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "summary.json"


class RecordingProvider:
    """Record provider call shape and normalized responses without secrets."""

    def __init__(self, provider: Any) -> None:
        self.provider = provider
        self.model = getattr(provider, "model", "")
        self.calls: list[dict[str, Any]] = []

    async def complete(self, messages: list[dict[str, Any]], *, tools=None) -> Any:
        call: dict[str, Any] = {
            "message_count": len(messages),
            "tool_names": _tool_names(tools),
        }
        try:
            response = await self.provider.complete(messages, tools=tools)
        except BaseException as exc:
            call["error"] = f"{type(exc).__name__}: {exc}"
            self.calls.append(call)
            raise
        call["response"] = _response_summary(response)
        self.calls.append(call)
        return response


ProviderFactory = Callable[..., Any]


def build_provider(profile: str) -> Any:
    """Build the selected provider from the existing MyClaw environment.

    ``fake`` is intentionally available for unit tests and offline smoke
    runs.  ``live`` never silently falls back to it: a missing key is a clear
    configuration error rather than a false live evaluation.
    """

    load_env_file()
    normalized = profile.strip().lower()
    if normalized in {"fake", "test", "offline"}:
        return FakeProvider()
    if normalized != "live":
        raise ProviderConfigurationError(f"unknown eval profile: {profile}")

    api_key = os.environ.get(OPENAI_API_KEY_ENV_VAR, "").strip()
    if not api_key or api_key in {"你的key", "your-key", "changeme", "change-me"}:
        raise ProviderConfigurationError(
            f"{OPENAI_API_KEY_ENV_VAR} is required for --profile live"
        )
    return OpenAICompatibleProvider(
        api_key=api_key,
        base_url=os.environ.get(OPENAI_BASE_URL_ENV_VAR, DEFAULT_OPENAI_BASE_URL),
        model=os.environ.get(OPENAI_MODEL_ENV_VAR, DEFAULT_OPENAI_MODEL),
    )


def run_evaluation(
    dataset: str | Path,
    *,
    profile: str = "live",
    repeat: int = 1,
    case_filters: Iterable[str] | None = None,
    tag_filters: Iterable[str] | None = None,
    output: str | Path = "eval-output",
    provider_factory: ProviderFactory | Any | None = None,
) -> HarnessResult:
    """Run filtered cases and write ``runs.jsonl``, ``summary.json`` and report."""

    if repeat < 1:
        raise EvalInputError("repeat must be at least 1")
    dataset_model = load_jsonl_dataset(dataset)
    selected = filter_cases(
        dataset_model.cases,
        case_filters=case_filters,
        tag_filters=tag_filters,
    )
    if not selected:
        raise EvalInputError("filters selected no evaluation cases")

    output_dir = Path(output).expanduser()
    if output_dir.exists() and not output_dir.is_dir():
        raise EvalInputError(f"output is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    factory = provider_factory or build_provider
    runs: list[dict[str, Any]] = []
    for case in selected:
        for repeat_index in range(1, repeat + 1):
            runs.append(
                asyncio.run(
                    _run_case(
                        case,
                        profile=profile,
                        repeat_index=repeat_index,
                        provider_factory=factory,
                    )
                )
            )

    summary = build_summary(
        runs,
        dataset=Path(dataset).expanduser(),
        profile=profile,
        repeat=repeat,
        case_filters=list(case_filters or []),
        tag_filters=list(tag_filters or []),
    )
    write_outputs(output_dir, runs, summary)
    return HarnessResult(output_dir=output_dir, runs=runs, summary=summary)


async def _run_case(
    case: EvalCase,
    *,
    profile: str,
    repeat_index: int,
    provider_factory: ProviderFactory | Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    session_key = f"eval:{case.case_id}:{repeat_index}"
    progress_events: list[dict[str, Any]] = []
    provider_calls: list[dict[str, Any]] = []
    error: str | None = None
    result: Any = None
    fixture_files: dict[str, str] = {}
    workspace_files: dict[str, str] = {}
    session_messages: list[dict[str, Any]] = []
    ask_events: list[dict[str, Any]] = []
    turn_errors: list[dict[str, Any]] = []
    fault_recovered = False
    allowed_tools: list[str] = []
    registry: Any = None

    with tempfile.TemporaryDirectory(prefix="myclaw-eval-") as temporary:
        workspace = Path(temporary) / "workspace"
        fixture_root = workspace / "fixture"
        session_manager = SessionManager(workspace)
        write_fixture_setup(fixture_root, case.setup.files, case.setup.directories)
        fixture_files = snapshot_fixture(fixture_root)
        try:
            provider = await _maybe_await(
                _invoke_provider_factory(
                    provider_factory,
                    case=case,
                    profile=profile,
                    repeat_index=repeat_index,
                    workspace=workspace,
                    fixture_root=fixture_root,
                )
            )
            recorder = RecordingProvider(provider)
            registry = build_fixture_registry(
                fixture_root,
                workspace=workspace,
                disabled_tools=case.disabled_tools,
                cancel_on_tool_call=case.fault.cancel_on_tool_call,
                fault_phase=case.fault.phase,
            )
            allowed_tools = registry.tool_names
            loop = AgentLoop(
                recorder,
                AgentConfig(
                    system_prompt=case.system_prompt or DEFAULT_SYSTEM_PROMPT,
                    model=getattr(provider, "model", ""),
                    max_turns=case.max_turns,
                    max_context_messages=case.max_context_messages,
                    max_context_tokens=case.max_context_tokens,
                    auto_title=False,
                    idle_compact_after_minutes=0,
                    dream_interval_minutes=0,
                ),
                session_manager=session_manager,
                tool_registry=registry,
                observability=ObservabilityRuntime(
                    workspace,
                    ObservabilityConfig(enabled=False),
                ),
            )
            async def on_progress(event: dict[str, Any]) -> None:
                progress_events.append(_json_safe(dict(event)))

            async def on_ask(question: str, choices: list[str]) -> str:
                answer_index = len(ask_events)
                answer = (
                    case.ask_answers[answer_index]
                    if answer_index < len(case.ask_answers)
                    else ""
                )
                ask_events.append({
                    "question": question,
                    "choices": list(choices),
                    "answer": answer,
                })
                return answer

            turns = case.turns or [{"role": "user", "content": case.prompt}]
            for turn_index, turn in enumerate(turns, start=1):
                if isinstance(turn, dict):
                    role = turn.get("role", "user")
                    content = turn.get("content", "")
                else:
                    role = getattr(turn, "role", "user")
                    content = getattr(turn, "content", "")
                if role == "user":
                    try:
                        # Replay explicit user mode commands through the same
                        # controller used by the dispatcher, never as model text.
                        action = content.strip().split(maxsplit=1)[0].lower() if content.strip() else ""
                        if action in {"/plan", "/plans", "/execute", "/exit-plan"}:
                            command_result = loop.plan_command(session_key, content)
                            if command_result.run_prompt is None:
                                result = RunResult(
                                    content=command_result.content, messages=[], model=getattr(provider, "model", ""),
                                )
                                continue
                            content = command_result.run_prompt
                        result = await loop.run(
                            content,
                            session_key=session_key,
                            channel="eval",
                            chat_id=case.case_id,
                            metadata={
                                "eval_case_id": case.case_id,
                                "eval_repeat": repeat_index,
                                "eval_profile": profile,
                            },
                            progress_callback=on_progress,
                            ask_callback=on_ask,
                        )
                        if (
                            registry.injected
                            and turn_errors
                            and not getattr(result, "error", None)
                        ):
                            fault_recovered = True
                    except asyncio.CancelledError as exc:
                        if registry.injected:
                            turn_errors.append({
                                "turn": turn_index,
                                "error": f"CancelledError: {exc}",
                            })
                            continue
                        raise
                else:
                    # Role/content turns are useful for recovery/context
                    # fixtures.  They seed persisted history; only user turns
                    # invoke the real AgentLoop.
                    session = session_manager.get_or_create(session_key)
                    session.add_message(role, content)
                    session_manager.save(session)
            session_messages = [
                _json_safe(dict(message))
                for message in session_manager.get_or_create(session_key).messages
            ]
            provider_calls = recorder.calls
            workspace_files = snapshot_workspace(workspace)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if "recorder" in locals():
                provider_calls = recorder.calls
            if registry is not None and registry.injected:
                turn_errors.append({"error": error})
        session_messages = [
            _json_safe(dict(message))
            for message in session_manager.get_or_create(session_key).messages
        ]
        if registry is not None and registry.injected and not fault_recovered and error is None:
            error = "CancelledError: eval fault had no successful next user turn"
        fixture_files = snapshot_fixture(fixture_root)
        workspace_files = snapshot_workspace(workspace)

    run: dict[str, Any] = {
        "case_id": case.case_id,
        "family": case.family,
        "category": case.category,
        "capability": case.capability,
        "claim": case.claim,
        "metric_type": case.metric_type,
        "pytest_node_id": case.pytest_node_id,
        "tags": list(case.tags),
        "repeat_index": repeat_index,
        "profile": profile,
        "execution_scope": "fixture_agent_loop",
        "recovery_mode": (
            "checkpoint_turn_recovery"
            if fault_recovered
            else "fixture_state_only"
        ),
        "final": _result_content(result),
        "stop_reason": getattr(result, "stop_reason", "error" if error else "completed"),
        "error": error or getattr(result, "error", None),
        "messages": _json_safe(getattr(result, "messages", [])),
        "provider_calls": _json_safe(provider_calls),
        "usage": _aggregate_usage(provider_calls),
        "ask_events": _json_safe(ask_events),
        "turn_errors": _json_safe(turn_errors),
        "fault_injection": {
            "cancel_on_tool_call": case.fault.cancel_on_tool_call,
            "phase": case.fault.phase,
            "injected": bool(registry is not None and registry.injected),
            "injected_call": registry.injected_call if registry is not None else None,
            "recovered_on_next_user_turn": fault_recovered,
        },
        "progress_events": progress_events,
        "tool_trajectory": _tool_trajectory(getattr(result, "messages", []), progress_events),
        "turn_count": len(case.turns or [case.prompt]),
        "allowed_tools": allowed_tools,
        "disabled_tools": list(case.disabled_tools),
        "final_state": {
            "files": {**workspace_files, **fixture_files},
            "workspace_files": workspace_files,
            "session_messages": session_messages,
        },
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    score = score_case(case, run)
    run["score"] = score.to_dict()
    run["passed"] = score.passed
    return _json_safe(run)


def filter_cases(
    cases: Sequence[EvalCase],
    *,
    case_filters: Iterable[str] | None = None,
    tag_filters: Iterable[str] | None = None,
) -> list[EvalCase]:
    """Filter by exact/glob case IDs and require all requested tags."""

    case_patterns = _split_filters(case_filters)
    tags = set(_split_filters(tag_filters))
    selected: list[EvalCase] = []
    for case in cases:
        if case_patterns and not any(fnmatch.fnmatchcase(case.case_id, pattern) for pattern in case_patterns):
            continue
        if tags and not tags.issubset(set(case.tags)):
            continue
        selected.append(case)
    return selected


def build_summary(
    runs: Sequence[dict[str, Any]],
    *,
    dataset: Path,
    profile: str,
    repeat: int,
    case_filters: Sequence[str],
    tag_filters: Sequence[str],
) -> dict[str, Any]:
    passed = sum(1 for run in runs if run.get("passed"))
    failed = len(runs) - passed
    errors = sum(1 for run in runs if run.get("error"))
    scores = [float(run.get("score", {}).get("score", 0.0)) for run in runs]
    tool_call_count = sum(len(run.get("tool_trajectory", [])) for run in runs)
    disabled_call_count = sum(
        sum(
            1
            for item in run.get("tool_trajectory", [])
            if item.get("name") not in set(run.get("allowed_tools", []))
        )
        for run in runs
    )
    all_successful_durations = [
        float(run.get("duration_ms", 0.0)) for run in runs if run.get("passed")
    ]
    by_case: dict[str, dict[str, Any]] = {}
    by_family: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        entry = by_case.setdefault(
            str(run.get("case_id")),
            {"runs": 0, "passed": 0, "failed": 0, "scores": []},
        )
        entry["runs"] += 1
        entry["passed"] += int(bool(run.get("passed")))
        entry["failed"] += int(not bool(run.get("passed")))
        entry["scores"].append(float(run.get("score", {}).get("score", 0.0)))
        by_family.setdefault(str(run.get("family", "default")), []).append(run)
    for entry in by_case.values():
        entry["average_score"] = round(sum(entry["scores"]) / len(entry["scores"]), 6)

    ordered_cases = [
        sorted(case_runs, key=lambda item: int(item.get("repeat_index", 0)))
        for case_runs in (
            [run for run in runs if str(run.get("case_id")) == case_id]
            for case_id in sorted(by_case)
        )
    ]
    pass_at_1_count = sum(bool(case_runs and case_runs[0].get("passed")) for case_runs in ordered_cases)
    pass_at_3_count = sum(any(run.get("passed") for run in case_runs[:3]) for case_runs in ordered_cases)
    strict_pass_at_3_count = sum(
        len(case_runs[:3]) == 3 and all(run.get("passed") for run in case_runs[:3])
        for case_runs in ordered_cases
    )
    total_case_count = len(ordered_cases)
    pass_at_1_interval = _wilson_interval(pass_at_1_count, total_case_count)

    family_summary: dict[str, dict[str, Any]] = {}
    for family, family_runs in by_family.items():
        cases: dict[str, list[dict[str, Any]]] = {}
        for run in family_runs:
            cases.setdefault(str(run.get("case_id")), []).append(run)
        first_passes = 0
        any_three_passes = 0
        strict_three_passes = 0
        for case_runs in cases.values():
            ordered = sorted(case_runs, key=lambda item: int(item.get("repeat_index", 0)))
            first_passes += int(bool(ordered and ordered[0].get("passed")))
            first_three = ordered[:3]
            any_three_passes += int(any(run.get("passed") for run in first_three))
            strict_three_passes += int(len(first_three) >= 3 and all(run.get("passed") for run in first_three))
        family_case_count = len(cases)
        successful = sum(1 for run in family_runs if run.get("passed"))
        tool_calls = sum(len(run.get("tool_trajectory", [])) for run in family_runs)
        disabled_calls = sum(
            sum(
                1
                for item in run.get("tool_trajectory", [])
                if item.get("name") not in set(run.get("allowed_tools", []))
            )
            for run in family_runs
        )
        successful_durations = [
            float(run.get("duration_ms", 0.0)) for run in family_runs if run.get("passed")
        ]
        family_summary[family] = {
            "cases": family_case_count,
            "runs": len(family_runs),
            "task_success": round(successful / len(family_runs), 6) if family_runs else 0.0,
            "pass_at_1": round(first_passes / family_case_count, 6) if family_case_count else 0.0,
            "pass_at_3": round(any_three_passes / family_case_count, 6) if family_case_count else 0.0,
            "strict_pass_at_3": round(strict_three_passes / family_case_count, 6) if family_case_count else 0.0,
            "pass@1": round(first_passes / family_case_count, 6) if family_case_count else 0.0,
            "pass@3": round(any_three_passes / family_case_count, 6) if family_case_count else 0.0,
            "strict_pass@3": round(strict_three_passes / family_case_count, 6) if family_case_count else 0.0,
            "tool_calls": tool_calls,
            "disabled_calls": disabled_calls,
            "tool_call_count": tool_calls,
            "disabled_call_count": disabled_calls,
            "successful_duration_ms": round(
                sum(successful_durations) / len(successful_durations), 3
            ) if successful_durations else None,
            "success_duration_ms": round(
                sum(successful_durations) / len(successful_durations), 3
            ) if successful_durations else None,
        }

    family_rates = [metrics["task_success"] for metrics in family_summary.values()]
    trajectories = [run.get("tool_trajectory", []) for run in runs]
    runs_with_tools = [items for items in trajectories if items]
    first_successes = sum(bool(items[0].get("succeeded")) for items in runs_with_tools)
    eventual_successes = sum(any(item.get("succeeded") for item in items) for items in runs_with_tools)
    correction_candidates = [items for items in runs_with_tools if not items[0].get("succeeded")]
    corrected = sum(any(item.get("succeeded") for item in items[1:]) for items in correction_candidates)
    usage = {
        name: sum(int((run.get("usage") or {}).get(name) or 0) for run in runs)
        for name in ("prompt_tokens", "completion_tokens", "total_tokens")
    }

    return {
        "schema": "myclaw.eval.summary.v1",
        "recovery_validation": (
            "checkpoint_turn_recovery_with_one_shot_cancel"
            if any(run.get("fault_injection", {}).get("injected") for run in runs)
            else "fixture_state_only_no_fault_injection"
        ),
        "dataset": str(dataset),
        "profile": profile,
        "repeat": repeat,
        "case_filters": list(case_filters),
        "tag_filters": list(tag_filters),
        "total_runs": len(runs),
        "passed_runs": passed,
        "failed_runs": failed,
        "error_runs": errors,
        "average_score": round(sum(scores) / len(scores), 6) if scores else 0.0,
        "task_success": round(passed / len(runs), 6) if runs else 0.0,
        "macro_family_success": round(sum(family_rates) / len(family_rates), 6) if family_rates else 0.0,
        "pass_at_1": round(pass_at_1_count / total_case_count, 6) if total_case_count else 0.0,
        "pass_at_3": round(pass_at_3_count / total_case_count, 6) if total_case_count else 0.0,
        "strict_pass_at_3": round(strict_pass_at_3_count / total_case_count, 6) if total_case_count else 0.0,
        "pass_at_1_wilson_95": pass_at_1_interval,
        "tool_calls": tool_call_count,
        "disabled_calls": disabled_call_count,
        "tool_call_count": tool_call_count,
        "disabled_call_count": disabled_call_count,
        "first_tool_success_rate": round(first_successes / len(runs_with_tools), 6) if runs_with_tools else None,
        "eventual_tool_success_rate": round(eventual_successes / len(runs_with_tools), 6) if runs_with_tools else None,
        "tool_error_correction_rate": round(corrected / len(correction_candidates), 6) if correction_candidates else None,
        "successful_duration_ms": {
            "p50": _percentile(all_successful_durations, 50),
            "p95": _percentile(all_successful_durations, 95),
        },
        "usage": usage if any(usage.values()) else None,
        "by_case": by_case,
        "by_family": family_summary,
        "exit_code": EXIT_OK if failed == 0 else EXIT_FAILED,
    }


def write_outputs(output_dir: Path, runs: Sequence[dict[str, Any]], summary: dict[str, Any]) -> None:
    """Write the stable machine- and human-readable artifacts."""

    with (output_dir / "runs.jsonl").open("w", encoding="utf-8") as handle:
        for run in runs:
            handle.write(json.dumps(_json_safe(run), ensure_ascii=False, sort_keys=True) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(_json_safe(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        render_report(runs, summary),
        encoding="utf-8",
    )


def render_report(runs: Sequence[dict[str, Any]], summary: dict[str, Any]) -> str:
    """Render a compact report with failed rule details."""

    lines = [
        "# MyClaw Agent Eval Report",
        "",
        f"- Profile: `{summary.get('profile', '')}`",
        f"- Dataset: `{summary.get('dataset', '')}`",
        f"- Runs: **{summary.get('passed_runs', 0)} passed / {summary.get('total_runs', 0)} total**",
        f"- Average score: **{summary.get('average_score', 0.0):.3f}**",
        f"- Task success: **{summary.get('task_success', 0.0):.3f}**; "
        f"tool calls: **{summary.get('tool_calls', 0)}**; disabled calls: **{summary.get('disabled_calls', 0)}**",
        f"- Recovery validation: `{summary.get('recovery_validation', '')}`.",
        "",
    ]
    family_summary = summary.get("by_family", {})
    if isinstance(family_summary, dict) and family_summary:
        lines.extend(
            [
                "",
                "## Family metrics",
                "",
                "| Family | Task success | Pass@1 | Pass@3 | Strict pass@3 | Tool calls | Disabled calls | Successful ms |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for family, metrics in sorted(family_summary.items()):
            lines.append(
                f"| `{family}` | {metrics.get('task_success', 0):.3f} | "
                f"{metrics.get('pass_at_1', 0):.3f} | {metrics.get('pass_at_3', 0):.3f} | "
                f"{metrics.get('strict_pass_at_3', 0):.3f} | {metrics.get('tool_calls', 0)} | "
                f"{metrics.get('disabled_calls', 0)} | {metrics.get('successful_duration_ms') or '-'} |"
            )
    lines.extend(["", "## Cases", "", "| Case | Repeat | Status | Score | Tools | Final |", "| --- | ---: | --- | ---: | --- | --- |"])
    for run in runs:
        status = "PASS" if run.get("passed") else "FAIL"
        score = float(run.get("score", {}).get("score", 0.0))
        tools = ", ".join(str(item.get("name", "")) for item in run.get("tool_trajectory", [])) or "-"
        final = _single_line(run.get("final", ""), limit=100)
        lines.append(
            f"| `{run.get('case_id', '')}` | {run.get('repeat_index', '')} | **{status}** "
            f"| {score:.3f} | `{_single_line(tools, limit=60)}` | {_single_line(final, limit=100)} |"
        )

    failures = [run for run in runs if not run.get("passed")]
    if failures:
        lines.extend(["", "## Failed rules", ""])
        for run in failures:
            lines.extend(
                [
                    f"### `{run.get('case_id', '')}` repeat {run.get('repeat_index', '')}",
                    "",
                ]
            )
            if run.get("error"):
                lines.append(f"- Execution error: `{_single_line(run['error'], limit=240)}`")
            for rule in run.get("score", {}).get("rules", []):
                if not rule.get("passed"):
                    lines.append(
                        f"- `{rule.get('name', '')}`: {rule.get('detail') or 'rule failed'} "
                        f"(expected `{_single_line(rule.get('expected'), limit=120)}`, "
                        f"actual `{_single_line(rule.get('actual'), limit=120)}`)"
                    )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run rule-scored MyClaw Agent evaluations.")
    parser.add_argument("--dataset", required=True, help="Live JSONL dataset path.")
    parser.add_argument("--profile", default="live", help="Provider profile (live or fake).")
    parser.add_argument("--repeat", type=int, default=1, help="Runs per selected case.")
    parser.add_argument("--case", dest="case_filters", action="append", help="Case ID or glob filter.")
    parser.add_argument("--tag", dest="tag_filters", action="append", help="Required case tag; repeatable.")
    parser.add_argument("--output", default="eval-output", help="Output directory for eval artifacts.")
    args = parser.parse_args(argv)
    try:
        result = run_evaluation(
            args.dataset,
            profile=args.profile,
            repeat=args.repeat,
            case_filters=args.case_filters,
            tag_filters=args.tag_filters,
            output=args.output,
        )
    except (DatasetValidationError, EvalInputError, ProviderConfigurationError, OSError) as exc:
        print(f"eval error: {exc}", file=os.sys.stderr)
        return EXIT_INPUT_ERROR if not isinstance(exc, ProviderConfigurationError) else EXIT_FAILED
    print(
        f"eval: {result.summary['passed_runs']}/{result.summary['total_runs']} passed; "
        f"artifacts written to {result.output_dir}"
    )
    return result.exit_code


def _invoke_provider_factory(factory: ProviderFactory | Any, **kwargs: Any) -> Any:
    if not callable(factory):
        return factory
    # Keep the public factory ergonomic for tests: accept a full keyword-rich
    # factory, or a one/two-argument lambda without swallowing errors raised by
    # its implementation.
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory()
    parameters = signature.parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return factory(**kwargs)
    selected = {
        name: value
        for name, value in kwargs.items()
        if name in parameters
    }
    positional = [
        parameter
        for parameter in parameters.values()
        if parameter.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
    ]
    if not selected and positional:
        # The common test form is ``lambda case: ...``.
        return factory(kwargs.get(positional[0].name))
    return factory(**selected)


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _tool_names(tools: Any) -> list[str]:
    names: list[str] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.append(function["name"])
    return names


def _response_summary(response: Any) -> dict[str, Any]:
    if isinstance(response, LLMResponse):
        return {
            "type": "llm_response",
            "content": response.content,
            "final": response.final,
            "stop_reason": response.stop_reason,
            "tool_calls": [
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": _json_safe(call.arguments),
                }
                for call in response.tool_calls
            ],
            "usage": (
                {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                if response.usage is not None
                else None
            ),
        }
    return {"type": "text", "content": str(response)}


def _aggregate_usage(provider_calls: Sequence[dict[str, Any]]) -> dict[str, int] | None:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    available = False
    for call in provider_calls:
        response = call.get("response") if isinstance(call, dict) else None
        usage = response.get("usage") if isinstance(response, dict) else None
        if not isinstance(usage, dict):
            continue
        available = True
        for name in totals:
            value = usage.get(name)
            if isinstance(value, int):
                totals[name] += value
    return totals if available else None


def _percentile(values: Sequence[float], percentile: int) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] * (1 - fraction) + ordered[upper] * fraction, 3)


def _wilson_interval(successes: int, total: int) -> dict[str, float] | None:
    if total < 1:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * (
        (proportion * (1 - proportion) / total + z * z / (4 * total * total)) ** 0.5
    ) / denominator
    return {"low": round(max(0.0, centre - margin), 6), "high": round(min(1.0, centre + margin), 6)}


def _tool_trajectory(messages: Any, progress_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return []
    calls: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            for raw_call in message.get("tool_calls") or []:
                if not isinstance(raw_call, dict):
                    continue
                call_id = str(raw_call.get("id") or f"call_{len(order) + 1}")
                function = raw_call.get("function")
                if not isinstance(function, dict):
                    function = {}
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"_raw": arguments}
                item = {
                    "id": call_id,
                    "name": str(function.get("name") or raw_call.get("name") or ""),
                    "arguments": _json_safe(arguments if isinstance(arguments, dict) else {"_raw": arguments}),
                    "result": None,
                    "succeeded": None,
                }
                calls[call_id] = item
                order.append(call_id)
        elif message.get("role") == "tool":
            call_id = str(message.get("tool_call_id") or "")
            item = calls.get(call_id)
            if item is None:
                continue
            result = message.get("content", "")
            item["result"] = result
            item["succeeded"] = not (isinstance(result, str) and result.startswith("Error"))
            if not item["succeeded"]:
                item["error"] = result

    started_ids = {
        str(event.get("tool_call_id"))
        for event in progress_events
        if event.get("event") == "tool_started" and event.get("tool_call_id")
    }
    # Include progress timing/ordering without replacing the message-derived
    # result.  This preserves calls even if an interrupted run has no tool
    # message yet.
    for event in progress_events:
        if event.get("event") != "tool_started":
            continue
        call_id = str(event.get("tool_call_id") or f"progress_{len(order) + 1}")
        if call_id in calls:
            continue
        calls[call_id] = {
            "id": call_id,
            "name": event.get("tool_name", ""),
            "arguments": _json_safe(event.get("arguments", {})),
            "result": None,
            "succeeded": False,
            "error": "tool did not complete",
        }
        order.append(call_id)
    selected = [calls[call_id] for call_id in order]
    # A multi-turn AgentLoop returns context plus newly generated messages on
    # every turn.  Progress IDs identify calls belonging to this evaluation;
    # discard historical tool messages from later turns.
    if started_ids:
        selected = [item for item in selected if str(item.get("id")) in started_ids]
    return selected


def _result_content(result: Any) -> str:
    content = getattr(result, "content", "")
    return content if isinstance(content, str) else str(content or "")


def _split_filters(values: Iterable[str] | None) -> list[str]:
    output: list[str] = []
    for value in values or []:
        output.extend(part.strip() for part in str(value).split(",") if part.strip())
    return output


def _single_line(value: Any, *, limit: int) -> str:
    text = str(value).replace("\n", " ").replace("|", "\\|")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return str(value)


__all__ = [
    "EXIT_FAILED",
    "EXIT_INPUT_ERROR",
    "EXIT_OK",
    "EvalInputError",
    "HarnessResult",
    "ProviderConfigurationError",
    "RecordingProvider",
    "build_provider",
    "build_summary",
    "filter_cases",
    "main",
    "render_report",
    "run_evaluation",
    "write_outputs",
]
