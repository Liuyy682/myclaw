"""Local deterministic Agent evaluation harness."""

from myclaw.evals.harness import (
    EXIT_FAILED,
    EXIT_INPUT_ERROR,
    EXIT_OK,
    EvalInputError,
    HarnessResult,
    ProviderConfigurationError,
    build_provider,
    filter_cases,
    main,
    render_report,
    run_evaluation,
)
from myclaw.evals.schema import (
    DatasetValidationError,
    EvalCase,
    EvalExpectations,
    EvalTurn,
    FaultConfig,
    FixtureSetup,
    LiveJSONLDataset,
    RuleExpectation,
    ToolCallExpectation,
    live_jsonl_schema,
    load_jsonl_dataset,
)
from myclaw.evals.scoring import RuleResult, ScoreResult, score_case

# Small compatibility aliases for callers that use the shorter harness
# vocabulary.
load_dataset = load_jsonl_dataset
score_run = score_case

__all__ = [
    "DatasetValidationError",
    "EXIT_FAILED",
    "EXIT_INPUT_ERROR",
    "EXIT_OK",
    "EvalCase",
    "EvalExpectations",
    "EvalInputError",
    "EvalTurn",
    "FaultConfig",
    "FixtureSetup",
    "HarnessResult",
    "LiveJSONLDataset",
    "ProviderConfigurationError",
    "RuleExpectation",
    "RuleResult",
    "ScoreResult",
    "ToolCallExpectation",
    "build_provider",
    "filter_cases",
    "live_jsonl_schema",
    "load_jsonl_dataset",
    "load_dataset",
    "main",
    "render_report",
    "run_evaluation",
    "score_case",
    "score_run",
]
