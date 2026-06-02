from __future__ import annotations

import inspect
from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(slots=True)
class ToolRuntimeContext:
    session_key: str = ""
    channel: str = ""
    chat_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    workspace: Path | None = None
    tool_names: list[str] = field(default_factory=list)


_CURRENT_TOOL_CONTEXT: ContextVar[ToolRuntimeContext] = ContextVar(
    "myclaw_tool_runtime_context",
    default=ToolRuntimeContext(),
)


def get_current_tool_context() -> ToolRuntimeContext:
    return _CURRENT_TOOL_CONTEXT.get()


@contextmanager
def tool_context(context: ToolRuntimeContext):
    token = _CURRENT_TOOL_CONTEXT.set(context)
    try:
        yield
    finally:
        _CURRENT_TOOL_CONTEXT.reset(token)


class Tool(Protocol):
    read_only: bool
    exclusive: bool

    @property
    def name(self) -> str:
        """Tool name exposed to the model."""

    @property
    def description(self) -> str:
        """Tool description exposed to the model."""

    @property
    def parameters(self) -> dict[str, Any]:
        """OpenAI-compatible JSON schema for tool parameters."""

    async def execute(self, **kwargs: Any) -> Any:
        """Run the tool with model-supplied keyword arguments."""

    def cast_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Coerce model-supplied JSON arguments before validation."""

    def validate_params(self, params: dict[str, Any]) -> None:
        """Raise ValueError when arguments are invalid."""

    def to_schema(self) -> dict[str, Any]:
        """Return an OpenAI-compatible tool definition."""

    def set_context(self, context: ToolRuntimeContext) -> None:
        """Receive per-run context before execution."""


@dataclass(slots=True)
class FunctionTool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
    read_only: bool = False
    exclusive: bool = False
    context: ToolRuntimeContext | None = None

    def cast_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return dict(params)

    def validate_params(self, params: dict[str, Any]) -> None:
        return None

    def to_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def set_context(self, context: ToolRuntimeContext) -> None:
        self.context = context

    async def execute(self, **kwargs: Any) -> Any:
        result = self.func(**kwargs)
        if inspect.isawaitable(result):
            return await result
        return result
