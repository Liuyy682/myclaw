from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol


class Tool(Protocol):
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


@dataclass(slots=True)
class FunctionTool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]

    async def execute(self, **kwargs: Any) -> Any:
        result = self.func(**kwargs)
        if inspect.isawaitable(result):
            return await result
        return result
