from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from myclaw.tools import ToolRegistry

Message = dict[str, Any]
CheckpointCallback = Callable[[dict[str, Any]], Awaitable[None]]
ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass(slots=True)
class AgentConfig:
    system_prompt: str = "You are a helpful personal assistant."
    model: str = ""
    max_turns: int = 4
    max_tool_result_chars: int = 16_000
    history: list[Message] = field(default_factory=list)


@dataclass(slots=True)
class RunResult:
    content: str
    messages: list[Message]
    model: str


@dataclass(slots=True)
class AgentRunSpec:
    messages: list[Message]
    model: str
    max_iterations: int = 4
    tools: ToolRegistry | None = None
    checkpoint_callback: CheckpointCallback | None = None
    progress_callback: ProgressCallback | None = None


@dataclass(slots=True)
class AgentRunResult:
    content: str
    messages: list[Message]
    stop_reason: str = "completed"
    error: str | None = None
