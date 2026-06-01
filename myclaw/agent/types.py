from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from myclaw.config import (
    DEFAULT_AGENT_AUTO_TITLE,
    DEFAULT_AGENT_MAX_TURNS,
    DEFAULT_AGENT_MODEL,
    DEFAULT_AGENT_RUN_MAX_ITERATIONS,
    DEFAULT_MAX_TOOL_RESULT_CHARS,
    DEFAULT_SYSTEM_PROMPT,
)

if TYPE_CHECKING:
    from myclaw.tools import ToolRegistry

Message = dict[str, Any]
CheckpointCallback = Callable[[dict[str, Any]], Awaitable[None]]
ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]
StreamCallback = Callable[[str], Awaitable[None]]


@dataclass(slots=True)
class AgentConfig:
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    model: str = DEFAULT_AGENT_MODEL
    max_turns: int = DEFAULT_AGENT_MAX_TURNS
    max_tool_result_chars: int = DEFAULT_MAX_TOOL_RESULT_CHARS
    auto_title: bool = DEFAULT_AGENT_AUTO_TITLE
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
    max_iterations: int = DEFAULT_AGENT_RUN_MAX_ITERATIONS
    tools: ToolRegistry | None = None
    max_tool_result_chars: int | None = None
    checkpoint_callback: CheckpointCallback | None = None
    progress_callback: ProgressCallback | None = None
    stream_callback: StreamCallback | None = None


@dataclass(slots=True)
class AgentRunResult:
    content: str
    messages: list[Message]
    stop_reason: str = "completed"
    error: str | None = None
