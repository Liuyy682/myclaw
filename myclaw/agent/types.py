from __future__ import annotations

from dataclasses import dataclass, field

Message = dict[str, str]


@dataclass(slots=True)
class AgentConfig:
    system_prompt: str = "You are a helpful personal assistant."
    model: str = ""
    max_turns: int = 1
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
    max_iterations: int = 1


@dataclass(slots=True)
class AgentRunResult:
    content: str
    messages: list[Message]
    stop_reason: str = "completed"
    error: str | None = None
