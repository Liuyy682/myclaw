from __future__ import annotations

from dataclasses import dataclass, field

from myclaw.providers.base import LLMProvider, Message


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


class Agent:
    """A minimal agent loop without tools or channel abstractions."""

    def __init__(self, provider: LLMProvider, config: AgentConfig | None = None) -> None:
        self.provider = provider
        self.config = config or AgentConfig()
        if self.config.max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self._messages: list[Message] = self._initial_messages(self.config)

    @property
    def messages(self) -> list[Message]:
        return [dict(message) for message in self._messages]

    async def run(self, text: str) -> RunResult:
        user_text = text.strip()
        if not user_text:
            raise ValueError("user input cannot be empty")

        self._messages.append({"role": "user", "content": user_text})

        try:
            content = await self.provider.complete(self.messages)
        except Exception as exc:
            content = f"Error: {exc}"

        assistant_text = content.strip() if content.strip() else "(empty response)"
        self._messages.append({"role": "assistant", "content": assistant_text})
        return RunResult(
            content=assistant_text,
            messages=self.messages,
            model=self.config.model or self.provider.model,
        )

    @staticmethod
    def _initial_messages(config: AgentConfig) -> list[Message]:
        messages: list[Message] = []
        if config.system_prompt.strip():
            messages.append({"role": "system", "content": config.system_prompt.strip()})
        messages.extend(dict(message) for message in config.history)
        return messages
