import asyncio

from myclaw import AgentRunSpec, AgentRunner, FakeProvider, LLMResponse


def test_runner_returns_assistant_message_for_single_model_call():
    runner = AgentRunner(FakeProvider(prefix="Echo"))
    spec = AgentRunSpec(
        messages=[{"role": "user", "content": "hello"}],
        model="fake",
        max_iterations=1,
    )

    result = asyncio.run(runner.run(spec))

    assert result.content == "Echo: hello"
    assert result.stop_reason == "completed"
    assert result.error is None
    assert result.messages == [
        {"role": "assistant", "content": "Echo: hello"},
    ]


class MultiStepProvider:
    model = "multi"

    def __init__(self):
        self.calls = []
        self.responses = [
            LLMResponse(content="thinking", final=False, stop_reason="continue"),
            LLMResponse(content="done", final=True),
        ]

    async def complete(self, messages):
        self.calls.append([dict(message) for message in messages])
        return self.responses[len(self.calls) - 1]


def test_runner_continues_until_provider_returns_final_response():
    provider = MultiStepProvider()
    runner = AgentRunner(provider)

    result = asyncio.run(runner.run(AgentRunSpec(
        messages=[{"role": "user", "content": "solve"}],
        model="multi",
        max_iterations=3,
    )))

    assert result.content == "done"
    assert result.stop_reason == "completed"
    assert result.messages == [
        {"role": "assistant", "content": "thinking"},
        {"role": "assistant", "content": "done"},
    ]
    assert len(provider.calls) == 2
    assert provider.calls[1] == [
        {"role": "user", "content": "solve"},
        {"role": "assistant", "content": "thinking"},
    ]


class NeverFinalProvider:
    model = "loop"

    def __init__(self):
        self.calls = 0

    async def complete(self, messages):
        self.calls += 1
        return LLMResponse(content=f"step {self.calls}", final=False, stop_reason="continue")


def test_runner_stops_at_max_iterations():
    provider = NeverFinalProvider()
    runner = AgentRunner(provider)

    result = asyncio.run(runner.run(AgentRunSpec(
        messages=[{"role": "user", "content": "solve"}],
        model="loop",
        max_iterations=2,
    )))

    assert provider.calls == 2
    assert result.content == "step 2"
    assert result.stop_reason == "max_iterations"
    assert result.messages == [
        {"role": "assistant", "content": "step 1"},
        {"role": "assistant", "content": "step 2"},
    ]


class FailingProvider:
    model = "broken"

    async def complete(self, messages):
        raise RuntimeError("provider unavailable")


def test_runner_returns_error_state_when_provider_fails():
    runner = AgentRunner(FailingProvider())
    spec = AgentRunSpec(
        messages=[{"role": "user", "content": "hello"}],
        model="broken",
        max_iterations=1,
    )

    result = asyncio.run(runner.run(spec))

    assert result.content == "Error: provider unavailable"
    assert result.stop_reason == "error"
    assert result.error == "provider unavailable"
    assert result.messages == [
        {"role": "assistant", "content": "Error: provider unavailable"},
    ]


def test_runner_does_not_persist_history_between_calls():
    runner = AgentRunner(FakeProvider(prefix="Echo"))

    first = asyncio.run(runner.run(AgentRunSpec(
        messages=[{"role": "user", "content": "first"}],
        model="fake",
        max_iterations=1,
    )))
    second = asyncio.run(runner.run(AgentRunSpec(
        messages=[{"role": "user", "content": "second"}],
        model="fake",
        max_iterations=1,
    )))

    assert first.messages == [{"role": "assistant", "content": "Echo: first"}]
    assert second.messages == [{"role": "assistant", "content": "Echo: second"}]
