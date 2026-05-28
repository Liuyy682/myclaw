import asyncio

from myclaw import AgentRunSpec, AgentRunner, FakeProvider


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
