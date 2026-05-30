import asyncio

from myclaw import AgentRunSpec, AgentRunner, FakeProvider, FunctionTool, LLMResponse, ToolCallRequest, ToolRegistry


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

    async def complete(self, messages, *, tools=None):
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

    async def complete(self, messages, *, tools=None):
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

    async def complete(self, messages, *, tools=None):
        raise RuntimeError("provider unavailable")


class ToolCallingProvider:
    model = "tools"

    def __init__(self):
        self.calls = []

    async def complete(self, messages, *, tools=None):
        self.calls.append({"messages": [dict(message) for message in messages], "tools": tools})
        if len(self.calls) == 1:
            return LLMResponse(
                content="",
                final=False,
                stop_reason="tool_calls",
                tool_calls=[ToolCallRequest(id="call_add", name="add", arguments={"a": 2, "b": 3})],
            )
        return LLMResponse(content=f"sum is {messages[-1]['content']}", final=True)


def test_runner_executes_tool_call_and_sends_tool_result_to_next_model_call():
    registry = ToolRegistry()
    registry.register(
        FunctionTool(
            "add",
            "Add two numbers",
            {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
            lambda a, b: a + b,
        )
    )
    provider = ToolCallingProvider()
    runner = AgentRunner(provider)

    result = asyncio.run(runner.run(AgentRunSpec(
        messages=[{"role": "user", "content": "add 2 and 3"}],
        model="tools",
        max_iterations=3,
        tools=registry,
    )))

    assert result.content == "sum is 5"
    assert result.stop_reason == "completed"
    assert result.messages == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_add",
                    "type": "function",
                    "function": {
                        "name": "add",
                        "arguments": '{"a": 2, "b": 3}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_add", "name": "add", "content": "5"},
        {"role": "assistant", "content": "sum is 5"},
    ]
    assert provider.calls[0]["tools"] == registry.definitions()
    assert provider.calls[1]["messages"] == [
        {"role": "user", "content": "add 2 and 3"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_add",
                    "type": "function",
                    "function": {
                        "name": "add",
                        "arguments": '{"a": 2, "b": 3}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_add", "name": "add", "content": "5"},
    ]


class MultipleToolCallProvider:
    model = "tools"

    def __init__(self):
        self.calls = []

    async def complete(self, messages, *, tools=None):
        self.calls.append([dict(message) for message in messages])
        if len(self.calls) == 1:
            return LLMResponse(
                content="",
                final=False,
                stop_reason="tool_calls",
                tool_calls=[
                    ToolCallRequest(id="call_add", name="add", arguments={"a": 1, "b": 2}),
                    ToolCallRequest(id="call_double", name="double", arguments={"value": 4}),
                ],
            )
        tool_results = [message["content"] for message in messages if message["role"] == "tool"]
        return LLMResponse(content=", ".join(tool_results), final=True)


def test_runner_executes_multiple_tool_calls_before_follow_up_model_call():
    registry = ToolRegistry()
    registry.register(FunctionTool("add", "Add", {"type": "object"}, lambda a, b: a + b))
    registry.register(FunctionTool("double", "Double", {"type": "object"}, lambda value: value * 2))
    provider = MultipleToolCallProvider()
    runner = AgentRunner(provider)

    result = asyncio.run(runner.run(AgentRunSpec(
        messages=[{"role": "user", "content": "use tools"}],
        model="tools",
        max_iterations=3,
        tools=registry,
    )))

    assert result.content == "3, 8"
    assert [message["content"] for message in provider.calls[1] if message["role"] == "tool"] == ["3", "8"]


class NeverFinalToolProvider:
    model = "tools"

    def __init__(self):
        self.calls = 0

    async def complete(self, messages, *, tools=None):
        self.calls += 1
        return LLMResponse(
            content="",
            final=False,
            stop_reason="tool_calls",
            tool_calls=[ToolCallRequest(id=f"call_{self.calls}", name="noop", arguments={})],
        )


def test_runner_stops_tool_loop_at_max_iterations():
    registry = ToolRegistry()
    registry.register(FunctionTool("noop", "No-op", {"type": "object"}, lambda: "ok"))
    provider = NeverFinalToolProvider()
    runner = AgentRunner(provider)

    result = asyncio.run(runner.run(AgentRunSpec(
        messages=[{"role": "user", "content": "loop"}],
        model="tools",
        max_iterations=2,
        tools=registry,
    )))

    assert provider.calls == 2
    assert result.stop_reason == "max_iterations"
    assert result.content == ""
    assert result.messages == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "noop", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "noop", "content": "ok"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "noop", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_2", "name": "noop", "content": "ok"},
    ]


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
