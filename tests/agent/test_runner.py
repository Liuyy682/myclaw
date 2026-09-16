import asyncio

from myclaw.agent import AgentRunSpec, AgentRunner
from myclaw.providers import FakeProvider, LLMResponse, LLMServiceUnavailableError
from myclaw.tools import FunctionTool, ToolCallRequest, ToolRegistry


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


def test_runner_checkpoints_each_complete_model_response_before_final_check():
    provider = MultiStepProvider()
    checkpoints = []

    async def record_checkpoint(payload):
        checkpoints.append(payload)

    result = asyncio.run(AgentRunner(provider).run(AgentRunSpec(
        messages=[{"role": "user", "content": "solve"}],
        model="multi",
        max_iterations=3,
        checkpoint_callback=record_checkpoint,
    )))

    assert result.content == "done"
    assert [(checkpoint["phase"], checkpoint["iteration"]) for checkpoint in checkpoints] == [
        ("model_response_received", 0),
        ("model_response_received", 1),
    ]
    assert [
        [message["content"] for message in checkpoint["messages"]]
        for checkpoint in checkpoints
    ] == [["thinking"], ["thinking", "done"]]
    assert all(checkpoint["pending_tool_calls"] == [] for checkpoint in checkpoints)


class NeverFinalProvider:
    model = "loop"

    def __init__(self):
        self.calls = 0

    async def complete(self, messages, *, tools=None):
        self.calls += 1
        return LLMResponse(content=f"step {self.calls}", final=False, stop_reason="continue")


class StreamingProvider:
    model = "stream"

    def __init__(self):
        self.complete_calls = 0
        self.stream_calls = 0

    async def complete(self, messages, *, tools=None):
        self.complete_calls += 1
        return "complete fallback"

    async def stream_complete(self, messages, *, tools=None, delta_callback=None):
        self.stream_calls += 1
        if delta_callback is not None:
            await delta_callback("hel")
            await delta_callback("lo")
        return "hello"


class CompleteOnlyProvider:
    model = "complete"

    def __init__(self):
        self.calls = 0

    async def complete(self, messages, *, tools=None):
        self.calls += 1
        return "complete only"


def test_runner_stops_at_max_iterations():
    provider = NeverFinalProvider()
    runner = AgentRunner(provider)
    checkpoints = []

    async def record_checkpoint(payload):
        checkpoints.append(payload)

    result = asyncio.run(runner.run(AgentRunSpec(
        messages=[{"role": "user", "content": "solve"}],
        model="loop",
        max_iterations=2,
        checkpoint_callback=record_checkpoint,
    )))

    assert provider.calls == 2
    assert result.content == "step 2"
    assert result.stop_reason == "max_iterations"
    assert result.messages == [
        {"role": "assistant", "content": "step 1"},
        {"role": "assistant", "content": "step 2"},
    ]
    assert checkpoints[-1] == {
        "phase": "model_response_received",
        "iteration": 1,
        "messages": result.messages,
        "pending_tool_calls": [],
    }


def test_runner_uses_streaming_provider_when_stream_callback_is_available():
    provider = StreamingProvider()
    runner = AgentRunner(provider)
    deltas = []

    async def record_delta(delta):
        deltas.append(delta)

    result = asyncio.run(runner.run(AgentRunSpec(
        messages=[{"role": "user", "content": "hello"}],
        model="stream",
        max_iterations=1,
        stream_callback=record_delta,
    )))

    assert result.content == "hello"
    assert deltas == ["hel", "lo"]
    assert provider.stream_calls == 1
    assert provider.complete_calls == 0


def test_runner_checkpoints_streamed_response_only_after_stream_completes():
    provider = StreamingProvider()
    events = []

    async def record_delta(delta):
        events.append(f"delta:{delta}")

    async def record_checkpoint(payload):
        events.append(payload["phase"])

    asyncio.run(AgentRunner(provider).run(AgentRunSpec(
        messages=[{"role": "user", "content": "hello"}],
        model="stream",
        max_iterations=1,
        stream_callback=record_delta,
        checkpoint_callback=record_checkpoint,
    )))

    assert events == ["delta:hel", "delta:lo", "model_response_received"]


def test_runner_falls_back_to_complete_when_provider_has_no_streaming_method():
    provider = CompleteOnlyProvider()
    runner = AgentRunner(provider)
    deltas = []

    async def record_delta(delta):
        deltas.append(delta)

    result = asyncio.run(runner.run(AgentRunSpec(
        messages=[{"role": "user", "content": "hello"}],
        model="complete",
        max_iterations=1,
        stream_callback=record_delta,
    )))

    assert result.content == "complete only"
    assert deltas == []
    assert provider.calls == 1


class FailingProvider:
    model = "broken"

    async def complete(self, messages, *, tools=None):
        raise RuntimeError("provider unavailable")


class FailsAfterResponseProvider:
    model = "broken"

    def __init__(self):
        self.calls = 0

    async def complete(self, messages, *, tools=None):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(content="partial", final=False, stop_reason="continue")
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


class ToolThenFailingProvider:
    model = "tools"

    def __init__(self):
        self.calls = 0

    async def complete(self, messages, *, tools=None):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content="",
                final=False,
                stop_reason="tool_calls",
                tool_calls=[ToolCallRequest(id="call_add", name="add", arguments={"a": 2, "b": 3})],
            )
        raise RuntimeError("provider unavailable")


def test_runner_executes_tool_call_and_sends_tool_result_to_next_model_call():
    registry = ToolRegistry()
    registry.register(
        FunctionTool(
            "add",
            "Add two numbers",
            {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
            lambda a, b: a + b,
            read_only=True,
            effect="local_read",
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


def test_runner_keeps_tool_messages_when_follow_up_model_call_fails():
    registry = ToolRegistry()
    registry.register(FunctionTool("add", "Add", {"type": "object"}, lambda a, b: a + b, read_only=True, effect="local_read"))

    result = asyncio.run(AgentRunner(ToolThenFailingProvider()).run(AgentRunSpec(
        messages=[{"role": "user", "content": "add 2 and 3"}],
        model="tools",
        max_iterations=2,
        tools=registry,
    )))

    assert result.messages == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_add",
                    "type": "function",
                    "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_add", "name": "add", "content": "5"},
        {"role": "assistant", "content": "Error: provider unavailable"},
    ]


class LongToolResultProvider:
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
                tool_calls=[ToolCallRequest(id="call_long", name="long", arguments={})],
            )
        return LLMResponse(content=messages[-1]["content"], final=True)


def test_runner_truncates_tool_result_before_model_call_and_generated_messages():
    registry = ToolRegistry()
    registry.register(FunctionTool("long", "Long result", {"type": "object"}, lambda: "abcdef", read_only=True, effect="local_read"))
    provider = LongToolResultProvider()
    runner = AgentRunner(provider)

    result = asyncio.run(runner.run(AgentRunSpec(
        messages=[{"role": "user", "content": "use tool"}],
        model="tools",
        max_iterations=3,
        tools=registry,
        max_tool_result_chars=3,
    )))

    expected = "abc\n[tool result truncated: 3 chars omitted]"
    assert provider.calls[1][-1] == {"role": "tool", "tool_call_id": "call_long", "name": "long", "content": expected}
    assert result.messages[1]["content"] == expected
    assert result.content == expected


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
    registry.register(FunctionTool("add", "Add", {"type": "object"}, lambda a, b: a + b, read_only=True, effect="local_read"))
    registry.register(FunctionTool("double", "Double", {"type": "object"}, lambda value: value * 2, read_only=True, effect="local_read"))
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


def test_runner_emits_tool_progress_checkpoints():
    registry = ToolRegistry()
    registry.register(FunctionTool("add", "Add", {"type": "object"}, lambda a, b: a + b, read_only=True, effect="local_read"))
    registry.register(FunctionTool("double", "Double", {"type": "object"}, lambda value: value * 2, read_only=True, effect="local_read"))
    provider = MultipleToolCallProvider()
    runner = AgentRunner(provider)
    checkpoints = []

    async def record_checkpoint(payload):
        checkpoints.append(payload)

    asyncio.run(runner.run(AgentRunSpec(
        messages=[{"role": "user", "content": "use tools"}],
        model="tools",
        max_iterations=3,
        tools=registry,
        checkpoint_callback=record_checkpoint,
    )))

    assert [checkpoint["phase"] for checkpoint in checkpoints] == [
        "awaiting_tools",
        "tools_in_progress",
        "tools_completed",
        "model_response_received",
    ]
    assert [message["role"] for message in checkpoints[0]["messages"]] == ["assistant"]
    assert [call["id"] for call in checkpoints[0]["pending_tool_calls"]] == ["call_add", "call_double"]
    assert [message["role"] for message in checkpoints[1]["messages"]] == ["assistant", "tool"]
    assert [call["id"] for call in checkpoints[1]["pending_tool_calls"]] == ["call_double"]
    assert [message["role"] for message in checkpoints[2]["messages"]] == ["assistant", "tool", "tool"]
    assert checkpoints[2]["pending_tool_calls"] == []
    assert [message["role"] for message in checkpoints[3]["messages"]] == [
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]
    assert checkpoints[3]["pending_tool_calls"] == []


def test_runner_emits_tool_progress_callbacks_around_each_tool_call():
    registry = ToolRegistry()
    registry.register(FunctionTool("add", "Add", {"type": "object"}, lambda a, b: a + b, read_only=True, effect="local_read"))
    registry.register(FunctionTool("double", "Double", {"type": "object"}, lambda value: value * 2, read_only=True, effect="local_read"))
    provider = MultipleToolCallProvider()
    runner = AgentRunner(provider)
    progress = []

    async def record_progress(payload):
        progress.append(payload)

    asyncio.run(runner.run(AgentRunSpec(
        messages=[{"role": "user", "content": "use tools"}],
        model="tools",
        max_iterations=3,
        tools=registry,
        progress_callback=record_progress,
    )))

    assert [(event["event"], event["tool_name"], event["index"], event["total"]) for event in progress] == [
        ("tool_started", "add", 1, 2),
        ("tool_completed", "add", 1, 2),
        ("tool_started", "double", 2, 2),
        ("tool_completed", "double", 2, 2),
    ]
    assert [event["tool_call_id"] for event in progress] == [
        "call_add",
        "call_add",
        "call_double",
        "call_double",
    ]


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
    registry.register(FunctionTool("noop", "No-op", {"type": "object"}, lambda: "ok", read_only=True, effect="local_read"))
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


def test_runner_keeps_generated_messages_when_a_later_model_call_fails():
    result = asyncio.run(AgentRunner(FailsAfterResponseProvider()).run(AgentRunSpec(
        messages=[{"role": "user", "content": "hello"}],
        model="broken",
        max_iterations=2,
    )))

    assert result.content == "Error: provider unavailable"
    assert result.stop_reason == "error"
    assert result.error == "provider unavailable"
    assert result.messages == [
        {"role": "assistant", "content": "partial"},
        {"role": "assistant", "content": "Error: provider unavailable"},
    ]


def test_runner_does_not_swallow_cancellation():
    class CancellingProvider:
        model = "cancelled"

        async def complete(self, messages, *, tools=None):
            raise asyncio.CancelledError()

    try:
        asyncio.run(AgentRunner(CancellingProvider()).run(AgentRunSpec(
            messages=[{"role": "user", "content": "hello"}],
            model="cancelled",
            max_iterations=1,
        )))
    except asyncio.CancelledError:
        return
    raise AssertionError("CancelledError was swallowed")


def test_runner_returns_stable_message_for_temporary_llm_outage():
    class UnavailableProvider:
        model = "broken"

        async def complete(self, messages, *, tools=None):
            raise LLMServiceUnavailableError()

    result = asyncio.run(AgentRunner(UnavailableProvider()).run(AgentRunSpec(
        messages=[{"role": "user", "content": "hello"}],
        model="broken",
        max_iterations=1,
    )))

    assert result.content == "Error: LLM service is temporarily unavailable. Please retry later."
    assert result.error == "LLM service is temporarily unavailable. Please retry later."


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
