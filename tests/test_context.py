from myclaw import AgentConfig, ContextBuilder
from myclaw.agent.context import TokenEstimator


def test_context_builder_preserves_structured_tool_history_without_timestamps():
    builder = ContextBuilder()
    tool_calls = [
        {
            "id": "call_add",
            "type": "function",
            "function": {"name": "add", "arguments": "{}"},
        }
    ]
    history = [
        {"role": "user", "content": "add numbers", "timestamp": "ignored"},
        {"role": "assistant", "content": "", "tool_calls": tool_calls, "timestamp": "ignored"},
        {"role": "tool", "content": "5", "tool_call_id": "call_add", "name": "add", "timestamp": "ignored"},
        {"role": "assistant", "content": "sum is 5", "timestamp": "ignored"},
    ]

    messages = builder.build_messages(AgentConfig(system_prompt=""), history, "next")

    assert messages == [
        {"role": "user", "content": "add numbers"},
        {"role": "assistant", "content": "", "tool_calls": tool_calls},
        {"role": "tool", "content": "5", "tool_call_id": "call_add", "name": "add"},
        {"role": "assistant", "content": "sum is 5"},
        {"role": "user", "content": "next"},
    ]


def test_context_builder_drops_orphan_tool_results():
    builder = ContextBuilder()
    history = [
        {"role": "tool", "content": "orphan", "tool_call_id": "missing", "name": "read_file"},
        {"role": "assistant", "content": "visible"},
    ]

    messages = builder.build_messages(AgentConfig(system_prompt=""), history, "next")

    assert messages == [
        {"role": "assistant", "content": "visible"},
        {"role": "user", "content": "next"},
    ]


def test_context_builder_backfills_missing_tool_results():
    builder = ContextBuilder()
    tool_calls = [
        {
            "id": "call_missing",
            "type": "function",
            "function": {"name": "read_file", "arguments": "{}"},
        }
    ]
    history = [{"role": "assistant", "content": "", "tool_calls": tool_calls}]

    messages = builder.build_messages(AgentConfig(system_prompt=""), history, "next")

    assert messages == [
        {"role": "assistant", "content": "", "tool_calls": tool_calls},
        {
            "role": "tool",
            "tool_call_id": "call_missing",
            "name": "read_file",
            "content": "[tool result unavailable in persisted history]",
        },
        {"role": "user", "content": "next"},
    ]


def test_context_builder_truncates_tool_results_for_prompt_only():
    builder = ContextBuilder()
    full_content = "x" * 20
    tool_calls = [
        {
            "id": "call_read",
            "type": "function",
            "function": {"name": "read_file", "arguments": "{}"},
        }
    ]
    history = [
        {"role": "assistant", "content": "", "tool_calls": tool_calls},
        {"role": "tool", "content": full_content, "tool_call_id": "call_read", "name": "read_file"},
    ]

    messages = builder.build_messages(
        AgentConfig(system_prompt="", max_tool_result_chars=8),
        history,
        "next",
    )

    assert messages[1]["content"] == "x" * 8 + "\n[tool result truncated: 12 chars omitted]"
    assert history[1]["content"] == full_content


def test_context_builder_inserts_summary_and_skips_covered_messages():
    builder = ContextBuilder()
    history = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "recent question"},
        {"role": "assistant", "content": "recent answer"},
    ]

    messages = builder.build_messages(
        AgentConfig(system_prompt=""),
        history,
        "next",
        context_summary={
            "content": "Old topic summary.",
            "covered_message_count": 2,
        },
    )

    assert messages == [
        {"role": "system", "content": "Summary of earlier conversation:\nOld topic summary."},
        {"role": "user", "content": "recent question"},
        {"role": "assistant", "content": "recent answer"},
        {"role": "user", "content": "next"},
    ]


class FakeEncoding:
    def encode(self, text):
        return list(text)


def test_token_estimator_uses_supplied_encoding():
    estimator = TokenEstimator("demo-model", encoding=FakeEncoding())

    assert estimator.estimate_text("hello") == 5


def test_token_estimator_fallback_counts_ascii_and_non_ascii_text():
    estimator = TokenEstimator("demo-model", encoding=None)

    assert estimator.estimate_text("abcdefgh") == 2
    assert estimator.estimate_text("你好") == 2


def test_token_estimator_estimates_chat_messages_with_overhead():
    estimator = TokenEstimator("demo-model", encoding=FakeEncoding())

    tokens = estimator.estimate_messages([{"role": "user", "content": "hello"}])

    assert tokens > estimator.estimate_text("hello")


def test_context_builder_cleans_remaining_history_after_summary_boundary():
    builder = ContextBuilder()
    tool_calls = [
        {
            "id": "call_missing",
            "type": "function",
            "function": {"name": "read_file", "arguments": "{}"},
        }
    ]
    history = [
        {"role": "user", "content": "covered"},
        {"role": "assistant", "content": "covered answer"},
        {"role": "assistant", "content": "", "tool_calls": tool_calls},
    ]

    messages = builder.build_messages(
        AgentConfig(system_prompt=""),
        history,
        "next",
        context_summary={"content": "Covered summary.", "covered_message_count": 2},
    )

    assert messages == [
        {"role": "system", "content": "Summary of earlier conversation:\nCovered summary."},
        {"role": "assistant", "content": "", "tool_calls": tool_calls},
        {
            "role": "tool",
            "tool_call_id": "call_missing",
            "name": "read_file",
            "content": "[tool result unavailable in persisted history]",
        },
        {"role": "user", "content": "next"},
    ]
