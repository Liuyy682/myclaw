import asyncio
import io
import json
import os
import subprocess
import sys

from myclaw.bus import MessageBus, OutboundMessage
from myclaw.gateway import run_gateway


class RecordingDispatcher:
    def __init__(self):
        self.bus = MessageBus()
        self.run_calls = 0
        self.received = []

    async def run(self):
        self.run_calls += 1
        while True:
            msg = await self.bus.consume_inbound()
            self.received.append((msg.channel, msg.chat_id, msg.content, msg.metadata, msg.session_key_override))
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=f"ack: {msg.content}",
                    metadata=dict(msg.metadata),
                    event_type="control" if msg.content.startswith("/") else "message",
                )
            )


class ProgressDispatcher:
    def __init__(self):
        self.bus = MessageBus()
        self.run_calls = 0

    async def run(self):
        self.run_calls += 1
        await self.bus.consume_inbound()
        await self.bus.publish_outbound(
            OutboundMessage(
                channel="gateway",
                chat_id="direct",
                content="Running tool add (1/1)",
                metadata={"request_id": "req-progress"},
                terminal=False,
                event_type="tool_progress",
            )
        )
        await self.bus.publish_outbound(
            OutboundMessage(
                channel="gateway",
                chat_id="direct",
                content="done",
                metadata={"request_id": "req-progress"},
            )
        )
        await asyncio.Event().wait()


def _events(output):
    return [json.loads(line) for line in output.getvalue().splitlines() if line]


def test_gateway_reuses_one_dispatcher_runtime_for_multiple_jsonl_messages():
    dispatcher = RecordingDispatcher()
    input_stream = io.StringIO(
        "\n".join(
            [
                json.dumps({"id": "one", "chat_id": "alpha", "content": "hello"}),
                json.dumps({"id": "two", "chat_id": "beta", "content": "/status", "session_key": "shared:beta"}),
                "",
            ]
        )
    )
    output_stream = io.StringIO()

    asyncio.run(run_gateway(dispatcher, input_stream=input_stream, output_stream=output_stream))

    events = _events(output_stream)
    assert dispatcher.run_calls == 1
    assert dispatcher.received == [
        ("gateway", "alpha", "hello", {"request_id": "one"}, None),
        ("gateway", "beta", "/status", {"request_id": "two"}, "shared:beta"),
    ]
    assert events == [
        {
            "type": "message",
            "id": "one",
            "chat_id": "alpha",
            "content": "ack: hello",
            "terminal": True,
            "metadata": {"request_id": "one"},
        },
        {
            "type": "control",
            "id": "two",
            "chat_id": "beta",
            "content": "ack: /status",
            "terminal": True,
            "metadata": {"request_id": "two"},
        },
    ]


def test_gateway_reports_invalid_json_and_continues_to_next_message():
    dispatcher = RecordingDispatcher()
    input_stream = io.StringIO('not json\n{"id": "ok", "content": "hello"}\n')
    output_stream = io.StringIO()

    asyncio.run(run_gateway(dispatcher, input_stream=input_stream, output_stream=output_stream))

    events = _events(output_stream)
    assert events[0]["type"] == "error"
    assert events[0]["terminal"] is True
    assert events[0]["content"].startswith("Error: invalid JSON:")
    assert events[1] == {
        "type": "message",
        "id": "ok",
        "chat_id": "direct",
        "content": "ack: hello",
        "terminal": True,
        "metadata": {"request_id": "ok"},
    }


def test_gateway_reports_schema_errors_without_stopping():
    dispatcher = RecordingDispatcher()
    input_stream = io.StringIO('{"id": "bad", "content": 42}\n{"id": "ok", "content": "hello"}\n')
    output_stream = io.StringIO()

    asyncio.run(run_gateway(dispatcher, input_stream=input_stream, output_stream=output_stream))

    events = _events(output_stream)
    assert events[0] == {
        "type": "error",
        "id": "bad",
        "chat_id": "direct",
        "content": "Error: gateway input must include string content",
        "terminal": True,
        "metadata": {},
    }
    assert events[1]["type"] == "message"
    assert events[1]["content"] == "ack: hello"


def test_gateway_preserves_tool_progress_events_as_non_terminal_jsonl():
    dispatcher = ProgressDispatcher()
    input_stream = io.StringIO('{"id": "req-progress", "content": "use tool"}\n')
    output_stream = io.StringIO()

    asyncio.run(run_gateway(dispatcher, input_stream=input_stream, output_stream=output_stream))

    events = _events(output_stream)
    assert events == [
        {
            "type": "tool_progress",
            "id": "req-progress",
            "chat_id": "direct",
            "content": "Running tool add (1/1)",
            "terminal": False,
            "metadata": {"request_id": "req-progress"},
        },
        {
            "type": "message",
            "id": "req-progress",
            "chat_id": "direct",
            "content": "done",
            "terminal": True,
            "metadata": {"request_id": "req-progress"},
        },
    ]


def test_gateway_cli_command_reads_jsonl_and_exits_after_eof(tmp_path):
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env["MYCLAW_ENV_FILE"] = str(tmp_path / "missing.env")
    env["MYCLAW_WORKSPACE"] = str(tmp_path / "workspace")

    result = subprocess.run(
        [sys.executable, "-m", "myclaw", "gateway"],
        input='{"id": "req-1", "chat_id": "direct", "content": "hello"}\n',
        check=True,
        cwd="/root/myclaw",
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.stderr == ""
    assert [json.loads(line) for line in result.stdout.splitlines()] == [
        {
            "type": "message",
            "id": "req-1",
            "chat_id": "direct",
            "content": "Echo: hello",
            "terminal": True,
            "metadata": {"request_id": "req-1"},
        }
    ]
    assert (tmp_path / "workspace" / "sessions" / "gateway_direct.jsonl").exists()
