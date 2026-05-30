import asyncio
import subprocess
import sys
import os
import json

from myclaw.bus import MessageBus, OutboundMessage
from myclaw.cli.commands import build_agent_loop, dispatch_text, run_interactive
from myclaw.config.env import load_env_file


def test_cli_single_turn_uses_fake_provider_without_api_key(tmp_path):
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env["MYCLAW_ENV_FILE"] = str(tmp_path / "missing.env")
    env["MYCLAW_WORKSPACE"] = str(tmp_path / "workspace")

    result = subprocess.run(
        [sys.executable, "-m", "myclaw", "hello"],
        check=True,
        cwd="/root/myclaw",
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "Echo: hello"


def test_cli_single_turn_persists_and_reuses_local_session(tmp_path):
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env["MYCLAW_ENV_FILE"] = str(tmp_path / "missing.env")
    env["MYCLAW_WORKSPACE"] = str(tmp_path / "workspace")

    subprocess.run(
        [sys.executable, "-m", "myclaw", "first"],
        check=True,
        cwd="/root/myclaw",
        env=env,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [sys.executable, "-m", "myclaw", "second"],
        check=True,
        cwd="/root/myclaw",
        env=env,
        capture_output=True,
        text=True,
    )

    session_file = tmp_path / "workspace" / "sessions" / "cli_direct.jsonl"
    messages = [
        json.loads(line)
        for line in session_file.read_text(encoding="utf-8").splitlines()
        if line and json.loads(line).get("_type") != "metadata"
    ]

    assert [message["content"] for message in messages] == [
        "first",
        "Echo: first",
        "second",
        "Echo: second",
    ]


class ProgressThenFinalDispatcher:
    def __init__(self):
        self.bus = MessageBus()

    async def run(self):
        await self.bus.consume_inbound()
        await self.bus.publish_outbound(
            OutboundMessage(
                channel="cli",
                chat_id="direct",
                content="Running tool add (1/1)",
                terminal=False,
                event_type="tool_progress",
            )
        )
        await self.bus.publish_outbound(
            OutboundMessage(channel="cli", chat_id="direct", content="final answer")
        )
        await asyncio.Event().wait()


def test_dispatch_text_waits_for_terminal_outbound_message():
    dispatcher = ProgressThenFinalDispatcher()

    outbound = asyncio.run(dispatch_text(dispatcher, "hello"))

    assert outbound.content == "final answer"
    assert outbound.terminal is True


class RecordingDispatcher:
    def __init__(self):
        self.bus = MessageBus()
        self.run_calls = 0
        self.received = []

    async def run(self):
        self.run_calls += 1
        while True:
            msg = await self.bus.consume_inbound()
            self.received.append(msg.content)
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=f"ack: {msg.content}",
                    event_type="control" if msg.content.startswith("/") else "message",
                )
            )


def test_interactive_cli_keeps_one_dispatcher_running_for_multiple_inputs(monkeypatch):
    dispatcher = RecordingDispatcher()
    inputs = iter(["hello", "/status", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    asyncio.run(run_interactive(dispatcher))

    assert dispatcher.run_calls == 1
    assert dispatcher.received == ["hello", "/status"]


def test_cli_interactive_persists_two_turns(tmp_path):
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env["MYCLAW_ENV_FILE"] = str(tmp_path / "missing.env")
    env["MYCLAW_WORKSPACE"] = str(tmp_path / "workspace")

    result = subprocess.run(
        [sys.executable, "-m", "myclaw"],
        input="first\nsecond\nexit\n",
        check=True,
        cwd="/root/myclaw",
        env=env,
        capture_output=True,
        text=True,
    )

    assert "Assistant: Echo: first" in result.stdout
    assert "Assistant: Echo: second" in result.stdout
    assert result.stdout.startswith("You: Assistant: Echo: first\nYou: Assistant: Echo: second\nYou: ")

    session_file = tmp_path / "workspace" / "sessions" / "cli_direct.jsonl"
    messages = [
        json.loads(line)
        for line in session_file.read_text(encoding="utf-8").splitlines()
        if line and json.loads(line).get("_type") != "metadata"
    ]
    assert [message["content"] for message in messages] == [
        "first",
        "Echo: first",
        "second",
        "Echo: second",
    ]


def test_build_agent_loop_registers_default_file_tools(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("MYCLAW_ENV_FILE", str(tmp_path / "missing.env"))
    monkeypatch.setenv("MYCLAW_WORKSPACE", str(tmp_path / "workspace"))

    loop = build_agent_loop()

    assert loop.tool_registry is not None
    assert [definition["function"]["name"] for definition in loop.tool_registry.definitions()] == [
        "list_dir",
        "read_file",
        "write_file",
    ]


def test_load_env_file_reads_project_env_without_overwriting_existing_values(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=from-file",
                "OPENAI_BASE_URL='https://example.test/v1'",
                'OPENAI_MODEL="demo-model"',
                "IGNORED_LINE",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "from-shell")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    load_env_file(env_file)

    assert env_file.read_text(encoding="utf-8").startswith("OPENAI_API_KEY")
    assert "OPENAI_API_KEY=from-file" in env_file.read_text(encoding="utf-8")
    assert __import__("os").environ["OPENAI_API_KEY"] == "from-shell"
    assert __import__("os").environ["OPENAI_BASE_URL"] == "https://example.test/v1"
    assert __import__("os").environ["OPENAI_MODEL"] == "demo-model"
