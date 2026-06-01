import asyncio
import re
import subprocess
import sys
import os
import json

from myclaw import DispatcherRuntime
from myclaw.bus import MessageBus, OutboundMessage
from myclaw.cli.commands import build_agent_loop, dispatch_text, run_interactive
from myclaw.config.env import load_env_file
from myclaw.session import SessionManager


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


def test_cli_single_turn_accepts_session_option(tmp_path):
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env["MYCLAW_ENV_FILE"] = str(tmp_path / "missing.env")
    env["MYCLAW_WORKSPACE"] = str(tmp_path / "workspace")

    subprocess.run(
        [sys.executable, "-m", "myclaw", "--session", "work", "hello"],
        check=True,
        cwd="/root/myclaw",
        env=env,
        capture_output=True,
        text=True,
    )

    assert (tmp_path / "workspace" / "sessions" / "cli_work.jsonl").exists()
    assert not (tmp_path / "workspace" / "sessions" / "cli_direct.jsonl").exists()


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


class StreamingCliDispatcher:
    def __init__(self):
        self.bus = MessageBus()
        self.loop = type("Loop", (), {"session_manager": None})()
        self.received = []

    async def run(self):
        msg = await self.bus.consume_inbound()
        self.received.append((msg.chat_id, msg.content, msg.metadata))
        await self.bus.publish_outbound(
            OutboundMessage(
                channel="cli",
                chat_id=msg.chat_id,
                content="hel",
                terminal=False,
                event_type="message_delta",
            )
        )
        await self.bus.publish_outbound(
            OutboundMessage(
                channel="cli",
                chat_id=msg.chat_id,
                content="lo",
                terminal=False,
                event_type="message_delta",
            )
        )
        await self.bus.publish_outbound(
            OutboundMessage(channel="cli", chat_id=msg.chat_id, content="hello")
        )
        await asyncio.Event().wait()


def test_dispatch_text_waits_for_terminal_outbound_message():
    dispatcher = ProgressThenFinalDispatcher()

    async def scenario():
        async with DispatcherRuntime(dispatcher) as runtime:
            return await dispatch_text(runtime, "hello")

    outbound = asyncio.run(scenario())
    assert outbound.content == "final answer"
    assert outbound.terminal is True


class RecordingDispatcher:
    def __init__(self, session_manager=None):
        self.bus = MessageBus()
        self.loop = type("Loop", (), {"session_manager": session_manager})()
        self.run_calls = 0
        self.received = []

    async def run(self):
        self.run_calls += 1
        while True:
            msg = await self.bus.consume_inbound()
            self.received.append((msg.chat_id, msg.content))
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=f"ack: {msg.content}",
                    event_type="control" if msg.content.startswith("/") else "message",
                )
            )


def test_dispatch_text_uses_existing_runtime_for_multiple_messages():
    dispatcher = RecordingDispatcher()

    async def scenario():
        async with DispatcherRuntime(dispatcher) as runtime:
            first = await dispatch_text(runtime, "first")
            second = await dispatch_text(runtime, "second")
            return first, second

    first, second = asyncio.run(scenario())

    assert dispatcher.run_calls == 1
    assert dispatcher.received == [("direct", "first"), ("direct", "second")]
    assert first.content == "ack: first"
    assert second.content == "ack: second"


def test_interactive_cli_streams_message_deltas_inline(monkeypatch, capsys):
    dispatcher = StreamingCliDispatcher()
    inputs = iter(["hello", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    asyncio.run(run_interactive(dispatcher, session_name="direct"))

    output = capsys.readouterr().out
    assert output == "Assistant: hello\n"
    assert dispatcher.received == [("direct", "hello", {"stream": True})]


def test_interactive_cli_keeps_one_dispatcher_running_for_multiple_inputs(monkeypatch):
    dispatcher = RecordingDispatcher()
    inputs = iter(["hello", "/status", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    asyncio.run(run_interactive(dispatcher, session_name="direct"))

    assert dispatcher.run_calls == 1
    assert dispatcher.received == [("direct", "hello"), ("direct", "/status")]


def test_interactive_cli_resume_switches_sessions_and_prompt_labels(tmp_path, monkeypatch):
    manager = SessionManager(tmp_path)
    dispatcher = RecordingDispatcher(manager)
    inputs = iter(["/resume work", "hello", "/resume direct", "again", "exit"])
    prompts = []

    def fake_input(prompt=""):
        prompts.append(prompt)
        return next(inputs)

    monkeypatch.setattr("builtins.input", fake_input)

    asyncio.run(run_interactive(dispatcher, session_name="direct"))

    assert dispatcher.received == [("work", "hello"), ("direct", "again")]
    assert prompts == [
        "You[direct]: ",
        "You[work]: ",
        "You[work]: ",
        "You[direct]: ",
        "You[direct]: ",
    ]


def test_interactive_cli_resume_lists_sessions_with_titles(tmp_path, monkeypatch, capsys):
    manager = SessionManager(tmp_path)
    direct = manager.get_or_create("cli:direct")
    direct.metadata["title"] = "Direct chat"
    manager.save(direct)
    work = manager.get_or_create("cli:work")
    work.metadata["title"] = "Work chat"
    manager.save(work)
    dispatcher = RecordingDispatcher(manager)
    inputs = iter(["/resume", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    asyncio.run(run_interactive(dispatcher, session_name="direct"))

    output = capsys.readouterr().out
    assert "* direct - Direct chat" in output
    assert "  work - Work chat" in output
    assert dispatcher.received == []


def test_interactive_cli_new_creates_and_switches_to_new_session(tmp_path, monkeypatch):
    manager = SessionManager(tmp_path)
    dispatcher = RecordingDispatcher(manager)
    inputs = iter(["/new", "hello", "exit"])
    prompts = []

    def fake_input(prompt=""):
        prompts.append(prompt)
        return next(inputs)

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr("myclaw.cli.commands.new_cli_session_name", lambda manager: "chat-test")

    asyncio.run(run_interactive(dispatcher, session_name="direct"))

    assert dispatcher.received == [("chat-test", "hello")]
    assert prompts == ["You[direct]: ", "You[chat-test]: ", "You[chat-test]: "]
    assert (tmp_path / "sessions" / "cli_chat-test.jsonl").exists()


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

    assert re.search(
        r"You\[chat-\d{8}-\d{6}(?:-\d+)?\]: Assistant: Echo: first\n"
        r"You\[first\]: Assistant: Echo: second\n"
        r"You\[first\]: ",
        result.stdout,
    )

    session_files = list((tmp_path / "workspace" / "sessions").glob("cli_chat-*.jsonl"))
    assert len(session_files) == 1
    assert not (tmp_path / "workspace" / "sessions" / "cli_direct.jsonl").exists()
    session_file = session_files[0]
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
        "edit_file",
        "glob",
        "grep",
        "list_dir",
        "read_file",
        "remember",
        "write_file",
    ]
    remember = loop.tool_registry.get("remember")
    assert remember is not None
    result = asyncio.run(remember.execute(content="User likes CLI memory."))
    assert result == "Remembered."
    assert "User likes CLI memory." in (
        tmp_path / "workspace" / "memory" / "MEMORY.md"
    ).read_text(encoding="utf-8")


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
