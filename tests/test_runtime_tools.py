import asyncio
import json
from datetime import datetime, timedelta

import pytest

from myclaw.bus import MessageBus
from myclaw.tools.ask import AskUserTool
from myclaw.tools.cron import CronTool
from myclaw.tools.message import MessageTool
from myclaw.tools.notebook import NotebookEditTool
from myclaw.tools.self import MyTool
from myclaw.tools.shell import ExecTool, _detect_bwrap
from myclaw.tools.spawn import SpawnTool
from myclaw.tools.tasks import TaskCreateTool, TaskGetTool, TaskListTool, TaskUpdateTool
from myclaw.tools.web import WebFetchTool, WebSearchTool
from myclaw.tasks import TaskStore
from myclaw.cron import CronStore
from myclaw.tools.base import ToolRuntimeContext, tool_context


def _bwrap_ready() -> bool:
    return asyncio.run(_detect_bwrap())


requires_bwrap = pytest.mark.skipif(not _bwrap_ready(), reason="bwrap sandbox not available")


def test_message_and_my_tools_use_runtime_context(tmp_path):
    context = ToolRuntimeContext(
        session_key="cli:direct",
        channel="cli",
        chat_id="direct",
        metadata={"request_id": "req-1"},
        workspace=tmp_path,
        tool_names=["ask_user", "message", "my", "spawn"],
    )

    async def scenario():
        with tool_context(context):
            asked = await AskUserTool().execute(question="Choose?", choices=["a", "b"])
            messaged = await MessageTool().execute(content="hello")
            spawned = await SpawnTool().execute(prompt="summarize this", name="summary")
            mine = await MyTool().execute()
        return asked, messaged, spawned, mine

    asked, messaged, spawned, mine = asyncio.run(scenario())

    # Without host callbacks wired into the context, ask/spawn report unavailability
    # instead of pretending to succeed.
    assert asked == "Error: interactive user prompts are not available in this context"
    assert spawned == "Error: sub-agent execution is not available in this context"
    assert messaged == {
        "status": "queued",
        "channel": "cli",
        "chat_id": "direct",
        "content": "hello",
    }
    assert mine["session_key"] == "cli:direct"
    assert mine["workspace"] == str(tmp_path)
    assert mine["tools"] == ["ask_user", "message", "my", "spawn"]


def test_spawn_tool_runs_subagent_when_context_provides_callback(tmp_path):
    async def fake_spawn(prompt, name):
        return f"sub[{name}]: {prompt}"

    context = ToolRuntimeContext(session_key="cli:direct", workspace=tmp_path, spawn=fake_spawn)

    async def scenario():
        with tool_context(context):
            return await SpawnTool().execute(prompt="summarize this", name="summary")

    spawned = asyncio.run(scenario())

    assert spawned["status"] == "completed"
    assert spawned["name"] == "summary"
    assert spawned["result"] == "sub[summary]: summarize this"


def test_ask_tool_returns_answer_when_context_provides_callback(tmp_path):
    async def fake_ask(question, choices):
        return f"answer to {question} from {choices}"

    context = ToolRuntimeContext(session_key="cli:direct", workspace=tmp_path, ask=fake_ask)

    async def scenario():
        with tool_context(context):
            return await AskUserTool().execute(question="Choose?", choices=["a", "b"])

    answered = asyncio.run(scenario())

    assert answered == {
        "status": "answered",
        "question": "Choose?",
        "choices": ["a", "b"],
        "answer": "answer to Choose? from ['a', 'b']",
    }


def test_exec_tool_runs_workspace_bounded_commands_and_blocks_destructive_commands(tmp_path):
    tool = ExecTool(tmp_path)

    ok = asyncio.run(tool.execute(cmd="python3 -c \"print('hi')\""))
    blocked = asyncio.run(tool.execute(cmd="rm -rf ."))
    blocked_nested = asyncio.run(tool.execute(cmd="rm -rf nested"))
    outside = asyncio.run(tool.execute(cmd="pwd", cwd=str(tmp_path.parent)))

    assert ok["exit_code"] == 0
    assert ok["stdout"] == "hi\n"
    assert ok["stderr"] == ""
    assert blocked.startswith("Error: command is blocked")
    assert blocked_nested.startswith("Error: command is blocked")
    assert outside.startswith("Error: Path is outside workspace:")


@requires_bwrap
def test_exec_sandbox_reports_sandboxed_and_can_write_workspace(tmp_path):
    tool = ExecTool(tmp_path)

    result = asyncio.run(tool.execute(cmd="echo data > out.txt && cat out.txt"))

    assert result["sandboxed"] is True
    assert result["exit_code"] == 0
    assert result["stdout"] == "data\n"
    # The write lands in the real workspace because it is bind-mounted read-write.
    assert (tmp_path / "out.txt").read_text() == "data\n"


@requires_bwrap
def test_exec_sandbox_makes_system_paths_read_only(tmp_path):
    tool = ExecTool(tmp_path)

    result = asyncio.run(tool.execute(cmd="echo x > /usr/sandbox_probe"))

    assert result["sandboxed"] is True
    assert result["exit_code"] != 0
    assert "Read-only file system" in result["stderr"]


@requires_bwrap
def test_exec_sandbox_blocks_network_by_default_and_allows_when_requested(tmp_path):
    tool = ExecTool(tmp_path)
    probe = "python3 -c \"import socket; socket.gethostbyname('example.com')\""

    blocked = asyncio.run(tool.execute(cmd=probe))
    allowed = asyncio.run(tool.execute(cmd=probe, allow_network=True))

    assert blocked["exit_code"] != 0
    # With network namespace shared, resolution either succeeds or fails for
    # reasons unrelated to the sandbox cutting it off.
    assert "Temporary failure in name resolution" not in allowed["stderr"]


def test_exec_falls_back_to_blacklist_when_bwrap_unavailable(tmp_path, monkeypatch):
    import myclaw.tools.shell as shell

    monkeypatch.setattr(shell, "_BWRAP_AVAILABLE", False)
    tool = ExecTool(tmp_path)

    ok = asyncio.run(tool.execute(cmd="echo fallback-ok"))
    blocked = asyncio.run(tool.execute(cmd="rm -rf /"))

    assert ok["sandboxed"] is False
    assert ok["exit_code"] == 0
    assert ok["stdout"] == "fallback-ok\n"
    # The command blacklist remains the second line of defense in fallback mode.
    assert blocked.startswith("Error: command is blocked")


def test_notebook_edit_replaces_existing_cell_source(tmp_path):
    notebook = tmp_path / "note.ipynb"
    notebook.write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "markdown", "metadata": {}, "source": ["old\n"]},
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )

    result = asyncio.run(NotebookEditTool(tmp_path).execute(path="note.ipynb", cell_index=0, source="new\n"))
    data = json.loads(notebook.read_text(encoding="utf-8"))

    assert result == "Edited note.ipynb cell 0"
    assert data["cells"][0]["source"] == ["new\n"]


def test_task_tools_persist_create_list_get_and_update(tmp_path):
    store = TaskStore(tmp_path)
    created = asyncio.run(TaskCreateTool(store).execute(title="Ship tools", description="Add parity"))
    task_id = created["id"]

    listed = asyncio.run(TaskListTool(store).execute())
    fetched = asyncio.run(TaskGetTool(store).execute(id=task_id))
    updated = asyncio.run(TaskUpdateTool(store).execute(id=task_id, status="done"))
    reloaded = TaskStore(tmp_path).get(task_id)

    assert created["status"] == "open"
    assert listed["tasks"][0]["id"] == task_id
    assert fetched["title"] == "Ship tools"
    assert updated["status"] == "done"
    assert reloaded["status"] == "done"
    assert (tmp_path / "tasks" / "tasks.json").exists()


def test_cron_tool_persists_supported_schedules_and_rejects_full_cron(tmp_path):
    store = CronStore(tmp_path)
    created = asyncio.run(CronTool(store).execute(name="heartbeat", prompt="check", every_seconds=60))
    unsupported = asyncio.run(CronTool(store).execute(name="bad", prompt="check", cron="* * * * *"))

    assert created["name"] == "heartbeat"
    assert created["every_seconds"] == 60
    assert created["enabled"] is True
    assert store.get(created["id"])["prompt"] == "check"
    assert unsupported == "Error: cron expressions are not supported; use every_seconds or at"


class DummyResponse:
    def __init__(self, payload):
        self.payload = payload
        self.headers = {"content-type": "text/html; charset=utf-8"}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return self.payload


def test_web_tools_validate_urls_and_parse_simple_results(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout=0):
        calls.append(request.full_url)
        if "duckduckgo" in request.full_url:
            return DummyResponse(
                b'<a class="result__a" href="https://example.com/doc">Example Result</a>'
            )
        return DummyResponse(b"<html><title>Doc</title><p>Hello</p></html>")

    monkeypatch.setattr("myclaw.tools.web.urlopen", fake_urlopen)

    fetched = asyncio.run(WebFetchTool().execute(url="https://example.com/doc"))
    searched = asyncio.run(WebSearchTool().execute(query="example"))
    blocked = asyncio.run(WebFetchTool().execute(url="http://127.0.0.1/private"))

    assert fetched["url"] == "https://example.com/doc"
    assert "Hello" in fetched["content"]
    assert searched["results"] == [{"title": "Example Result", "url": "https://example.com/doc"}]
    assert blocked == "Error: blocked private or local address"


def test_cron_store_claims_due_jobs_and_reschedules_intervals(tmp_path):
    store = CronStore(tmp_path)
    now = datetime.now()
    interval = store.create(
        name="interval",
        prompt="repeat",
        every_seconds=30,
        next_run_at=now - timedelta(seconds=1),
    )
    one_shot = store.create(
        name="once",
        prompt="run once",
        at=now - timedelta(seconds=1),
        next_run_at=now - timedelta(seconds=1),
    )

    due = store.claim_due(now=now)

    assert [job["id"] for job in due] == [interval["id"], one_shot["id"]]
    assert store.get(interval["id"])["enabled"] is True
    assert datetime.fromisoformat(store.get(interval["id"])["next_run_at"]) > now
    assert store.get(one_shot["id"])["enabled"] is False
