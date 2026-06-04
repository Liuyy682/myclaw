import asyncio
import json
import shutil

import pytest

from myclaw import AgentConfig
from myclaw.agent.dream import DreamManager
from myclaw.memory import MemoryStore
from myclaw.providers import LLMResponse
from myclaw.providers.base import ToolCallRequest
from myclaw.session import SessionManager


class ScriptedDreamProvider:
    """Phase 1 returns a checklist; Phase 2 writes a file then finishes."""

    model = "dream"

    def __init__(self, checklist):
        self.checklist = checklist
        self.calls = []

    async def complete(self, messages, *, tools=None):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        # Phase 1 is the tool-free call (tools is None).
        if tools is None:
            return self.checklist
        # Phase 2 runs through AgentRunner with file tools. First iteration:
        # call write_file; second: finish.
        if len(self.calls) == 2:
            return LLMResponse(
                content="",
                final=False,
                stop_reason="tool_calls",
                tool_calls=[
                    ToolCallRequest(
                        id="call_write",
                        name="write_file",
                        arguments={"path": "USER.md", "content": "- Name is Sam.\n"},
                    )
                ],
            )
        return LLMResponse(content="done", final=True)


def _make_dream(tmp_path, provider, *, interval=15):
    manager = SessionManager(tmp_path)
    store = MemoryStore(tmp_path)
    config = AgentConfig(system_prompt="", dream_interval_minutes=interval)
    return DreamManager(manager, provider, store, config, model="dream"), store


def test_dream_disabled_when_interval_zero(tmp_path):
    dream, store = _make_dream(tmp_path, ScriptedDreamProvider("(nothing)"), interval=0)
    store.append_history("something happened")

    assert dream.enabled is False
    assert dream.should_run_now() is False


def test_should_run_now_false_without_new_history(tmp_path):
    dream, _ = _make_dream(tmp_path, ScriptedDreamProvider("(nothing)"))

    assert dream.should_run_now() is False  # no history at all


def test_phase1_input_includes_history_with_id_tags(tmp_path):
    provider = ScriptedDreamProvider("(nothing)")
    dream, store = _make_dream(tmp_path, provider)
    store.append_history("user adopted a cat named Luna")

    assert asyncio.run(dream.run_once()) is True

    phase1_user = provider.calls[0]["messages"][1]["content"]
    assert provider.calls[0]["tools"] is None
    assert "⟨0⟩" in phase1_user
    assert "Luna" in phase1_user
    # Cursor advanced past the consumed batch.
    cursor = json.loads((tmp_path / "memory" / ".dream_cursor").read_text(encoding="utf-8"))
    assert cursor["last_id"] == 0


def test_phase2_applies_checklist_via_file_tools(tmp_path):
    provider = ScriptedDreamProvider("[USER] Name is Sam.")
    dream, store = _make_dream(tmp_path, provider)
    store.append_history("the user introduced themselves as Sam")

    assert asyncio.run(dream.run_once()) is True

    # Phase 2 ran the write_file tool, creating USER.md in the memory dir.
    assert store.read_user() == "- Name is Sam."
    # Three calls: phase 1 + two phase-2 iterations.
    assert len(provider.calls) == 3
    assert provider.calls[1]["tools"] is not None


def test_nothing_checklist_skips_phase2_but_advances_cursor(tmp_path):
    provider = ScriptedDreamProvider("(nothing)")
    dream, store = _make_dream(tmp_path, provider)
    store.append_history("idle chatter")

    assert asyncio.run(dream.run_once()) is True

    # Only Phase 1 ran; no file tool calls.
    assert len(provider.calls) == 1
    assert not (tmp_path / "memory" / "USER.md").exists()
    cursor = json.loads((tmp_path / "memory" / ".dream_cursor").read_text(encoding="utf-8"))
    assert cursor["last_id"] == 0


def test_run_once_advances_cursor_even_when_phase1_raises(tmp_path):
    class BoomProvider:
        model = "dream"

        async def complete(self, messages, *, tools=None):
            raise RuntimeError("phase 1 boom")

    dream, store = _make_dream(tmp_path, BoomProvider())
    store.append_history("entry one")
    store.append_history("entry two")

    # Failure is swallowed (returns False) and never propagates.
    assert asyncio.run(dream.run_once()) is False
    cursor = json.loads((tmp_path / "memory" / ".dream_cursor").read_text(encoding="utf-8"))
    assert cursor["last_id"] == 1


def test_run_once_returns_false_when_no_new_entries(tmp_path):
    dream, _ = _make_dream(tmp_path, ScriptedDreamProvider("(nothing)"))

    assert asyncio.run(dream.run_once()) is False


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_run_once_auto_commits_memory_changes(tmp_path):
    provider = ScriptedDreamProvider("[USER] Name is Sam.")
    dream, store = _make_dream(tmp_path, provider)
    store.append_history("the user introduced themselves as Sam")

    assert asyncio.run(dream.run_once()) is True

    # The memory dir became a git repo with one dream commit carrying the
    # checklist as the body.
    assert asyncio.run(dream.git.is_repo()) is True
    entries = asyncio.run(dream.git.log(5))
    assert len(entries) == 1
    assert entries[0]["subject"].startswith("dream:")
    code, body, _ = asyncio.run(dream.git._run("log", "-1", "--format=%B"))
    assert "[USER] Name is Sam." in body


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_nothing_checklist_produces_no_commit(tmp_path):
    provider = ScriptedDreamProvider("(nothing)")
    dream, store = _make_dream(tmp_path, provider)
    store.append_history("idle chatter")

    assert asyncio.run(dream.run_once()) is True

    # Phase 2 never ran, so no repo/commit was created.
    assert asyncio.run(dream.git.log(5)) == []
