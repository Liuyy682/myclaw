import asyncio
import json

import pytest

from myclaw.tools import ToolCallRequest, ToolRegistry
from myclaw.memory import MemoryStore
from myclaw.tools import MemoryWriteTool


def test_memory_store_writes_markdown_without_history_jsonl(tmp_path):
    store = MemoryStore(tmp_path)

    result = store.remember("  User likes\nshort answers.  ")

    assert result == "Remembered."
    memory_path = tmp_path / "memory" / "MEMORY.md"
    history_path = tmp_path / "memory" / "history.jsonl"
    memory_text = memory_path.read_text(encoding="utf-8")
    assert memory_text.startswith("# Memory\n\n")
    assert "User likes short answers." in memory_text
    assert not history_path.exists()
    assert store.read_memory() == memory_text.strip()


def test_memory_store_rejects_blank_content(tmp_path):
    store = MemoryStore(tmp_path)

    with pytest.raises(ValueError, match="content must be non-empty"):
        store.remember(" \n\t ")

    assert not (tmp_path / "memory").exists()


def test_memory_write_tool_executes_through_registry(tmp_path):
    registry = ToolRegistry()
    registry.register(MemoryWriteTool(MemoryStore(tmp_path)))

    result = asyncio.run(
        registry.execute(
            ToolCallRequest(
                id="call_remember",
                name="remember",
                arguments={"content": "User prefers concise answers."},
            )
        )
    )

    assert result == "Remembered."
    assert "User prefers concise answers." in (tmp_path / "memory" / "MEMORY.md").read_text(encoding="utf-8")
    assert not (tmp_path / "memory" / "history.jsonl").exists()


def test_memory_write_tool_returns_readable_error_for_blank_content(tmp_path):
    tool = MemoryWriteTool(MemoryStore(tmp_path))

    result = asyncio.run(tool.execute(content=" "))

    assert result == "Error: content must be non-empty"


def test_memory_store_appends_compact_history_with_monotonic_ids(tmp_path):
    store = MemoryStore(tmp_path)

    store.append_history("  compressed\nold turns  ")
    store.append_history("second entry")

    history_path = tmp_path / "memory" / "history.jsonl"
    history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
    assert [entry["id"] for entry in history] == [0, 1]
    assert history[0]["source"] == "compact"
    assert history[0]["content"] == "compressed old turns"
    assert "timestamp" in history[0]
    assert not (tmp_path / "memory" / "MEMORY.md").exists()


def test_read_history_since_returns_only_newer_entries(tmp_path):
    store = MemoryStore(tmp_path)
    store.append_history("a")
    store.append_history("b")
    store.append_history("c")

    assert [entry["content"] for entry in store.read_history_since(0)] == ["b", "c"]
    assert store.read_history_since(2) == []
    assert [entry["id"] for entry in store.read_history_since(-1)] == [0, 1, 2]


def test_read_history_backfills_ids_for_legacy_entries(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    # Legacy entries written before ids existed (no "id" key).
    (memory_dir / "history.jsonl").write_text(
        json.dumps({"timestamp": "t0", "source": "compact", "content": "old0"}) + "\n"
        + json.dumps({"timestamp": "t1", "source": "compact", "content": "old1"}) + "\n",
        encoding="utf-8",
    )
    store = MemoryStore(tmp_path)

    entries = store.read_history()
    assert [entry["id"] for entry in entries] == [0, 1]
    # Next append continues past the backfilled ids without rewriting the file.
    store.append_history("new")
    assert store.read_history()[-1]["id"] == 2


def test_read_user_and_soul_return_empty_when_missing(tmp_path):
    store = MemoryStore(tmp_path)

    assert store.read_user() == ""
    assert store.read_soul() == ""

    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "USER.md").write_text("  likes cats  \n", encoding="utf-8")
    (tmp_path / "memory" / "SOUL.md").write_text("warm and concise\n", encoding="utf-8")
    assert store.read_user() == "likes cats"
    assert store.read_soul() == "warm and concise"
