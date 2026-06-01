import asyncio
import json

import pytest

from myclaw import ToolCallRequest, ToolRegistry
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


def test_memory_store_appends_compact_history_jsonl(tmp_path):
    store = MemoryStore(tmp_path)

    store.append_history("  compressed\nold turns  ")

    history_path = tmp_path / "memory" / "history.jsonl"
    history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
    assert len(history) == 1
    assert history[0]["source"] == "compact"
    assert history[0]["content"] == "compressed old turns"
    assert "timestamp" in history[0]
    assert not (tmp_path / "memory" / "MEMORY.md").exists()


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
