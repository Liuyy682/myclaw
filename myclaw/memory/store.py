from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class MemoryStore:
    """Persist workspace-level long-term memory."""

    def __init__(self, workspace: Path | str) -> None:
        self.workspace = Path(workspace).expanduser()
        self.memory_dir = self.workspace / "memory"
        self.memory_path = self.memory_dir / "MEMORY.md"
        self.history_path = self.memory_dir / "history.jsonl"

    def read_memory(self) -> str:
        if not self.memory_path.exists() or not self.memory_path.is_file():
            return ""
        return self.memory_path.read_text(encoding="utf-8").strip()

    def remember(self, content: str) -> str:
        normalized = self._normalize_content(content)
        timestamp = datetime.now().isoformat()
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._append_memory(timestamp, normalized)
        return "Remembered."

    def append_history(self, entry: str) -> None:
        normalized = self._normalize_content(entry)
        timestamp = datetime.now().isoformat()
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._append_history(timestamp, normalized)

    @staticmethod
    def _normalize_content(content: str) -> str:
        normalized = " ".join(str(content).split())
        if not normalized:
            raise ValueError("content must be non-empty")
        return normalized

    def _append_memory(self, timestamp: str, content: str) -> None:
        existing = self.memory_path.read_text(encoding="utf-8") if self.memory_path.exists() else ""
        with self.memory_path.open("a", encoding="utf-8") as handle:
            if not existing.strip():
                handle.write("# Memory\n\n")
            elif not existing.endswith("\n"):
                handle.write("\n")
            handle.write(f"- {timestamp} {content}\n")

    def _append_history(self, timestamp: str, content: str) -> None:
        record = {
            "timestamp": timestamp,
            "source": "compact",
            "content": content,
        }
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
