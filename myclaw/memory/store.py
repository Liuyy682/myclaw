from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class MemoryStore:
    """Persist workspace-level long-term memory.

    Three markdown files live under ``memory/``:

    - ``MEMORY.md`` — project knowledge and contextual facts (ages, gets pruned).
    - ``USER.md`` — user identity and preferences (durable).
    - ``SOUL.md`` — the assistant's persona and tone (durable).

    ``history.jsonl`` is an append-only stream of compacted conversation
    summaries. Each entry carries a monotonic ``id`` so long-term memory can
    point back to its source even if the stream is later truncated.
    """

    def __init__(self, workspace: Path | str) -> None:
        self.workspace = Path(workspace).expanduser()
        self.memory_dir = self.workspace / "memory"
        self.memory_path = self.memory_dir / "MEMORY.md"
        self.user_path = self.memory_dir / "USER.md"
        self.soul_path = self.memory_dir / "SOUL.md"
        self.history_path = self.memory_dir / "history.jsonl"

    def read_memory(self) -> str:
        return self._read_file(self.memory_path)

    def read_user(self) -> str:
        return self._read_file(self.user_path)

    def read_soul(self) -> str:
        return self._read_file(self.soul_path)

    @staticmethod
    def _read_file(path: Path) -> str:
        if not path.exists() or not path.is_file():
            return ""
        return path.read_text(encoding="utf-8").strip()

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
            "id": self._next_history_id(),
            "timestamp": timestamp,
            "source": "compact",
            "content": content,
        }
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read_history(self) -> list[dict]:
        """Return all history entries, each with a stable ``id``.

        Legacy entries written before ids existed are assigned a derived id by
        line order on read; the file itself is never rewritten.
        """
        if not self.history_path.exists():
            return []
        entries: list[dict] = []
        for index, raw_line in enumerate(self.history_path.read_text(encoding="utf-8").splitlines()):
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            if not isinstance(data.get("id"), int):
                # Backfill a derived id for legacy entries without persisting it.
                data = {**data, "id": index}
            entries.append(data)
        return entries

    def read_history_since(self, after_id: int) -> list[dict]:
        """Return history entries with ``id`` greater than ``after_id``."""
        return [entry for entry in self.read_history() if entry["id"] > after_id]

    def _next_history_id(self) -> int:
        entries = self.read_history()
        if not entries:
            return 0
        return max(entry["id"] for entry in entries) + 1
