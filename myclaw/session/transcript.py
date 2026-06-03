from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TranscriptStore:
    """Append-only, full-fidelity JSONL log of every conversation message.

    Unlike the session file (which auto-compaction truncates and rewrites),
    this stream is only ever appended to, so it preserves the verbatim
    original of every user message, assistant reply, tool call, and tool
    result. One file per session, named with the same scheme as the session
    file so the two correspond.
    """

    def __init__(self, workspace: Path | str, *, safe_key: Callable[[str], str]) -> None:
        self.workspace = Path(workspace).expanduser()
        self.transcripts_dir = self.workspace / "transcripts"
        self._safe_key = safe_key

    def _path(self, session_key: str) -> Path:
        return self.transcripts_dir / f"{self._safe_key(session_key)}.jsonl"

    def append(self, session_key: str, message: dict[str, Any]) -> None:
        self.append_many(session_key, (message,))

    def append_many(self, session_key: str, messages: Iterable[dict[str, Any]]) -> None:
        records = [self._record(session_key, message) for message in messages]
        if not records:
            return
        try:
            self.transcripts_dir.mkdir(parents=True, exist_ok=True)
            with self._path(session_key).open("a", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            # The transcript is a side-channel audit log; never let a disk
            # problem interrupt the live conversation.
            logger.warning("Failed to append transcript for %s", session_key, exc_info=True)

    @staticmethod
    def _record(session_key: str, message: dict[str, Any]) -> dict[str, Any]:
        return {
            "session_key": session_key,
            "logged_at": datetime.now().isoformat(),
            "message": message,
        }
