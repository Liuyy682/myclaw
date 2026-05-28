from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from myclaw.agent.types import Message

WORKSPACE_ENV_VAR = "MYCLAW_WORKSPACE"


def get_default_workspace() -> Path:
    configured = os.environ.get(WORKSPACE_ENV_VAR)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".myclaw" / "workspace"


@dataclass(slots=True)
class Session:
    key: str
    messages: list[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str) -> None:
        self.messages.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
            }
        )
        self.updated_at = datetime.now()


class SessionManager:
    def __init__(self, workspace: Path | str | None = None) -> None:
        self.workspace = Path(workspace).expanduser() if workspace is not None else get_default_workspace()
        self.sessions_dir = self.workspace / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Session] = {}

    @staticmethod
    def safe_key(key: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", key).strip("_") or "session"

    def get_or_create(self, key: str) -> Session:
        if key in self._cache:
            return self._cache[key]

        session = self._load(key)
        if session is None:
            session = Session(key=key)
        self._cache[key] = session
        return session

    def save(self, session: Session) -> None:
        path = self._get_session_path(session.key)
        tmp_path = path.with_suffix(".jsonl.tmp")
        session.updated_at = datetime.now()

        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(self._metadata_line(session), ensure_ascii=False) + "\n")
                for message in session.messages:
                    handle.write(json.dumps(message, ensure_ascii=False) + "\n")
            os.replace(tmp_path, path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

        self._cache[session.key] = session

    def _get_session_path(self, key: str) -> Path:
        return self.sessions_dir / f"{self.safe_key(key)}.jsonl"

    def _load(self, key: str) -> Session | None:
        path = self._get_session_path(key)
        if not path.exists():
            return None

        messages: list[Message] = []
        metadata: dict[str, Any] = {}
        created_at: datetime | None = None
        updated_at: datetime | None = None

        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if data.get("_type") == "metadata":
                metadata = data.get("metadata", {})
                created_at = self._parse_datetime(data.get("created_at"))
                updated_at = self._parse_datetime(data.get("updated_at"))
                continue

            if isinstance(data.get("role"), str) and isinstance(data.get("content"), str):
                messages.append(data)

        return Session(
            key=key,
            messages=messages,
            created_at=created_at or datetime.now(),
            updated_at=updated_at or datetime.now(),
            metadata=metadata,
        )

    @staticmethod
    def _metadata_line(session: Session) -> dict[str, Any]:
        return {
            "_type": "metadata",
            "key": session.key,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "metadata": session.metadata,
        }

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
