from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


class CronStore:
    def __init__(self, workspace: Path | str) -> None:
        self.workspace = Path(workspace).expanduser()
        self.cron_dir = self.workspace / "cron"
        self.path = self.cron_dir / "jobs.json"
        self.cron_dir.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        *,
        name: str,
        prompt: str,
        every_seconds: int | None = None,
        at: datetime | str | None = None,
        next_run_at: datetime | str | None = None,
        session_key: str | None = None,
        enabled: bool = True,
    ) -> dict[str, Any]:
        if not name.strip():
            raise ValueError("name is required")
        if not prompt.strip():
            raise ValueError("prompt is required")
        if every_seconds is not None and every_seconds < 1:
            raise ValueError("every_seconds must be positive")

        at_text = self._isoformat(at) if at is not None else None
        if next_run_at is None:
            if every_seconds is not None:
                next_run_at = datetime.now() + timedelta(seconds=every_seconds)
            else:
                next_run_at = at
        if next_run_at is None:
            raise ValueError("at or every_seconds is required")

        now = datetime.now().isoformat()
        job = {
            "id": uuid.uuid4().hex[:12],
            "name": name.strip(),
            "prompt": prompt,
            "every_seconds": every_seconds,
            "at": at_text,
            "next_run_at": self._isoformat(next_run_at),
            "session_key": session_key,
            "enabled": bool(enabled),
            "created_at": now,
            "updated_at": now,
        }
        jobs = self.list()
        jobs.append(job)
        self._save(jobs)
        return job

    def list(self) -> list[dict[str, Any]]:
        return self._load()

    def get(self, job_id: str) -> dict[str, Any]:
        for job in self._load():
            if job.get("id") == job_id:
                return job
        raise KeyError(job_id)

    def claim_due(self, *, now: datetime | None = None, max_jobs: int = 10) -> list[dict[str, Any]]:
        now = now or datetime.now()
        jobs = self._load()
        due: list[dict[str, Any]] = []
        changed = False

        for index, job in enumerate(jobs):
            if len(due) >= max_jobs:
                break
            if not job.get("enabled", True):
                continue
            next_run = self._parse_datetime(job.get("next_run_at"))
            if next_run is None or next_run > now:
                continue

            due.append(dict(job))
            updated = dict(job)
            every_seconds = updated.get("every_seconds")
            if isinstance(every_seconds, int) and every_seconds > 0:
                updated["next_run_at"] = (now + timedelta(seconds=every_seconds)).isoformat()
            else:
                updated["enabled"] = False
            updated["updated_at"] = datetime.now().isoformat()
            jobs[index] = updated
            changed = True

        if changed:
            self._save(jobs)
        return due

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []
        return [job for job in data if isinstance(job, dict)]

    def _save(self, jobs: list[dict[str, Any]]) -> None:
        self.cron_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".json.tmp")
        try:
            tmp_path.write_text(json.dumps(jobs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp_path, self.path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _isoformat(value: datetime | str) -> str:
        if isinstance(value, datetime):
            return value.isoformat()
        parsed = datetime.fromisoformat(value)
        return parsed.isoformat()

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
