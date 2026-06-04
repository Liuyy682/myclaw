from __future__ import annotations

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_GIT_TIMEOUT_SECONDS = 15
_DEFAULT_LOG_LIMIT = 10
_COMMIT_AUTHOR_NAME = "myclaw-dream"
_COMMIT_AUTHOR_EMAIL = "dream@myclaw.local"
_LOG_FORMAT = "%h%x1f%cI%x1f%s"  # short-hash, committer ISO date, subject (unit-separated)


class MemoryGit:
    """Minimal async git wrapper scoped to the ``memory/`` directory.

    Gives the Dream consolidation an auditable, revertible history of how the
    memory files evolve. Every operation swallows its errors and returns a
    benign value — git being missing or failing must never disrupt Dream, which
    is a side-channel capability.
    """

    def __init__(self, memory_dir: Path | str) -> None:
        self.memory_dir = Path(memory_dir).expanduser()

    async def _run(self, *argv: str) -> tuple[int, str, str]:
        try:
            process = await asyncio.create_subprocess_exec(
                "git",
                *argv,
                cwd=str(self.memory_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (OSError, ValueError) as exc:
            logger.warning("git %s failed to start: %s", argv[:1], exc)
            return 1, "", str(exc)
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=_GIT_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            logger.warning("git %s timed out", argv[:1])
            return 1, "", "timeout"
        return (
            process.returncode or 0,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )

    async def is_repo(self) -> bool:
        if not self.memory_dir.exists():
            return False
        code, out, _ = await self._run("rev-parse", "--is-inside-work-tree")
        return code == 0 and out.strip() == "true"

    async def ensure_repo(self) -> bool:
        """Initialise the repo (with a local identity) if it is not one yet."""
        try:
            self.memory_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("cannot create memory dir for git: %s", exc)
            return False
        if await self.is_repo():
            return True
        code, _, err = await self._run("init")
        if code != 0:
            logger.warning("git init failed: %s", err.strip())
            return False
        # Local identity only — never touch the user's global git config.
        await self._run("config", "user.name", _COMMIT_AUTHOR_NAME)
        await self._run("config", "user.email", _COMMIT_AUTHOR_EMAIL)
        # The dream cursor is transient bookkeeping, not memory content.
        gitignore = self.memory_dir / ".gitignore"
        if not gitignore.exists():
            try:
                gitignore.write_text(".dream_cursor\n", encoding="utf-8")
            except OSError:
                pass
        return True

    async def has_changes(self) -> bool:
        code, out, _ = await self._run("status", "--porcelain")
        if code != 0:
            return False
        return bool(out.strip())

    async def changed_count(self) -> int:
        code, out, _ = await self._run("status", "--porcelain")
        if code != 0:
            return 0
        return len([line for line in out.splitlines() if line.strip()])

    async def commit_all(self, title: str, body: str = "") -> bool:
        """Stage everything and commit. Returns False if there is nothing to commit."""
        if not await self.ensure_repo():
            return False
        if not await self.has_changes():
            return False
        add_code, _, add_err = await self._run("add", "-A")
        if add_code != 0:
            logger.warning("git add failed: %s", add_err.strip())
            return False
        argv = ["commit", "-m", title]
        if body.strip():
            argv += ["-m", body]
        code, _, err = await self._run(*argv)
        if code != 0:
            logger.warning("git commit failed: %s", err.strip())
            return False
        return True

    async def log(self, limit: int = _DEFAULT_LOG_LIMIT) -> list[dict]:
        if not await self.is_repo():
            return []
        code, out, _ = await self._run(
            "log", f"-{max(1, limit)}", f"--format={_LOG_FORMAT}"
        )
        if code != 0:
            return []
        entries: list[dict] = []
        for line in out.splitlines():
            if not line.strip():
                continue
            parts = line.split("\x1f")
            if len(parts) != 3:
                continue
            entries.append({"hash": parts[0], "date": parts[1], "subject": parts[2]})
        return entries
