from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from myclaw.tools.base import Tool
from myclaw.tools.filesystem import _is_blocked_device, _is_under


class ExecTool(Tool):
    read_only = False
    exclusive = True

    _BLOCKED_PATTERNS = (
        re.compile(r"\brm\s+(?=[^;&|]*-[A-Za-z]*r)(?=[^;&|]*-[A-Za-z]*f)[^;&|]*", re.IGNORECASE),
        re.compile(r"\bgit\s+reset\s+--hard\b", re.IGNORECASE),
        re.compile(r"\bdd\s+.*\bof=", re.IGNORECASE),
        re.compile(r"\bmkfs(?:\.[A-Za-z0-9_+-]+)?\b", re.IGNORECASE),
        re.compile(r"\bshutdown\b|\breboot\b|\bpoweroff\b", re.IGNORECASE),
        re.compile(r":\(\)\s*\{"),
    )

    def __init__(self, workspace: Path | str | None = None) -> None:
        self._workspace = Path(workspace).expanduser().resolve() if workspace is not None else Path.cwd().resolve()

    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return "Run a timeout-limited shell command inside the workspace."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "description": "Shell command to run"},
                "cwd": {"type": "string", "description": "Optional workspace-relative working directory"},
                "timeout_seconds": {"type": "integer", "description": "Timeout in seconds"},
                "max_output_chars": {"type": "integer", "description": "Maximum stdout/stderr chars each"},
            },
            "required": ["cmd"],
        }

    async def execute(
        self,
        cmd: str | None = None,
        cwd: str | None = None,
        timeout_seconds: int = 30,
        max_output_chars: int = 12000,
        **kwargs: Any,
    ) -> dict[str, Any] | str:
        if cmd is None or not str(cmd).strip():
            return "Error: cmd is required"
        command = str(cmd).strip()
        if self._is_blocked_command(command):
            return "Error: command is blocked by the exec safety policy"
        try:
            working_dir = self._resolve_cwd(cwd or ".")
        except PermissionError as exc:
            return f"Error: {exc}"
        try:
            timeout = max(1, int(timeout_seconds))
            output_limit = max(1, int(max_output_chars))
        except (TypeError, ValueError):
            return "Error: timeout_seconds and max_output_chars must be integers"

        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(working_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        timed_out = False
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            timed_out = True
            process.kill()
            stdout, stderr = await process.communicate()

        return {
            "exit_code": process.returncode,
            "timed_out": timed_out,
            "stdout": self._decode(stdout, output_limit),
            "stderr": self._decode(stderr, output_limit),
            "cwd": working_dir.relative_to(self._workspace).as_posix() or ".",
        }

    def _resolve_cwd(self, cwd: str) -> Path:
        if _is_blocked_device(cwd):
            raise PermissionError(f"Device path is blocked: {cwd}")
        candidate = Path(cwd).expanduser()
        if not candidate.is_absolute():
            candidate = self._workspace / candidate
        resolved = candidate.resolve()
        if _is_blocked_device(resolved):
            raise PermissionError(f"Device path is blocked: {cwd}")
        if not _is_under(resolved, self._workspace):
            raise PermissionError(f"Path is outside workspace: {cwd}")
        if not resolved.exists():
            raise PermissionError(f"Directory not found: {cwd}")
        if not resolved.is_dir():
            raise PermissionError(f"Not a directory: {cwd}")
        return resolved

    @classmethod
    def _is_blocked_command(cls, command: str) -> bool:
        return any(pattern.search(command) for pattern in cls._BLOCKED_PATTERNS)

    @staticmethod
    def _decode(payload: bytes, limit: int) -> str:
        text = payload.decode("utf-8", errors="replace")
        if len(text) <= limit:
            return text
        return text[:limit] + f"\n[truncated: {len(text) - limit} chars omitted]"
