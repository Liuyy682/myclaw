from __future__ import annotations

import asyncio
import re
import shutil
from pathlib import Path
from typing import Any

from myclaw.tools.base import Tool
from myclaw.tools.filesystem import _is_blocked_device, _is_under

_BWRAP_AVAILABLE: bool | None = None


def _bwrap_path() -> str | None:
    return shutil.which("bwrap")


async def _detect_bwrap() -> bool:
    """Return True if bwrap exists and can actually create a sandbox here.

    User namespaces are disabled in some containers/CI, where bwrap is present
    but every invocation fails. Probe once and cache the result.
    """
    global _BWRAP_AVAILABLE
    if _BWRAP_AVAILABLE is not None:
        return _BWRAP_AVAILABLE
    path = _bwrap_path()
    if path is None:
        _BWRAP_AVAILABLE = False
        return False
    try:
        process = await asyncio.create_subprocess_exec(
            path,
            "--ro-bind", "/", "/",
            "--unshare-all",
            "--die-with-parent",
            "--",
            "/bin/true",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(process.communicate(), timeout=10)
        _BWRAP_AVAILABLE = process.returncode == 0
    except (OSError, asyncio.TimeoutError):
        _BWRAP_AVAILABLE = False
    return _BWRAP_AVAILABLE



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
        return (
            "Run a timeout-limited shell command. When bubblewrap (bwrap) is available "
            "the command runs in a sandbox: read-only system, writable workspace only, "
            "PID/IPC/UTS isolation, and no network unless allow_network is set."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "description": "Shell command to run"},
                "cwd": {"type": "string", "description": "Optional workspace-relative working directory"},
                "timeout_seconds": {"type": "integer", "description": "Timeout in seconds"},
                "max_output_chars": {"type": "integer", "description": "Maximum stdout/stderr chars each"},
                "allow_network": {
                    "type": "boolean",
                    "description": "Allow network access inside the sandbox (default false)",
                },
            },
            "required": ["cmd"],
        }

    async def execute(
        self,
        cmd: str | None = None,
        cwd: str | None = None,
        timeout_seconds: int = 30,
        max_output_chars: int = 12000,
        allow_network: bool = False,
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

        sandboxed = await _detect_bwrap()
        if sandboxed:
            argv = self._build_bwrap_argv(command, working_dir, allow_network=bool(allow_network))
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        else:
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
            "sandboxed": sandboxed,
            "stdout": self._decode(stdout, output_limit),
            "stderr": self._decode(stderr, output_limit),
            "cwd": working_dir.relative_to(self._workspace).as_posix() or ".",
        }

    def _build_bwrap_argv(self, command: str, working_dir: Path, *, allow_network: bool) -> list[str]:
        bwrap = _bwrap_path() or "bwrap"
        argv = [
            bwrap,
            "--ro-bind", "/", "/",
            "--bind", str(self._workspace), str(self._workspace),
            "--dev", "/dev",
            "--proc", "/proc",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--die-with-parent",
        ]
        # A private /tmp keeps temp files out of the host, but only when it does
        # not shadow a workspace that itself lives under /tmp.
        if not _is_under(self._workspace, Path("/tmp")) and self._workspace != Path("/tmp"):
            argv += ["--tmpfs", "/tmp"]
        if not allow_network:
            argv.append("--unshare-net")
        argv += ["--chdir", str(working_dir), "--", "/bin/sh", "-c", command]
        return argv

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
