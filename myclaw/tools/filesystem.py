from __future__ import annotations

from pathlib import Path
from typing import Any

from myclaw.tools.base import Tool
from myclaw.tools.registry import ToolRegistry

_BLOCKED_DEVICE_PATHS = frozenset(
    {
        "/dev/zero",
        "/dev/random",
        "/dev/urandom",
        "/dev/full",
        "/dev/stdin",
        "/dev/stdout",
        "/dev/stderr",
        "/dev/tty",
        "/dev/console",
        "/dev/fd/0",
        "/dev/fd/1",
        "/dev/fd/2",
    }
)
_IGNORED_DIRS = frozenset({".git", "node_modules", ".pytest_cache", "__pycache__"})


def build_default_tool_registry(workspace: Path | str | None = None) -> ToolRegistry:
    root = Path(workspace).expanduser() if workspace is not None else Path.cwd()
    registry = ToolRegistry()
    registry.register(ReadFileTool(root))
    registry.register(ListDirTool(root))
    registry.register(WriteFileTool(root))
    return registry


def _is_under(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _is_blocked_device(path: str | Path) -> bool:
    raw = str(path)
    try:
        resolved = str(Path(raw).resolve())
    except (OSError, ValueError):
        resolved = raw
    return raw in _BLOCKED_DEVICE_PATHS or resolved in _BLOCKED_DEVICE_PATHS or resolved.startswith("/dev/")


class _FilesystemTool(Tool):
    def __init__(self, workspace: Path | str | None = None) -> None:
        self._workspace = Path(workspace).expanduser().resolve() if workspace is not None else Path.cwd().resolve()

    def _resolve(self, path: str) -> Path:
        if _is_blocked_device(path):
            raise PermissionError(f"Device path is blocked: {path}")
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self._workspace / candidate
        resolved = candidate.resolve()
        if _is_blocked_device(resolved):
            raise PermissionError(f"Device path is blocked: {path}")
        if not _is_under(resolved, self._workspace):
            raise PermissionError(f"Path is outside workspace: {path}")
        return resolved

    @staticmethod
    def _error(exc: PermissionError) -> str:
        return f"Error: {exc}"


class ReadFileTool(_FilesystemTool):
    _DEFAULT_LIMIT = 2000

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read a UTF-8 text file under the workspace. Output format is LINE|CONTENT."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read"},
                "offset": {"type": "integer", "description": "1-indexed line number to start from"},
                "limit": {"type": "integer", "description": "Maximum number of lines to read"},
            },
            "required": ["path"],
        }

    async def execute(self, path: str | None = None, offset: int = 1, limit: int | None = None, **kwargs: Any) -> str:
        if not path:
            return "Error reading file: Unknown path"
        try:
            file_path = self._resolve(path)
        except PermissionError as exc:
            return self._error(exc)

        if not file_path.exists():
            return f"Error: File not found: {path}"
        if not file_path.is_file():
            return f"Error: Not a file: {path}"

        raw = file_path.read_bytes()
        if not raw:
            return f"(Empty file: {path})"
        try:
            text = raw.decode("utf-8").replace("\r\n", "\n")
        except UnicodeDecodeError:
            return f"Error: Cannot read binary file: {path}"

        try:
            start = max(1, int(offset))
            count = self._DEFAULT_LIMIT if limit is None else max(1, int(limit))
        except (TypeError, ValueError):
            return "Error: offset and limit must be integers"

        lines = text.splitlines()
        if start > len(lines):
            return f"Error: offset {start} is beyond end of file ({len(lines)} lines)"
        selected = lines[start - 1 : start - 1 + count]
        return "\n".join(f"{line_no}|{line}" for line_no, line in enumerate(selected, start=start))


class ListDirTool(_FilesystemTool):
    _DEFAULT_MAX_ENTRIES = 200

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return "List files and directories under the workspace."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path to list"},
                "recursive": {"type": "boolean", "description": "Whether to list files recursively"},
                "max_entries": {"type": "integer", "description": "Maximum entries to return"},
            },
        }

    async def execute(
        self,
        path: str | None = ".",
        recursive: bool = False,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        **kwargs: Any,
    ) -> str:
        path = path or "."
        try:
            directory = self._resolve(path)
        except PermissionError as exc:
            return self._error(exc)

        if not directory.exists():
            return f"Error: Directory not found: {path}"
        if not directory.is_dir():
            return f"Error: Not a directory: {path}"

        entries = self._recursive_entries(directory) if recursive else self._direct_entries(directory)
        if not entries:
            return f"(Empty directory: {path})"

        try:
            limit = max(1, int(max_entries))
        except (TypeError, ValueError):
            return "Error: max_entries must be an integer"
        shown = entries[:limit]
        result = "\n".join(shown)
        if len(entries) > limit:
            result += f"\n[truncated: showing {limit} of {len(entries)} entries]"
        return result

    @staticmethod
    def _direct_entries(directory: Path) -> list[str]:
        entries = []
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            if child.name in _IGNORED_DIRS:
                continue
            entries.append(f"{child.name}/" if child.is_dir() else child.name)
        return entries

    @staticmethod
    def _recursive_entries(directory: Path) -> list[str]:
        entries = []
        for child in sorted(directory.rglob("*"), key=lambda item: item.relative_to(directory).as_posix()):
            relative = child.relative_to(directory)
            if any(part in _IGNORED_DIRS for part in relative.parts) or child.is_dir():
                continue
            entries.append(relative.as_posix())
        return entries


class WriteFileTool(_FilesystemTool):
    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write UTF-8 text content to a file under the workspace, creating parent directories."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write"},
                "content": {"type": "string", "description": "UTF-8 text content to write"},
            },
            "required": ["path", "content"],
        }

    async def execute(self, path: str | None = None, content: str = "", **kwargs: Any) -> str:
        if not path:
            return "Error writing file: Unknown path"
        try:
            file_path = self._resolve(path)
        except PermissionError as exc:
            return self._error(exc)

        if file_path.exists() and file_path.is_dir():
            return f"Error: Cannot write to directory: {path}"

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(str(content), encoding="utf-8")
        return f"Wrote {len(str(content).encode('utf-8'))} bytes to {path}"
