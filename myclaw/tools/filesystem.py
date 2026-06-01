from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from myclaw.config import (
    FILESYSTEM_BLOCKED_DEVICE_PATHS,
    FILESYSTEM_IGNORED_DIRS,
    GLOB_DEFAULT_MAX_MATCHES,
    GREP_DEFAULT_MAX_MATCHES,
    LIST_DIR_DEFAULT_MAX_ENTRIES,
    READ_FILE_DEFAULT_LIMIT,
)
from myclaw.memory import MemoryStore
from myclaw.tools.base import Tool
from myclaw.tools.memory import MemoryWriteTool
from myclaw.tools.registry import ToolRegistry


def build_default_tool_registry(
    workspace: Path | str | None = None,
    *,
    memory_workspace: Path | str | None = None,
) -> ToolRegistry:
    root = Path(workspace).expanduser() if workspace is not None else Path.cwd()
    memory_root = Path(memory_workspace).expanduser() if memory_workspace is not None else root
    registry = ToolRegistry()
    registry.register(EditFileTool(root))
    registry.register(GlobTool(root))
    registry.register(GrepTool(root))
    registry.register(ReadFileTool(root))
    registry.register(ListDirTool(root))
    registry.register(MemoryWriteTool(MemoryStore(memory_root)))
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
    return (
        raw in FILESYSTEM_BLOCKED_DEVICE_PATHS
        or resolved in FILESYSTEM_BLOCKED_DEVICE_PATHS
        or resolved.startswith("/dev/")
    )


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

    def _relative_path(self, path: Path) -> str:
        return path.relative_to(self._workspace).as_posix()

    def _is_ignored(self, path: Path) -> bool:
        try:
            relative = path.relative_to(self._workspace)
        except ValueError:
            return True
        return any(part in FILESYSTEM_IGNORED_DIRS for part in relative.parts)

    def _is_safe_existing_path(self, path: Path) -> bool:
        try:
            resolved = path.resolve()
        except (OSError, ValueError):
            return False
        return _is_under(resolved, self._workspace)

    def _recursive_files(self, directory: Path) -> list[Path]:
        if self._is_ignored(directory):
            return []
        files: list[Path] = []
        for child in sorted(directory.rglob("*"), key=lambda item: self._relative_path(item)):
            if self._is_ignored(child) or not child.is_file() or not self._is_safe_existing_path(child):
                continue
            files.append(child)
        return files


class ReadFileTool(_FilesystemTool):
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
            count = READ_FILE_DEFAULT_LIMIT if limit is None else max(1, int(limit))
        except (TypeError, ValueError):
            return "Error: offset and limit must be integers"

        lines = text.splitlines()
        if start > len(lines):
            return f"Error: offset {start} is beyond end of file ({len(lines)} lines)"
        selected = lines[start - 1 : start - 1 + count]
        return "\n".join(f"{line_no}|{line}" for line_no, line in enumerate(selected, start=start))


class ListDirTool(_FilesystemTool):
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
        max_entries: int = LIST_DIR_DEFAULT_MAX_ENTRIES,
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
            if child.name in FILESYSTEM_IGNORED_DIRS:
                continue
            entries.append(f"{child.name}/" if child.is_dir() else child.name)
        return entries

    @staticmethod
    def _recursive_entries(directory: Path) -> list[str]:
        entries = []
        for child in sorted(directory.rglob("*"), key=lambda item: item.relative_to(directory).as_posix()):
            relative = child.relative_to(directory)
            if any(part in FILESYSTEM_IGNORED_DIRS for part in relative.parts) or child.is_dir():
                continue
            entries.append(relative.as_posix())
        return entries


class EditFileTool(_FilesystemTool):
    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return "Replace exact text in an existing UTF-8 file under the workspace. old_text must match exactly once."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to edit"},
                "old_text": {"type": "string", "description": "Existing exact text to replace"},
                "new_text": {"type": "string", "description": "Replacement text"},
            },
            "required": ["path", "old_text", "new_text"],
        }

    async def execute(
        self,
        path: str | None = None,
        old_text: str | None = None,
        new_text: str = "",
        **kwargs: Any,
    ) -> str:
        if not path:
            return "Error editing file: Unknown path"
        if old_text is None:
            return "Error: old_text is required"
        old_text = str(old_text)
        if old_text == "":
            return "Error: old_text must be non-empty"
        new_text = str(new_text)

        try:
            file_path = self._resolve(path)
        except PermissionError as exc:
            return self._error(exc)

        if not file_path.exists():
            return f"Error: File not found: {path}"
        if not file_path.is_file():
            return f"Error: Not a file: {path}"

        try:
            text = file_path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            return f"Error: Cannot edit binary file: {path}"

        matches = text.count(old_text)
        if matches == 0:
            return f"Error: old_text not found in {path}"
        if matches > 1:
            return f"Error: old_text matched {matches} times in {path}; expected exactly 1"

        file_path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}: replaced 1 occurrence"


class GrepTool(_FilesystemTool):
    @property
    def name(self) -> str:
        return "grep"

    @property
    def description(self) -> str:
        return "Regex-search UTF-8 text files under the workspace. Output format is PATH:LINE|CONTENT."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Python regular expression to search for"},
                "path": {"type": "string", "description": "File or directory path to search"},
                "case_sensitive": {"type": "boolean", "description": "Whether matching is case-sensitive"},
                "max_matches": {"type": "integer", "description": "Maximum matches to return"},
            },
            "required": ["pattern"],
        }

    async def execute(
        self,
        pattern: str | None = None,
        path: str | None = ".",
        case_sensitive: bool = True,
        max_matches: int = GREP_DEFAULT_MAX_MATCHES,
        **kwargs: Any,
    ) -> str:
        if pattern is None:
            return "Error: pattern is required"
        try:
            limit = max(1, int(max_matches))
        except (TypeError, ValueError):
            return "Error: max_matches must be an integer"
        try:
            regex = re.compile(str(pattern), 0 if case_sensitive else re.IGNORECASE)
        except re.error as exc:
            return f"Error: Invalid regex: {exc}"

        display_path = path or "."
        try:
            target = self._resolve(display_path)
        except PermissionError as exc:
            return self._error(exc)

        if not target.exists():
            return f"Error: Path not found: {display_path}"

        if target.is_file():
            candidates = [] if self._is_ignored(target) else [target]
        elif target.is_dir():
            candidates = self._recursive_files(target)
        else:
            return f"Error: Not a file or directory: {display_path}"

        matches: list[str] = []
        for file_path in candidates:
            try:
                text = file_path.read_bytes().decode("utf-8").replace("\r\n", "\n")
            except (OSError, UnicodeDecodeError):
                continue
            relative = self._relative_path(file_path)
            for line_no, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    matches.append(f"{relative}:{line_no}|{line}")

        if not matches:
            return f"(No matches for pattern in {display_path})"

        shown = matches[:limit]
        result = "\n".join(shown)
        if len(matches) > limit:
            result += f"\n[truncated: showing {limit} of {len(matches)} matches]"
        return result


class GlobTool(_FilesystemTool):
    @property
    def name(self) -> str:
        return "glob"

    @property
    def description(self) -> str:
        return "List workspace-relative files and directories matching a glob pattern."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern such as *.py or **/*.md"},
                "path": {"type": "string", "description": "Directory path to search from"},
                "max_matches": {"type": "integer", "description": "Maximum matches to return"},
            },
            "required": ["pattern"],
        }

    async def execute(
        self,
        pattern: str | None = None,
        path: str | None = ".",
        max_matches: int = GLOB_DEFAULT_MAX_MATCHES,
        **kwargs: Any,
    ) -> str:
        if pattern is None or str(pattern) == "":
            return "Error: pattern is required"
        try:
            limit = max(1, int(max_matches))
        except (TypeError, ValueError):
            return "Error: max_matches must be an integer"

        display_path = path or "."
        try:
            directory = self._resolve(display_path)
        except PermissionError as exc:
            return self._error(exc)

        if not directory.exists():
            return f"Error: Directory not found: {display_path}"
        if not directory.is_dir():
            return f"Error: Not a directory: {display_path}"

        try:
            raw_matches = list(directory.glob(str(pattern)))
        except (NotImplementedError, ValueError) as exc:
            return f"Error: Invalid glob pattern: {exc}"

        safe_matches = sorted(
            (
                candidate
                for candidate in raw_matches
                if not self._is_ignored(candidate) and self._is_safe_existing_path(candidate)
            ),
            key=lambda item: self._relative_path(item),
        )
        matches = [self._format_glob_match(candidate) for candidate in safe_matches]
        if not matches:
            return f"(No matches for pattern in {display_path})"

        shown = matches[:limit]
        result = "\n".join(shown)
        if len(matches) > limit:
            result += f"\n[truncated: showing {limit} of {len(matches)} matches]"
        return result

    def _format_glob_match(self, path: Path) -> str:
        relative = self._relative_path(path)
        return f"{relative}/" if path.is_dir() else relative


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
