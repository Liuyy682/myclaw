"""Restricted, deterministic tools used by eval cases.

The normal MyClaw registry intentionally contains powerful local-development
tools.  Evaluation cases must not inherit those capabilities, so this module
registers only file operations rooted below a per-run temporary directory.
"""

from __future__ import annotations

import json
import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any

from myclaw.cron import CronStore
from myclaw.memory import MemoryStore
from myclaw.tasks import TaskStore
from myclaw.tools import (
    AskUserTool,
    CronTool,
    FunctionTool,
    MemoryWriteTool,
    SpawnTool,
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskUpdateTool,
    ToolRegistry,
    SecurityStore,
)
from myclaw.tools.base import ToolRuntimeContext


class FixturePathError(ValueError):
    """Raised when a model-supplied fixture path escapes the run root."""


def resolve_fixture_path(root: Path, relative_path: str) -> Path:
    """Resolve a relative fixture path while rejecting traversal/symlinks."""

    if not isinstance(relative_path, str) or not relative_path.strip():
        raise FixturePathError("path must be a non-empty relative string")
    candidate = Path(relative_path)
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        raise FixturePathError("path must stay inside the fixture root")
    root = root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise FixturePathError("path must stay inside the fixture root") from exc
    # A pre-existing symlink can otherwise point outside the root.  ``resolve``
    # above catches existing links; rejecting links in the final path also
    # prevents a model from replacing a safe file with an escape link.
    current = root
    for part in candidate.parts:
        current = current / part
        if current.is_symlink():
            raise FixturePathError("symlinks are not allowed in the fixture")
    return resolved


def _text_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class EvalToolRegistry:
    """Delegate a restricted registry and inject one deterministic fault."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        fixture_root: Path,
        cancel_on_tool_call: int | None = None,
        fault_phase: str = "before",
    ) -> None:
        if cancel_on_tool_call is not None and cancel_on_tool_call < 1:
            raise ValueError("cancel_on_tool_call must be at least 1")
        if fault_phase not in {"before", "after"}:
            raise ValueError("fault_phase must be 'before' or 'after'")
        self._registry = registry
        self._fixture_root = fixture_root.resolve()
        self.cancel_on_tool_call = cancel_on_tool_call
        self.fault_phase = fault_phase
        self.call_count = 0
        self.injected = False
        self.injected_call: int | None = None

    @property
    def tool_names(self) -> list[str]:
        return sorted(self._registry.tool_names)

    def get(self, name: str) -> Any:
        return self._registry.get(name)

    def has(self, name: str) -> bool:
        return self._registry.has(name)

    def definitions(self) -> list[dict[str, Any]]:
        return self._registry.definitions()

    def prepare_call(self, request):
        return self._registry.prepare_call(request)

    async def execute(self, request, *, max_result_chars=None, context=None) -> str:
        self.call_count += 1
        should_inject = (
            self.cancel_on_tool_call is not None
            and not self.injected
            and self.call_count == self.cancel_on_tool_call
        )
        if should_inject and self.fault_phase == "before":
            self.injected = True
            self.injected_call = self.call_count
            raise asyncio.CancelledError(
                f"eval fault: cancelled before tool call {self.call_count}"
            )
        base_context = context or ToolRuntimeContext()
        scoped_context = replace(
            base_context,
            workspace=self._fixture_root,
            subject=base_context.subject or "internal:eval",
            security_subject=base_context.security_subject or "internal:eval",
            approval_callback=lambda *_: "allow",
            allowed_tools=self.tool_names,
            resource_scopes=[str(self._fixture_root)],
        )
        result = await self._registry.execute(
            request,
            max_result_chars=max_result_chars,
            context=scoped_context,
        )
        if should_inject and self.fault_phase == "after":
            self.injected = True
            self.injected_call = self.call_count
            raise asyncio.CancelledError(
                f"eval fault: cancelled after tool call {self.call_count}"
            )
        return result

    def unregister(self, name: str) -> None:
        self._registry.unregister(name)

    def __len__(self) -> int:
        return len(self._registry)

    def __contains__(self, name: str) -> bool:
        return name in self._registry


def build_fixture_registry(
    root: Path,
    *,
    workspace: Path | None = None,
    disabled_tools: list[str] | None = None,
    cancel_on_tool_call: int | None = None,
    fault_phase: str = "before",
) -> EvalToolRegistry:
    """Build restricted fixture tools plus real persistent MyClaw tools.

    File tools are rooted at ``root``.  Task, cron and memory stores use the
    temporary ``workspace`` root, never the user's normal workspace.
    """

    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    def read_file(path: str, max_chars: int = 12000) -> str:
        target = resolve_fixture_path(root, path)
        if not target.is_file():
            raise FileNotFoundError(path)
        if max_chars < 1:
            raise ValueError("max_chars must be positive")
        return target.read_text(encoding="utf-8")[:max_chars]

    def write_file(path: str, content: Any) -> str:
        target = resolve_fixture_path(root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        text = _text_value(content)
        target.write_text(text, encoding="utf-8")
        return f"wrote {path} ({len(text)} chars)"

    def append_file(path: str, content: Any) -> str:
        target = resolve_fixture_path(root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        text = _text_value(content)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(text)
        return f"appended {path} ({len(text)} chars)"

    def list_dir(path: str = ".", recursive: bool = False, max_entries: int = 200) -> list[str]:
        target = resolve_fixture_path(root, path) if path != "." else root
        if not target.is_dir():
            raise NotADirectoryError(path)
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        entries = target.rglob("*") if recursive else target.iterdir()
        names: list[str] = []
        for entry in sorted(entries, key=lambda item: str(item)):
            if entry.is_symlink():
                continue
            names.append(entry.relative_to(root).as_posix())
            if len(names) >= max_entries:
                break
        return names

    state_root = Path(workspace or root).expanduser().resolve()
    registry = ToolRegistry(SecurityStore(state_root))
    registry.register(
        FunctionTool(
            name="read_file",
            description="Read UTF-8 text from a file inside the evaluation fixture.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative fixture path."},
                    "max_chars": {"type": "integer", "minimum": 1, "default": 12000},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            func=read_file,
            read_only=True,
            effect="local_read",
        )
    )
    registry.register(
        FunctionTool(
            name="write_file",
            description="Write UTF-8 text to a file inside the evaluation fixture.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative fixture path."},
                    "content": {"description": "Text or JSON-compatible value to write."},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            func=write_file,
            effect="local_write",
        )
    )
    registry.register(
        FunctionTool(
            name="append_file",
            description="Append UTF-8 text to a file inside the evaluation fixture.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative fixture path."},
                    "content": {"description": "Text or JSON-compatible value to append."},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            func=append_file,
            effect="local_write",
        )
    )
    registry.register(
        FunctionTool(
            name="list_dir",
            description="List files and directories inside the evaluation fixture.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                    "recursive": {"type": "boolean", "default": False},
                    "max_entries": {"type": "integer", "minimum": 1, "default": 200},
                },
                "additionalProperties": False,
            },
            func=list_dir,
            read_only=True,
            effect="local_read",
        )
    )
    task_store = TaskStore(state_root)
    cron_store = CronStore(state_root)
    registry.register(AskUserTool())
    registry.register(CronTool(cron_store))
    registry.register(MemoryWriteTool(MemoryStore(state_root)))
    registry.register(SpawnTool())
    registry.register(TaskCreateTool(task_store))
    registry.register(TaskGetTool(task_store))
    registry.register(TaskListTool(task_store))
    registry.register(TaskUpdateTool(task_store))

    for name in disabled_tools or []:
        registry.unregister(str(name))
    return EvalToolRegistry(
        registry,
        fixture_root=root,
        cancel_on_tool_call=cancel_on_tool_call,
        fault_phase=fault_phase,
    )


def write_fixture_setup(root: Path, files: dict[str, Any], directories: list[str]) -> None:
    """Materialise initial fixture files and directories safely."""

    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for directory in directories:
        resolve_fixture_path(root, directory).mkdir(parents=True, exist_ok=True)
    for relative_path, value in files.items():
        target = resolve_fixture_path(root, str(relative_path))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_text_value(value), encoding="utf-8")


def snapshot_workspace(root: Path) -> dict[str, str]:
    """Return deterministic UTF-8 file contents below any temp root."""

    root = root.resolve()
    files: dict[str, str] = {}
    if not root.exists():
        return files
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            path.relative_to(root)
            files[path.relative_to(root).as_posix()] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            # Fixture cases are text-oriented.  Do not make binary or
            # transient files abort a completed run; represent them clearly.
            files[path.relative_to(root).as_posix()] = "<unreadable fixture file>"
    return files


def snapshot_fixture(root: Path) -> dict[str, str]:
    """Return deterministic UTF-8 fixture file contents."""

    return snapshot_workspace(root)


__all__ = [
    "FixturePathError",
    "EvalToolRegistry",
    "build_fixture_registry",
    "resolve_fixture_path",
    "snapshot_fixture",
    "snapshot_workspace",
    "write_fixture_setup",
]
