"""Runtime input models for built-in tools.

These models intentionally describe only the public argument types and
requiredness already exposed by each tool's hand-written OpenAI schema.  Tool
business rules remain in the corresponding ``execute`` methods.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from myclaw.config import (
    GLOB_DEFAULT_MAX_MATCHES,
    GREP_DEFAULT_MAX_MATCHES,
    LIST_DIR_DEFAULT_MAX_ENTRIES,
)


class ToolInputModel(BaseModel):
    """Common Pydantic configuration for built-in tool arguments."""

    model_config = ConfigDict(extra="forbid", coerce_numbers_to_str=True)


class AskUserInput(ToolInputModel):
    question: str
    choices: list[str] | None = None


class CronInput(ToolInputModel):
    name: str
    prompt: str
    every_seconds: int | None = None
    at: str | None = None
    cron: str | None = None
    session_key: str | None = None
    # A scheduled run carries an explicit least-privilege tool/resource
    # scope.  ``None`` is kept distinct from an empty list so old jobs can be
    # recognised and downgraded by the dispatcher.
    allowed_tools: list[str] | None = None
    resource_scopes: dict[str, Any] | list[str] | None = None


class EditFileInput(ToolInputModel):
    path: str
    old_text: str
    new_text: str


class ExecInput(ToolInputModel):
    cmd: str
    cwd: str | None = None
    timeout_seconds: int = 30
    max_output_chars: int = 12000
    allow_network: bool = False


class ReadFileInput(ToolInputModel):
    path: str
    offset: int = 1
    limit: int | None = None


class ListDirInput(ToolInputModel):
    path: str = "."
    recursive: bool = False
    max_entries: int = LIST_DIR_DEFAULT_MAX_ENTRIES


class GrepInput(ToolInputModel):
    pattern: str
    path: str = "."
    case_sensitive: bool = True
    max_matches: int = GREP_DEFAULT_MAX_MATCHES


class GlobInput(ToolInputModel):
    pattern: str
    path: str = "."
    max_matches: int = GLOB_DEFAULT_MAX_MATCHES


class WriteFileInput(ToolInputModel):
    path: str
    content: str


class NotebookEditInput(ToolInputModel):
    path: str
    cell_index: int
    source: str
    cell_type: str = "code"


class MemoryWriteInput(ToolInputModel):
    content: str


class RecallInput(ToolInputModel):
    id: str


class MessageInput(ToolInputModel):
    content: str
    channel: str | None = None
    chat_id: str | None = None


class MyInput(ToolInputModel):
    pass


class SpawnInput(ToolInputModel):
    prompt: str
    name: str | None = None


class TaskCreateInput(ToolInputModel):
    title: str
    description: str = ""
    approach: str = ""
    acceptance_criteria: str = ""
    depends_on: list[str] | None = None


class TaskListInput(ToolInputModel):
    plan_id: str | None = None
    status: str | None = None
    legacy: bool = False


class TaskGetInput(ToolInputModel):
    id: str
    legacy: bool = False


class TaskUpdateInput(ToolInputModel):
    id: str
    title: str | None = None
    description: str | None = None
    approach: str | None = None
    acceptance_criteria: str | None = None
    depends_on: list[str] | None = None
    cancel: bool = False


class TaskProgressInput(ToolInputModel):
    id: str
    status: str | None = None
    # Progress is an execution report or a short evidence note.
    progress: str | None = None


class WebFetchInput(ToolInputModel):
    url: str
    max_chars: int = 6000


class WebSearchInput(ToolInputModel):
    query: str
    max_results: int = 5


class SkillLoadInput(ToolInputModel):
    name: str


__all__ = [
    "AskUserInput",
    "CronInput",
    "EditFileInput",
    "ExecInput",
    "GlobInput",
    "GrepInput",
    "ListDirInput",
    "MemoryWriteInput",
    "MessageInput",
    "MyInput",
    "NotebookEditInput",
    "ReadFileInput",
    "RecallInput",
    "SkillLoadInput",
    "SpawnInput",
    "TaskCreateInput",
    "TaskGetInput",
    "TaskListInput",
    "TaskProgressInput",
    "TaskUpdateInput",
    "ToolInputModel",
    "WebFetchInput",
    "WebSearchInput",
    "WriteFileInput",
]
