from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from myclaw.agent.runner import AgentRunner
from myclaw.agent.types import AgentConfig, AgentRunSpec, Message
from myclaw.memory import MemoryGit, MemoryStore
from myclaw.providers.base import LLMProvider, LLMResponse
from myclaw.session import SessionManager
from myclaw.tools import ToolRegistry
from myclaw.tools.base import ToolRuntimeContext
from myclaw.tools.filesystem import EditFileTool, ReadFileTool, WriteFileTool

logger = logging.getLogger(__name__)

_HISTORY_ENTRY_MAX_CHARS = 4_000
_PHASE2_MAX_ITERATIONS = 15

_PHASE1_SYSTEM = (
    "You consolidate long-term memory for a personal assistant. You are given new "
    "conversation-history entries (each tagged with a source id) and the current "
    "contents of three memory files: SOUL.md (the assistant's persona/tone), "
    "USER.md (the user's identity/preferences), and MEMORY.md (project knowledge "
    "and facts).\n\n"
    "Produce a plain-text checklist, one instruction per line, using these forms:\n"
    "  [SOUL] <fact>            add a persona/behaviour fact to SOUL.md\n"
    "  [USER] <fact>            add a user identity/preference fact to USER.md\n"
    "  [MEMORY] <fact> ⟨id⟩   add a project fact to MEMORY.md, tagging its source id\n"
    "  [SOUL-REMOVE] <text> -- <reason>\n"
    "  [USER-REMOVE] <text> -- <reason>\n"
    "  [MEMORY-REMOVE] <text> -- <reason>\n\n"
    "Rules:\n"
    "- Extract only durable, reusable facts. Skip transient chatter.\n"
    "- Actively de-duplicate: if a fact already exists in any memory file, do NOT re-add it; "
    "instead emit a *-REMOVE line for redundant or superseded existing entries.\n"
    "- User preferences, identity, and persona are permanent regardless of age.\n"
    "- Only prune things that are objectively obsolete (resolved issues, superseded decisions).\n"
    "- For [MEMORY] facts, always append the source id in angle brackets, e.g. ⟨42⟩.\n"
    "- If nothing should change, output exactly: (nothing)"
)

_PHASE2_SYSTEM = (
    "You are editing the assistant's long-term memory files using file tools. "
    "The memory directory contains SOUL.md, USER.md, and MEMORY.md. You are given a "
    "consolidation checklist. Apply it precisely:\n"
    "- Use read_file to inspect a file, then edit_file for surgical incremental edits. "
    "Use write_file ONLY to create a file that does not exist yet.\n"
    "- NEVER rewrite a whole file from scratch; make minimal additions/removals.\n"
    "- Add [MEMORY] facts to MEMORY.md keeping their ⟨id⟩ source tag verbatim.\n"
    "- Add [USER] facts to USER.md and [SOUL] facts to SOUL.md.\n"
    "- For *-REMOVE lines, delete the matching line(s) from the named file.\n"
    "- Keep each file a tidy markdown bullet list. When done, reply with a one-line summary."
)


class DreamManager:
    """Periodically consolidate the history stream into long-term memory files.

    Two phases, run off the main conversation loop on a timer:
      Phase 1 — a tool-free LLM call analyses unconsumed history + current memory
                files and emits a line-by-line consolidation checklist.
      Phase 2 — a sub-agent equipped with file tools applies the checklist with
                incremental edits to SOUL.md / USER.md / MEMORY.md.
    The cursor always advances past the consumed batch, so a failed Phase 2 never
    wedges the system on the same entries.
    """

    def __init__(
        self,
        session_manager: SessionManager,
        provider: LLMProvider,
        memory_store: MemoryStore,
        config: AgentConfig,
        *,
        model: str,
    ) -> None:
        self.session_manager = session_manager
        self.provider = provider
        self.memory_store = memory_store
        self.config = config
        self.model = model
        self.runner = AgentRunner(provider)
        self.memory_dir = memory_store.memory_dir
        self.git = MemoryGit(self.memory_dir)
        self.cursor_path = self.memory_dir / ".dream_cursor"
        self.running = False

    @property
    def enabled(self) -> bool:
        return self.config.dream_interval_minutes > 0

    def should_run_now(self, now: datetime | None = None) -> bool:
        if not self.enabled or self.running:
            return False
        cursor = self._read_cursor()
        if not self.memory_store.read_history_since(cursor["last_id"]):
            return False
        last_run = self._parse_datetime(cursor.get("last_run_at"))
        if last_run is None:
            return True
        elapsed = ((now or datetime.now()) - last_run).total_seconds()
        return elapsed >= self.config.dream_interval_minutes * 60

    async def run_once(self) -> bool:
        if self.running:
            return False
        self.running = True
        try:
            return await self._run_once()
        except Exception:
            logger.exception("Dream consolidation failed")
            return False
        finally:
            self.running = False

    async def _run_once(self) -> bool:
        cursor = self._read_cursor()
        entries = self.memory_store.read_history_since(cursor["last_id"])
        if not entries:
            return False
        last_id = max(entry["id"] for entry in entries)
        try:
            checklist = await self._phase1_analyze(entries)
            if checklist and checklist.strip().lower() != "(nothing)":
                await self._phase2_apply(checklist)
                await self._commit_memory(checklist)
        finally:
            # The cursor always advances so we never re-chew the same batch,
            # even if a phase failed midway (mirrors nanobot's Dream).
            self._write_cursor(last_id)
        return True

    async def _commit_memory(self, checklist: str) -> None:
        """Commit the memory files so each consolidation is auditable/revertible.

        Best-effort: MemoryGit swallows its own errors, so a missing or failing
        git never disrupts the consolidation flow.
        """
        if not await self.git.ensure_repo():
            return
        count = await self.git.changed_count()
        if count == 0:
            return
        title = f"dream: {datetime.now().isoformat(timespec='seconds')}, {count} change(s)"
        await self.git.commit_all(title, checklist)

    async def _phase1_analyze(self, entries: list[dict]) -> str:
        history_block = "\n".join(
            f"⟨{entry['id']}⟩ {str(entry.get('content', ''))[:_HISTORY_ENTRY_MAX_CHARS]}"
            for entry in entries
        )
        memory_block = (
            f"=== SOUL.md ===\n{self.memory_store.read_soul() or '(empty)'}\n\n"
            f"=== USER.md ===\n{self.memory_store.read_user() or '(empty)'}\n\n"
            f"=== MEMORY.md ===\n{self.memory_store.read_memory() or '(empty)'}"
        )
        messages: list[Message] = [
            {"role": "system", "content": _PHASE1_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"New history entries to consolidate:\n{history_block}\n\n"
                    f"Current memory files:\n{memory_block}"
                ),
            },
        ]
        response = await self.provider.complete(messages)
        return self._response_text(response)

    async def _phase2_apply(self, checklist: str) -> None:
        registry = ToolRegistry()
        registry.register(ReadFileTool(self.memory_dir))
        registry.register(EditFileTool(self.memory_dir))
        registry.register(WriteFileTool(self.memory_dir))
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        messages: list[Message] = [
            {"role": "system", "content": _PHASE2_SYSTEM},
            {"role": "user", "content": f"Consolidation checklist:\n{checklist}"},
        ]
        await self.runner.run(
            AgentRunSpec(
                messages=messages,
                model=self.model,
                max_iterations=_PHASE2_MAX_ITERATIONS,
                tools=registry,
                max_tool_result_chars=self.config.max_tool_result_chars,
                tool_context=ToolRuntimeContext(
                    session_key="dream",
                    channel="dream",
                    workspace=self.memory_dir,
                    tool_names=sorted(registry.tool_names),
                ),
            )
        )

    @staticmethod
    def _response_text(response: str | LLMResponse) -> str:
        if isinstance(response, LLMResponse):
            return response.content
        return response if isinstance(response, str) else ""

    def _read_cursor(self) -> dict:
        if not self.cursor_path.exists():
            return {"last_id": -1, "last_run_at": None}
        try:
            data = json.loads(self.cursor_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"last_id": -1, "last_run_at": None}
        if not isinstance(data, dict) or not isinstance(data.get("last_id"), int):
            return {"last_id": -1, "last_run_at": None}
        return data

    def _write_cursor(self, last_id: int) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        payload = {"last_id": last_id, "last_run_at": datetime.now().isoformat()}
        self.cursor_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    @staticmethod
    def _parse_datetime(value: object) -> datetime | None:
        if not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
