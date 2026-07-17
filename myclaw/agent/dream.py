from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
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
from myclaw.observability import ObservabilityConfig, ObservabilityRuntime, current_trace_context

logger = logging.getLogger(__name__)

_HISTORY_ENTRY_MAX_CHARS = 4_000
_PHASE2_MAX_ITERATIONS = 15
_CHECKLIST_PATTERN = re.compile(r"^\[(SOUL|USER|MEMORY)(-REMOVE)?\]\s+(.+?)\s*$")
_SOURCE_TAG_PATTERN = re.compile(r"⟨(\d+)⟩")
_TARGET_PRIORITY = {"SOUL": 0, "USER": 1, "MEMORY": 2}

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
    "Classification rules (the three files are mutually exclusive):\n"
    "1. [MEMORY] is the only destination for projects, repositories, tasks, technical designs, "
    "runtime configuration, scheduled jobs, operational state, business context, and decisions.\n"
    "2. [USER] is only for stable, cross-project user identity, habits, and communication or "
    "working preferences. Never put project-specific information in USER.md.\n"
    "3. [SOUL] is only for persona, tone, or lasting assistant behaviour that the user explicitly "
    "asked the assistant to adopt. Never infer SOUL facts from how one task was performed, and "
    "never put user or project information in SOUL.md.\n"
    "4. Skip temporary status, one-off requests, execution output, and uncertain facts.\n"
    "Apply this routing order to every candidate: project/task/technical/operational -> MEMORY; "
    "otherwise stable cross-project user fact -> USER; otherwise explicit assistant-persona rule "
    "-> SOUL; otherwise skip. One fact must have exactly one authoritative destination.\n\n"
    "Examples:\n"
    "- 'A cron job reports CPU and memory every 10 minutes' -> MEMORY only, not USER or SOUL.\n"
    "- 'The user is developing MyClaw with a Python agent loop' -> MEMORY only.\n"
    "- 'The user prefers reviewing plans before implementation' -> USER.\n"
    "- 'The assistant should always answer calmly and directly' -> SOUL only when explicitly requested.\n\n"
    "Consolidation and migration rules:\n"
    "- Extract only durable, reusable facts. Skip transient chatter.\n"
    "- Actively audit all three current files for wrong-file and semantic duplicates. If a fact "
    "already exists in its correct file, do not re-add it and keep that authoritative copy.\n"
    "- Remove redundant or superseded copies from other files. For a project/task fact incorrectly "
    "stored in USER.md or SOUL.md, keep/add the MEMORY.md copy and emit the corresponding REMOVE lines.\n"
    "- Never remove a wrong-file fact unless an equivalent authoritative copy already exists or a "
    "new [MEMORY] line in this checklist is supported by a supplied source id.\n"
    "- Stable user identity/preferences and explicit persona rules do not expire merely because they are old.\n"
    "- Only prune things that are objectively obsolete (resolved issues, superseded decisions).\n"
    "- For [MEMORY] facts, always append the source id in angle brackets, e.g. ⟨42⟩.\n"
    "- If nothing should change, output exactly: (nothing)"
)

_PHASE2_SYSTEM = (
    "You are editing the assistant's long-term memory files using file tools. "
    "The memory directory contains SOUL.md, USER.md, and MEMORY.md. You are given a "
    "validated consolidation checklist. Apply it precisely without reclassifying facts:\n"
    "- Use read_file to inspect a file, then edit_file for surgical incremental edits. "
    "Use write_file ONLY to create a file that does not exist yet.\n"
    "- NEVER rewrite a whole file from scratch; make minimal additions/removals.\n"
    "- Add [MEMORY] facts to MEMORY.md keeping their ⟨id⟩ source tag verbatim.\n"
    "- Add [USER] facts to USER.md and [SOUL] facts to SOUL.md.\n"
    "- For *-REMOVE lines, delete the matching line(s) from the named file.\n"
    "- Keep each file a tidy markdown bullet list. When done, reply with a one-line summary."
)


@dataclass(frozen=True)
class _ChecklistInstruction:
    line: str
    target: str
    action: str
    normalized_fact: str


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
        observability: ObservabilityRuntime | None = None,
    ) -> None:
        self.session_manager = session_manager
        self.provider = provider
        self.memory_store = memory_store
        self.config = config
        self.model = model
        self.observability = observability or ObservabilityRuntime(
            session_manager.workspace, ObservabilityConfig(enabled=False)
        )
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
            scope = (
                self.observability.span("dream.run", "background")
                if current_trace_context() is not None
                else self.observability.trace("dream.run", "dream", model=self.model)
            )
            with scope as span:
                changed = await self._run_once()
                span.set_attribute("changed", changed)
                return changed
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
            raw_checklist = await self._phase1_analyze(entries)
            checklist = self._validate_checklist(
                raw_checklist,
                {int(entry["id"]) for entry in entries},
            )
            if checklist != "(nothing)":
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

    def _validate_checklist(self, checklist: str, valid_source_ids: set[int]) -> str:
        if checklist.strip().lower() == "(nothing)":
            return "(nothing)"

        instructions: list[_ChecklistInstruction] = []
        invalid_memory_adds: set[str] = set()
        rejected = 0
        for raw_line in checklist.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            instruction = self._parse_instruction(line, valid_source_ids)
            if instruction is None:
                match = _CHECKLIST_PATTERN.fullmatch(line)
                if match and match.group(1) == "MEMORY" and match.group(2) is None:
                    invalid_memory_adds.add(self._normalize_fact(match.group(3)))
                rejected += 1
                continue
            instructions.append(instruction)

        actions_by_fact = {}
        for instruction in instructions:
            key = (instruction.target, instruction.normalized_fact)
            actions_by_fact.setdefault(key, set()).add(instruction.action)
        conflicts = {key for key, actions in actions_by_fact.items() if len(actions) > 1}

        preferred_add_targets = {}
        for instruction in instructions:
            if instruction.action != "add":
                continue
            current = preferred_add_targets.get(instruction.normalized_fact)
            if current is None or _TARGET_PRIORITY[instruction.target] > _TARGET_PRIORITY[current]:
                preferred_add_targets[instruction.normalized_fact] = instruction.target

        kept: list[str] = []
        seen: set[tuple[str, str, str]] = set()
        for instruction in instructions:
            key = (instruction.target, instruction.normalized_fact)
            signature = (instruction.action, instruction.target, instruction.normalized_fact)
            if key in conflicts or signature in seen:
                rejected += 1
                continue
            if (
                instruction.action == "add"
                and preferred_add_targets[instruction.normalized_fact] != instruction.target
            ):
                rejected += 1
                continue
            if (
                instruction.action == "remove"
                and instruction.target in {"USER", "SOUL"}
                and instruction.normalized_fact in invalid_memory_adds
            ):
                rejected += 1
                continue
            seen.add(signature)
            kept.append(instruction.line)

        if rejected:
            logger.warning("Dream checklist validation discarded %d invalid or conflicting line(s)", rejected)
        return "\n".join(kept) if kept else "(nothing)"

    @staticmethod
    def _parse_instruction(
        line: str,
        valid_source_ids: set[int],
    ) -> _ChecklistInstruction | None:
        match = _CHECKLIST_PATTERN.fullmatch(line)
        if match is None:
            return None

        target, remove_suffix, payload = match.groups()
        action = "remove" if remove_suffix else "add"
        if action == "remove":
            fact, separator, reason = payload.partition(" -- ")
            if not separator or not fact.strip() or not reason.strip():
                return None
        else:
            fact = payload

        source_ids = {int(value) for value in _SOURCE_TAG_PATTERN.findall(fact)}
        if action == "add" and target == "MEMORY":
            if not source_ids or not source_ids.issubset(valid_source_ids):
                return None
        elif action == "add" and source_ids:
            return None

        normalized_fact = DreamManager._normalize_fact(fact)
        if not normalized_fact:
            return None
        return _ChecklistInstruction(line, target, action, normalized_fact)

    @staticmethod
    def _normalize_fact(fact: str) -> str:
        without_sources = _SOURCE_TAG_PATTERN.sub("", fact)
        return re.sub(r"[\W_]+", "", without_sources.casefold())

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
