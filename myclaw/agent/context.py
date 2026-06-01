from __future__ import annotations

from collections.abc import Callable
import json
import math
from datetime import datetime
from typing import Any

from myclaw.agent.types import AgentConfig, Message
from myclaw.config import FAKE_PROVIDER_MODEL, TOOL_RESULT_TRUNCATED_TEMPLATE

CONTEXT_SUMMARY_METADATA_KEY = "context_summary"
SUMMARY_MESSAGE_PREFIX = "Summary of earlier conversation:"
_SUMMARY_SYSTEM_PROMPT = (
    "Summarize older conversation turns for future context. "
    "Preserve user goals, decisions, files, tool outcomes, and open tasks. "
    "Return concise plain text only."
)
_OMISSION_NOTE = "Older raw conversation turns were compressed to fit the context budget."
_ENCODING_UNSET = object()

MISSING_TOOL_RESULT_CONTENT = "[tool result unavailable in persisted history]"


class TokenEstimator:
    """Estimate chat-message tokens with tiktoken when available."""

    def __init__(self, model: str, encoding: Any = _ENCODING_UNSET) -> None:
        self.model = model
        self.encoding = self._load_encoding(model) if encoding is _ENCODING_UNSET else encoding

    def estimate_messages(self, messages: list[Message]) -> int:
        return 2 + sum(self.estimate_message(message) for message in messages)

    def estimate_message(self, message: Message) -> int:
        tokens = 4
        for key in ("role", "content", "name", "tool_call_id"):
            value = message.get(key)
            if isinstance(value, str):
                tokens += self.estimate_text(value)
        if "tool_calls" in message:
            tokens += self.estimate_text(json.dumps(message["tool_calls"], ensure_ascii=False))
        return tokens

    def estimate_text(self, text: str) -> int:
        if not text:
            return 0
        if self.encoding is not None:
            try:
                return len(self.encoding.encode(text))
            except Exception:
                pass
        ascii_chars = sum(1 for char in text if ord(char) < 128)
        non_ascii_chars = len(text) - ascii_chars
        return math.ceil(ascii_chars / 4) + non_ascii_chars

    @staticmethod
    def _load_encoding(model: str) -> Any:
        try:
            import tiktoken
        except ImportError:
            return None

        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")


class ContextBudgetManager:
    """Compress old session turns into a reusable session summary."""

    def __init__(self, provider: Any, context_builder: Any) -> None:
        self.provider = provider
        self.context_builder = context_builder

    async def ensure_budget(
        self,
        session: Any,
        config: AgentConfig,
        current_user_text: str,
        *,
        model: str,
        memory_text: str | None = None,
        archive_history: Callable[[str], None] | None = None,
    ) -> bool:
        estimator = TokenEstimator(model)
        updated = False

        while True:
            summary = self.summary_metadata(session.messages, session.metadata.get(CONTEXT_SUMMARY_METADATA_KEY))
            prompt_messages = self.context_builder.build_messages(
                config,
                session.messages,
                current_user_text,
                context_summary=summary,
                memory_text=memory_text,
            )
            if self._within_budget(prompt_messages, config, estimator):
                return updated

            covered_count = self._covered_count(summary, len(session.messages))
            next_cover_count = self._next_cover_count(session.messages, covered_count, config)
            if next_cover_count <= covered_count:
                self._store_summary(
                    session,
                    self._summary_with_omission_note(summary, config.context_summary_max_chars),
                    covered_count,
                    estimator,
                )
                return True

            selected_messages = session.messages[covered_count:next_cover_count]
            summary_content = await self._summarize_messages(
                self._summary_content(summary),
                selected_messages,
                config,
                estimator,
            )
            self._store_summary(session, summary_content, next_cover_count, estimator)
            if archive_history is not None:
                archive_history(summary_content)
            updated = True

    @staticmethod
    def summary_metadata(session_messages: list[Message], value: Any) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        content = value.get("content")
        covered_count = value.get("covered_message_count")
        if not isinstance(content, str) or not content.strip() or not isinstance(covered_count, int):
            return None
        return {
            **value,
            "content": content.strip(),
            "covered_message_count": max(0, min(covered_count, len(session_messages))),
        }

    @staticmethod
    def _covered_count(summary: dict[str, Any] | None, message_count: int) -> int:
        if summary is None:
            return 0
        return max(0, min(int(summary.get("covered_message_count", 0)), message_count))

    @staticmethod
    def _summary_content(summary: dict[str, Any] | None) -> str:
        if summary is None:
            return ""
        content = summary.get("content")
        return content.strip() if isinstance(content, str) else ""

    @staticmethod
    def _within_budget(messages: list[Message], config: AgentConfig, estimator: TokenEstimator) -> bool:
        return (
            len(messages) <= config.max_context_messages
            and estimator.estimate_messages(messages) <= config.max_context_tokens
        )

    def _next_cover_count(self, messages: list[Message], covered_count: int, config: AgentConfig) -> int:
        if covered_count >= len(messages):
            return covered_count

        target_recent = max(1, config.max_context_messages // 2)
        desired_boundary = max(covered_count + 1, len(messages) - target_recent)
        boundaries = self._turn_boundaries(messages, covered_count)
        for boundary in boundaries:
            if boundary >= desired_boundary:
                return boundary
        return boundaries[-1] if boundaries else len(messages)

    @staticmethod
    def _turn_boundaries(messages: list[Message], covered_count: int) -> list[int]:
        boundaries = [
            index
            for index in range(covered_count + 1, len(messages))
            if messages[index].get("role") == "user"
        ]
        boundaries.append(len(messages))
        return [boundary for boundary in boundaries if boundary > covered_count]

    async def _summarize_messages(
        self,
        existing_summary: str,
        messages: list[Message],
        config: AgentConfig,
        estimator: TokenEstimator,
    ) -> str:
        if getattr(self.provider, "model", "") == FAKE_PROVIDER_MODEL:
            return self._fallback_summary(existing_summary, messages, config.context_summary_max_chars)

        summary = existing_summary
        for chunk in self._message_chunks(messages, config.context_summary_chunk_tokens, estimator):
            try:
                response = await self.provider.complete(
                    self._summary_prompt(summary, chunk, config.context_summary_max_chars),
                    tools=None,
                )
                content = self._response_content(response)
            except Exception:
                return self._fallback_summary(existing_summary, messages, config.context_summary_max_chars)
            if not content:
                return self._fallback_summary(existing_summary, messages, config.context_summary_max_chars)
            summary = self._truncate(content, config.context_summary_max_chars)
        return summary or self._fallback_summary(existing_summary, messages, config.context_summary_max_chars)

    def _message_chunks(
        self,
        messages: list[Message],
        token_limit: int,
        estimator: TokenEstimator,
    ) -> list[list[Message]]:
        chunks: list[list[Message]] = []
        current: list[Message] = []
        current_tokens = 0

        for message in messages:
            message_tokens = estimator.estimate_message(message)
            if current and current_tokens + message_tokens > token_limit:
                chunks.append(current)
                current = []
                current_tokens = 0
            current.append(message)
            current_tokens += message_tokens

        if current:
            chunks.append(current)
        return chunks

    def _summary_prompt(self, existing_summary: str, messages: list[Message], max_chars: int) -> list[Message]:
        existing = existing_summary if existing_summary else "(none)"
        transcript = self._transcript(messages)
        return [
            {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Existing summary:\n{existing}\n\n"
                    f"New older turns:\n{transcript}\n\n"
                    f"Update the summary in at most {max_chars} characters."
                ),
            },
        ]

    def _fallback_summary(self, existing_summary: str, messages: list[Message], max_chars: int) -> str:
        parts = []
        if existing_summary:
            parts.append(existing_summary)
        if messages:
            parts.append(self._transcript(messages))
        return self._truncate("\n".join(parts), max_chars)

    @staticmethod
    def _response_content(response: Any) -> str:
        if isinstance(response, str):
            return response.strip()
        content = getattr(response, "content", "")
        return content.strip() if isinstance(content, str) else ""

    @staticmethod
    def _transcript(messages: list[Message]) -> str:
        lines = []
        for message in messages:
            role = str(message.get("role") or "message")
            content = str(message.get("content") or "")
            if role == "assistant" and not content and message.get("tool_calls"):
                content = f"[assistant requested tools: {ContextBudgetManager._tool_names(message)}]"
            if role == "tool" and message.get("name"):
                role = f"tool:{message['name']}"
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _tool_names(message: Message) -> str:
        names = []
        for tool_call in message.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                names.append(function["name"])
        return ", ".join(names) if names else "tool"

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        normalized = text.strip()
        if len(normalized) <= max_chars:
            return normalized
        if max_chars <= 3:
            return normalized[:max_chars]
        return normalized[: max_chars - 3].rstrip() + "..."

    def _summary_with_omission_note(self, summary: dict[str, Any] | None, max_chars: int) -> str:
        content = self._summary_content(summary)
        if _OMISSION_NOTE in content:
            return content
        return self._truncate("\n\n".join(part for part in (content, _OMISSION_NOTE) if part), max_chars)

    def _store_summary(
        self,
        session: Any,
        content: str,
        covered_message_count: int,
        estimator: TokenEstimator,
    ) -> None:
        session.metadata[CONTEXT_SUMMARY_METADATA_KEY] = {
            "content": content.strip(),
            "covered_message_count": max(0, min(covered_message_count, len(session.messages))),
            "updated_at": datetime.now().isoformat(),
            "token_estimate": estimator.estimate_text(content.strip()),
        }


class ContextBuilder:
    """Build model-ready messages from persisted session history."""

    def build_messages(
        self,
        config: AgentConfig,
        session_messages: list[Message],
        current_user_text: str,
        *,
        context_summary: dict[str, Any] | None = None,
        memory_text: str | None = None,
    ) -> list[Message]:
        messages = self._initial_messages(config, memory_text)
        summary_message, covered_count = self._summary_message(context_summary, len(session_messages))
        if summary_message is not None:
            messages.append(summary_message)
        messages.extend(self._history_messages(session_messages[covered_count:], config.max_tool_result_chars))
        messages.append({"role": "user", "content": current_user_text})
        return messages

    @staticmethod
    def _summary_message(context_summary: dict[str, Any] | None, session_message_count: int) -> tuple[Message | None, int]:
        if not isinstance(context_summary, dict):
            return None, 0
        content = context_summary.get("content")
        covered_count = context_summary.get("covered_message_count")
        if not isinstance(content, str) or not content.strip() or not isinstance(covered_count, int):
            return None, 0
        covered_count = max(0, min(covered_count, session_message_count))
        return {"role": "system", "content": f"{SUMMARY_MESSAGE_PREFIX}\n{content.strip()}"}, covered_count

    @staticmethod
    def _initial_messages(config: AgentConfig, memory_text: str | None = None) -> list[Message]:
        messages: list[Message] = []
        system_prompt = ContextBuilder._system_prompt(config.system_prompt, memory_text)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(dict(message) for message in config.history)
        return messages

    @staticmethod
    def _system_prompt(system_prompt: str, memory_text: str | None) -> str:
        base = system_prompt.strip()
        memory = memory_text.strip() if isinstance(memory_text, str) else ""
        if not memory:
            return base
        memory_block = f"Long-term memory:\n{memory}"
        return "\n\n".join(part for part in (base, memory_block) if part)

    def _history_messages(self, session_messages: list[Message], max_tool_result_chars: int) -> list[Message]:
        messages = self._drop_orphan_tool_results(
            self._sanitize_history(session_messages, max_tool_result_chars)
        )
        return self._backfill_missing_tool_results(messages)

    def _sanitize_history(self, session_messages: list[Message], max_tool_result_chars: int) -> list[Message]:
        sanitized: list[Message] = []
        for message in session_messages:
            role = message.get("role")
            content = message.get("content")
            if role not in {"user", "assistant", "tool"} or not isinstance(content, str):
                continue

            entry: Message = {"role": role, "content": content}
            if role == "assistant":
                tool_calls = self._copy_tool_calls(message.get("tool_calls"))
                if tool_calls:
                    entry["tool_calls"] = tool_calls
                elif not content:
                    continue
            elif role == "tool":
                tool_call_id = message.get("tool_call_id")
                if not isinstance(tool_call_id, str) or not tool_call_id:
                    continue
                entry["tool_call_id"] = tool_call_id
                name = message.get("name")
                if isinstance(name, str) and name:
                    entry["name"] = name
                entry["content"] = self._truncate_tool_result(content, max_tool_result_chars)

            sanitized.append(entry)
        return sanitized

    @staticmethod
    def _copy_tool_calls(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        return [dict(tool_call) for tool_call in value if isinstance(tool_call, dict)]

    @staticmethod
    def _truncate_tool_result(content: str, max_chars: int) -> str:
        if len(content) <= max_chars:
            return content
        omitted = len(content) - max_chars
        return f"{content[:max_chars]}\n{TOOL_RESULT_TRUNCATED_TEMPLATE.format(omitted=omitted)}"

    @staticmethod
    def _drop_orphan_tool_results(messages: list[Message]) -> list[Message]:
        declared: set[str] = set()
        updated: list[Message] = []
        for message in messages:
            if message.get("role") == "assistant":
                for tool_call in message.get("tool_calls") or []:
                    if isinstance(tool_call, dict) and tool_call.get("id"):
                        declared.add(str(tool_call["id"]))
                updated.append(message)
                continue

            if message.get("role") == "tool":
                tool_call_id = message.get("tool_call_id")
                if not tool_call_id or str(tool_call_id) not in declared:
                    continue
            updated.append(message)
        return updated

    @staticmethod
    def _backfill_missing_tool_results(messages: list[Message]) -> list[Message]:
        declared: list[tuple[int, str, str]] = []
        fulfilled: set[str] = set()
        for idx, message in enumerate(messages):
            if message.get("role") == "assistant":
                for tool_call in message.get("tool_calls") or []:
                    if not isinstance(tool_call, dict) or not tool_call.get("id"):
                        continue
                    name = ""
                    function = tool_call.get("function")
                    if isinstance(function, dict) and isinstance(function.get("name"), str):
                        name = function["name"]
                    declared.append((idx, str(tool_call["id"]), name))
            elif message.get("role") == "tool" and message.get("tool_call_id"):
                fulfilled.add(str(message["tool_call_id"]))

        missing = [(idx, call_id, name) for idx, call_id, name in declared if call_id not in fulfilled]
        if not missing:
            return messages

        updated = [dict(message) for message in messages]
        offset = 0
        for assistant_idx, call_id, name in missing:
            insert_at = assistant_idx + 1 + offset
            while insert_at < len(updated) and updated[insert_at].get("role") == "tool":
                insert_at += 1
            updated.insert(
                insert_at,
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": MISSING_TOOL_RESULT_CONTENT,
                },
            )
            offset += 1
        return updated
