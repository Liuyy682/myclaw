from __future__ import annotations

from typing import Any

from myclaw.agent.types import AgentConfig, Message

MISSING_TOOL_RESULT_CONTENT = "[tool result unavailable in persisted history]"
TOOL_RESULT_TRUNCATED_TEMPLATE = "[tool result truncated: {omitted} chars omitted]"


class ContextBuilder:
    """Build model-ready messages from persisted session history."""

    def build_messages(
        self,
        config: AgentConfig,
        session_messages: list[Message],
        current_user_text: str,
    ) -> list[Message]:
        messages = self._initial_messages(config)
        messages.extend(self._history_messages(session_messages, config.max_tool_result_chars))
        messages.append({"role": "user", "content": current_user_text})
        return messages

    @staticmethod
    def _initial_messages(config: AgentConfig) -> list[Message]:
        messages: list[Message] = []
        if config.system_prompt.strip():
            messages.append({"role": "system", "content": config.system_prompt.strip()})
        messages.extend(dict(message) for message in config.history)
        return messages

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
