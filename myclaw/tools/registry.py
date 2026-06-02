from __future__ import annotations

import json
from typing import Any

from myclaw.providers.base import ToolCallRequest
from myclaw.config import TOOL_RESULT_TRUNCATED_TEMPLATE
from myclaw.tools.base import Tool, ToolRuntimeContext, tool_context


class ToolRegistry:
    """Registry for OpenAI-style function tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._cached_definitions: list[dict[str, Any]] | None = None

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        self._cached_definitions = None

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)
        self._cached_definitions = None

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    def definitions(self) -> list[dict[str, Any]]:
        if self._cached_definitions is not None:
            return self._cached_definitions

        self._cached_definitions = [
            self._tool_schema(tool)
            for tool in sorted(self._tools.values(), key=lambda candidate: candidate.name)
        ]
        return self._cached_definitions

    def prepare_call(self, request: ToolCallRequest) -> tuple[Tool | None, dict[str, Any], str | None]:
        tool = self._tools.get(request.name)
        if tool is None:
            return None, {}, f"Error: Tool '{request.name}' not found. Available: {', '.join(self.tool_names)}"
        if not isinstance(request.arguments, dict):
            return None, {}, (
                f"Error: Tool '{request.name}' arguments must be a JSON object, "
                f"got {type(request.arguments).__name__}"
            )
        arguments = dict(request.arguments)
        cast_params = getattr(tool, "cast_params", None)
        if callable(cast_params):
            try:
                cast_arguments = cast_params(arguments)
            except Exception as exc:
                return None, {}, f"Error casting {request.name}: {exc}"
            if cast_arguments is None:
                cast_arguments = arguments
            arguments = cast_arguments
            if not isinstance(arguments, dict):
                return None, {}, f"Error casting {request.name}: cast_params must return a dict"
        validate_params = getattr(tool, "validate_params", None)
        if callable(validate_params):
            try:
                validate_params(arguments)
            except Exception as exc:
                return None, {}, f"Error validating {request.name}: {exc}"
        return tool, arguments, None

    async def execute(
        self,
        request: ToolCallRequest,
        *,
        max_result_chars: int | None = None,
        context: ToolRuntimeContext | None = None,
    ) -> str:
        tool, arguments, error = self.prepare_call(request)
        if error is not None:
            return self._truncate_result(error, max_result_chars)

        try:
            assert tool is not None
            runtime_context = self._runtime_context(context)
            set_context = getattr(tool, "set_context", None)
            if callable(set_context):
                set_context(runtime_context)
            with tool_context(runtime_context):
                result = await tool.execute(**arguments)
        except Exception as exc:
            return self._truncate_result(f"Error executing {request.name}: {exc}", max_result_chars)
        return self._truncate_result(self._normalize_result(result), max_result_chars)

    @staticmethod
    def _tool_schema(tool: Tool) -> dict[str, Any]:
        to_schema = getattr(tool, "to_schema", None)
        if callable(to_schema):
            schema = to_schema()
            if isinstance(schema, dict):
                return schema
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    def _runtime_context(self, context: ToolRuntimeContext | None) -> ToolRuntimeContext:
        runtime_context = context or ToolRuntimeContext()
        if not runtime_context.tool_names:
            runtime_context.tool_names = sorted(self.tool_names)
        return runtime_context

    @staticmethod
    def _normalize_result(result: Any) -> str:
        if result is None:
            return "(empty)"
        if isinstance(result, str):
            return result if result else "(empty)"
        try:
            return json.dumps(result, ensure_ascii=False)
        except TypeError:
            return str(result) if str(result) else "(empty)"

    @staticmethod
    def _truncate_result(result: str, max_result_chars: int | None) -> str:
        if max_result_chars is None or len(result) <= max_result_chars:
            return result
        omitted = len(result) - max_result_chars
        return f"{result[:max_result_chars]}\n{TOOL_RESULT_TRUNCATED_TEMPLATE.format(omitted=omitted)}"

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
