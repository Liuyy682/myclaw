from __future__ import annotations

import json
from typing import Any

from myclaw.tools.base import Tool, get_current_tool_context
from myclaw.tools.models import RecallInput


class RecallTool(Tool):
    """Resolve an observation or reflection back to its stored evidence."""

    read_only = True
    exclusive = False
    effect = "local_read"
    input_model = RecallInput

    def __init__(self, store: Any) -> None:
        self.store = store

    @property
    def name(self) -> str:
        return "recall"

    @property
    def description(self) -> str:
        return "Recall a session observation or reflection and its original evidence by ID."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Observation or reflection ID",
                }
            },
            "required": ["id"],
        }

    async def execute(self, id: str, **kwargs: Any) -> str:
        identifier = id.strip()
        if not identifier:
            return "Error: id is required"
        context = get_current_tool_context()
        result = self.store.recall(identifier, session_id=context.session_key)
        if result is None or result.get("status") != "found":
            return f"Error: memory not found or not visible: {identifier}"
        return json.dumps(result, ensure_ascii=False, indent=2)
