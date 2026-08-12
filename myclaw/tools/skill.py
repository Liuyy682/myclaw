from __future__ import annotations

from typing import Any

from myclaw.skills import SkillCatalog
from myclaw.tools.base import Tool
from myclaw.tools.models import SkillLoadInput


class SkillLoadTool(Tool):
    read_only = True
    exclusive = False
    input_model = SkillLoadInput

    def __init__(self, catalog: SkillCatalog) -> None:
        self._catalog = catalog

    @property
    def name(self) -> str:
        return "skill_load"

    @property
    def description(self) -> str:
        return "Load the instructions for an available skill when it matches the current task."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the skill to load",
                    "enum": list(self._catalog.names),
                }
            },
            "required": ["name"],
        }

    async def execute(self, name: str | None = None, **kwargs: Any) -> str:
        if not isinstance(name, str) or not name:
            return "Error: skill name is required"
        try:
            return self._catalog.load(name).body
        except (OSError, UnicodeError, ValueError) as exc:
            return f"Error loading skill '{name}': {exc}"
