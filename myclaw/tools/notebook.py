from __future__ import annotations

import json
from typing import Any

from myclaw.tools.filesystem import _FilesystemTool


class NotebookEditTool(_FilesystemTool):
    read_only = False
    exclusive = True

    @property
    def name(self) -> str:
        return "notebook_edit"

    @property
    def description(self) -> str:
        return "Edit or append a cell in a Jupyter .ipynb notebook under the workspace."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Notebook path"},
                "cell_index": {"type": "integer", "description": "Zero-based cell index"},
                "source": {"type": "string", "description": "New cell source"},
                "cell_type": {"type": "string", "description": "Optional cell type for appended cells"},
            },
            "required": ["path", "cell_index", "source"],
        }

    async def execute(
        self,
        path: str | None = None,
        cell_index: int | None = None,
        source: str | None = None,
        cell_type: str = "code",
        **kwargs: Any,
    ) -> str:
        if not path:
            return "Error: path is required"
        if cell_index is None:
            return "Error: cell_index is required"
        if source is None:
            return "Error: source is required"
        try:
            index = int(cell_index)
        except (TypeError, ValueError):
            return "Error: cell_index must be an integer"
        if index < 0:
            return "Error: cell_index must be non-negative"

        try:
            notebook_path = self._resolve(path)
        except PermissionError as exc:
            return self._error(exc)
        if notebook_path.exists() and not notebook_path.is_file():
            return f"Error: Not a file: {path}"

        if notebook_path.exists():
            try:
                data = json.loads(notebook_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                return f"Error: Invalid notebook JSON: {exc}"
        else:
            data = {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}

        cells = data.get("cells")
        if not isinstance(cells, list):
            return "Error: notebook cells must be a list"
        if index > len(cells):
            return f"Error: cell_index {index} is beyond end of notebook ({len(cells)} cells)"

        source_lines = str(source).splitlines(keepends=True)
        if not source_lines and source == "":
            source_lines = []
        if index == len(cells):
            new_cell = {
                "cell_type": cell_type,
                "metadata": {},
                "source": source_lines,
            }
            if cell_type == "code":
                new_cell["outputs"] = []
                new_cell["execution_count"] = None
            cells.append(new_cell)
            action = "Appended"
        else:
            if not isinstance(cells[index], dict):
                return f"Error: cell {index} is invalid"
            cells[index]["source"] = source_lines
            action = "Edited"

        notebook_path.parent.mkdir(parents=True, exist_ok=True)
        notebook_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return f"{action} {path} cell {index}"
