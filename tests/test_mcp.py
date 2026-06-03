import asyncio
import json
import sys
from pathlib import Path

import pytest

from myclaw.mcp import McpManager, McpServerConfig, load_mcp_configs, mcp_tool_name
from myclaw.tools import ToolRegistry

pytest.importorskip("mcp")

_ECHO_SERVER = str(Path(__file__).parent / "mcp_echo_server.py")


def _echo_config() -> McpServerConfig:
    return McpServerConfig(name="echo", command=sys.executable, args=[_ECHO_SERVER])


def test_mcp_manager_connects_registers_and_calls_tool():
    registry = ToolRegistry()

    async def scenario():
        manager = McpManager()
        try:
            await manager.connect([_echo_config()])
            manager.register_into(registry)
            tool_name = mcp_tool_name("echo", "echo")
            tool = registry.get(tool_name)
            result = await tool.execute(text="hello") if tool is not None else None
            return [t.name for t in manager.tools], tool_name, result
        finally:
            await manager.aclose()

    names, tool_name, result = asyncio.run(scenario())

    assert tool_name in names
    assert registry.has(tool_name)
    assert result == "echo: hello"


def test_mcp_tool_definition_uses_remote_input_schema():
    async def scenario():
        manager = McpManager()
        try:
            await manager.connect([_echo_config()])
            tool = manager.tools[0]
            return tool.name, tool.parameters
        finally:
            await manager.aclose()

    name, parameters = asyncio.run(scenario())

    assert name == mcp_tool_name("echo", "echo")
    assert parameters.get("type") == "object"
    assert "text" in parameters.get("properties", {})


def test_load_mcp_configs_reads_optional_file(tmp_path):
    assert load_mcp_configs(tmp_path) == []

    (tmp_path / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "echo": {"command": "python", "args": ["server.py"], "env": {"K": "v"}},
                    "bad": {"args": ["no-command"]},
                }
            }
        ),
        encoding="utf-8",
    )

    configs = load_mcp_configs(tmp_path)

    assert len(configs) == 1
    assert configs[0].name == "echo"
    assert configs[0].command == "python"
    assert configs[0].args == ["server.py"]
    assert configs[0].env == {"K": "v"}


def test_load_mcp_configs_handles_invalid_json(tmp_path):
    (tmp_path / "mcp.json").write_text("{not json", encoding="utf-8")
    assert load_mcp_configs(tmp_path) == []
