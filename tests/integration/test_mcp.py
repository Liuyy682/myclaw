import json

from myclaw.mcp import load_mcp_configs


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
