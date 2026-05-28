import subprocess
import sys
import os

from myclaw.__main__ import load_env_file


def test_cli_single_turn_uses_fake_provider_without_api_key(tmp_path):
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env["MYCLAW_ENV_FILE"] = str(tmp_path / "missing.env")

    result = subprocess.run(
        [sys.executable, "-m", "myclaw", "hello"],
        check=True,
        cwd="/root/myclaw",
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "Echo: hello"


def test_load_env_file_reads_project_env_without_overwriting_existing_values(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=from-file",
                "OPENAI_BASE_URL='https://example.test/v1'",
                'OPENAI_MODEL="demo-model"',
                "IGNORED_LINE",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "from-shell")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    load_env_file(env_file)

    assert env_file.read_text(encoding="utf-8").startswith("OPENAI_API_KEY")
    assert "OPENAI_API_KEY=from-file" in env_file.read_text(encoding="utf-8")
    assert __import__("os").environ["OPENAI_API_KEY"] == "from-shell"
    assert __import__("os").environ["OPENAI_BASE_URL"] == "https://example.test/v1"
    assert __import__("os").environ["OPENAI_MODEL"] == "demo-model"
