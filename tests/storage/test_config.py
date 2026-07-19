import os

from myclaw.config import (
    CLI_EXIT_COMMANDS,
    DEFAULT_CLI_SESSION_NAME,
    DEFAULT_GATEWAY_CHAT_ID,
    DEFAULT_GATEWAY_HOST,
    DEFAULT_GATEWAY_PORT,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENAI_TIMEOUT_SECONDS,
    ENV_FILE_VAR,
    FILESYSTEM_IGNORED_DIRS,
    GATEWAY_CHANNEL,
    GATEWAY_MAX_BODY_BYTES,
    OPENAI_API_KEY_ENV_VAR,
    OPENAI_BASE_URL_ENV_VAR,
    OPENAI_MODEL_ENV_VAR,
    READ_FILE_DEFAULT_LIMIT,
    TOOL_RESULT_TRUNCATED_TEMPLATE,
    WORKSPACE_ENV_VAR,
    load_env_file,
)


def test_config_exports_shared_settings():
    assert ENV_FILE_VAR == "MYCLAW_ENV_FILE"
    assert WORKSPACE_ENV_VAR == "MYCLAW_WORKSPACE"
    assert OPENAI_API_KEY_ENV_VAR == "OPENAI_API_KEY"
    assert OPENAI_BASE_URL_ENV_VAR == "OPENAI_BASE_URL"
    assert OPENAI_MODEL_ENV_VAR == "OPENAI_MODEL"
    assert DEFAULT_OPENAI_BASE_URL == "https://api.openai.com/v1"
    assert DEFAULT_OPENAI_MODEL == "gpt-4o-mini"
    assert DEFAULT_OPENAI_TIMEOUT_SECONDS == 120
    assert DEFAULT_CLI_SESSION_NAME == "direct"
    assert CLI_EXIT_COMMANDS == frozenset({"exit", "quit"})
    assert GATEWAY_CHANNEL == "gateway"
    assert DEFAULT_GATEWAY_CHAT_ID == "direct"
    assert DEFAULT_GATEWAY_HOST == "127.0.0.1"
    assert DEFAULT_GATEWAY_PORT == 8765
    assert GATEWAY_MAX_BODY_BYTES == 64 * 1024
    assert READ_FILE_DEFAULT_LIMIT == 2000
    assert "node_modules" in FILESYSTEM_IGNORED_DIRS
    assert TOOL_RESULT_TRUNCATED_TEMPLATE.format(omitted=3) == (
        "[tool result truncated: 3 chars omitted]"
    )


def test_load_env_file_uses_configured_env_file_without_overwriting_existing_values(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                f"{OPENAI_API_KEY_ENV_VAR}=from-file",
                f"{OPENAI_MODEL_ENV_VAR}=from-file-model",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(ENV_FILE_VAR, str(env_file))
    monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "from-shell")
    monkeypatch.delenv(OPENAI_MODEL_ENV_VAR, raising=False)

    load_env_file()

    assert os.environ[OPENAI_API_KEY_ENV_VAR] == "from-shell"
    assert os.environ[OPENAI_MODEL_ENV_VAR] == "from-file-model"
