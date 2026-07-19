import json
from datetime import datetime

from myclaw.config import WORKSPACE_ENV_VAR
from myclaw.session import Session, SessionManager, TranscriptStore, get_default_workspace


def test_default_workspace_uses_env_var_or_expanded_home(tmp_path, monkeypatch):
    monkeypatch.delenv(WORKSPACE_ENV_VAR, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    assert get_default_workspace() == tmp_path / "home" / ".myclaw" / "workspace"

    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path / "custom"))

    assert get_default_workspace() == tmp_path / "custom"


def test_save_creates_jsonl_with_metadata_and_messages(tmp_path):
    manager = SessionManager(tmp_path)
    session = Session(key="cli:direct")
    session.add_message("user", "hello")
    session.add_message("assistant", "hi")

    manager.save(session)

    path = tmp_path / "sessions" / "cli_direct.jsonl"
    lines = path.read_text(encoding="utf-8").strip().splitlines()

    assert len(lines) == 3
    metadata = json.loads(lines[0])
    assert metadata["_type"] == "metadata"
    assert metadata["key"] == "cli:direct"
    assert metadata["metadata"] == {}

    first_message = json.loads(lines[1])
    assert first_message["role"] == "user"
    assert first_message["content"] == "hello"
    assert "timestamp" in first_message


def test_session_add_message_preserves_structured_tool_fields(tmp_path):
    manager = SessionManager(tmp_path)
    session = Session(key="cli:direct")
    tool_calls = [
        {
            "id": "call_add",
            "type": "function",
            "function": {"name": "add", "arguments": "{}"},
        }
    ]
    session.add_message("assistant", "", tool_calls=tool_calls)
    session.add_message("tool", "5", tool_call_id="call_add", name="add")

    manager.save(session)
    reloaded = SessionManager(tmp_path).get_or_create("cli:direct")

    assert reloaded.messages[0]["tool_calls"] == tool_calls
    assert reloaded.messages[1]["tool_call_id"] == "call_add"
    assert reloaded.messages[1]["name"] == "add"


def test_get_or_create_loads_existing_history(tmp_path):
    manager = SessionManager(tmp_path)
    session = Session(key="cli:direct")
    session.add_message("user", "first")
    session.add_message("assistant", "Echo: first")
    manager.save(session)

    reloaded = SessionManager(tmp_path).get_or_create("cli:direct")

    assert [message["content"] for message in reloaded.messages] == ["first", "Echo: first"]
    assert reloaded.key == "cli:direct"


def test_reset_clears_history_and_metadata_but_keeps_key(tmp_path):
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("cli:direct")
    session.add_message("user", "hello")
    session.metadata["runtime_checkpoint"] = {"phase": "awaiting_tools"}
    manager.save(session)

    reset = manager.reset("cli:direct")

    assert reset.key == "cli:direct"
    assert reset.messages == []
    assert reset.metadata == {}
    reloaded = SessionManager(tmp_path).get_or_create("cli:direct")
    assert reloaded.key == "cli:direct"
    assert reloaded.messages == []
    assert reloaded.metadata == {}


def test_list_sessions_returns_saved_sessions_sorted_by_updated_at(tmp_path):
    manager = SessionManager(tmp_path)
    first = manager.get_or_create("cli:direct")
    first.metadata["title"] = "Direct chat"
    manager.save(first)
    second = manager.get_or_create("cli:work")
    second.metadata["title"] = "Work chat"
    manager.save(second)

    sessions = manager.list_sessions()

    assert [session.key for session in sessions] == ["cli:work", "cli:direct"]
    assert [session.metadata["title"] for session in sessions] == ["Work chat", "Direct chat"]


def test_safe_key_keeps_default_cli_file_and_avoids_non_ascii_collisions():
    assert SessionManager.safe_key("cli:direct") == "cli_direct"
    assert SessionManager.safe_key("cli:工作") != SessionManager.safe_key("cli:旅行")


def test_session_key_maps_to_safe_filename(tmp_path):
    manager = SessionManager(tmp_path)
    session = Session(key="cli:direct")

    manager.save(session)

    assert (tmp_path / "sessions" / "cli_direct.jsonl").exists()


def test_save_does_not_leave_tmp_file_after_success(tmp_path):
    manager = SessionManager(tmp_path)
    session = Session(key="cli:direct")

    manager.save(session)

    assert list((tmp_path / "sessions").glob("*.tmp")) == []


def test_load_recovers_valid_lines_from_corrupt_jsonl(tmp_path):
    manager = SessionManager(tmp_path)
    path = tmp_path / "sessions" / "cli_direct.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "_type": "metadata",
                        "key": "cli:direct",
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat(),
                        "metadata": {},
                    }
                ),
                "not valid json",
                json.dumps({"role": "user", "content": "survived"}),
                '{"role": "assistant", "content": "partial',
            ]
        ),
        encoding="utf-8",
    )

    session = manager.get_or_create("cli:direct")

    assert session.messages == [{"role": "user", "content": "survived"}]


def _read_transcript(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_transcript_store_appends_messages_per_session(tmp_path):
    store = TranscriptStore(tmp_path, safe_key=SessionManager.safe_key)

    store.append("cli:direct", {"role": "user", "content": "hi"})
    store.append_many(
        "cli:direct",
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "add", "arguments": '{"a": 1}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "add", "content": "1"},
        ],
    )

    path = tmp_path / "transcripts" / f"{SessionManager.safe_key('cli:direct')}.jsonl"
    records = _read_transcript(path)

    assert [record["message"]["role"] for record in records] == ["user", "assistant", "tool"]
    assert all(record["session_key"] == "cli:direct" for record in records)
    assert all(record["logged_at"] for record in records)
    # Tool call name/arguments and tool result are preserved verbatim.
    assert records[1]["message"]["tool_calls"][0]["function"] == {"name": "add", "arguments": '{"a": 1}'}
    assert records[2]["message"] == {"role": "tool", "tool_call_id": "c1", "name": "add", "content": "1"}


def test_transcript_store_isolates_sessions(tmp_path):
    store = TranscriptStore(tmp_path, safe_key=SessionManager.safe_key)

    store.append("cli:a", {"role": "user", "content": "a"})
    store.append("cli:b", {"role": "user", "content": "b"})

    path_a = tmp_path / "transcripts" / f"{SessionManager.safe_key('cli:a')}.jsonl"
    path_b = tmp_path / "transcripts" / f"{SessionManager.safe_key('cli:b')}.jsonl"

    assert _read_transcript(path_a)[0]["message"]["content"] == "a"
    assert _read_transcript(path_b)[0]["message"]["content"] == "b"


def test_transcript_store_append_failure_does_not_raise(tmp_path):
    # Pre-create a regular file where the transcripts directory should be so
    # mkdir/open fails; the store must swallow the error rather than propagate.
    (tmp_path / "transcripts").write_text("not a dir", encoding="utf-8")
    store = TranscriptStore(tmp_path, safe_key=SessionManager.safe_key)

    store.append("cli:direct", {"role": "user", "content": "hi"})  # must not raise


def test_transcript_store_append_many_empty_is_noop(tmp_path):
    store = TranscriptStore(tmp_path, safe_key=SessionManager.safe_key)

    store.append_many("cli:direct", [])

    assert not (tmp_path / "transcripts").exists()
