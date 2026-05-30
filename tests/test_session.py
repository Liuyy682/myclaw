import json
from datetime import datetime

from myclaw.session import Session, SessionManager


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
