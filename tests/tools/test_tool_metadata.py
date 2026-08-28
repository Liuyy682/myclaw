from myclaw.tools import build_default_tool_registry


def test_builtin_tools_expose_explicit_security_metadata(tmp_path):
    registry = build_default_tool_registry(tmp_path)

    expected = {
        "ask_user": (False, True, "ask"),
        "cron": (False, False, "cron"),
        "edit_file": (False, True, "local_write"),
        "exec": (False, True, "exec"),
        "glob": (True, False, "local_read"),
        "grep": (True, False, "local_read"),
        "list_dir": (True, False, "local_read"),
        "message": (False, False, "message"),
        "my": (True, False, "local_read"),
        "notebook_edit": (False, True, "local_write"),
        "read_file": (True, False, "local_read"),
        "remember": (False, True, "local_write"),
        "spawn": (False, False, "spawn"),
        "task_create": (False, False, "local_write"),
        "task_get": (True, False, "local_read"),
        "task_list": (True, False, "local_read"),
        "task_update": (False, True, "local_write"),
        "web_fetch": (True, False, "network_read"),
        "web_search": (True, False, "network_read"),
        "write_file": (False, True, "local_write"),
    }

    assert set(registry.tool_names) == set(expected)
    for name, (read_only, exclusive, effect) in expected.items():
        tool = registry.get(name)
        assert tool is not None
        assert (tool.read_only, tool.exclusive, tool.effect) == (read_only, exclusive, effect)


def test_default_registry_uses_workspace_security_database(tmp_path):
    registry = build_default_tool_registry(tmp_path)

    assert registry.security_store is not None
    assert registry.security_store.path == tmp_path / "security" / "tool_security.db"
