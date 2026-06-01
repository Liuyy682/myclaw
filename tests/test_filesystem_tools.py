import asyncio

from myclaw import EditFileTool, GlobTool, GrepTool, ListDirTool, ReadFileTool, WriteFileTool, build_default_tool_registry


def test_read_file_returns_line_numbered_text_and_supports_ranges(tmp_path):
    (tmp_path / "sample.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    tool = ReadFileTool(tmp_path)

    full = asyncio.run(tool.execute(path="sample.txt"))
    partial = asyncio.run(tool.execute(path="sample.txt", offset=2, limit=1))

    assert full == "1|one\n2|two\n3|three"
    assert partial == "2|two"


def test_read_file_returns_clear_errors_for_invalid_targets(tmp_path):
    (tmp_path / "dir").mkdir()
    (tmp_path / "binary.dat").write_bytes(b"\xff\xfe\x00")
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    tool = ReadFileTool(tmp_path)

    assert asyncio.run(tool.execute(path="missing.txt")) == "Error: File not found: missing.txt"
    assert asyncio.run(tool.execute(path="dir")) == "Error: Not a file: dir"
    assert asyncio.run(tool.execute(path="binary.dat")) == "Error: Cannot read binary file: binary.dat"
    assert asyncio.run(tool.execute(path="empty.txt")) == "(Empty file: empty.txt)"


def test_read_file_blocks_paths_outside_workspace_and_symlink_escape(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = workspace / "link.txt"
    link.symlink_to(outside)
    tool = ReadFileTool(workspace)

    outside_result = asyncio.run(tool.execute(path=str(outside)))
    link_result = asyncio.run(tool.execute(path="link.txt"))

    assert outside_result.startswith("Error: Path is outside workspace:")
    assert link_result.startswith("Error: Path is outside workspace:")


def test_list_dir_lists_entries_recursively_ignores_noise_and_truncates(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("pass", encoding="utf-8")
    (tmp_path / "README.md").write_text("hi", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("x", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "pkg").write_text("x", encoding="utf-8")
    tool = ListDirTool(tmp_path)

    direct = asyncio.run(tool.execute(path="."))
    recursive = asyncio.run(tool.execute(path=".", recursive=True))
    truncated = asyncio.run(tool.execute(path=".", recursive=True, max_entries=1))

    assert direct == "README.md\nsrc/"
    assert "src/main.py" in recursive
    assert ".git" not in recursive
    assert "node_modules" not in recursive
    assert "truncated: showing 1 of 2 entries" in truncated


def test_list_dir_returns_clear_errors_for_invalid_targets_and_outside_paths(tmp_path):
    (tmp_path / "file.txt").write_text("x", encoding="utf-8")
    outside = tmp_path.parent / "outside-list.txt"
    outside.write_text("x", encoding="utf-8")
    tool = ListDirTool(tmp_path)

    assert asyncio.run(tool.execute(path="missing")) == "Error: Directory not found: missing"
    assert asyncio.run(tool.execute(path="file.txt")) == "Error: Not a directory: file.txt"
    assert asyncio.run(tool.execute(path=str(outside))).startswith("Error: Path is outside workspace:")


def test_write_file_creates_parent_dirs_overwrites_and_blocks_invalid_targets(tmp_path):
    tool = WriteFileTool(tmp_path)

    first = asyncio.run(tool.execute(path="nested/out.txt", content="hello"))
    second = asyncio.run(tool.execute(path="nested/out.txt", content="updated"))
    directory_target = asyncio.run(tool.execute(path="nested", content="nope"))
    outside = asyncio.run(tool.execute(path=str(tmp_path.parent / "outside-write.txt"), content="nope"))

    assert first == "Wrote 5 bytes to nested/out.txt"
    assert second == "Wrote 7 bytes to nested/out.txt"
    assert (tmp_path / "nested" / "out.txt").read_text(encoding="utf-8") == "updated"
    assert directory_target == "Error: Cannot write to directory: nested"
    assert outside.startswith("Error: Path is outside workspace:")


def test_edit_file_replaces_exactly_one_match_and_preserves_rest(tmp_path):
    (tmp_path / "sample.txt").write_text("before\nneedle\nafter\n", encoding="utf-8")
    tool = EditFileTool(tmp_path)

    result = asyncio.run(tool.execute(path="sample.txt", old_text="needle", new_text="thread"))

    assert result == "Edited sample.txt: replaced 1 occurrence"
    assert (tmp_path / "sample.txt").read_text(encoding="utf-8") == "before\nthread\nafter\n"


def test_edit_file_returns_clear_errors_for_invalid_targets_and_match_counts(tmp_path):
    (tmp_path / "dir").mkdir()
    (tmp_path / "binary.dat").write_bytes(b"\xff\xfe\x00")
    (tmp_path / "sample.txt").write_text("alpha\nbeta\nalpha\n", encoding="utf-8")
    tool = EditFileTool(tmp_path)

    assert asyncio.run(tool.execute(path="missing.txt", old_text="x", new_text="y")) == "Error: File not found: missing.txt"
    assert asyncio.run(tool.execute(path="dir", old_text="x", new_text="y")) == "Error: Not a file: dir"
    assert asyncio.run(tool.execute(path="binary.dat", old_text="x", new_text="y")) == "Error: Cannot edit binary file: binary.dat"
    assert asyncio.run(tool.execute(path="sample.txt", old_text="gamma", new_text="delta")) == (
        "Error: old_text not found in sample.txt"
    )
    assert asyncio.run(tool.execute(path="sample.txt", old_text="alpha", new_text="delta")) == (
        "Error: old_text matched 2 times in sample.txt; expected exactly 1"
    )


def test_grep_finds_matches_with_line_numbers_case_options_and_ignored_dirs(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("Alpha\nbeta\nALPHA\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("Alpha\n", encoding="utf-8")
    (tmp_path / "binary.dat").write_bytes(b"\xff\xfeAlpha")
    tool = GrepTool(tmp_path)

    case_sensitive = asyncio.run(tool.execute(pattern="Alpha", path="."))
    case_insensitive = asyncio.run(tool.execute(pattern="alpha", path=".", case_sensitive=False))

    assert case_sensitive == "src/app.py:1|Alpha"
    assert case_insensitive == "src/app.py:1|Alpha\nsrc/app.py:3|ALPHA"
    assert ".git" not in case_insensitive
    assert "binary.dat" not in case_insensitive


def test_grep_reports_invalid_regex_no_matches_and_respects_max_matches(tmp_path):
    (tmp_path / "notes.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    tool = GrepTool(tmp_path)

    invalid = asyncio.run(tool.execute(pattern="[", path="."))
    missing = asyncio.run(tool.execute(pattern="delta", path="."))
    truncated = asyncio.run(tool.execute(pattern="a", path=".", max_matches=2))

    assert invalid.startswith("Error: Invalid regex:")
    assert missing == "(No matches for pattern in .)"
    assert truncated == "notes.txt:1|alpha\nnotes.txt:2|beta\n[truncated: showing 2 of 3 matches]"


def test_glob_returns_relative_matches_recursively_skips_ignored_dirs_and_truncates(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("pass", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("guide", encoding="utf-8")
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "hidden.md").write_text("hidden", encoding="utf-8")
    tool = GlobTool(tmp_path)

    root_markdown = asyncio.run(tool.execute(pattern="*.md", path="."))
    recursive_markdown = asyncio.run(tool.execute(pattern="**/*.md", path="."))
    truncated = asyncio.run(tool.execute(pattern="**/*", path=".", max_matches=2))

    assert root_markdown == "README.md"
    assert recursive_markdown == "README.md\ndocs/guide.md"
    assert "node_modules" not in recursive_markdown
    assert truncated == "README.md\ndocs/\n[truncated: showing 2 of 5 matches]"


def test_default_tool_registry_contains_default_tools_in_stable_order(tmp_path):
    registry = build_default_tool_registry(tmp_path)

    assert [definition["function"]["name"] for definition in registry.definitions()] == [
        "edit_file",
        "glob",
        "grep",
        "list_dir",
        "read_file",
        "remember",
        "write_file",
    ]
