import asyncio
import shutil

import pytest

from myclaw.memory import MemoryGit

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")


def _run(coro):
    return asyncio.run(coro)


def test_ensure_repo_initialises_with_local_identity(tmp_path):
    memory_dir = tmp_path / "memory"
    git = MemoryGit(memory_dir)

    assert _run(git.is_repo()) is False
    assert _run(git.ensure_repo()) is True
    assert _run(git.is_repo()) is True
    # Idempotent: a second call on an existing repo still succeeds.
    assert _run(git.ensure_repo()) is True
    # A .gitignore excluding the transient cursor is created.
    assert ".dream_cursor" in (memory_dir / ".gitignore").read_text(encoding="utf-8")


def test_commit_all_commits_changes_and_skips_when_clean(tmp_path):
    memory_dir = tmp_path / "memory"
    git = MemoryGit(memory_dir)

    _run(git.ensure_repo())
    (memory_dir / "USER.md").write_text("- Name is Sam.\n", encoding="utf-8")

    assert _run(git.has_changes()) is True
    assert _run(git.commit_all("dream: t, 1 change(s)", "[USER] Name is Sam.")) is True
    # Nothing left to commit -> no empty commit.
    assert _run(git.has_changes()) is False
    assert _run(git.commit_all("noop", "body")) is False


def test_log_returns_recent_commits(tmp_path):
    memory_dir = tmp_path / "memory"
    git = MemoryGit(memory_dir)
    _run(git.ensure_repo())

    (memory_dir / "MEMORY.md").write_text("- one\n", encoding="utf-8")
    _run(git.commit_all("dream: t1, 1 change(s)", "first"))
    (memory_dir / "MEMORY.md").write_text("- one\n- two\n", encoding="utf-8")
    _run(git.commit_all("dream: t2, 1 change(s)", "second"))

    entries = _run(git.log(5))
    assert [entry["subject"] for entry in entries] == [
        "dream: t2, 1 change(s)",
        "dream: t1, 1 change(s)",
    ]
    assert all(entry["hash"] and entry["date"] for entry in entries)


def test_dream_cursor_is_not_tracked(tmp_path):
    memory_dir = tmp_path / "memory"
    git = MemoryGit(memory_dir)
    _run(git.ensure_repo())

    (memory_dir / "MEMORY.md").write_text("- fact\n", encoding="utf-8")
    (memory_dir / ".dream_cursor").write_text('{"last_id": 0}\n', encoding="utf-8")
    _run(git.commit_all("dream: t, 1 change(s)", "body"))

    code, out, _ = _run(git._run("ls-files"))
    tracked = out.split()
    assert "MEMORY.md" in tracked
    assert ".dream_cursor" not in tracked


def test_operations_are_safe_outside_a_repo(tmp_path):
    # A plain directory that is not a git repo: read ops must not raise.
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    git = MemoryGit(memory_dir)

    assert _run(git.is_repo()) is False
    assert _run(git.has_changes()) is False
    assert _run(git.changed_count()) == 0
    assert _run(git.log(5)) == []
