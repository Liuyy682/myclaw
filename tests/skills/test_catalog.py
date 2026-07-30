import asyncio
from pathlib import Path

import pytest

from myclaw.skills import SkillCatalog, SkillDefinition, SkillMetadata
from myclaw.tools import SkillLoadTool


def _write_skill(
    root: Path,
    directory: str,
    *,
    name: str,
    description: str = "Useful instructions",
    platforms: list[str] | None = None,
    body: str = "Follow these steps.",
) -> Path:
    skill_dir = root / directory
    skill_dir.mkdir(parents=True)
    platform_line = f"platforms: [{', '.join(platforms)}]\n" if platforms is not None else ""
    path = skill_dir / "SKILL.md"
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n{platform_line}---\n{body}\n",
        encoding="utf-8",
    )
    return path


def test_catalog_discovers_metadata_without_loading_body(tmp_path):
    root = tmp_path / "skills"
    path = _write_skill(
        root,
        "review",
        name="code-review",
        description="Review code safely",
        platforms=["linux", "darwin"],
        body="Do not reveal this in the catalog.",
    )

    catalog = SkillCatalog.discover(root, platform="linux")

    assert catalog.names == ("code-review",)
    assert catalog.metadata == (
        SkillMetadata(
            name="code-review",
            description="Review code safely",
            platforms=("linux", "darwin"),
        ),
    )
    rendered = catalog.render_for_prompt()
    assert "- code-review: Review code safely" in rendered
    assert "Do not reveal this" not in rendered

    payload = path.read_bytes()
    path.write_bytes(payload.replace(b"Do not reveal this in the catalog.", b"\xff"))
    assert SkillCatalog.discover(root, platform="linux").names == ("code-review",)


def test_catalog_loads_body_on_demand_and_revalidates_metadata(tmp_path):
    root = tmp_path / "skills"
    path = _write_skill(root, "review", name="code-review", body="Inspect the real call chain.")
    catalog = SkillCatalog.discover(root, platform="linux")

    definition = catalog.load("code-review")

    assert definition == SkillDefinition(
        metadata=SkillMetadata("code-review", "Useful instructions"),
        body="Inspect the real call chain.",
        source=path.resolve(),
    )

    path.write_text(
        "---\nname: renamed\ndescription: Useful instructions\n---\nChanged.\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="metadata changed"):
        catalog.load("code-review")


@pytest.mark.parametrize(
    ("frontmatter", "message"),
    [
        ("name: Uppercase\ndescription: valid", "lowercase ASCII slug"),
        ("name: valid", "description is required"),
        ("- not\n- a mapping", "must be a mapping"),
        ("name: valid\ndescription: ok\nplatforms: linux", "list of strings"),
        ("name: valid\ndescription: ok\nplatforms: [plan9]", "unknown skill platforms"),
    ],
)
def test_catalog_isolates_invalid_metadata(tmp_path, caplog, frontmatter, message):
    root = tmp_path / "skills"
    skill_dir = root / "bad"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(f"---\n{frontmatter}\n---\nbody\n", encoding="utf-8")

    catalog = SkillCatalog.discover(root, platform="linux")

    assert len(catalog) == 0
    assert message in caplog.text


@pytest.mark.parametrize(
    "content",
    [
        "name: missing-opening\ndescription: bad\n---\nbody\n",
        "---\nname: missing-close\ndescription: bad\nbody\n",
        "---\nname: [invalid\ndescription: bad\n---\nbody\n",
    ],
)
def test_catalog_isolates_malformed_frontmatter(tmp_path, caplog, content):
    root = tmp_path / "skills"
    skill_dir = root / "bad"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")

    assert len(SkillCatalog.discover(root, platform="linux")) == 0
    assert "Ignoring invalid skill" in caplog.text


def test_catalog_filters_disabled_and_incompatible_skills(tmp_path):
    root = tmp_path / "skills"
    _write_skill(root, "one", name="disabled")
    _write_skill(root, "two", name="linux-only", platforms=["linux"])
    _write_skill(root, "three", name="mac-only", platforms=["darwin"])

    catalog = SkillCatalog.discover(root, disabled={"disabled"}, platform="linux")

    assert catalog.names == ("linux-only",)
    assert SkillCatalog.discover(root, platform="linux-wsl").names == (
        "disabled",
        "linux-only",
    )
    assert SkillCatalog.discover(root, platform="win32").names == ("disabled",)


def test_catalog_excludes_all_duplicate_names_but_keeps_other_skills(tmp_path, caplog):
    root = tmp_path / "skills"
    _write_skill(root, "a", name="duplicate")
    _write_skill(root, "b", name="duplicate")
    _write_skill(root, "c", name="unique")

    catalog = SkillCatalog.discover(root, platform="linux")

    assert catalog.names == ("unique",)
    assert "Ignoring duplicate skill name duplicate" in caplog.text
    assert str(root / "a" / "SKILL.md") in caplog.text
    assert str(root / "b" / "SKILL.md") in caplog.text


def test_catalog_rejects_oversized_and_escaping_files(tmp_path, caplog):
    root = tmp_path / "skills"
    oversized = _write_skill(root, "large", name="large")
    oversized.write_text(oversized.read_text(encoding="utf-8") + ("x" * 12_000), encoding="utf-8")

    outside = _write_skill(tmp_path / "outside", "source", name="escape")
    linked_dir = root / "linked"
    linked_dir.mkdir(parents=True)
    (linked_dir / "SKILL.md").symlink_to(outside)

    catalog = SkillCatalog.discover(root, platform="linux")

    assert len(catalog) == 0
    assert "exceeds 12000 bytes" in caplog.text
    assert "escapes skills root" in caplog.text


def test_catalog_rechecks_path_escape_when_loading(tmp_path):
    root = tmp_path / "skills"
    path = _write_skill(root, "review", name="review")
    catalog = SkillCatalog.discover(root, platform="linux")
    outside = _write_skill(tmp_path / "outside", "source", name="review")
    path.unlink()
    path.symlink_to(outside)

    with pytest.raises(ValueError, match="escapes skills root"):
        catalog.load("review")


def test_skill_load_tool_returns_body_and_readable_errors(tmp_path):
    root = tmp_path / "skills"
    _write_skill(root, "review", name="review", body="Review the implementation.")
    catalog = SkillCatalog.discover(root, platform="linux")
    tool = SkillLoadTool(catalog)

    assert tool.parameters["properties"]["name"]["enum"] == ["review"]
    assert asyncio.run(tool.execute(name="review")) == "Review the implementation."
    assert asyncio.run(tool.execute(name="missing")).startswith("Error loading skill 'missing':")


def test_missing_skills_directory_is_an_empty_catalog(tmp_path):
    catalog = SkillCatalog.discover(tmp_path / "missing", platform="linux")

    assert len(catalog) == 0
    assert catalog.names == ()
    assert catalog.render_for_prompt() == ""
