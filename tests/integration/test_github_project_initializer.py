from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "bootstrap_github_project.py"
SPEC = importlib.util.spec_from_file_location("bootstrap_github_project", SCRIPT)
assert SPEC and SPEC.loader
bootstrap = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bootstrap
SPEC.loader.exec_module(bootstrap)


def manifest(**overrides):
    value = {
        "phases": ["build", "harden"],
        "project": {"title": "MyClaw Roadmap", "visibility": "PUBLIC", "description": "Roadmap"},
        "fields": {"Status": ["Backlog", "Ready", "In progress", "In review", "Blocked", "Done"]},
        "epics": [{
            "key": "E-001", "title": "Foundation", "priority": "P0", "status": "planned",
            "area": "runtime", "type": "epic", "effort": "L", "phase": "build",
            "summary": "Build the foundation.", "outcome": "A usable foundation.",
            "out_of_scope": "No hosted deployment.", "labels": ["area:runtime"],
            "children": [{
                "key": "E-001-C01", "title": "Implement runner", "type": "feature", "effort": "M",
                "summary": "Implement the runner.", "acceptance": ["It runs.", "It fails safely."],
                "validation": ["Run the focused tests."],
            }],
        }],
    }
    value.update(overrides)
    return value


class FakeGh:
    def __init__(self):
        self.calls = []

    def run(self, args):
        args = list(args)
        self.calls.append(args)
        if args[:2] == ["auth", "status"]:
            return "Logged in\nToken scopes: project, repo\n"
        if args[:2] == ["repo", "view"]:
            return json.dumps({"nameWithOwner": "acme/myclaw"})
        if args[:3] == ["project", "list", "--owner"]:
            return "[]"
        if args[:2] == ["label", "list"]:
            return "[]"
        if args[:2] == ["project", "field-list"]:
            return json.dumps({"fields": [{"id": "STATUS-1", "name": "Status", "options": [{"id": "TODO", "name": "Todo"}]}]})
        if args[:2] == ["issue", "list"]:
            return "[]"
        if args[:2] == ["project", "item-list"]:
            return "[]"
        if args[:2] == ["issue", "view"]:
            if args[2].startswith("https://"):
                number = int(args[2].rsplit("/", 1)[1])
                return json.dumps({"id": f"I-{number}", "number": number, "title": "created", "body": "<!-- myclaw-roadmap:E-001 -->", "url": args[2]})
            return json.dumps({"subIssues": {"nodes": [], "totalCount": 0}, "id": "I-1"})
        if args[:2] == ["project", "create"]:
            return json.dumps({"id": "P-1", "number": 7, "title": "MyClaw Roadmap"})
        if args[:2] in (["project", "edit"], ["project", "link"]):
            return "{}"
        if args[:2] == ["project", "field-create"]:
            return json.dumps({"id": "F-1", "name": args[args.index("--name") + 1], "options": []})
        if args[:2] == ["label", "create"]:
            return "{}"
        if args[:2] == ["issue", "create"]:
            title = args[args.index("--title") + 1]
            number = 10 + sum(call[:2] == ["issue", "create"] for call in self.calls)
            return f"https://github.com/acme/myclaw/issues/{number}\n"
        if args[:2] == ["project", "item-add"]:
            return json.dumps({"id": "ITEM-1"})
        if args[:2] == ["api", "graphql"]:
            return "{}"
        raise AssertionError(f"unexpected gh call: {args}")


def test_validate_inherits_child_fields_and_rejects_bad_values():
    config = bootstrap.validate_config(manifest())
    child = config["epics"][0]["children"][0]
    assert child["priority"] == "P0"
    assert child["area"] == "runtime"
    assert child["phase"] == "build"
    with pytest.raises(bootstrap.ConfigError, match="P0 or P1"):
        bootstrap.validate_config(manifest(epics=[{**manifest()["epics"][0], "priority": "P2"}]))
    bad_child = {**child, "effort": "XL"}
    with pytest.raises(bootstrap.ConfigError, match="cannot be XL"):
        bootstrap.validate_config(manifest(epics=[{**manifest()["epics"][0], "children": [bad_child]}]))


def test_validate_requires_unique_keys_and_acceptance():
    epic = manifest()["epics"][0]
    duplicate = {**epic, "key": "E-001"}
    with pytest.raises(bootstrap.ConfigError, match="duplicate key"):
        bootstrap.validate_config(manifest(epics=[epic, duplicate]))
    missing_acceptance = {**epic, "children": [{**epic["children"][0], "acceptance": ["only one"]}]}
    with pytest.raises(bootstrap.ConfigError, match="acceptance"):
        bootstrap.validate_config(manifest(epics=[missing_acceptance]))


def test_dry_run_does_not_execute_mutations_and_renders_markers():
    fake = FakeGh()
    config = bootstrap.validate_config(manifest())
    report = bootstrap.ProjectBootstrapper(config, "acme/myclaw", runner=fake, dry_run=True).run()
    mutating = {"project", "label", "field-create", "issue", "api"}
    assert not any(call[0] in mutating and call[1] in {"create", "item-add", "graphql"} for call in fake.calls)
    bodies = [action.get("command", []) for action in report.actions if action["operation"] == "issue.create"]
    assert bodies and bootstrap._marker("E-001") in bodies[0][bodies[0].index("--body") + 1]
    assert any(action["operation"] == "project.create" for action in report.actions)


def test_apply_uses_argument_arrays_and_is_idempotent_for_existing_markers():
    fake = FakeGh()
    config = bootstrap.validate_config(manifest())
    report = bootstrap.ProjectBootstrapper(config, "acme/myclaw", runner=fake, dry_run=False).run()
    assert report.issues
    assert all(isinstance(call, list) for call in fake.calls)
    assert any(call[:2] == ["project", "create"] for call in fake.calls)
    assert any(call[:2] == ["project", "edit"] for call in fake.calls)
    assert any(call[:2] == ["project", "link"] for call in fake.calls)
    project_create = next(call for call in fake.calls if call[:2] == ["project", "create"])
    assert project_create[project_create.index("--title") + 1] == "MyClaw Roadmap"
    assert any(call[:2] == ["issue", "create"] for call in fake.calls)
    issue_create = next(call for call in fake.calls if call[:2] == ["issue", "create"])
    assert "--body" in issue_create
    assert bootstrap._marker("E-001") in issue_create[issue_create.index("--body") + 1]
    assert any(call[:2] == ["issue", "view"] and call[2].startswith("https://") for call in fake.calls)
    field_create = next(call for call in fake.calls if call[:2] == ["project", "field-create"])
    assert "--name" in field_create and "--title" not in field_create
    created_field_names = {
        call[call.index("--name") + 1]
        for call in fake.calls
        if call[:2] == ["project", "field-create"]
    }
    assert "Work Type" in created_field_names
    assert "Type" not in created_field_names
    assert sum(call[:2] == ["project", "item-list"] for call in fake.calls) == 1
    assert any(call[:2] == ["api", "graphql"] for call in fake.calls)


def test_main_requires_exactly_one_mode(tmp_path):
    path = tmp_path / "roadmap.yaml"
    path.write_text("epics: []\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        bootstrap.build_parser().parse_args(["--repo", "acme/myclaw"])


def test_real_manifest_loads_its_two_declared_phases():
    config = bootstrap.load_config(Path(__file__).parents[2] / ".github/project/myclaw-roadmap.yaml")
    assert config["phases"] == ["Phase 1 - Core workflow", "Phase 2 - Productization"]
    assert len(config["epics"]) == 9
    assert any(child["key"] == "memory-recovery/recovery-tests" for epic in config["epics"] for child in epic["children"])
