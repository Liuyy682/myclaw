#!/usr/bin/env python3
"""Create and reconcile the MyClaw roadmap GitHub Project.

The script deliberately uses the public ``gh`` command line interface rather
than making GitHub API requests itself.  All calls go through ``GhRunner`` so
offline tests can provide a small in-memory runner.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


DEFAULT_MANIFEST = ".github/project/myclaw-roadmap.yaml"
DEFAULT_PHASES = ("phase-1", "phase-2")
MARKER_PREFIX = "myclaw-roadmap"
FIELD_VALUES = {
    "Priority": ("P0", "P1"),
    "Effort": ("S", "M", "L", "XL"),
}
REQUIRED_EPIC_FIELDS = (
    "key", "title", "priority", "status", "area", "type", "effort",
    "phase", "summary", "outcome", "out_of_scope", "labels", "children",
)
REQUIRED_CHILD_FIELDS = ("key", "title", "type", "effort", "summary", "acceptance", "validation")


class ConfigError(ValueError):
    """Raised when the roadmap manifest does not satisfy its schema."""


class GhError(RuntimeError):
    """Raised when gh cannot complete a command."""


@dataclass
class GhRunner:
    """Small injectable adapter around ``gh``.

    ``run`` accepts arguments *without* the executable.  Keeping this boundary
    list-based prevents shell interpolation and makes command assertions easy.
    """

    executable: str = "gh"
    transient_attempts: int = 4

    def run(self, args: Sequence[str]) -> str:
        command = [self.executable, *[str(arg) for arg in args]]
        transient_markers = (
            "connection attempt failed",
            "connection reset by peer",
            "context deadline exceeded",
            "dial tcp",
            "i/o timeout",
            "tls handshake timeout",
        )
        for attempt in range(1, self.transient_attempts + 1):
            completed = subprocess.run(command, text=True, capture_output=True)
            if not completed.returncode:
                # ``gh auth status`` writes its useful status (including scopes)
                # to stderr even on success; retain both streams for preflight.
                return "\n".join(part for part in (completed.stdout, completed.stderr) if part)
            detail = (completed.stderr or completed.stdout or "").strip()
            transient = any(marker in detail.lower() for marker in transient_markers)
            if not transient or attempt == self.transient_attempts:
                raise GhError(f"{' '.join(command)} failed: {detail}")
            time.sleep(min(2 ** (attempt - 1), 4))
        raise AssertionError("unreachable")


@dataclass
class IssueSpec:
    key: str
    title: str
    body: str
    labels: list[str]
    epic_key: str | None
    priority: str
    area: str
    phase: str
    type: str
    effort: str
    status: str


@dataclass
class BootstrapReport:
    dry_run: bool
    repo: str
    project: dict[str, Any] = field(default_factory=dict)
    actions: list[dict[str, Any]] = field(default_factory=list)
    issues: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def action(self, operation: str, **details: Any) -> None:
        self.actions.append({"operation": operation, **details})

    def as_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "repo": self.repo,
            "project": self.project,
            "actions": self.actions,
            "issues": self.issues,
            "errors": self.errors,
        }


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path} must be a non-empty string")
    return value.strip()


def _list_of_text(value: Any, path: str, minimum: int = 0) -> list[str]:
    if not isinstance(value, list) or len(value) < minimum:
        raise ConfigError(f"{path} must be a list with at least {minimum} item(s)")
    return [_text(item, f"{path}[{index}]") for index, item in enumerate(value)]


def _phase_values(raw: Mapping[str, Any]) -> tuple[str, str]:
    declared = raw.get("phases")
    if declared is None and isinstance(raw.get("project"), Mapping):
        declared = raw["project"].get("phases")
    if declared is None and isinstance(raw.get("fields"), Mapping):
        declared = raw["fields"].get("Phase")
    if declared is None:
        return DEFAULT_PHASES
    values = _list_of_text(declared, "phases")
    if len(values) != 2 or len(set(values)) != 2:
        raise ConfigError("phases must declare exactly two unique values")
    return values[0], values[1]


def validate_config(raw: Any) -> dict[str, Any]:
    """Validate and normalize a roadmap manifest.

    The returned object is the original structure with normalized inherited
    child values; callers should use it rather than reimplementing inheritance.
    """

    if not isinstance(raw, Mapping):
        raise ConfigError("manifest root must be a mapping")
    epics = raw.get("epics")
    if not isinstance(epics, list) or not epics:
        raise ConfigError("epics must be a non-empty list")
    phases = _phase_values(raw)
    phase_set = set(phases)
    seen: set[str] = set()
    normalized_epics: list[dict[str, Any]] = []
    for epic_index, source in enumerate(epics):
        path = f"epics[{epic_index}]"
        if not isinstance(source, Mapping):
            raise ConfigError(f"{path} must be a mapping")
        missing = [key for key in REQUIRED_EPIC_FIELDS if key not in source]
        if missing:
            raise ConfigError(f"{path} missing required fields: {', '.join(missing)}")
        epic = dict(source)
        key = _text(epic["key"], f"{path}.key")
        if key in seen:
            raise ConfigError(f"duplicate key: {key}")
        seen.add(key)
        for name in ("title", "priority", "status", "area", "type", "effort", "phase", "summary", "outcome", "out_of_scope"):
            epic[name] = _text(epic[name], f"{path}.{name}")
        if epic["priority"] not in {"P0", "P1"}:
            raise ConfigError(f"{path}.priority must be P0 or P1")
        if epic["phase"] not in phase_set:
            raise ConfigError(f"{path}.phase must be one of {sorted(phase_set)}")
        epic["labels"] = _list_of_text(epic["labels"], f"{path}.labels")
        children = epic["children"]
        if not isinstance(children, list):
            raise ConfigError(f"{path}.children must be a list")
        normalized_children: list[dict[str, Any]] = []
        for child_index, child_source in enumerate(children):
            child_path = f"{path}.children[{child_index}]"
            if not isinstance(child_source, Mapping):
                raise ConfigError(f"{child_path} must be a mapping")
            missing = [name for name in REQUIRED_CHILD_FIELDS if name not in child_source]
            if missing:
                raise ConfigError(f"{child_path} missing required fields: {', '.join(missing)}")
            child = dict(child_source)
            child_key = _text(child["key"], f"{child_path}.key")
            for name in ("title", "type", "effort", "summary"):
                child[name] = _text(child[name], f"{child_path}.{name}")
            if child["effort"].upper() == "XL":
                raise ConfigError(f"{child_path}.effort cannot be XL")
            child["acceptance"] = _list_of_text(child["acceptance"], f"{child_path}.acceptance", 2)
            child["validation"] = _list_of_text(child["validation"], f"{child_path}.validation", 1)
            for optional in ("out_of_scope", "security"):
                if optional in child:
                    child[optional] = _text(child[optional], f"{child_path}.{optional}")
            if "dependencies" in child:
                child["dependencies"] = _list_of_text(child["dependencies"], f"{child_path}.dependencies")
            child["priority"] = child.get("priority", epic["priority"])
            child["area"] = child.get("area", epic["area"])
            child["phase"] = child.get("phase", epic["phase"])
            child["status"] = child.get("status", epic["status"])
            for name in ("priority", "area", "phase", "status"):
                child[name] = _text(child[name], f"{child_path}.{name}")
            if child["priority"] not in {"P0", "P1"}:
                raise ConfigError(f"{child_path}.priority must be P0 or P1")
            if child["phase"] not in phase_set:
                raise ConfigError(f"{child_path}.phase must be one of {sorted(phase_set)}")
            child["labels"] = _list_of_text(child.get("labels", []), f"{child_path}.labels")
            child["manifest_key"] = child_key
            child["key"] = f"{key}/{child_key}"
            if child["key"] in seen:
                raise ConfigError(f"duplicate key: {child['key']}")
            seen.add(child["key"])
            normalized_children.append(child)
        epic["children"] = normalized_children
        normalized_epics.append(epic)
    result = dict(raw)
    result["phases"] = list(phases)
    result["epics"] = normalized_epics
    return result


def load_config(path: str | os.PathLike[str] = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest_path = Path(path)
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except OSError as exc:
        raise ConfigError(f"cannot read manifest {manifest_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {manifest_path}: {exc}") from exc
    return validate_config(raw)


def _json_output(value: str) -> Any:
    try:
        return json.loads(value) if value.strip() else None
    except json.JSONDecodeError as exc:
        raise GhError(f"gh returned invalid JSON: {exc}") from exc


def _records(value: Any, key: str | None = None) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    if isinstance(value, Mapping):
        if key and isinstance(value.get(key), list):
            return [item for item in value[key] if isinstance(item, Mapping)]
        for candidate in ("projects", "fields", "items", "issues", "labels", "nodes"):
            if isinstance(value.get(candidate), list):
                return [item for item in value[candidate] if isinstance(item, Mapping)]
    return []


def _marker(key: str) -> str:
    return f"<!-- {MARKER_PREFIX}:{key} -->"


def _run(runner: Any, args: Sequence[str]) -> str:
    """Call either a GhRunner-like object or a plain mock callable."""
    if hasattr(runner, "run"):
        return runner.run(args)
    return runner(args)


def render_epic_body(epic: Mapping[str, Any]) -> str:
    children = epic["children"]
    acceptance = "\n".join(f"- Child issue `{child['key']}` is completed and linked." for child in children)
    validation = "\n".join(f"- {item}" for child in children for item in child["validation"])
    labels = ", ".join(epic["labels"]) or "(none)"
    return "\n".join([
        _marker(epic["key"]), "", "## Summary", epic["summary"], "",
        "## Outcome", epic["outcome"], "", "## Scope", "### Out of scope",
        epic["out_of_scope"], "", "## Acceptance", acceptance or "- No child issues declared.",
        "", "## Validation", validation or "- Validate the completed child issue set.", "",
        "## Metadata", f"- Priority: `{epic['priority']}`", f"- Phase: `{epic['phase']}`",
        f"- Area: `{epic['area']}`", f"- Type: `{epic['type']}`", f"- Effort: `{epic['effort']}`",
        f"- Labels: {labels}", "",
    ])


def render_child_body(child: Mapping[str, Any], parent_key: str) -> str:
    lines = [_marker(child["key"]), "", "## Summary", child["summary"], "", "## Acceptance"]
    lines.extend(f"- {item}" for item in child["acceptance"])
    lines.extend(["", "## Validation"])
    lines.extend(f"- {item}" for item in child["validation"])
    lines.extend(["", "## Scope", "### Out of scope", child.get("out_of_scope", "None specified.")])
    if child.get("security"):
        lines.extend(["", "### Security", child["security"]])
    if child.get("dependencies"):
        lines.extend(["", "### Dependencies"])
        lines.extend(f"- {item}" for item in child["dependencies"])
    lines.extend([
        "", "## Metadata", f"- Parent epic: `{parent_key}`", f"- Priority: `{child['priority']}`",
        f"- Phase: `{child['phase']}`", f"- Area: `{child['area']}`", f"- Type: `{child['type']}`",
        f"- Effort: `{child['effort']}`", "",
    ])
    return "\n".join(lines)


def _project_title(config: Mapping[str, Any], override: str | None = None) -> str:
    project = config.get("project")
    base = project.get("title") if isinstance(project, Mapping) else None
    return override or base or "MyClaw Roadmap"


class ProjectBootstrapper:
    def __init__(self, config: Mapping[str, Any], repo: str, runner: Any | None = None, dry_run: bool = False, *, owner: str | None = None, project_title: str | None = None) -> None:
        self.config = config
        self.repo = repo
        self.owner = owner or repo.split("/", 1)[0]
        self.project_title = project_title
        self.runner = runner or GhRunner()
        self.dry_run = dry_run
        self.report = BootstrapReport(dry_run=dry_run, repo=repo)
        self._project: dict[str, Any] = {}
        self._fields: dict[str, dict[str, Any]] = {}
        self._project_created = False
        self._managed_project = False
        self._project_items: list[dict[str, Any]] | None = None

    def _query(self, args: Sequence[str]) -> Any:
        return _json_output(_run(self.runner, args))

    def _mutate(self, operation: str, args: Sequence[str], **details: Any) -> Any:
        self.report.action(operation, command=["gh", *args], **details)
        if self.dry_run:
            return None
        output = _run(self.runner, args)
        try:
            return _json_output(output)
        except GhError:
            # gh label/issue create return a URL or prose on success.
            return None

    def _mutate_raw(self, operation: str, args: Sequence[str], **details: Any) -> str:
        self.report.action(operation, command=["gh", *args], **details)
        if self.dry_run:
            return ""
        return _run(self.runner, args)

    def preflight(self) -> None:
        if self.repo.count("/") != 1 or any(not part.strip() for part in self.repo.split("/")):
            raise GhError("repo must be in OWNER/NAME form")
        if isinstance(self.runner, GhRunner) and shutil.which(self.runner.executable) is None:
            raise GhError("gh is not installed or not on PATH")
        try:
            auth_output = _run(self.runner, ["auth", "status"])
        except GhError as exc:
            raise GhError(f"gh authentication unavailable: {exc}") from exc
        auth_text = auth_output.lower()
        if any(token in auth_text for token in ("not logged in", "authentication required", "no oauth token")):
            raise GhError("gh authentication unavailable; run `gh auth login`")
        scope_match = re.search(r"scopes?[^\n:]*:\s*([^\n]+)", auth_output, re.IGNORECASE)
        if scope_match:
            scopes = {item.strip(" `'\"") for item in re.split(r"[, ]+", scope_match.group(1)) if item.strip()}
            if "project" not in scopes or not ({"repo", "public_repo"} & scopes):
                raise GhError("gh token is missing required project/repo scope")
        try:
            repository = self._query(["repo", "view", self.repo, "--json", "nameWithOwner"])
        except GhError as exc:
            raise GhError(f"repository is unavailable: {self.repo}") from exc
        actual = repository.get("nameWithOwner") if isinstance(repository, Mapping) else None
        if actual and actual.lower() != self.repo.lower():
            raise GhError(f"repository mismatch: requested {self.repo}, got {actual}")

    def _ensure_project(self) -> None:
        title = _project_title(self.config, self.project_title)
        listed = self._query(["project", "list", "--owner", self.owner, "--format", "json", "--limit", "100"])
        matches = [item for item in _records(listed) if item.get("title") == title]
        if matches:
            self._project = dict(matches[0])
            project_config = self.config.get("project", {})
            expected_description = project_config.get("description") if isinstance(project_config, Mapping) else None
            self._managed_project = bool(expected_description and self._project.get("shortDescription") == expected_description)
            self.report.project = self._project
            self.report.action("project.exists", number=self._project.get("number"), title=title)
            return
        created = self._mutate("project.create", ["project", "create", "--owner", self.owner, "--title", title, "--format", "json"], title=title)
        self._project = dict(created or {"title": title, "number": "<new-project>", "id": "<new-project>"})
        self._project.setdefault("title", title)
        self._project_created = True
        self._managed_project = True
        self.report.project = self._project

    def _ensure_project_settings(self) -> None:
        number = str(self._project.get("number", "<new-project>"))
        project = self.config.get("project", {})
        visibility = str(project.get("visibility", "PUBLIC")) if isinstance(project, Mapping) else "PUBLIC"
        description = str(project.get("description", "")) if isinstance(project, Mapping) else ""
        if self.dry_run and number.startswith("<"):
            self.report.action("project.edit", visibility=visibility, description=description, planned=True)
            self.report.action("project.link", repo=self.repo, planned=True)
            return
        edit_args = ["project", "edit", number, "--owner", self.owner, "--visibility", visibility, "--format", "json"]
        if description:
            edit_args.extend(["--description", description])
        self._mutate("project.edit", edit_args, visibility=visibility)
        self._mutate("project.link", ["project", "link", number, "--owner", self.owner, "--repo", self.repo], repo=self.repo)

    def _ensure_labels(self, labels: Mapping[str, Mapping[str, Any]]) -> None:
        existing_raw = self._query(["label", "list", "--repo", self.repo, "--limit", "1000", "--json", "name,description,color"])
        existing = {str(item.get("name")): item for item in _records(existing_raw)}
        for name in sorted(labels):
            if name in existing:
                continue
            spec = labels[name]
            self._mutate(
                "label.create", ["label", "create", name, "--repo", self.repo, "--color", str(spec.get("color", "1D76DB")), "--description", str(spec.get("description", f"{MARKER_PREFIX} label"))],
                name=name,
            )

    def _ensure_fields(self) -> None:
        number = str(self._project.get("number", "<new-project>"))
        if self.dry_run and number.startswith("<"):
            self._fields = {}
            for name in ("Priority", "Phase", "Area", "Work Type", "Effort"):
                self.report.action("field.create", name=name, planned=True)
            return
        fields_raw = self._query(["project", "field-list", number, "--owner", self.owner, "--format", "json", "--limit", "100"])
        fields = {str(item.get("name")): dict(item) for item in _records(fields_raw)}
        declared_fields = self.config.get("fields", {})
        values: dict[str, list[str]] = {
            "Status": [],
            "Priority": list(FIELD_VALUES["Priority"]),
            "Phase": list(self.config["phases"]),
            "Area": sorted({epic["area"] for epic in self.config["epics"]}),
            "Work Type": sorted({epic["type"] for epic in self.config["epics"]} | {child["type"] for epic in self.config["epics"] for child in epic["children"]}),
            "Effort": list(FIELD_VALUES["Effort"]),
        }
        if isinstance(declared_fields, Mapping):
            for name in values:
                configured = declared_fields.get(name)
                if configured is not None:
                    values[name] = _list_of_text(configured, f"fields.{name}")
        status = fields.get("Status")
        desired_status = values["Status"]
        status_changed = False
        if status and desired_status:
            current_status = [str(option.get("name")) for option in status.get("options", []) if isinstance(option, Mapping)]
            if current_status != desired_status:
                if self._project_created or self._managed_project:
                    colors = ("GRAY", "BLUE", "YELLOW", "PURPLE", "RED", "GREEN")
                    options = ",".join(
                        "{name:%s,description:\"\",color:%s}" % (json.dumps(name), color)
                        for name, color in zip(desired_status, colors, strict=False)
                    )
                    query = (
                        "mutation{updateProjectV2Field(input:{fieldId:%s,singleSelectOptions:[%s]})"
                        "{projectV2Field{... on ProjectV2SingleSelectField{id name options{id name}}}}}"
                    ) % (json.dumps(str(status.get("id"))), options)
                    self._mutate("field.update", ["api", "graphql", "-f", f"query={query}"], name="Status")
                    status_changed = True
                else:
                    self.report.action("field.drift", name="Status", current=current_status, desired=desired_status)
        created_any = False
        for name, options in values.items():
            if name == "Status":
                continue
            if name not in fields:
                created_any = True
                created = self._mutate(
                    "field.create",
                    ["project", "field-create", number, "--owner", self.owner, "--name", name, "--data-type", "SINGLE_SELECT", "--single-select-options", ",".join(options), "--format", "json"],
                    name=name,
                )
                if isinstance(created, Mapping):
                    fields[name] = dict(created)
        if (created_any or status_changed) and not self.dry_run:
            refreshed = self._query(["project", "field-list", number, "--owner", self.owner, "--format", "json", "--limit", "100"])
            fields.update({str(item.get("name")): dict(item) for item in _records(refreshed)})
        self._fields = fields

    def _issue_list(self) -> list[dict[str, Any]]:
        result = self._query(["issue", "list", "--repo", self.repo, "--state", "all", "--limit", "1000", "--json", "number,title,body,url,id,labels"])
        return _records(result)

    def _create_or_find_issue(self, spec: IssueSpec, existing: list[dict[str, Any]]) -> dict[str, Any]:
        marker = _marker(spec.key)
        for issue in existing:
            if marker in str(issue.get("body", "")):
                found = dict(issue)
                found["key"] = spec.key
                self.report.action("issue.exists", key=spec.key, number=found.get("number"))
                self._ensure_issue_labels(found, spec)
                return found
        labels = list(spec.labels)
        create_args = ["issue", "create", "--repo", self.repo, "--title", spec.title, "--body", spec.body]
        for label in labels:
            create_args.extend(["--label", label])
        created_raw = self._mutate_raw("issue.create", create_args, key=spec.key)
        try:
            created = _json_output(created_raw)
        except GhError:
            created = None
        if not isinstance(created, Mapping):
            # ``gh issue create`` intentionally prints a URL, not JSON.  Read
            # that URL with issue view to obtain canonical fields; marker scan
            # is a fallback for runners/older gh versions that omit it.
            issue_url_match = re.search(r"https://github\.com/[^\s]+/issues/\d+", created_raw)
            if issue_url_match:
                try:
                    viewed = self._query(["issue", "view", issue_url_match.group(0), "--repo", self.repo, "--json", "number,title,body,url,id"])
                    if isinstance(viewed, Mapping):
                        created = viewed
                except GhError:
                    pass
            if not isinstance(created, Mapping):
                refreshed = self._issue_list()
                created = next((item for item in refreshed if marker in str(item.get("body", ""))), None)
        issue = dict(created or {"key": spec.key, "number": f"<new:{spec.key}>", "url": f"https://github.com/{self.repo}/issues/<new:{spec.key}>"})
        issue["key"] = spec.key
        issue.setdefault("title", spec.title)
        return issue

    def _ensure_issue_labels(self, issue: Mapping[str, Any], spec: IssueSpec) -> None:
        number = issue.get("number")
        if not isinstance(number, int):
            return
        desired = list(spec.labels)
        current = issue.get("labels", [])
        current_names = {item.get("name") for item in current if isinstance(item, Mapping)}
        current_names.update(item for item in current if isinstance(item, str))
        for label in desired:
            if label not in current_names:
                self._mutate("issue.label-add", ["issue", "edit", str(number), "--repo", self.repo, "--add-label", label], key=spec.key, label=label)

    def _ensure_subissue(self, parent: Mapping[str, Any], child: Mapping[str, Any]) -> None:
        parent_number, child_number = parent.get("number"), child.get("number")
        if not isinstance(parent_number, int) or not isinstance(child_number, int):
            self.report.action("subissue.plan", parent=parent.get("key"), child=child.get("key"))
            return
        try:
            current = self._query(["issue", "view", str(parent_number), "--repo", self.repo, "--json", "subIssues"])
        except GhError:
            current = {}
        subissues = current.get("subIssues", []) if isinstance(current, Mapping) else []
        if isinstance(subissues, Mapping):
            subissues = subissues.get("nodes", [])
        if any(item.get("number") == child_number for item in subissues if isinstance(item, Mapping)):
            self.report.action("subissue.exists", parent=parent.get("key"), child=child.get("key"))
            return
        parent_id = self._query(["issue", "view", str(parent_number), "--repo", self.repo, "--json", "id"]).get("id")
        child_id = self._query(["issue", "view", str(child_number), "--repo", self.repo, "--json", "id"]).get("id")
        if not parent_id or not child_id:
            raise GhError(f"cannot resolve issue node ids for {parent_number} and {child_number}")
        self._mutate(
            "subissue.create",
            ["api", "graphql", "-f", "query=mutation($parentId:ID!,$childId:ID!){addSubIssue(input:{issueId:$parentId,subIssueId:$childId}){issue{id}}}", "-f", f"parentId={parent_id}", "-f", f"childId={child_id}"],
            parent=parent.get("key"), child=child.get("key"),
        )

    def _add_and_set_item(self, issue: Mapping[str, Any], spec: IssueSpec) -> None:
        number = str(self._project.get("number", "<new-project>"))
        if self.dry_run and number.startswith("<"):
            self.report.action("project.item-add", key=spec.key, planned=True)
            self.report.action("project.fields.plan", key=spec.key)
            return
        if self._project_items is None:
            existing_raw = self._query(["project", "item-list", number, "--owner", self.owner, "--format", "json", "--limit", "200"])
            self._project_items = _records(existing_raw)
        item = next((item for item in self._project_items if item.get("content", {}).get("number") == issue.get("number") or item.get("content", {}).get("url") == issue.get("url")), None)
        if item is None:
            item = self._mutate("project.item-add", ["project", "item-add", number, "--owner", self.owner, "--url", str(issue.get("url")), "--format", "json"], key=spec.key) or {"id": f"<new-item:{spec.key}>"}
            if isinstance(item, Mapping):
                cached_item = dict(item)
                cached_item.setdefault("content", {"number": issue.get("number"), "url": issue.get("url")})
                self._project_items.append(cached_item)
        else:
            self.report.action("project.item.exists", key=spec.key, item_id=item.get("id"))
        item_id = item.get("id") if isinstance(item, Mapping) else None
        project_id = self._project.get("id")
        if not item_id or not project_id or str(item_id).startswith("<") or str(project_id).startswith("<"):
            self.report.action("project.fields.plan", key=spec.key)
            return
        values = {"Status": spec.status, "Priority": spec.priority, "Phase": spec.phase, "Area": spec.area, "Work Type": spec.type, "Effort": spec.effort}
        for name, value in values.items():
            field_info = self._fields.get(name, {})
            field_id = field_info.get("id")
            options = {str(option.get("name")): option.get("id") for option in field_info.get("options", []) if isinstance(option, Mapping)}
            option_id = options.get(value)
            if field_id and option_id:
                self._mutate("project.item-edit", ["project", "item-edit", "--id", str(item_id), "--project-id", str(project_id), "--field-id", str(field_id), "--single-select-option-id", str(option_id)], key=spec.key, field=name, value=value)
            else:
                self.report.action("project.field-unavailable", key=spec.key, field=name, value=value)

    def run(self) -> BootstrapReport:
        self._ensure_project()
        self._ensure_project_settings()
        label_specs: dict[str, dict[str, Any]] = {}
        configured_labels = self.config.get("labels", [])
        if isinstance(configured_labels, list):
            for item in configured_labels:
                if isinstance(item, Mapping):
                    name = _text(item.get("name"), "labels[].name")
                    label_specs[name] = dict(item)
                else:
                    name = _text(item, "labels[]")
                    label_specs[name] = {}
        specs: list[IssueSpec] = []
        for epic in self.config["epics"]:
            for name in epic["labels"]:
                label_specs.setdefault(name, {})
            specs.append(IssueSpec(epic["key"], epic["title"], render_epic_body(epic), epic["labels"], None, epic["priority"], epic["area"], epic["phase"], epic["type"], epic["effort"], epic["status"]))
            for child in epic["children"]:
                for name in child["labels"]:
                    label_specs.setdefault(name, {})
                specs.append(IssueSpec(child["key"], child["title"], render_child_body(child, epic["key"]), child["labels"], epic["key"], child["priority"], child["area"], child["phase"], child["type"], child["effort"], child["status"]))
        self._ensure_labels(label_specs)
        self._ensure_fields()
        existing = self._issue_list()
        created: dict[str, dict[str, Any]] = {}
        for spec in specs:
            issue = self._create_or_find_issue(spec, existing)
            created[spec.key] = issue
            self.report.issues.append({"key": spec.key, "number": issue.get("number"), "url": issue.get("url"), "existing": any(_marker(spec.key) in str(item.get("body", "")) for item in existing)})
            existing.append(issue)
            self._add_and_set_item(issue, spec)
        for epic in self.config["epics"]:
            for child in epic["children"]:
                self._ensure_subissue(created[epic["key"]], created[child["key"]])
        return self.report


def preflight(repo: str, runner: Any | None = None) -> None:
    ProjectBootstrapper({"epics": [], "phases": list(DEFAULT_PHASES)}, repo, runner=runner).preflight()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", help="GitHub repository in OWNER/NAME form (defaults to manifest project.repository)")
    parser.add_argument("--owner", help="Project owner (defaults to manifest project.owner or repo owner)")
    parser.add_argument("--project-title", help="Project title (defaults to manifest project.title)")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--apply", action="store_true")
    parser.add_argument("--json-report", nargs="?", const="-", help="write the JSON report to PATH, or stdout when omitted")
    return parser


def main(argv: Sequence[str] | None = None, *, runner: Any | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.manifest)
        project = config.get("project", {})
        repo = args.repo or (project.get("repository") if isinstance(project, Mapping) else None) or os.environ.get("GH_REPO")
        if not repo:
            raise ConfigError("repo is required (pass --repo or set project.repository)")
        owner = args.owner or (project.get("owner") if isinstance(project, Mapping) else None) or repo.split("/", 1)[0]
        bootstrapper = ProjectBootstrapper(config, repo, runner=runner, dry_run=args.dry_run, owner=owner, project_title=args.project_title)
        bootstrapper.preflight()
        report = bootstrapper.run().as_dict()
    except (ConfigError, GhError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json_report:
        rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.json_report == "-":
            print(rendered, end="")
        else:
            Path(args.json_report).write_text(rendered, encoding="utf-8")
    else:
        print(f"{'planned' if args.dry_run else 'applied'} {len(report['issues'])} issues for {args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
