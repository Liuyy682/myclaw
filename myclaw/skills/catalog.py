from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
import sys

import yaml

from myclaw.config import SKILL_MAX_FILE_BYTES


logger = logging.getLogger(__name__)

SKILL_FILENAME = "SKILL.md"
_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_SUPPORTED_PLATFORMS = frozenset({"linux", "darwin", "windows"})


@dataclass(frozen=True, slots=True)
class SkillMetadata:
    name: str
    description: str
    platforms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SkillDefinition:
    metadata: SkillMetadata
    body: str
    source: Path


@dataclass(frozen=True, slots=True)
class _SkillEntry:
    metadata: SkillMetadata
    path: Path


class SkillCatalog:
    """Discover safe SKILL.md metadata and load skill bodies on demand."""

    def __init__(
        self,
        root: Path,
        entries: dict[str, _SkillEntry] | None = None,
        *,
        max_file_bytes: int = SKILL_MAX_FILE_BYTES,
    ) -> None:
        self.root = root
        self._entries = dict(entries or {})
        self.max_file_bytes = max_file_bytes

    @classmethod
    def discover(
        cls,
        root: Path | str,
        disabled: set[str] | tuple[str, ...] | list[str] = (),
        platform: str | None = None,
        *,
        max_file_bytes: int = SKILL_MAX_FILE_BYTES,
    ) -> SkillCatalog:
        skill_root = Path(root).expanduser()
        if not skill_root.exists():
            return cls(skill_root, max_file_bytes=max_file_bytes)
        if not skill_root.is_dir():
            logger.warning("Skills root is not a directory: %s", skill_root)
            return cls(skill_root, max_file_bytes=max_file_bytes)

        current_platform = _normalize_platform(platform or sys.platform)
        disabled_names = set(disabled)
        candidates: dict[str, list[_SkillEntry]] = {}
        for path in sorted(skill_root.glob(f"*/{SKILL_FILENAME}")):
            try:
                resolved = _validated_path(path, skill_root)
                metadata = _read_metadata(resolved, max_file_bytes=max_file_bytes)
            except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
                logger.warning("Ignoring invalid skill at %s: %s", path, exc)
                continue
            if metadata.name in disabled_names:
                continue
            if metadata.platforms and current_platform not in metadata.platforms:
                continue
            candidates.setdefault(metadata.name, []).append(_SkillEntry(metadata, path))

        entries: dict[str, _SkillEntry] = {}
        for name, matches in candidates.items():
            if len(matches) > 1:
                sources = ", ".join(str(match.path) for match in matches)
                logger.warning("Ignoring duplicate skill name %s from: %s", name, sources)
                continue
            entries[name] = matches[0]
        return cls(skill_root, entries, max_file_bytes=max_file_bytes)

    @property
    def metadata(self) -> tuple[SkillMetadata, ...]:
        return tuple(self._entries[name].metadata for name in sorted(self._entries))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def render_for_prompt(self) -> str:
        if not self._entries:
            return ""
        lines = [
            "Available skills:",
            "Call skill_load only when a skill description matches the current task.",
        ]
        lines.extend(f"- {item.name}: {item.description}" for item in self.metadata)
        return "\n".join(lines)

    def load(self, name: str) -> SkillDefinition:
        entry = self._entries.get(name)
        if entry is None:
            available = ", ".join(self.names) or "(none)"
            raise ValueError(f"Skill '{name}' not found. Available: {available}")

        resolved = _validated_path(entry.path, self.root)
        text = _read_document(resolved, max_file_bytes=self.max_file_bytes)
        frontmatter, body = _split_document(text)
        try:
            metadata = _parse_metadata(frontmatter)
        except yaml.YAMLError as exc:
            raise ValueError(f"Skill frontmatter is invalid: {exc}") from exc
        if metadata != entry.metadata:
            raise ValueError(f"Skill metadata changed after discovery: {name}")
        return SkillDefinition(metadata=metadata, body=body.strip(), source=resolved)

    def __len__(self) -> int:
        return len(self._entries)


def _normalize_platform(value: str) -> str:
    normalized = value.strip().lower()
    if normalized.startswith("linux"):
        return "linux"
    if normalized == "darwin":
        return "darwin"
    if normalized.startswith(("win32", "cygwin", "msys", "windows")):
        return "windows"
    raise ValueError(f"Unsupported runtime platform: {value}")


def _validated_path(path: Path, root: Path) -> Path:
    root_resolved = root.resolve()
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"Skill path escapes skills root: {path}") from exc
    if not resolved.is_file():
        raise ValueError(f"Skill path is not a file: {path}")
    return resolved


def _read_metadata(path: Path, *, max_file_bytes: int) -> SkillMetadata:
    _validate_size(path, max_file_bytes)
    with path.open("rb") as handle:
        first = handle.readline()
        bytes_read = len(first)
        if first.rstrip(b"\r\n") != b"---":
            raise ValueError("SKILL.md must start with YAML frontmatter")
        frontmatter_lines: list[bytes] = []
        for line in handle:
            bytes_read += len(line)
            if bytes_read > max_file_bytes:
                raise ValueError(f"SKILL.md exceeds {max_file_bytes} bytes")
            if line.rstrip(b"\r\n") == b"---":
                return _parse_metadata(b"".join(frontmatter_lines).decode("utf-8"))
            frontmatter_lines.append(line)
    raise ValueError("SKILL.md frontmatter is not closed")


def _read_document(path: Path, *, max_file_bytes: int) -> str:
    _validate_size(path, max_file_bytes)
    payload = path.read_bytes()
    if len(payload) > max_file_bytes:
        raise ValueError(f"SKILL.md exceeds {max_file_bytes} bytes")
    return payload.decode("utf-8")


def _validate_size(path: Path, max_file_bytes: int) -> None:
    if max_file_bytes < 1:
        raise ValueError("max_file_bytes must be at least 1")
    if path.stat().st_size > max_file_bytes:
        raise ValueError(f"SKILL.md exceeds {max_file_bytes} bytes")


def _split_document(text: str) -> tuple[str, str]:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        raise ValueError("SKILL.md must start with YAML frontmatter")
    for index, line in enumerate(lines[1:], start=1):
        if line.rstrip("\r\n") == "---":
            return "".join(lines[1:index]), "".join(lines[index + 1 :])
    raise ValueError("SKILL.md frontmatter is not closed")


def _parse_metadata(frontmatter: str) -> SkillMetadata:
    raw = yaml.safe_load(frontmatter)
    if not isinstance(raw, dict):
        raise ValueError("SKILL.md frontmatter must be a mapping")

    name = raw.get("name")
    if not isinstance(name, str) or not _NAME_PATTERN.fullmatch(name):
        raise ValueError("skill name must be a lowercase ASCII slug of at most 64 characters")

    description = raw.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("skill description is required")
    description = description.strip()
    if len(description) > 500:
        raise ValueError("skill description must be at most 500 characters")

    raw_platforms = raw.get("platforms")
    if raw_platforms is None:
        platforms: tuple[str, ...] = ()
    else:
        if not isinstance(raw_platforms, list) or not all(isinstance(item, str) for item in raw_platforms):
            raise ValueError("skill platforms must be a list of strings")
        normalized = tuple(dict.fromkeys(item.strip().lower() for item in raw_platforms))
        unknown = sorted(set(normalized) - _SUPPORTED_PLATFORMS)
        if unknown:
            raise ValueError(f"unknown skill platforms: {', '.join(unknown)}")
        platforms = normalized
    return SkillMetadata(name=name, description=description, platforms=platforms)
