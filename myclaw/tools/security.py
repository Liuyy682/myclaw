"""Central authorization and side-effect bookkeeping for tool calls.

The security layer deliberately keeps the data it persists small.  Arguments
and tool results are never written to the security database; only canonical
hashes, argument keys, and result size/hash metadata are retained.  The
operation table is a duplicate-call guard and a recovery hint, not a claim of
exactly-once execution for an external side effect.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, TypedDict


DEFAULT_APPROVAL_TTL_SECONDS = 300.0
DEFAULT_TOOL_TIMEOUT_SECONDS = 120.0

_ALLOW = "allow"
_ASK = "ask"
_DENY = "deny"
ALLOW = _ALLOW
ASK = _ASK
DENY = _DENY


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _timestamp(value: datetime | str | None = None) -> str:
    if value is None:
        value = _utc_now()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat()
    return str(value)


def _parse_timestamp(value: datetime | str | float | int | None) -> datetime:
    if value is None:
        return _utc_now()
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=UTC)
    if isinstance(value, datetime):
        return value.replace(tzinfo=value.tzinfo or UTC).astimezone(UTC)
    text = str(value).strip()
    if not text:
        return _utc_now()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        # A malformed expiry is treated as already expired by callers that use
        # it for authorization; never turn it into a non-expiring approval.
        return datetime.fromtimestamp(0, tz=UTC)
    return parsed.replace(tzinfo=parsed.tzinfo or UTC).astimezone(UTC)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _canonical_value(value: Any) -> Any:
    """Return a JSON-compatible value with deterministic mapping order.

    JSON object ordering is normalized by ``canonical_args``.  Mapping keys
    are required to be strings, matching the tool-call wire format; accepting
    arbitrary Python keys would make two callers disagree about the operation
    identity.
    """

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in value:
            if not isinstance(key, str):
                raise TypeError("tool argument object keys must be strings")
            result[key] = _canonical_value(value[key])
        return dict(sorted(result.items()))
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"tool arguments are not JSON-compatible: {type(value).__name__}")


def canonical_args(args: Mapping[str, Any] | None) -> str:
    """Serialize tool arguments into a stable, compact JSON representation."""

    if args is None:
        args = {}
    if not isinstance(args, Mapping):
        raise TypeError("tool arguments must be a mapping")
    normalized = _canonical_value(args)
    return _json(dict(sorted(normalized.items())))


def canonical_args_hash(args: Mapping[str, Any] | None) -> str:
    """Return the SHA-256 hash used by approvals and operation identities."""

    return hashlib.sha256(canonical_args(args).encode("utf-8")).hexdigest()


# Short aliases are useful to adapters which already call this an argument
# digest.  They intentionally point to the same implementation.
canonicalize_args = canonical_args
args_hash = canonical_args_hash


def operation_id(
    session_key: str,
    tool_call_id: str,
    tool: str,
    args_or_hash: Mapping[str, Any] | str | None,
) -> str:
    """Build a stable operation id from session/call/tool/canonical args.

    A call id is intentionally part of the identity: two deliberate calls
    with equal arguments are allowed, while a replay of one call id is
    rejected by the operation ledger.
    """

    digest = (
        canonical_args_hash(args_or_hash)
        if not isinstance(args_or_hash, str) or len(args_or_hash) != 64
        else args_or_hash
    )
    payload = _json(
        {
            "session_key": str(session_key),
            "tool_call_id": str(tool_call_id),
            "tool": str(tool),
            "args_hash": digest,
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


make_operation_id = operation_id


def result_digest(result: str | bytes | Any) -> tuple[str, int]:
    """Return a result hash and logical length without persisting result data."""

    if not isinstance(result, (str, bytes)):
        result = _json(result)
    raw = result if isinstance(result, bytes) else result.encode("utf-8")
    return hashlib.sha256(raw).hexdigest(), len(result)


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class ResultEnvelope(TypedDict):
    status: str
    ok: bool
    tool: str
    tool_call_id: str
    operation_id: str | None
    args_hash: str | None
    decision: str | None
    result: str
    content: str
    error: str | None
    error_type: str | None
    result_hash: str | None
    result_len: int | None
    untrusted: bool


@dataclass(slots=True, eq=False)
class PolicyDecision:
    action: str
    reason: str = ""
    tool: str = ""
    args_hash: str = ""
    approval_id: str | None = None

    @property
    def decision(self) -> str:
        return self.action

    @property
    def allowed(self) -> bool:
        return self.action == _ALLOW

    @property
    def requires_approval(self) -> bool:
        return self.action == _ASK

    def __str__(self) -> str:
        return self.action

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.action == other
        if not isinstance(other, PolicyDecision):
            return NotImplemented
        return (
            self.action,
            self.reason,
            self.tool,
            self.args_hash,
            self.approval_id,
        ) == (
            other.action,
            other.reason,
            other.tool,
            other.args_hash,
            other.approval_id,
        )


ApprovalCallback = Callable[[str, list[str]], Awaitable[str] | str]


class SecurityStore:
    """SQLite store for approvals, operation states, audit metadata, and MCP approvals."""

    def __init__(
        self,
        workspace: Path | str | None = None,
        *,
        path: Path | str | None = None,
        database_path: Path | str | None = None,
    ) -> None:
        configured_path = path if path is not None else database_path
        if configured_path is not None:
            candidate = Path(configured_path).expanduser()
            # A direct .db path is convenient for tests and registry injection;
            # a directory still follows the standard workspace layout.
            self.path = candidate if candidate.suffix == ".db" else candidate / "security" / "tool_security.db"
        else:
            root = Path(workspace).expanduser() if workspace is not None else Path.cwd()
            self.path = root / "security" / "tool_security.db"
        self.directory = self.path.parent
        self.initialize()

    def initialize(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS approvals (
                    approval_id TEXT PRIMARY KEY,
                    security_subject TEXT NOT NULL DEFAULT '',
                    session_key TEXT NOT NULL DEFAULT '',
                    channel TEXT NOT NULL DEFAULT '',
                    tool_call_id TEXT NOT NULL DEFAULT '',
                    tool TEXT NOT NULL,
                    args_hash TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    consumed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_approvals_lookup
                    ON approvals(security_subject, session_key, channel, tool_call_id, tool, args_hash);
                CREATE TABLE IF NOT EXISTS operations (
                    operation_id TEXT PRIMARY KEY,
                    security_subject TEXT NOT NULL DEFAULT '',
                    session_key TEXT NOT NULL DEFAULT '',
                    channel TEXT NOT NULL DEFAULT '',
                    tool_call_id TEXT NOT NULL DEFAULT '',
                    tool TEXT NOT NULL,
                    args_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    result_hash TEXT,
                    result_len INTEGER,
                    error_type TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_operations_session
                    ON operations(session_key, created_at DESC);
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    security_subject TEXT NOT NULL DEFAULT '',
                    session_key TEXT NOT NULL DEFAULT '',
                    channel TEXT NOT NULL DEFAULT '',
                    tool_call_id TEXT NOT NULL DEFAULT '',
                    tool TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    operation_id TEXT,
                    argument_keys_json TEXT NOT NULL DEFAULT '[]',
                    args_hash TEXT NOT NULL DEFAULT '',
                    result_hash TEXT,
                    result_len INTEGER,
                    error_type TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_audit_events_time
                    ON audit_events(timestamp DESC);
                CREATE TABLE IF NOT EXISTS mcp_configs (
                    server TEXT PRIMARY KEY,
                    config_hash TEXT NOT NULL,
                    env_keys_json TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL,
                    approved_at TEXT,
                    revoked_at TEXT
                );
                """
            )

    def recover_incomplete_operations(self) -> int:
        """Mark operations left prepared/running by a prior process unknown.

        Recovery is explicit on purpose.  A second ``SecurityStore`` opened
        by a live component must not invalidate an operation currently being
        executed by the first component.
        """

        now = _timestamp()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE operations
                   SET status = 'unknown', finished_at = COALESCE(finished_at, ?)
                 WHERE status IN ('prepared', 'running')
                """,
                (now,),
            )
            return cursor.rowcount

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    # --- approvals ---------------------------------------------------------

    def create_approval(
        self,
        *,
        security_subject: str = "",
        subject: str | None = None,
        session_key: str = "",
        channel: str = "",
        tool_call_id: str = "",
        tool: str,
        args_hash: str,
        expires_at: datetime | str | float | int | None = None,
        expiry: datetime | str | float | int | None = None,
        ttl_seconds: float = DEFAULT_APPROVAL_TTL_SECONDS,
        approval_id: str | None = None,
    ) -> str:
        if subject is not None:
            security_subject = subject
        if expires_at is None:
            expires_at = expiry
        if expires_at is None:
            expires_at = _utc_now() + timedelta(seconds=max(0.0, float(ttl_seconds)))
        approval_id = approval_id or uuid.uuid4().hex
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO approvals (
                    approval_id, security_subject, session_key, channel,
                    tool_call_id, tool, args_hash, expires_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    approval_id,
                    security_subject,
                    session_key,
                    channel,
                    tool_call_id,
                    tool,
                    args_hash,
                    _timestamp(expires_at if isinstance(expires_at, datetime) else _parse_timestamp(expires_at)),
                    _timestamp(),
                ),
            )
        return approval_id

    # Names used by different adapters all mean “create a bound approval”.
    issue_approval = create_approval
    add_approval = create_approval
    approve = create_approval
    grant_approval = create_approval

    def consume_approval(
        self,
        *,
        approval_id: str | None = None,
        security_subject: str = "",
        subject: str | None = None,
        session_key: str = "",
        channel: str = "",
        tool_call_id: str = "",
        tool: str,
        args_hash: str,
    ) -> bool:
        if subject is not None:
            security_subject = subject
        now = _utc_now()
        clauses = [
            "security_subject = ?", "session_key = ?", "channel = ?",
            "tool_call_id = ?", "tool = ?", "args_hash = ?",
            "consumed_at IS NULL", "expires_at > ?",
        ]
        params: list[Any] = [
            security_subject, session_key, channel, tool_call_id, tool,
            args_hash, _timestamp(now),
        ]
        if approval_id is not None:
            clauses.insert(0, "approval_id = ?")
            params.insert(0, approval_id)
        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE approvals SET consumed_at = ? WHERE {' AND '.join(clauses)}",
                [_timestamp(now), *params],
            )
            return cursor.rowcount == 1

    use_approval = consume_approval
    approve_call = create_approval

    def list_approvals(self, *, include_consumed: bool = True) -> list[dict[str, Any]]:
        query = "SELECT * FROM approvals"
        if not include_consumed:
            query += " WHERE consumed_at IS NULL"
        query += " ORDER BY created_at DESC"
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(query).fetchall()]

    # --- operation ledger --------------------------------------------------

    def prepare_operation(
        self,
        *,
        operation_id: str,
        security_subject: str = "",
        session_key: str = "",
        channel: str = "",
        tool_call_id: str = "",
        tool: str,
        args_hash: str,
    ) -> bool:
        """Reserve an operation id; return False when it was already seen."""

        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO operations (
                    operation_id, security_subject, session_key, channel,
                    tool_call_id, tool, args_hash, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'prepared', ?)
                """,
                (
                    operation_id, security_subject, session_key, channel,
                    tool_call_id, tool, args_hash, _timestamp(),
                ),
            )
            return cursor.rowcount == 1

    reserve_operation = prepare_operation
    begin_operation = prepare_operation

    def start_operation(self, operation_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE operations SET status = 'running', started_at = ?
                 WHERE operation_id = ? AND status = 'prepared'
                """,
                (_timestamp(), operation_id),
            )
            return cursor.rowcount == 1

    def finish_operation(
        self,
        operation_id: str,
        *,
        status: str,
        result: Any = None,
        result_hash: str | None = None,
        result_len: int | None = None,
        error_type: str | None = None,
    ) -> bool:
        if result_hash is None and result is not None:
            result_hash, result_len = result_digest(result)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE operations
                   SET status = ?, finished_at = ?, result_hash = ?,
                       result_len = ?, error_type = ?
                 WHERE operation_id = ?
                """,
                (status, _timestamp(), result_hash, result_len, error_type, operation_id),
            )
            return cursor.rowcount == 1

    complete_operation = finish_operation
    mark_operation = finish_operation

    def get_operation(self, operation_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM operations WHERE operation_id = ?", (operation_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def list_operations(self, *, session_key: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM operations"
        params: list[Any] = []
        if session_key is not None:
            query += " WHERE session_key = ?"
            params.append(session_key)
        query += " ORDER BY created_at DESC"
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(query, params).fetchall()]

    # --- privacy-preserving audit -----------------------------------------

    def record_audit_event(
        self,
        *,
        security_subject: str = "",
        subject: str | None = None,
        session_key: str = "",
        channel: str = "",
        tool_call_id: str = "",
        tool: str,
        decision: str,
        operation_id: str | None = None,
        argument_keys: list[str] | tuple[str, ...] | None = None,
        args_hash: str = "",
        result: Any = None,
        result_hash: str | None = None,
        result_len: int | None = None,
        error_type: str | None = None,
    ) -> int:
        if subject is not None:
            security_subject = subject
        if result_hash is None and result is not None:
            result_hash, result_len = result_digest(result)
        keys = sorted({str(key) for key in (argument_keys or [])})
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO audit_events (
                    timestamp, security_subject, session_key, channel,
                    tool_call_id, tool, decision, operation_id,
                    argument_keys_json, args_hash, result_hash, result_len, error_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _timestamp(), security_subject, session_key, channel,
                    tool_call_id, tool, decision, operation_id,
                    _json(keys), args_hash, result_hash, result_len, error_type,
                ),
            )
            return int(cursor.lastrowid)

    audit = record_audit_event
    audit_event = record_audit_event

    def list_audit_events(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM audit_events ORDER BY id DESC LIMIT ?", (max(1, int(limit)),)
            ).fetchall()
        events = []
        for row in rows:
            event = dict(row)
            try:
                event["argument_keys"] = json.loads(event.pop("argument_keys_json"))
            except (TypeError, json.JSONDecodeError):
                event["argument_keys"] = []
            events.append(event)
        return events

    # --- MCP configuration approval ---------------------------------------

    def mcp_status(self, server: str, config_hash: str | None = None) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM mcp_configs WHERE server = ?", (server,)
            ).fetchone()
        if row is None:
            return {
                "server": server,
                "config_hash": config_hash or "",
                "status": "unapproved",
                "approved": False,
                "env_keys": [],
            }
        stored_hash = str(row["config_hash"])
        status = str(row["status"])
        approved = status == "approved" and (config_hash is None or config_hash == stored_hash)
        if status == "approved" and config_hash is not None and config_hash != stored_hash:
            status = "config_changed"
        try:
            env_keys = json.loads(row["env_keys_json"])
        except (TypeError, json.JSONDecodeError):
            env_keys = []
        return {
            "server": server,
            "config_hash": stored_hash,
            "status": status,
            "approved": approved,
            "env_keys": env_keys if isinstance(env_keys, list) else [],
            "approved_at": row["approved_at"],
            "revoked_at": row["revoked_at"],
        }

    def approve_mcp(
        self,
        server: str,
        config_hash: str,
        env_keys: list[str] | tuple[str, ...] | set[str] = (),
    ) -> dict[str, Any]:
        keys = sorted({str(key) for key in env_keys})
        now = _timestamp()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO mcp_configs (
                    server, config_hash, env_keys_json, status, approved_at, revoked_at
                ) VALUES (?, ?, ?, 'approved', ?, NULL)
                ON CONFLICT(server) DO UPDATE SET
                    config_hash = excluded.config_hash,
                    env_keys_json = excluded.env_keys_json,
                    status = 'approved',
                    approved_at = excluded.approved_at,
                    revoked_at = NULL
                """,
                (server, str(config_hash), _json(keys), now),
            )
        return self.mcp_status(server, str(config_hash))

    def revoke_mcp(self, server: str) -> dict[str, Any]:
        now = _timestamp()
        with self._connect() as connection:
            connection.execute(
                "UPDATE mcp_configs SET status = 'revoked', revoked_at = ? WHERE server = ?",
                (now, server),
            )
        return self.mcp_status(server)

    # Explicit aliases keep the API convenient for MCP adapters that use a
    # noun-first naming convention.
    mcp_config_status = mcp_status
    approve_mcp_config = approve_mcp
    revoke_mcp_config = revoke_mcp
    get_mcp_status = mcp_status


class PolicyGate:
    """Fail-closed authorization for tool metadata and runtime scope."""

    # Only this explicit effect gets an automatic allow.  ``read_only`` alone
    # is intentionally insufficient because a network read is still an
    # externally controlled input and may contain prompt injection.
    LOCAL_READ_EFFECTS = frozenset(
        {
            "local_read",
            "local_pure_read",
            "local-filesystem-read",
            "filesystem_read",
            "read_local",
        }
    )

    def __init__(
        self,
        store: SecurityStore | None = None,
        *,
        security_store: SecurityStore | None = None,
        approval_callback: ApprovalCallback | None = None,
        ask_callback: ApprovalCallback | None = None,
        approval_ttl_seconds: float = DEFAULT_APPROVAL_TTL_SECONDS,
    ) -> None:
        self.store = security_store or store
        self.approval_callback = approval_callback or ask_callback
        self.approval_ttl_seconds = approval_ttl_seconds

    @staticmethod
    def _tool_name(tool: Any, tool_name: str | None = None) -> str:
        if isinstance(tool, Mapping):
            return str(tool_name or tool.get("name", ""))
        return str(tool_name or getattr(tool, "name", tool or ""))

    @staticmethod
    def _subject(context: Any) -> str:
        if context is None:
            return ""
        return str(
            getattr(context, "security_subject", "")
            or getattr(context, "subject", "")
            or getattr(context, "metadata", {}).get("security_subject", "")
            or ""
        )

    @staticmethod
    def _allowed_tools(context: Any) -> set[str] | None:
        if context is None:
            return None
        values = getattr(context, "allowed_tools", None)
        if values is None:
            return None
        return {str(value) for value in values}

    @staticmethod
    def _resource_scope_allows(args: Mapping[str, Any], context: Any) -> bool:
        scopes = getattr(context, "resource_scopes", None) if context is not None else None
        if scopes is None:
            return True
        workspace = Path(getattr(context, "workspace", None) or Path.cwd()).expanduser().resolve()
        path_scopes: list[str] = []
        domain_scopes: list[str] = []
        if isinstance(scopes, Mapping):
            raw_paths = scopes.get("paths", [])
            raw_domains = scopes.get("domains", [])
            if isinstance(raw_paths, (list, tuple, set)):
                path_scopes = [str(value) for value in raw_paths]
            if isinstance(raw_domains, (list, tuple, set)):
                domain_scopes = [str(value).lower() for value in raw_domains]
        elif isinstance(scopes, (list, tuple, set)):
            path_scopes = [str(value) for value in scopes]
        else:
            return False

        target = args.get("path", args.get("cwd"))
        if target is not None:
            candidate = Path(str(target)).expanduser()
            if not candidate.is_absolute():
                candidate = workspace / candidate
            candidate = candidate.resolve()
            roots = []
            for value in path_scopes:
                root = workspace if value == "workspace" else Path(value).expanduser()
                if not root.is_absolute():
                    root = workspace / root
                roots.append(root.resolve())
            if not roots or not any(_is_under(candidate, root) for root in roots):
                return False

        url = args.get("url")
        if url is not None:
            from urllib.parse import urlsplit

            host = (urlsplit(str(url)).hostname or "").lower()
            if not domain_scopes or not any(host == domain or host.endswith(f".{domain}") for domain in domain_scopes):
                return False
        return True

    @classmethod
    def is_explicit_local_read(cls, tool: Any) -> bool:
        """Return whether metadata explicitly permits a pure local read."""

        if isinstance(tool, Mapping):
            effect_value = tool.get("effect")
            read_only = bool(tool.get("read_only", False))
        else:
            effect_value = getattr(tool, "effect", None) if tool is not None else None
            read_only = bool(getattr(tool, "read_only", False))
        effect = str(effect_value).strip().lower() if effect_value is not None else ""
        return read_only and effect in cls.LOCAL_READ_EFFECTS

    def evaluate(
        self,
        tool: Any = None,
        args: Mapping[str, Any] | None = None,
        context: Any = None,
        *,
        tool_name: str | None = None,
    ) -> PolicyDecision:
        name = self._tool_name(tool, tool_name)
        try:
            digest = canonical_args_hash(args or {})
        except (TypeError, ValueError):
            return PolicyDecision(_DENY, "arguments are not canonical JSON", name)

        allowed_tools = self._allowed_tools(context)
        if allowed_tools is not None and name not in allowed_tools:
            return PolicyDecision(_DENY, "tool is outside the allowed_tools scope", name, digest)
        if not self._resource_scope_allows(args or {}, context):
            return PolicyDecision(_DENY, "tool arguments are outside the resource scope", name, digest)

        # ``ask_user`` is the one trusted path used to obtain approval.  It
        # must not ask for approval of itself or recurse through the callback.
        if name == "ask_user":
            return PolicyDecision(_ALLOW, "interactive approval tool", name, digest)

        if isinstance(tool, Mapping):
            effect_value = tool.get("effect")
            read_only = bool(tool.get("read_only", False))
        else:
            effect_value = getattr(tool, "effect", None) if tool is not None else None
            read_only = bool(getattr(tool, "read_only", False))
        effect = str(effect_value).strip().lower() if effect_value is not None else ""
        if effect in self.LOCAL_READ_EFFECTS:
            if read_only:
                return PolicyDecision(_ALLOW, "explicit local pure read", name, digest)
            return PolicyDecision(_ASK, "local read effect conflicts with read_only metadata", name, digest)
        if "network" in effect or "mcp" in effect:
            return PolicyDecision(_ASK, f"external effect requires approval: {effect}", name, digest)
        if not effect:
            return PolicyDecision(_ASK, "tool effect metadata is missing", name, digest)
        # Unknown metadata is high-risk.  Do not silently infer safety from
        # ``read_only`` or from a familiar tool name.
        return PolicyDecision(_ASK, f"unknown tool effect metadata: {effect}", name, digest)

    decide = evaluate
    check = evaluate

    async def authorize(
        self,
        tool: Any,
        args: Mapping[str, Any] | None = None,
        *,
        context: Any = None,
        tool_call_id: str = "",
        approval_id: str | None = None,
    ) -> PolicyDecision:
        decision = self.evaluate(tool, args, context)
        if decision.action != _ASK:
            return decision

        name = decision.tool
        session_key = str(getattr(context, "session_key", "") or "")
        channel = str(getattr(context, "channel", "") or "")
        subject = self._subject(context)
        if approval_id is None:
            approval_id = getattr(context, "approval_id", None)

        # A previously issued approval is consumed only after every binding
        # field matches.  One row can therefore never authorize a second call.
        if self.store is not None and approval_id:
            if self.store.consume_approval(
                approval_id=approval_id,
                security_subject=subject,
                session_key=session_key,
                channel=channel,
                tool_call_id=tool_call_id,
                tool=name,
                args_hash=decision.args_hash,
            ):
                decision.action = _ALLOW
                decision.reason = "bound approval consumed"
                decision.approval_id = approval_id
                return decision

        callback = (
            getattr(context, "approval_callback", None)
            or getattr(context, "approval", None)
            or getattr(context, "ask", None)
            or self.approval_callback
        )
        if callback is None:
            return PolicyDecision(_DENY, f"approval required: {decision.reason}", name, decision.args_hash)

        question = f"Approve tool '{name}'? (argument keys: {', '.join(sorted((args or {}).keys()))})"
        try:
            answer = callback(question, ["allow", "deny"])
            if inspect.isawaitable(answer):
                answer = await answer
        except TypeError:
            # A small adapter may expose a one-argument callback.  Retry only
            # on signature mismatch; callback exceptions remain a denial.
            try:
                answer = callback(question)  # type: ignore[misc,call-arg]
                if inspect.isawaitable(answer):
                    answer = await answer
            except Exception:
                return PolicyDecision(_DENY, "approval callback failed", name, decision.args_hash)
        except Exception:
            return PolicyDecision(_DENY, "approval callback failed", name, decision.args_hash)

        approved = (
            answer is True
            or (isinstance(answer, str) and answer.strip().lower() in {
                "allow", "allowed", "approve", "approved", "yes", "y", "ok", "true", "1",
            })
        )
        if not approved:
            return PolicyDecision(_DENY, "approval was not granted", name, decision.args_hash)

        # Persist and immediately consume the approval for this exact call.
        # This gives the audit trail a bound approval while keeping an approval
        # response one-shot.  If no store is configured, the callback still
        # authorizes this in-memory call but there is no durable replay guard.
        if self.store is not None:
            created_id = self.store.create_approval(
                security_subject=subject,
                session_key=session_key,
                channel=channel,
                tool_call_id=tool_call_id,
                tool=name,
                args_hash=decision.args_hash,
                ttl_seconds=self.approval_ttl_seconds,
            )
            consumed = self.store.consume_approval(
                approval_id=created_id,
                security_subject=subject,
                session_key=session_key,
                channel=channel,
                tool_call_id=tool_call_id,
                tool=name,
                args_hash=decision.args_hash,
            )
            if not consumed:
                return PolicyDecision(_DENY, "approval could not be consumed", name, decision.args_hash)
            decision.approval_id = created_id
        decision.action = _ALLOW
        decision.reason = "approval granted"
        return decision


__all__ = [
    "ApprovalCallback",
    "ALLOW",
    "ASK",
    "DENY",
    "DEFAULT_APPROVAL_TTL_SECONDS",
    "DEFAULT_TOOL_TIMEOUT_SECONDS",
    "PolicyDecision",
    "PolicyGate",
    "ResultEnvelope",
    "SecurityStore",
    "args_hash",
    "canonical_args",
    "canonical_args_hash",
    "canonicalize_args",
    "make_operation_id",
    "operation_id",
    "result_digest",
]
