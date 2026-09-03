from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from myclaw.gateway.http import HttpRequest


_BEARER_CREDENTIAL = re.compile(r"Bearer ([A-Za-z0-9._~+/-]+={0,})", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class SecuritySubject:
    """A server-authenticated identity. Raw credentials must never be stored here."""

    subject_id: str | None
    authenticated: bool

    def __post_init__(self) -> None:
        if self.authenticated:
            if self.subject_id is None or not self.subject_id.strip():
                raise ValueError("an authenticated subject requires a non-empty subject_id")
        elif self.subject_id is not None:
            raise ValueError("an anonymous subject cannot have a subject_id")

    @classmethod
    def authenticated_as(cls, subject_id: str) -> SecuritySubject:
        return cls(subject_id=subject_id, authenticated=True)

    @classmethod
    def anonymous(cls) -> SecuritySubject:
        return cls(subject_id=None, authenticated=False)


class Authenticator(Protocol):
    """Resolve one opaque credential to a server-confirmed subject."""

    def authenticate(self, credential: str, /) -> SecuritySubject | None: ...


class GatewayAuthError(Exception):
    """A stable, credential-free error safe to expose through the Gateway API."""

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.headers = dict(headers or {})

    @property
    def payload(self) -> dict[str, str]:
        return {"error": self.message, "code": self.code}


def resolve_gateway_subject(
    request: HttpRequest,
    authenticator: Authenticator,
    *,
    allow_anonymous: bool = False,
) -> SecuritySubject:
    """Resolve the request identity through the Gateway's single auth entry point."""

    authorization = request.headers.get("authorization")
    if authorization is None:
        if allow_anonymous:
            return SecuritySubject.anonymous()
        raise _unauthenticated()

    match = _BEARER_CREDENTIAL.fullmatch(authorization)
    if match is None:
        raise _unauthenticated()

    subject = authenticator.authenticate(match.group(1))
    if subject is None or not subject.authenticated:
        raise _unauthenticated()
    return subject


def require_subject_access(subject: SecuritySubject, owner_subject_id: str) -> None:
    """Require an authenticated subject to own a server-labelled resource."""

    if not subject.authenticated:
        raise _unauthenticated()
    if subject.subject_id != owner_subject_id:
        raise GatewayAuthError(403, "forbidden", "forbidden")


def _unauthenticated() -> GatewayAuthError:
    return GatewayAuthError(
        401,
        "unauthenticated",
        "authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


__all__ = [
    "Authenticator",
    "GatewayAuthError",
    "SecuritySubject",
    "require_subject_access",
    "resolve_gateway_subject",
]
