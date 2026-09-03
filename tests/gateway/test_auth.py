from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from myclaw.gateway import (
    GatewayAuthError,
    SecuritySubject,
    require_subject_access,
    resolve_gateway_subject,
)
from myclaw.gateway.http import HttpRequest, send_json


class RecordingAuthenticator:
    def __init__(self, subjects: dict[str, SecuritySubject] | None = None) -> None:
        self.subjects = subjects or {}
        self.credentials: list[str] = []

    def authenticate(self, credential: str, /) -> SecuritySubject | None:
        self.credentials.append(credential)
        return self.subjects.get(credential)


class BufferWriter:
    def __init__(self) -> None:
        self.data = bytearray()

    def write(self, data: bytes) -> None:
        self.data.extend(data)

    async def drain(self) -> None:
        return None


def _request(authorization: str | None = None) -> HttpRequest:
    headers = {} if authorization is None else {"authorization": authorization}
    return HttpRequest("GET", "/api/sessions", "/api/sessions", {}, headers, b"")


def _auth_error(request: HttpRequest, authenticator: RecordingAuthenticator) -> GatewayAuthError:
    with pytest.raises(GatewayAuthError) as raised:
        resolve_gateway_subject(request, authenticator)
    return raised.value


def test_resolve_gateway_subject_accepts_case_insensitive_bearer_once() -> None:
    subject = SecuritySubject.authenticated_as("operator")
    authenticator = RecordingAuthenticator({"valid-token": subject})

    resolved = resolve_gateway_subject(_request("bEaReR valid-token"), authenticator)

    assert resolved is subject
    assert authenticator.credentials == ["valid-token"]


@pytest.mark.parametrize(
    "authorization",
    [None, "", "Bearer", "Bearer ", "Basic value", "Bearer two parts", "Bearer bad@token"],
)
def test_missing_and_malformed_credentials_share_one_safe_error(authorization: str | None) -> None:
    authenticator = RecordingAuthenticator()

    error = _auth_error(_request(authorization), authenticator)

    assert error.status == 401
    assert error.payload == {"error": "authentication required", "code": "unauthenticated"}
    assert error.headers == {"WWW-Authenticate": "Bearer"}
    assert authenticator.credentials == []


def test_invalid_credential_uses_same_error_without_leaking_token() -> None:
    token = "secret-token"
    error = _auth_error(_request(f"Bearer {token}"), RecordingAuthenticator())

    exposed = " ".join((str(error), repr(error), repr(error.payload), repr(error.headers)))
    assert token not in exposed
    assert error.payload == {"error": "authentication required", "code": "unauthenticated"}


def test_anonymous_subject_requires_explicit_opt_in() -> None:
    authenticator = RecordingAuthenticator()

    subject = resolve_gateway_subject(_request(), authenticator, allow_anonymous=True)

    assert subject == SecuritySubject.anonymous()
    assert not subject.authenticated


def test_invalid_supplied_credential_is_not_downgraded_to_anonymous() -> None:
    error = _auth_error(_request("Bearer invalid"), RecordingAuthenticator())

    assert error.status == 401


def test_security_subject_is_immutable_and_rejects_inconsistent_states() -> None:
    subject = SecuritySubject.authenticated_as("operator")

    with pytest.raises(FrozenInstanceError):
        subject.subject_id = "other"  # type: ignore[misc]
    with pytest.raises(ValueError):
        SecuritySubject(subject_id=None, authenticated=True)
    with pytest.raises(ValueError):
        SecuritySubject(subject_id="claimed", authenticated=False)


def test_require_subject_access_allows_owner_and_hides_mismatch() -> None:
    subject = SecuritySubject.authenticated_as("operator")

    require_subject_access(subject, "operator")
    with pytest.raises(GatewayAuthError) as raised:
        require_subject_access(subject, "someone-else")

    assert raised.value.status == 403
    assert raised.value.payload == {"error": "forbidden", "code": "forbidden"}
    assert "someone-else" not in repr(raised.value)


def test_require_subject_access_rejects_anonymous_as_unauthenticated() -> None:
    with pytest.raises(GatewayAuthError) as raised:
        require_subject_access(SecuritySubject.anonymous(), "operator")

    assert raised.value.status == 401
    assert raised.value.code == "unauthenticated"


@pytest.mark.parametrize(
    ("status", "expected_status_line"),
    [(401, b"HTTP/1.1 401 Unauthorized"), (403, b"HTTP/1.1 403 Forbidden")],
)
def test_http_layer_emits_standard_auth_statuses(status: int, expected_status_line: bytes) -> None:
    writer = BufferWriter()
    error = (
        GatewayAuthError(
            401,
            "unauthenticated",
            "authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
        if status == 401
        else GatewayAuthError(403, "forbidden", "forbidden")
    )

    asyncio.run(send_json(cast(Any, writer), error.status, error.payload, headers=error.headers))

    response = bytes(writer.data)
    assert response.startswith(expected_status_line)
    if status == 401:
        assert b"WWW-Authenticate: Bearer\r\n" in response
