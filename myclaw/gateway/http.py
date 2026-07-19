from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlsplit

from myclaw.config import GATEWAY_MAX_BODY_BYTES


@dataclass(slots=True)
class HttpRequest:
    method: str
    target: str
    path: str
    query: dict[str, list[str]]
    headers: dict[str, str]
    body: bytes


async def read_http_request(reader: asyncio.StreamReader) -> HttpRequest | None:
    raw_head = await reader.readuntil(b"\r\n\r\n")
    lines = raw_head.decode("iso-8859-1").split("\r\n")
    try:
        method, target, _version = lines[0].split()
    except ValueError:
        return None

    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line:
            continue
        if ":" not in line:
            return None
        name, value = line.split(":", 1)
        headers[name.lower()] = value.strip()

    try:
        content_length = int(headers.get("content-length", "0"))
    except ValueError:
        return None
    if content_length < 0 or content_length > GATEWAY_MAX_BODY_BYTES:
        return None

    parsed = urlsplit(target)
    body = await reader.readexactly(content_length) if content_length else b""
    return HttpRequest(
        method=method.upper(),
        target=target,
        path=parsed.path,
        query=parse_qs(parsed.query),
        headers=headers,
        body=body,
    )


def json_body(request: HttpRequest) -> tuple[dict[str, Any], str | None]:
    try:
        payload = json.loads(request.body.decode("utf-8") if request.body else "{}")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {}, f"invalid JSON: {exc}"
    if not isinstance(payload, dict):
        return {}, "request body must be a JSON object"
    return payload, None


async def send_json(writer: asyncio.StreamWriter, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    await send_response(writer, status, body, "application/json; charset=utf-8")


async def send_response(writer: asyncio.StreamWriter, status: int, body: bytes, content_type: str) -> None:
    reason = {
        200: "OK",
        202: "Accepted",
        400: "Bad Request",
        404: "Not Found",
        405: "Method Not Allowed",
        503: "Service Unavailable",
    }.get(status, "Error")
    writer.write((
        f"HTTP/1.1 {status} {reason}\r\n"
        f"Content-Type: {content_type}\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Cache-Control: no-store\r\n"
        "Connection: close\r\n\r\n"
    ).encode("utf-8"))
    writer.write(body)
    await writer.drain()
