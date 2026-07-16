from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import mimetypes
import re
import sys
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from myclaw.agent import AgentDispatcher, DispatcherRuntime
from myclaw.bus import InboundMessage, OutboundMessage
from myclaw.config import (
    DEFAULT_GATEWAY_CHAT_ID,
    DEFAULT_GATEWAY_HOST,
    DEFAULT_GATEWAY_PORT,
    GATEWAY_CHANNEL,
    GATEWAY_MAX_BODY_BYTES,
)
from myclaw.session import Session


_WEB_DIST_DIR = Path(__file__).resolve().parent / "web" / "dist"
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _HttpRequest:
    method: str
    target: str
    path: str
    query: dict[str, list[str]]
    headers: dict[str, str]
    body: bytes


@dataclass(slots=True)
class _SseClient:
    chat_id: str
    writer: asyncio.StreamWriter
    queue: asyncio.Queue[dict[str, Any] | None] = field(default_factory=asyncio.Queue)


class HttpGatewayServer:
    """Local HTTP gateway for browser clients."""

    def __init__(
        self,
        dispatcher: AgentDispatcher,
        *,
        host: str = DEFAULT_GATEWAY_HOST,
        port: int = DEFAULT_GATEWAY_PORT,
    ) -> None:
        self.dispatcher = dispatcher
        self.host = host
        self.port = port
        self._runtime = DispatcherRuntime(dispatcher)
        self._server: asyncio.Server | None = None
        self._fanout_task: asyncio.Task[None] | None = None
        self._clients: list[_SseClient] = []

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    async def start(self) -> None:
        if self._server is not None:
            return
        await self._runtime.start()
        self._fanout_task = asyncio.create_task(self._fanout_outbound())
        self._server = await asyncio.start_server(self._handle_client, self.host, self.port)
        sockets = self._server.sockets or []
        if sockets:
            bound = sockets[0].getsockname()
            self.host = str(bound[0])
            self.port = int(bound[1])

    async def serve_forever(self) -> None:
        if self._server is None:
            await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._server.wait_closed(), timeout=0.2)
            self._server = None

        for client in list(self._clients):
            client.queue.put_nowait(None)
        await asyncio.sleep(0)
        for client in list(self._clients):
            client.writer.close()
            with contextlib.suppress(ConnectionError, OSError, asyncio.TimeoutError):
                await asyncio.wait_for(client.writer.wait_closed(), timeout=0.2)
        self._clients.clear()

        if self._fanout_task is not None:
            self._fanout_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._fanout_task
            self._fanout_task = None
        logger.info("Gateway stopped")
        await self._runtime.stop()

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        keep_open = False
        try:
            request = await _read_http_request(reader)
            if request is None:
                await _send_json(writer, 400, {"error": "bad request"})
                return

            if request.path == "/api/messages":
                if request.method != "POST":
                    await _send_json(writer, 405, {"error": "method not allowed"})
                    return
                await self._handle_post_message(writer, request)
                return

            if request.path == "/api/sessions":
                if request.method != "GET":
                    await _send_json(writer, 405, {"error": "method not allowed"})
                    return
                await self._handle_sessions(writer, request)
                return

            if request.path == "/api/memory":
                if request.method != "GET":
                    await _send_json(writer, 405, {"error": "method not allowed"})
                    return
                await self._handle_memory(writer)
                return

            if request.path == "/api/observability/summary":
                if request.method != "GET":
                    await _send_json(writer, 405, {"error": "method not allowed"})
                    return
                await self._handle_observability_summary(writer, request)
                return

            if request.path == "/api/observability/traces":
                if request.method != "GET":
                    await _send_json(writer, 405, {"error": "method not allowed"})
                    return
                await self._handle_observability_traces(writer, request)
                return

            if request.path.startswith("/api/observability/traces/"):
                if request.method != "GET":
                    await _send_json(writer, 405, {"error": "method not allowed"})
                    return
                await self._handle_observability_trace_detail(writer, request)
                return

            if request.path == "/api/observability/logs":
                if request.method != "GET":
                    await _send_json(writer, 405, {"error": "method not allowed"})
                    return
                await self._handle_observability_logs(writer, request)
                return

            if request.path == "/api/events":
                if request.method != "GET":
                    await _send_json(writer, 405, {"error": "method not allowed"})
                    return
                keep_open = True
                await self._handle_sse(writer, request)
                return

            if request.method == "GET" and await _send_web_asset(writer, request.path):
                return

            await _send_json(writer, 404, {"error": "not found"})
        except (ConnectionError, asyncio.IncompleteReadError):
            return
        finally:
            if not keep_open:
                writer.close()
                with contextlib.suppress(ConnectionError, OSError):
                    await writer.wait_closed()

    async def _handle_post_message(self, writer: asyncio.StreamWriter, request: _HttpRequest) -> None:
        payload, error = _json_body(request)
        if error is not None:
            await _send_json(writer, 400, {"error": error})
            return

        content = payload.get("content")
        if not isinstance(content, str) or not content.strip():
            await _send_json(writer, 400, {"error": "content must be a non-empty string"})
            return

        chat_id = payload.get("chat_id", DEFAULT_GATEWAY_CHAT_ID)
        if not isinstance(chat_id, str) or not chat_id:
            await _send_json(writer, 400, {"error": "chat_id must be a non-empty string"})
            return

        session_key = payload.get("session_key")
        if session_key is not None and not isinstance(session_key, str):
            await _send_json(writer, 400, {"error": "session_key must be a string"})
            return

        request_id = uuid.uuid4().hex
        trace_id = uuid.uuid4().hex
        await self.dispatcher.bus.publish_inbound(
            InboundMessage(
                channel=GATEWAY_CHANNEL,
                sender_id="user",
                chat_id=chat_id,
                content=content,
                metadata={"request_id": request_id, "trace_id": trace_id},
                session_key_override=session_key,
            )
        )
        logger.info(
            "Gateway request accepted",
            extra={"trace_id": trace_id, "request_id": request_id, "session_key": session_key or f"gateway:{chat_id}"},
        )
        await _send_json(
            writer,
            202,
            {"id": request_id, "trace_id": trace_id, "chat_id": chat_id, "accepted": True},
        )

    async def _handle_sessions(self, writer: asyncio.StreamWriter, request: _HttpRequest) -> None:
        sessions = self.dispatcher.loop.session_manager.list_sessions()
        key = _first_query_value(request.query, "key")
        if key is None:
            await _send_json(writer, 200, {"sessions": [_session_summary(session) for session in sessions]})
            return

        session = next((candidate for candidate in sessions if candidate.key == key), None)
        if session is None:
            await _send_json(writer, 404, {"error": "session not found"})
            return

        await _send_json(
            writer,
            200,
            {
                "key": session.key,
                "title": _session_title(session),
                "messages": _display_messages(session),
            },
        )

    async def _handle_memory(self, writer: asyncio.StreamWriter) -> None:
        store = self.dispatcher.loop.memory_store
        await _send_json(
            writer,
            200,
            {
                "memory": store.read_memory(),
                "user": store.read_user(),
                "soul": store.read_soul(),
            },
        )

    async def _handle_observability_summary(
        self, writer: asyncio.StreamWriter, request: _HttpRequest
    ) -> None:
        runtime = self._observability_runtime()
        if runtime is None or not runtime.config.enabled:
            await _send_json(writer, 503, {"error": "observability is disabled"})
            return
        try:
            window, since = _observability_window(request.query, default="24h")
        except ValueError as exc:
            await _send_json(writer, 400, {"error": str(exc)})
            return
        payload = await asyncio.to_thread(runtime.summary, since)
        payload.update({"window": window, "since": since, "generated_at": _utc_now()})
        await _send_json(writer, 200, payload)

    async def _handle_observability_traces(
        self, writer: asyncio.StreamWriter, request: _HttpRequest
    ) -> None:
        runtime = self._observability_runtime()
        if runtime is None or not runtime.config.enabled:
            await _send_json(writer, 503, {"error": "observability is disabled"})
            return
        try:
            window, since = _observability_window(request.query, default="24h")
            limit = _query_limit(request.query, default=50, maximum=200)
            status = _validated_choice(request.query, "status", {"running", "ok", "error", "cancelled", "abandoned"})
            kind = _validated_choice(
                request.query, "kind", {"conversation", "control", "cron", "dream", "autocompact"}
            )
            session_key = _first_query_value(request.query, "session_key")
            before = _validated_datetime(request.query, "before")
        except ValueError as exc:
            await _send_json(writer, 400, {"error": str(exc)})
            return
        traces = await asyncio.to_thread(
            runtime.list_traces,
            since,
            status=status,
            kind=kind,
            session_key=session_key,
            before=before,
            limit=limit,
        )
        await _send_json(
            writer,
            200,
            {
                "window": window,
                "traces": traces,
                "next_before": traces[-1]["started_at"] if len(traces) == limit else None,
            },
        )

    async def _handle_observability_trace_detail(
        self, writer: asyncio.StreamWriter, request: _HttpRequest
    ) -> None:
        runtime = self._observability_runtime()
        if runtime is None or not runtime.config.enabled:
            await _send_json(writer, 503, {"error": "observability is disabled"})
            return
        trace_id = request.path.removeprefix("/api/observability/traces/")
        if not re.fullmatch(r"[0-9a-f]{32}", trace_id):
            await _send_json(writer, 400, {"error": "trace_id must be 32 lowercase hexadecimal characters"})
            return
        detail = await asyncio.to_thread(runtime.trace_detail, trace_id)
        if detail is None:
            await _send_json(writer, 404, {"error": "trace not found"})
            return
        await _send_json(writer, 200, detail)

    async def _handle_observability_logs(
        self, writer: asyncio.StreamWriter, request: _HttpRequest
    ) -> None:
        runtime = self._observability_runtime()
        if runtime is None or not runtime.config.enabled:
            await _send_json(writer, 503, {"error": "observability is disabled"})
            return
        try:
            window, since = _observability_window(request.query, default="1h")
            limit = _query_limit(request.query, default=200, maximum=500)
            level = _validated_choice(request.query, "level", {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
            component = _first_query_value(request.query, "component")
            trace_id = _first_query_value(request.query, "trace_id")
            if trace_id is not None and not re.fullmatch(r"[0-9a-f]{32}", trace_id):
                raise ValueError("trace_id must be 32 lowercase hexadecimal characters")
            query = _first_query_value(request.query, "query")
            before = _validated_datetime(request.query, "before")
        except ValueError as exc:
            await _send_json(writer, 400, {"error": str(exc)})
            return
        logs = await asyncio.to_thread(
            runtime.list_logs,
            since,
            level=level,
            component=component,
            trace_id=trace_id,
            query=query,
            before=before,
            limit=limit,
        )
        await _send_json(
            writer,
            200,
            {
                "window": window,
                "logs": logs,
                "next_before": logs[-1]["timestamp"] if len(logs) == limit else None,
            },
        )

    def _observability_runtime(self):
        runtime = getattr(self.dispatcher, "observability", None)
        if runtime is not None:
            return runtime
        loop = getattr(self.dispatcher, "loop", None)
        return getattr(loop, "observability", None)

    async def _handle_sse(self, writer: asyncio.StreamWriter, request: _HttpRequest) -> None:
        chat_id = _first_query_value(request.query, "chat_id") or DEFAULT_GATEWAY_CHAT_ID
        client = _SseClient(chat_id=chat_id, writer=writer)
        self._clients.append(client)
        writer.write(
            (
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/event-stream; charset=utf-8\r\n"
                "Cache-Control: no-cache\r\n"
                "Connection: keep-alive\r\n"
                "\r\n"
            ).encode("utf-8")
        )
        await writer.drain()
        try:
            while True:
                event = await client.queue.get()
                if event is None:
                    return
                writer.write(f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8"))
                await writer.drain()
        except (asyncio.CancelledError, ConnectionError, OSError):
            return
        finally:
            with contextlib.suppress(ValueError):
                self._clients.remove(client)
            writer.close()
            with contextlib.suppress(ConnectionError, OSError, asyncio.TimeoutError):
                await asyncio.wait_for(writer.wait_closed(), timeout=0.2)

    async def _fanout_outbound(self) -> None:
        while True:
            outbound = await self.dispatcher.bus.consume_outbound()
            event = _outbound_event(outbound)
            for client in list(self._clients):
                if client.chat_id == outbound.chat_id:
                    await client.queue.put(event)


async def run_gateway(
    dispatcher: AgentDispatcher,
    *,
    host: str = DEFAULT_GATEWAY_HOST,
    port: int = DEFAULT_GATEWAY_PORT,
) -> None:
    server = HttpGatewayServer(dispatcher, host=host, port=port)
    await server.start()
    print(f"Gateway listening on {server.url}", file=sys.stderr, flush=True)
    logger.info("Gateway started on %s", server.url)
    try:
        await server.serve_forever()
    finally:
        await server.stop()


async def _read_http_request(reader: asyncio.StreamReader) -> _HttpRequest | None:
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

    length_text = headers.get("content-length", "0")
    try:
        content_length = int(length_text)
    except ValueError:
        return None
    if content_length < 0 or content_length > GATEWAY_MAX_BODY_BYTES:
        return None

    parsed = urlsplit(target)
    body = await reader.readexactly(content_length) if content_length else b""
    return _HttpRequest(
        method=method.upper(),
        target=target,
        path=parsed.path,
        query=parse_qs(parsed.query),
        headers=headers,
        body=body,
    )


def _json_body(request: _HttpRequest) -> tuple[dict[str, Any], str | None]:
    try:
        payload = json.loads(request.body.decode("utf-8") if request.body else "{}")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {}, f"invalid JSON: {exc}"
    if not isinstance(payload, dict):
        return {}, "request body must be a JSON object"
    return payload, None


async def _send_web_asset(writer: asyncio.StreamWriter, request_path: str) -> bool:
    decoded_path = unquote(request_path)
    relative_path = "index.html" if decoded_path == "/" else decoded_path.removeprefix("/")
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        return False

    candidate = _WEB_DIST_DIR.joinpath(*relative.parts).resolve()
    if not candidate.is_relative_to(_WEB_DIST_DIR) or not candidate.is_file():
        return False

    content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
    if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
        content_type += "; charset=utf-8"
    await _send_response(writer, 200, candidate.read_bytes(), content_type)
    return True


async def _send_json(writer: asyncio.StreamWriter, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    await _send_response(writer, status, body, "application/json; charset=utf-8")


async def _send_response(
    writer: asyncio.StreamWriter,
    status: int,
    body: bytes,
    content_type: str,
) -> None:
    reason = {
        200: "OK",
        202: "Accepted",
        400: "Bad Request",
        404: "Not Found",
        405: "Method Not Allowed",
        503: "Service Unavailable",
    }.get(status, "Error")
    writer.write(
        (
            f"HTTP/1.1 {status} {reason}\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Cache-Control: no-store\r\n"
            "Connection: close\r\n"
            "\r\n"
        ).encode("utf-8")
    )
    writer.write(body)
    await writer.drain()


def _outbound_event(outbound: OutboundMessage) -> dict[str, Any]:
    metadata = dict(outbound.metadata)
    event: dict[str, Any] = {
        "type": outbound.event_type,
        "chat_id": outbound.chat_id,
        "content": outbound.content,
        "terminal": outbound.terminal,
        "metadata": metadata,
    }
    request_id = metadata.get("request_id")
    if isinstance(request_id, str):
        event["id"] = request_id
    return event


def _session_summary(session: Session) -> dict[str, Any]:
    return {
        "key": session.key,
        "channel": _session_channel(session.key),
        "title": _session_title(session),
        "preview": _session_preview(session),
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
        "message_count": len(session.messages),
    }


def _session_channel(key: str) -> str:
    channel, separator, _name = key.partition(":")
    return channel if separator else ""


def _session_title(session: Session) -> str:
    title = session.metadata.get("title")
    if isinstance(title, str) and title.strip():
        return _truncate_text(title, 60)
    return _truncate_text(_first_message_content(session, roles={"user"}) or "Untitled", 60)


def _session_preview(session: Session) -> str:
    return _truncate_text(_first_message_content(session, roles={"user", "assistant"}) or "", 120)


def _first_message_content(session: Session, *, roles: set[str]) -> str:
    for message in session.messages:
        role = message.get("role")
        content = message.get("content")
        if role in roles and isinstance(content, str) and content.strip():
            return " ".join(content.split())
    return ""


def _display_messages(session: Session) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for message in session.messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
            continue
        messages.append({"role": role, "content": content})
    return messages


def _truncate_text(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _first_query_value(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    if not values:
        return None
    value = values[0]
    return value if value else None


def _observability_window(query: dict[str, list[str]], *, default: str) -> tuple[str, str]:
    window = _first_query_value(query, "window") or default
    match = re.fullmatch(r"([1-9][0-9]*)([hd])", window)
    if match is None:
        raise ValueError("window must use a positive hour/day value such as 1h, 24h, or 7d")
    amount = int(match.group(1))
    duration = timedelta(hours=amount) if match.group(2) == "h" else timedelta(days=amount)
    if duration > timedelta(days=30):
        raise ValueError("window cannot exceed 30 days")
    since = (datetime.now(UTC) - duration).isoformat()
    return window, since


def _query_limit(query: dict[str, list[str]], *, default: int, maximum: int) -> int:
    raw = _first_query_value(query, "limit")
    if raw is None:
        return default
    try:
        limit = int(raw)
    except ValueError as exc:
        raise ValueError("limit must be an integer") from exc
    if limit < 1 or limit > maximum:
        raise ValueError(f"limit must be between 1 and {maximum}")
    return limit


def _validated_choice(
    query: dict[str, list[str]], key: str, allowed: set[str]
) -> str | None:
    value = _first_query_value(query, key)
    if value is not None and value not in allowed:
        raise ValueError(f"{key} must be one of: {', '.join(sorted(allowed))}")
    return value


def _validated_datetime(query: dict[str, list[str]], key: str) -> str | None:
    value = _first_query_value(query, key)
    if value is None:
        return None
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{key} must be an ISO-8601 timestamp") from exc
    return value


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
