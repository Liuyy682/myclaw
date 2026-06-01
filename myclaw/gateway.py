from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, urlsplit

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
        await self._runtime.stop()

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        keep_open = False
        try:
            request = await _read_http_request(reader)
            if request is None:
                await _send_json(writer, 400, {"error": "bad request"})
                return

            if request.path == "/" and request.method == "GET":
                await _send_html(writer, _WEBUI_HTML)
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

            if request.path == "/api/events":
                if request.method != "GET":
                    await _send_json(writer, 405, {"error": "method not allowed"})
                    return
                keep_open = True
                await self._handle_sse(writer, request)
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
        await self.dispatcher.bus.publish_inbound(
            InboundMessage(
                channel=GATEWAY_CHANNEL,
                sender_id="user",
                chat_id=chat_id,
                content=content,
                metadata={"request_id": request_id},
                session_key_override=session_key,
            )
        )
        await _send_json(writer, 202, {"id": request_id, "chat_id": chat_id, "accepted": True})

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


async def _send_html(writer: asyncio.StreamWriter, html: str) -> None:
    await _send_response(writer, 200, html.encode("utf-8"), "text/html; charset=utf-8")


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


_WEBUI_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>myclaw Gateway</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f7f9;
      color: #1f2933;
    }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; }
	    #app {
	      min-height: 100vh;
	      display: grid;
	      grid-template-rows: auto 1fr auto;
	      max-width: 1120px;
	      margin: 0 auto;
	      background: #ffffff;
	      border-left: 1px solid #d7dde5;
	      border-right: 1px solid #d7dde5;
	    }
    header {
      display: flex;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
      padding: 14px 18px;
      border-bottom: 1px solid #d7dde5;
      background: #fbfcfd;
    }
    h1 { margin: 0; font-size: 18px; font-weight: 650; letter-spacing: 0; }
    .session { display: flex; gap: 8px; align-items: center; }
    label { font-size: 13px; color: #52606d; }
	    input, textarea, button {
	      font: inherit;
	      border: 1px solid #c9d2dd;
	      border-radius: 6px;
	    }
	    input { width: 160px; padding: 8px 10px; }
	    .workspace {
	      min-height: 0;
	      display: grid;
	      grid-template-columns: 270px minmax(0, 1fr);
	    }
	    aside {
	      min-height: 0;
	      overflow-y: auto;
	      padding: 12px;
	      border-right: 1px solid #d7dde5;
	      background: #fbfcfd;
	    }
	    .history-header {
	      display: flex;
	      align-items: center;
	      justify-content: space-between;
	      gap: 8px;
	      margin-bottom: 10px;
	      color: #52606d;
	      font-size: 13px;
	      font-weight: 650;
	    }
	    .history-header button {
	      min-width: 0;
	      padding: 5px 8px;
	      color: #1f2933;
	      background: #ffffff;
	      border-color: #c9d2dd;
	      font-size: 12px;
	      font-weight: 600;
	    }
	    .session-list {
	      display: flex;
	      flex-direction: column;
	      gap: 8px;
	    }
	    .session-item {
	      width: 100%;
	      min-width: 0;
	      padding: 9px 10px;
	      text-align: left;
	      color: #1f2933;
	      background: #ffffff;
	      border-color: #d7dde5;
	      cursor: pointer;
	    }
	    .session-item.active {
	      border-color: #1f6feb;
	      background: #e8f2ff;
	    }
	    .session-title {
	      display: block;
	      overflow: hidden;
	      text-overflow: ellipsis;
	      white-space: nowrap;
	      font-weight: 650;
	    }
	    .session-key, .session-preview, .history-empty {
	      display: block;
	      margin-top: 4px;
	      overflow: hidden;
	      text-overflow: ellipsis;
	      white-space: nowrap;
	      color: #52606d;
	      font-size: 12px;
	    }
	    main {
	      min-height: 0;
	      overflow-y: auto;
	      padding: 18px;
	      display: flex;
	      flex-direction: column;
      gap: 10px;
      background: #f6f7f9;
    }
    .row {
      max-width: 78%;
      padding: 10px 12px;
      border-radius: 8px;
      border: 1px solid #d7dde5;
      background: #ffffff;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .user { align-self: flex-end; background: #e8f2ff; border-color: #bfd7f2; }
    .assistant { align-self: flex-start; }
    .progress { align-self: center; max-width: 92%; color: #52606d; background: #f2f6f4; border-color: #c8d8ce; font-size: 13px; }
    .error { align-self: center; max-width: 92%; color: #8a1f11; background: #fff2ef; border-color: #f0c3ba; }
    .row h1, .row h2, .row h3, .row p, .row ul, .row ol, .row pre, .row blockquote { margin: 0 0 8px; }
    .row h1 { font-size: 20px; }
    .row h2 { font-size: 18px; }
    .row h3 { font-size: 16px; }
    .row ul, .row ol { padding-left: 22px; }
    .row code { padding: 1px 4px; border-radius: 4px; background: #eef2f7; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.92em; }
    .row pre { padding: 10px; overflow-x: auto; border-radius: 6px; background: #1f2933; color: #f8fafc; }
    .row pre code { padding: 0; background: transparent; color: inherit; }
    .row blockquote { padding-left: 10px; border-left: 3px solid #9fb3c8; color: #52606d; }
    form {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      padding: 14px 18px;
      border-top: 1px solid #d7dde5;
      background: #fbfcfd;
    }
	    textarea { min-height: 48px; max-height: 140px; resize: vertical; padding: 10px; }
	    form button {
	      min-width: 92px;
	      padding: 0 16px;
	      border-color: #1f6feb;
      background: #1f6feb;
      color: white;
      font-weight: 650;
      cursor: pointer;
    }
	    button:disabled { opacity: 0.55; cursor: default; }
	    @media (max-width: 640px) {
	      #app { border: 0; }
	      header { align-items: flex-start; flex-direction: column; }
	      .workspace { grid-template-columns: 1fr; grid-template-rows: auto 1fr; }
	      aside { max-height: 190px; border-right: 0; border-bottom: 1px solid #d7dde5; }
	      .session, input { width: 100%; }
	      form { grid-template-columns: 1fr; }
	      form button { min-height: 42px; }
	      .row { max-width: 94%; }
	    }
  </style>
</head>
<body>
  <div id="app">
	    <header>
	      <h1>myclaw Gateway</h1>
	      <div class="session">
	        <label for="chatId">Session</label>
	        <input id="chatId" value="direct" autocomplete="off">
	      </div>
	    </header>
	    <div class="workspace">
	      <aside>
	        <div class="history-header">
	          <span>History</span>
	          <button id="refreshSessions" type="button">Refresh</button>
	        </div>
	        <div id="sessions" class="session-list" aria-label="History sessions"></div>
	      </aside>
	      <main id="messages" aria-live="polite"></main>
	    </div>
	    <form id="composer">
      <textarea id="content" placeholder="Message" required></textarea>
      <button id="send" type="submit">Send</button>
    </form>
  </div>
  <script>
    const messages = document.querySelector('#messages');
	    const composer = document.querySelector('#composer');
	    const content = document.querySelector('#content');
	    const chatId = document.querySelector('#chatId');
	    const send = document.querySelector('#send');
	    const sessions = document.querySelector('#sessions');
	    const refreshSessions = document.querySelector('#refreshSessions');
	    let source;
	    let currentSessionKey = null;
	    let streamingAssistant = null;
	    let streamingAssistantText = '';

    function escapeHtml(value) {
      return value.replace(/[&<>"']/g, (char) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
      })[char]);
    }

    function renderInlineMarkdown(text) {
      let html = escapeHtml(text);
      html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
      html = html.replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>');
      html = html.replace(/\\*([^*]+)\\*/g, '<em>$1</em>');
      html = html.replace(/\\[([^\\]]+)\\]\\((https?:\\/\\/[^\\s)]+)\\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
      return html;
    }

    function renderMarkdown(text) {
      const lines = text.replace(/\\r\\n/g, '\\n').split('\\n');
      const output = [];
      let listType = '';
      let inCode = false;

      function closeList() {
        if (listType) {
          output.push(`</${listType}>`);
          listType = '';
        }
      }

      for (const line of lines) {
        if (line.startsWith('```')) {
          if (inCode) {
            output.push('</code></pre>');
            inCode = false;
          } else {
            closeList();
            output.push('<pre><code>');
            inCode = true;
          }
          continue;
        }
        if (inCode) {
          output.push(`${escapeHtml(line)}\\n`);
          continue;
        }
        if (!line.trim()) {
          closeList();
          continue;
        }
        const heading = line.match(/^(#{1,3})\\s+(.+)$/);
        if (heading) {
          closeList();
          const level = heading[1].length;
          output.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
          continue;
        }
        const quote = line.match(/^>\\s?(.+)$/);
        if (quote) {
          closeList();
          output.push(`<blockquote>${renderInlineMarkdown(quote[1])}</blockquote>`);
          continue;
        }
        const bullet = line.match(/^[-*]\\s+(.+)$/);
        if (bullet) {
          if (listType !== 'ul') {
            closeList();
            output.push('<ul>');
            listType = 'ul';
          }
          output.push(`<li>${renderInlineMarkdown(bullet[1])}</li>`);
          continue;
        }
        const numbered = line.match(/^\\d+\\.\\s+(.+)$/);
        if (numbered) {
          if (listType !== 'ol') {
            closeList();
            output.push('<ol>');
            listType = 'ol';
          }
          output.push(`<li>${renderInlineMarkdown(numbered[1])}</li>`);
          continue;
        }
        closeList();
        output.push(`<p>${renderInlineMarkdown(line)}</p>`);
      }
      closeList();
      if (inCode) output.push('</code></pre>');
      return output.join('');
    }

    function setRowContent(row, text, markdown = false) {
      if (markdown) {
        row.innerHTML = renderMarkdown(text);
      } else {
        row.textContent = text;
      }
    }

    function addRow(kind, text, markdown = false) {
      const row = document.createElement('div');
      row.className = `row ${kind}`;
      setRowContent(row, text, markdown);
      messages.append(row);
      messages.scrollTop = messages.scrollHeight;
      return row;
    }

    function appendAssistantDelta(delta) {
      if (!streamingAssistant) {
        streamingAssistant = addRow('assistant', '', true);
        streamingAssistantText = '';
      }
      streamingAssistantText += delta;
      setRowContent(streamingAssistant, streamingAssistantText, true);
      messages.scrollTop = messages.scrollHeight;
    }

    function finishAssistantMessage(text) {
      if (streamingAssistant) {
        setRowContent(streamingAssistant, text, true);
        streamingAssistant = null;
        streamingAssistantText = '';
        messages.scrollTop = messages.scrollHeight;
      } else {
        addRow('assistant', text, true);
      }
    }

    function resetStreamingAssistant() {
      streamingAssistant = null;
      streamingAssistantText = '';
    }

    function currentChatId() {
      return chatId.value.trim() || 'direct';
    }

    function chatIdForSessionKey(key) {
      return key.startsWith('gateway:') ? (key.slice('gateway:'.length) || 'direct') : key;
    }

    function setActiveSession(key) {
      for (const item of sessions.querySelectorAll('.session-item')) {
        item.classList.toggle('active', item.dataset.sessionKey === key);
      }
    }

    function renderSessions(items) {
      sessions.textContent = '';
      if (!Array.isArray(items) || items.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'history-empty';
        empty.textContent = 'No history';
        sessions.append(empty);
        return;
      }
      for (const session of items) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'session-item';
        button.dataset.sessionKey = session.key;
        const title = document.createElement('span');
        title.className = 'session-title';
        title.textContent = session.title || session.key;
        const key = document.createElement('span');
        key.className = 'session-key';
        key.textContent = session.key;
        const preview = document.createElement('span');
        preview.className = 'session-preview';
        preview.textContent = session.preview || session.updated_at || '';
        button.append(title, key, preview);
        button.addEventListener('click', () => loadSession(session.key));
        sessions.append(button);
      }
      setActiveSession(currentSessionKey);
    }

    async function loadSessions() {
      try {
        const response = await fetch('/api/sessions');
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || 'Could not load sessions');
        renderSessions(payload.sessions);
      } catch (error) {
        sessions.textContent = '';
        const row = document.createElement('div');
        row.className = 'history-empty';
        row.textContent = String(error);
        sessions.append(row);
      }
    }

    async function loadSession(key) {
      try {
        const response = await fetch(`/api/sessions?key=${encodeURIComponent(key)}`);
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || 'Could not load session');
        currentSessionKey = payload.key;
        chatId.value = chatIdForSessionKey(payload.key);
        messages.textContent = '';
        resetStreamingAssistant();
        for (const message of payload.messages || []) {
          addRow(message.role === 'user' ? 'user' : 'assistant', message.content, message.role === 'assistant');
        }
        setActiveSession(currentSessionKey);
        connect();
        content.focus();
      } catch (error) {
        addRow('error', String(error));
      }
    }

    function connect() {
      if (source) source.close();
      const session = encodeURIComponent(currentChatId());
      source = new EventSource(`/api/events?chat_id=${session}`);
      source.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        if (payload.type === 'message_delta') {
          appendAssistantDelta(payload.content);
        } else if (payload.type === 'tool_progress') {
          addRow('progress', payload.content);
        } else if (payload.type === 'error') {
          addRow('error', payload.content);
        } else if (payload.type === 'control') {
          addRow('progress', payload.content);
        } else {
          finishAssistantMessage(payload.content);
        }
        if (payload.terminal) loadSessions();
      };
      source.onerror = () => addRow('error', 'Connection lost. Reconnecting...');
    }

    refreshSessions.addEventListener('click', loadSessions);
    chatId.addEventListener('change', () => {
      currentSessionKey = null;
      messages.textContent = '';
      resetStreamingAssistant();
      setActiveSession(null);
      connect();
    });
    composer.addEventListener('submit', async (event) => {
      event.preventDefault();
      const text = content.value.trim();
      if (!text) return;
      addRow('user', text);
      content.value = '';
      send.disabled = true;
      try {
        const body = {chat_id: currentChatId(), content: text, session_key: currentSessionKey};
        if (!body.session_key) delete body.session_key;
        const response = await fetch('/api/messages', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(body)
        });
        if (!response.ok) {
          const payload = await response.json();
          addRow('error', payload.error || 'Request failed');
        }
      } catch (error) {
        addRow('error', String(error));
      } finally {
        send.disabled = false;
        content.focus();
      }
    });
    loadSessions();
    connect();
    content.focus();
  </script>
</body>
</html>
"""
