import asyncio
import json
import os
import select
import socket
import subprocess
import sys
from types import SimpleNamespace

from myclaw.bus import MessageBus, OutboundMessage
import myclaw.gateway as gateway_module
from myclaw.gateway import HttpGatewayServer
from myclaw.memory import MemoryStore
from myclaw.session import Session, SessionManager


class RecordingDispatcher:
    def __init__(self):
        self.bus = MessageBus()
        self.run_calls = 0
        self.received = []

    async def run(self):
        self.run_calls += 1
        while True:
            msg = await self.bus.consume_inbound()
            self.received.append((msg.channel, msg.chat_id, msg.content, msg.metadata, msg.session_key_override))
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=f"ack: {msg.content}",
                    metadata=dict(msg.metadata),
                    event_type="control" if msg.content.startswith("/") else "message",
                )
            )


class ProgressDispatcher:
    def __init__(self):
        self.bus = MessageBus()
        self.run_calls = 0

    async def run(self):
        self.run_calls += 1
        await self.bus.consume_inbound()
        await self.bus.publish_outbound(
            OutboundMessage(
                channel="gateway",
                chat_id="direct",
                content="Running tool add (1/1)",
                metadata={"request_id": "req-progress"},
                terminal=False,
                event_type="tool_progress",
            )
        )
        await self.bus.publish_outbound(
            OutboundMessage(
                channel="gateway",
                chat_id="direct",
                content="done",
                metadata={"request_id": "req-progress"},
            )
        )
        await asyncio.Event().wait()


class DeltaDispatcher:
    def __init__(self):
        self.bus = MessageBus()

    async def run(self):
        await self.bus.consume_inbound()
        await self.bus.publish_outbound(
            OutboundMessage(
                channel="gateway",
                chat_id="direct",
                content="hel",
                metadata={"request_id": "req-delta"},
                terminal=False,
                event_type="message_delta",
            )
        )
        await self.bus.publish_outbound(
            OutboundMessage(
                channel="gateway",
                chat_id="direct",
                content="hello",
                metadata={"request_id": "req-delta"},
            )
        )
        await asyncio.Event().wait()


def _history_dispatcher(manager):
    dispatcher = RecordingDispatcher()
    dispatcher.loop = SimpleNamespace(
        session_manager=manager,
        memory_store=MemoryStore(manager.workspace),
    )
    return dispatcher


def _save_session(manager, key, title, messages):
    session = Session(key=key)
    if title is not None:
        session.metadata["title"] = title
    for role, content in messages:
        session.add_message(role, content)
    manager.save(session)
    return session


async def _request(port, method, path, body=None, headers=None):
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    headers = dict(headers or {})
    body_bytes = b""
    if body is not None:
        body_bytes = body.encode("utf-8")
        headers.setdefault("Content-Type", "application/json")
        headers["Content-Length"] = str(len(body_bytes))
    header_lines = "".join(f"{name}: {value}\r\n" for name, value in headers.items())
    writer.write(f"{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n{header_lines}\r\n".encode("utf-8"))
    writer.write(body_bytes)
    await writer.drain()
    status, response_headers = await _read_response_head(reader)
    length = int(response_headers.get("content-length", "0"))
    response_body = await reader.readexactly(length) if length else b""
    writer.close()
    await writer.wait_closed()
    return status, response_headers, response_body.decode("utf-8")


async def _open_sse(port, chat_id="direct"):
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(
        (
            f"GET /api/events?chat_id={chat_id} HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            "Accept: text/event-stream\r\n"
            "\r\n"
        ).encode("utf-8")
    )
    await writer.drain()
    status, response_headers = await _read_response_head(reader)
    assert status == 200
    assert response_headers["content-type"].startswith("text/event-stream")
    return reader, writer


async def _read_response_head(reader):
    raw = await reader.readuntil(b"\r\n\r\n")
    lines = raw.decode("iso-8859-1").split("\r\n")
    status = int(lines[0].split()[1])
    headers = {}
    for line in lines[1:]:
        if not line:
            continue
        name, value = line.split(":", 1)
        headers[name.lower()] = value.strip()
    return status, headers


async def _read_sse_json(reader):
    chunk = b""
    while b"\n\n" not in chunk:
        chunk += await asyncio.wait_for(reader.readexactly(1), timeout=1)
    event, _, _ = chunk.partition(b"\n\n")
    data_lines = [
        line.removeprefix("data: ")
        for line in event.decode("utf-8").splitlines()
        if line.startswith("data: ")
    ]
    return json.loads("\n".join(data_lines))


async def _with_server(dispatcher, scenario):
    server = HttpGatewayServer(dispatcher, host="127.0.0.1", port=0)
    await server.start()
    try:
        return await scenario(server)
    finally:
        await server.stop()


def test_gateway_root_serves_built_webui(tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text(
        '<!doctype html><html><body><div id="root"></div><script src="/assets/app.js"></script></body></html>',
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_module, "_WEB_DIST_DIR", dist)

    async def scenario(server):
        status, headers, body = await _request(server.port, "GET", "/")
        return status, headers, body

    status, headers, body = asyncio.run(_with_server(RecordingDispatcher(), scenario))

    assert status == 200
    assert headers["content-type"].startswith("text/html")
    assert 'id="root"' in body
    assert 'src="/assets/app.js"' in body


def test_repository_contains_production_webui_assets():
    index = gateway_module._WEB_DIST_DIR / "index.html"

    assert index.is_file()
    html = index.read_text(encoding="utf-8")
    assert '<div id="root"></div>' in html
    assert "/assets/" in html


def test_gateway_serves_static_assets_with_content_type_and_blocks_traversal(tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (assets / "app.js").write_text("console.log('myclaw')", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    monkeypatch.setattr(gateway_module, "_WEB_DIST_DIR", dist)

    async def scenario(server):
        asset = await _request(server.port, "GET", "/assets/app.js")
        traversal = await _request(server.port, "GET", "/assets/%2e%2e/%2e%2e/secret.txt")
        return asset, traversal

    asset, traversal = asyncio.run(_with_server(RecordingDispatcher(), scenario))

    assert asset[0] == 200
    assert "javascript" in asset[1]["content-type"]
    assert asset[2] == "console.log('myclaw')"
    assert traversal[0] == 404
    assert json.loads(traversal[2]) == {"error": "not found"}


def test_gateway_post_message_publishes_inbound_and_streams_terminal_sse_event():
    dispatcher = RecordingDispatcher()

    async def scenario(server):
        reader, writer = await _open_sse(server.port, chat_id="alpha")
        status, headers, body = await _request(
            server.port,
            "POST",
            "/api/messages",
            json.dumps({"chat_id": "alpha", "content": "hello"}),
        )
        response = json.loads(body)
        event = await _read_sse_json(reader)
        writer.close()
        await writer.wait_closed()
        return status, response, event

    status, response, event = asyncio.run(_with_server(dispatcher, scenario))

    assert status == 202
    assert response["accepted"] is True
    assert response["chat_id"] == "alpha"
    request_id = response["id"]
    assert dispatcher.run_calls == 1
    assert dispatcher.received == [
        ("gateway", "alpha", "hello", {"request_id": request_id}, None),
    ]
    assert event == {
        "type": "message",
        "id": request_id,
        "chat_id": "alpha",
        "content": "ack: hello",
        "terminal": True,
        "metadata": {"request_id": request_id},
    }


def test_gateway_post_message_passes_session_key_override():
    dispatcher = RecordingDispatcher()

    async def scenario(server):
        reader, writer = await _open_sse(server.port, chat_id="cli:direct")
        status, _headers, body = await _request(
            server.port,
            "POST",
            "/api/messages",
            json.dumps({"chat_id": "cli:direct", "session_key": "cli:direct", "content": "resume"}),
        )
        response = json.loads(body)
        event = await _read_sse_json(reader)
        writer.close()
        await writer.wait_closed()
        return status, response, event

    status, response, event = asyncio.run(_with_server(dispatcher, scenario))

    assert status == 202
    request_id = response["id"]
    assert dispatcher.received == [
        ("gateway", "cli:direct", "resume", {"request_id": request_id}, "cli:direct"),
    ]
    assert event["chat_id"] == "cli:direct"
    assert event["content"] == "ack: resume"


def test_gateway_lists_saved_sessions_for_history_switching(tmp_path):
    manager = SessionManager(tmp_path)
    _save_session(
        manager,
        "cli:direct",
        "Direct CLI",
        [("user", "from cli"), ("assistant", "cli answer")],
    )
    _save_session(
        manager,
        "gateway:direct",
        "Gateway Direct",
        [("user", "from gateway")],
    )

    async def scenario(server):
        return await _request(server.port, "GET", "/api/sessions")

    status, headers, body = asyncio.run(_with_server(_history_dispatcher(manager), scenario))
    payload = json.loads(body)

    assert status == 200
    assert headers["content-type"].startswith("application/json")
    sessions = payload["sessions"]
    assert [session["key"] for session in sessions] == ["gateway:direct", "cli:direct"]
    assert set(sessions[0]) == {
        "key",
        "channel",
        "title",
        "preview",
        "created_at",
        "updated_at",
        "message_count",
    }
    assert sessions[0]["channel"] == "gateway"
    assert sessions[0]["title"] == "Gateway Direct"
    assert sessions[0]["preview"] == "from gateway"
    assert sessions[0]["message_count"] == 1
    assert "T" in sessions[0]["created_at"]
    assert sessions[1]["channel"] == "cli"
    assert sessions[1]["title"] == "Direct CLI"
    assert sessions[1]["preview"] == "from cli"
    assert sessions[1]["message_count"] == 2


def test_gateway_reads_saved_session_display_messages(tmp_path):
    manager = SessionManager(tmp_path)
    _save_session(
        manager,
        "cli:direct",
        "Direct CLI",
        [
            ("user", "hello"),
            ("assistant", "hi **there**"),
            ("tool", "internal tool output"),
            ("assistant", "   "),
        ],
    )

    async def scenario(server):
        existing = await _request(server.port, "GET", "/api/sessions?key=cli%3Adirect")
        missing = await _request(server.port, "GET", "/api/sessions?key=missing")
        return existing, missing

    existing, missing = asyncio.run(_with_server(_history_dispatcher(manager), scenario))

    assert existing[0] == 200
    assert json.loads(existing[2]) == {
        "key": "cli:direct",
        "title": "Direct CLI",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi **there**"},
        ],
    }
    assert missing[0] == 404
    assert json.loads(missing[2]) == {"error": "session not found"}


def test_gateway_reads_long_term_memory(tmp_path):
    manager = SessionManager(tmp_path)
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "MEMORY.md").write_text("# Memory\n\nProject fact.", encoding="utf-8")
    (memory_dir / "USER.md").write_text("# User\n\nPrefers concise answers.", encoding="utf-8")
    (memory_dir / "SOUL.md").write_text("# Soul\n\nCalm and direct.", encoding="utf-8")

    async def scenario(server):
        return await _request(server.port, "GET", "/api/memory")

    status, headers, body = asyncio.run(_with_server(_history_dispatcher(manager), scenario))

    assert status == 200
    assert headers["content-type"].startswith("application/json")
    assert json.loads(body) == {
        "memory": "# Memory\n\nProject fact.",
        "user": "# User\n\nPrefers concise answers.",
        "soul": "# Soul\n\nCalm and direct.",
    }


def test_gateway_returns_empty_strings_for_missing_memory_files(tmp_path):
    manager = SessionManager(tmp_path)

    async def scenario(server):
        return await _request(server.port, "GET", "/api/memory")

    status, _headers, body = asyncio.run(_with_server(_history_dispatcher(manager), scenario))

    assert status == 200
    assert json.loads(body) == {"memory": "", "user": "", "soul": ""}


def test_gateway_sse_streams_tool_progress_and_final_events():
    async def scenario(server):
        reader, writer = await _open_sse(server.port, chat_id="direct")
        await _request(server.port, "POST", "/api/messages", json.dumps({"content": "use tool"}))
        progress = await _read_sse_json(reader)
        final = await _read_sse_json(reader)
        writer.close()
        await writer.wait_closed()
        return progress, final

    progress, final = asyncio.run(_with_server(ProgressDispatcher(), scenario))

    assert progress == {
        "type": "tool_progress",
        "id": "req-progress",
        "chat_id": "direct",
        "content": "Running tool add (1/1)",
        "terminal": False,
        "metadata": {"request_id": "req-progress"},
    }
    assert final == {
        "type": "message",
        "id": "req-progress",
        "chat_id": "direct",
        "content": "done",
        "terminal": True,
        "metadata": {"request_id": "req-progress"},
    }


def test_gateway_sse_streams_message_delta_before_final_event():
    async def scenario(server):
        reader, writer = await _open_sse(server.port, chat_id="direct")
        await _request(server.port, "POST", "/api/messages", json.dumps({"content": "stream"}))
        delta = await _read_sse_json(reader)
        final = await _read_sse_json(reader)
        writer.close()
        await writer.wait_closed()
        return delta, final

    delta, final = asyncio.run(_with_server(DeltaDispatcher(), scenario))

    assert delta == {
        "type": "message_delta",
        "id": "req-delta",
        "chat_id": "direct",
        "content": "hel",
        "terminal": False,
        "metadata": {"request_id": "req-delta"},
    }
    assert final == {
        "type": "message",
        "id": "req-delta",
        "chat_id": "direct",
        "content": "hello",
        "terminal": True,
        "metadata": {"request_id": "req-delta"},
    }


def test_gateway_rejects_invalid_requests_with_json_errors():
    async def scenario(server):
        invalid_json = await _request(server.port, "POST", "/api/messages", "{")
        blank_content = await _request(server.port, "POST", "/api/messages", json.dumps({"content": "  "}))
        wrong_method = await _request(server.port, "GET", "/api/messages")
        wrong_memory_method = await _request(server.port, "POST", "/api/memory", "{}")
        missing_route = await _request(server.port, "GET", "/missing")
        return invalid_json, blank_content, wrong_method, wrong_memory_method, missing_route

    invalid_json, blank_content, wrong_method, wrong_memory_method, missing_route = asyncio.run(
        _with_server(RecordingDispatcher(), scenario)
    )

    assert invalid_json[0] == 400
    assert json.loads(invalid_json[2])["error"].startswith("invalid JSON")
    assert blank_content[0] == 400
    assert json.loads(blank_content[2]) == {"error": "content must be a non-empty string"}
    assert wrong_method[0] == 405
    assert json.loads(wrong_method[2]) == {"error": "method not allowed"}
    assert wrong_memory_method[0] == 405
    assert json.loads(wrong_memory_method[2]) == {"error": "method not allowed"}
    assert missing_route[0] == 404
    assert json.loads(missing_route[2]) == {"error": "not found"}


def test_gateway_cli_starts_http_server_and_terminates_cleanly(tmp_path):
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env["MYCLAW_ENV_FILE"] = str(tmp_path / "missing.env")
    env["MYCLAW_WORKSPACE"] = str(tmp_path / "workspace")
    proc = subprocess.Popen(
        [sys.executable, "-m", "myclaw", "gateway", "--host", "127.0.0.1", "--port", "0"],
        cwd="/root/myclaw",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        ready, _, _ = select.select([proc.stderr], [], [], 5)
        assert ready
        line = proc.stderr.readline().strip()
        assert line.startswith("Gateway listening on http://127.0.0.1:")
        assert proc.poll() is None
    finally:
        proc.terminate()
        proc.wait(timeout=5)

    assert proc.stdout.read() == ""


def test_gateway_cli_reports_port_in_use_without_traceback(tmp_path):
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env["MYCLAW_ENV_FILE"] = str(tmp_path / "missing.env")
    env["MYCLAW_WORKSPACE"] = str(tmp_path / "workspace")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        port = sock.getsockname()[1]

        result = subprocess.run(
            [sys.executable, "-m", "myclaw", "gateway", "--host", "127.0.0.1", "--port", str(port)],
            cwd="/root/myclaw",
            env=env,
            capture_output=True,
            text=True,
        )

    assert result.returncode == 1
    assert result.stdout == ""
    assert "Error: could not start gateway on 127.0.0.1:" in result.stderr
    assert "address already in use" in result.stderr
    assert "Traceback" not in result.stderr
