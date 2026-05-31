from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from dataclasses import dataclass
from typing import Any, TextIO

from myclaw.agent import AgentDispatcher, DispatcherRuntime
from myclaw.bus import InboundMessage, OutboundMessage

GATEWAY_CHANNEL = "gateway"
DEFAULT_GATEWAY_CHAT_ID = "direct"


@dataclass(slots=True)
class _GatewayPending:
    count: int
    changed: asyncio.Event


async def run_gateway(
    dispatcher: AgentDispatcher,
    *,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
) -> None:
    input_stream = input_stream or sys.stdin
    output_stream = output_stream or sys.stdout
    pending = _GatewayPending(count=0, changed=asyncio.Event())
    write_lock = asyncio.Lock()

    async with DispatcherRuntime(dispatcher):
        output_task = asyncio.create_task(_write_outbound_events(dispatcher, output_stream, pending, write_lock))
        try:
            while True:
                line = await asyncio.to_thread(input_stream.readline)
                if line == "":
                    break
                if not line.strip():
                    continue
                message, error = _parse_gateway_line(line)
                if error is not None:
                    await _write_json_line(output_stream, error, write_lock)
                    continue
                assert message is not None
                pending.count += 1
                pending.changed.clear()
                await dispatcher.bus.publish_inbound(message)
            await _wait_for_pending(pending)
        finally:
            output_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await output_task


def _parse_gateway_line(line: str) -> tuple[InboundMessage | None, dict[str, Any] | None]:
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as exc:
        return None, _error_event(f"Error: invalid JSON: {exc.msg}")

    if not isinstance(raw, dict):
        return None, _error_event("Error: gateway input must be a JSON object")

    request_id = raw.get("id")
    if request_id is not None and not isinstance(request_id, str):
        return None, _error_event("Error: gateway input id must be a string", chat_id=_chat_id_or_default(raw))

    chat_id = raw.get("chat_id", DEFAULT_GATEWAY_CHAT_ID)
    if not isinstance(chat_id, str) or not chat_id:
        return None, _error_event("Error: gateway input chat_id must be a non-empty string", request_id=request_id)

    content = raw.get("content")
    if not isinstance(content, str):
        return None, _error_event(
            "Error: gateway input must include string content",
            request_id=request_id,
            chat_id=chat_id,
        )

    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        return None, _error_event(
            "Error: gateway input metadata must be an object",
            request_id=request_id,
            chat_id=chat_id,
        )
    metadata = dict(metadata)
    if request_id is not None:
        metadata["request_id"] = request_id

    session_key = raw.get("session_key")
    if session_key is not None and not isinstance(session_key, str):
        return None, _error_event(
            "Error: gateway input session_key must be a string",
            request_id=request_id,
            chat_id=chat_id,
        )

    return (
        InboundMessage(
            channel=GATEWAY_CHANNEL,
            sender_id="user",
            chat_id=chat_id,
            content=content,
            metadata=metadata,
            session_key_override=session_key,
        ),
        None,
    )


async def _write_outbound_events(
    dispatcher: AgentDispatcher,
    output_stream: TextIO,
    pending: _GatewayPending,
    write_lock: asyncio.Lock,
) -> None:
    while True:
        outbound = await dispatcher.bus.consume_outbound()
        await _write_json_line(output_stream, _outbound_event(outbound), write_lock)
        if outbound.terminal:
            pending.count = max(0, pending.count - 1)
            pending.changed.set()


async def _wait_for_pending(pending: _GatewayPending) -> None:
    while pending.count > 0:
        await pending.changed.wait()
        pending.changed.clear()


async def _write_json_line(output_stream: TextIO, payload: dict[str, Any], write_lock: asyncio.Lock) -> None:
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    async with write_lock:
        output_stream.write(line)
        output_stream.flush()


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


def _error_event(
    content: str,
    *,
    request_id: str | None = None,
    chat_id: str = DEFAULT_GATEWAY_CHAT_ID,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "error",
        "chat_id": chat_id,
        "content": content,
        "terminal": True,
        "metadata": {},
    }
    if request_id is not None:
        event["id"] = request_id
    return event


def _chat_id_or_default(raw: dict[str, Any]) -> str:
    chat_id = raw.get("chat_id")
    return chat_id if isinstance(chat_id, str) and chat_id else DEFAULT_GATEWAY_CHAT_ID
