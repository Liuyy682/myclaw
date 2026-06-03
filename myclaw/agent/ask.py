from __future__ import annotations

import asyncio

from myclaw.bus import MessageBus, OutboundMessage

ASK_NO_RESPONSE = "(no response)"
DEFAULT_ASK_TIMEOUT_SECONDS = 300.0


class AskCoordinator:
    """Bridge a tool's ask() request to the next inbound message on the session."""

    def __init__(self, bus: MessageBus, *, timeout_seconds: float = DEFAULT_ASK_TIMEOUT_SECONDS) -> None:
        self.bus = bus
        self.timeout_seconds = timeout_seconds
        self._pending: dict[str, asyncio.Future[str]] = {}

    async def ask(self, session_key: str, question: str, choices: list[str] | None = None) -> str:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        # Only one outstanding question per session; a new one supersedes the old.
        previous = self._pending.get(session_key)
        if previous is not None and not previous.done():
            previous.set_result(ASK_NO_RESPONSE)
        self._pending[session_key] = future

        await self._publish_question(session_key, question, choices or [])
        try:
            if self.timeout_seconds and self.timeout_seconds > 0:
                return await asyncio.wait_for(asyncio.shield(future), timeout=self.timeout_seconds)
            return await future
        except asyncio.TimeoutError:
            return ASK_NO_RESPONSE
        finally:
            if self._pending.get(session_key) is future:
                del self._pending[session_key]

    def submit_answer(self, session_key: str, text: str) -> bool:
        future = self._pending.get(session_key)
        if future is None or future.done():
            return False
        future.set_result(text)
        return True

    async def _publish_question(self, session_key: str, question: str, choices: list[str]) -> None:
        channel, _, chat_id = session_key.partition(":")
        await self.bus.publish_outbound(
            OutboundMessage(
                channel=channel or "cli",
                chat_id=chat_id or session_key,
                content=question,
                metadata={"session_key": session_key, "choices": list(choices)},
                terminal=False,
                event_type="ask",
            )
        )
