from __future__ import annotations

import asyncio

from myclaw.bus.events import InboundMessage, OutboundMessage


class MessageBus:
    """Async in-memory bus between input surfaces and the agent core."""

    def __init__(self, *, inbound_maxsize: int = 64) -> None:
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue(maxsize=inbound_maxsize)
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()

    async def publish_inbound(self, msg: InboundMessage) -> None:
        await self.inbound.put(msg)

    def try_publish_inbound(self, msg: InboundMessage) -> bool:
        try:
            self.inbound.put_nowait(msg)
        except asyncio.QueueFull:
            return False
        return True

    async def consume_inbound(self) -> InboundMessage:
        return await self.inbound.get()

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        await self.outbound.put(msg)

    async def consume_outbound(self) -> OutboundMessage:
        return await self.outbound.get()

    @property
    def inbound_size(self) -> int:
        return self.inbound.qsize()

    @property
    def outbound_size(self) -> int:
        return self.outbound.qsize()
