from __future__ import annotations

from myclaw.agent.loop import AgentLoop
from myclaw.bus import MessageBus, OutboundMessage


class AgentDispatcher:
    """Bridge one inbound bus message to one outbound agent response."""

    def __init__(self, bus: MessageBus, loop: AgentLoop) -> None:
        self.bus = bus
        self.loop = loop

    async def process_next(self) -> None:
        msg = await self.bus.consume_inbound()
        try:
            result = await self.loop.process(msg.content)
            content = result.content
        except Exception as exc:
            content = f"Error: {exc}"
        await self.bus.publish_outbound(
            OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=content,
                metadata=dict(msg.metadata),
            )
        )
