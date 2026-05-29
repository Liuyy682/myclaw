from __future__ import annotations

from myclaw.agent.loop import AgentLoop
from myclaw.bus import MessageBus, OutboundMessage


class AgentDispatcher:
    """Continuously bridge inbound bus messages to outbound agent responses."""

    def __init__(self, bus: MessageBus, loop: AgentLoop) -> None:
        self.bus = bus
        self.loop = loop

    async def run(self) -> None:
        while True:
            msg = await self.bus.consume_inbound()
            try:
                result = await self.loop.run(msg.content, session_key=msg.session_key)
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
