from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class InboundMessage:
    """Message received from an input surface."""

    channel: str
    sender_id: str
    chat_id: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    session_key_override: str | None = None

    @property
    def session_key(self) -> str:
        return self.session_key_override or f"{self.channel}:{self.chat_id}"


@dataclass
class OutboundMessage:
    """Message produced by the agent for an output surface."""

    channel: str
    chat_id: str
    content: str
    reply_to: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    terminal: bool = True
    event_type: str = "message"
