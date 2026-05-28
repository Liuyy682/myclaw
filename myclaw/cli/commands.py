from __future__ import annotations

import argparse
import asyncio
import os

from myclaw.agent import AgentConfig, AgentDispatcher, AgentLoop
from myclaw.bus import InboundMessage, MessageBus, OutboundMessage
from myclaw.config.env import load_env_file
from myclaw.providers import FakeProvider, OpenAICompatibleProvider

EXIT_COMMANDS = {"exit", "quit"}


def build_agent_loop() -> AgentLoop:
    load_env_file()
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        provider = OpenAICompatibleProvider(
            api_key=api_key,
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=model,
        )
        return AgentLoop(provider, AgentConfig(model=model))
    return AgentLoop(FakeProvider(), AgentConfig(model="fake"))


def build_dispatcher() -> AgentDispatcher:
    return AgentDispatcher(MessageBus(), build_agent_loop())


async def dispatch_text(dispatcher: AgentDispatcher, text: str) -> OutboundMessage:
    await dispatcher.bus.publish_inbound(
        InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="direct",
            content=text,
        )
    )
    await dispatcher.process_next()
    return await dispatcher.bus.consume_outbound()


async def run_once(dispatcher: AgentDispatcher, text: str) -> None:
    outbound = await dispatch_text(dispatcher, text)
    print(outbound.content)


async def run_interactive(dispatcher: AgentDispatcher) -> None:
    while True:
        try:
            text = input("You: ")
        except EOFError:
            print()
            return
        if text.strip().lower() in EXIT_COMMANDS:
            return
        if not text.strip():
            continue
        outbound = await dispatch_text(dispatcher, text)
        print(f"Assistant: {outbound.content}")


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Run the myclaw assistant MVP.")
    parser.add_argument("message", nargs="*", help="Message to send in one-shot mode.")
    args = parser.parse_args()

    dispatcher = build_dispatcher()
    if args.message:
        await run_once(dispatcher, " ".join(args.message))
    else:
        await run_interactive(dispatcher)


def main() -> None:
    asyncio.run(async_main())
