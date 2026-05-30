from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
from pathlib import Path

from myclaw.agent import AgentConfig, AgentDispatcher, AgentLoop
from myclaw.bus import InboundMessage, MessageBus, OutboundMessage
from myclaw.config.env import load_env_file
from myclaw.providers import FakeProvider, OpenAICompatibleProvider
from myclaw.session import SessionManager
from myclaw.tools import build_default_tool_registry

EXIT_COMMANDS = {"exit", "quit"}
CLI_SESSION_KEY = "cli:direct"


def build_agent_loop() -> AgentLoop:
    load_env_file()
    session_manager = SessionManager()
    tool_registry = build_default_tool_registry(Path.cwd())
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        provider = OpenAICompatibleProvider(
            api_key=api_key,
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=model,
        )
        return AgentLoop(
            provider,
            AgentConfig(model=model),
            session_manager=session_manager,
            tool_registry=tool_registry,
        )
    return AgentLoop(
        FakeProvider(),
        AgentConfig(model="fake"),
        session_manager=session_manager,
        tool_registry=tool_registry,
    )


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
    task = asyncio.create_task(dispatcher.run())
    try:
        while True:
            outbound = await dispatcher.bus.consume_outbound()
            if outbound.terminal:
                return outbound
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def run_once(dispatcher: AgentDispatcher, text: str) -> None:
    outbound = await dispatch_text(dispatcher, text)
    print(outbound.content)


async def run_interactive(dispatcher: AgentDispatcher) -> None:
    pending = {"count": 0, "changed": asyncio.Event()}
    dispatcher_task = asyncio.create_task(dispatcher.run())
    output_task = asyncio.create_task(_print_interactive_output(dispatcher, pending))
    try:
        while True:
            try:
                text = await asyncio.to_thread(input, "You: ")
            except EOFError:
                print()
                await _wait_for_pending_output(pending)
                return
            if text.strip().lower() in EXIT_COMMANDS:
                await _wait_for_pending_output(pending)
                return
            if not text.strip():
                continue
            pending["count"] += 1
            pending["changed"].clear()
            await dispatcher.bus.publish_inbound(
                InboundMessage(
                    channel="cli",
                    sender_id="user",
                    chat_id="direct",
                    content=text,
                )
            )
            await asyncio.sleep(0)
            await _wait_for_pending_output(pending, timeout=None)
    finally:
        for task in (output_task, dispatcher_task):
            task.cancel()
        for task in (output_task, dispatcher_task):
            with contextlib.suppress(asyncio.CancelledError):
                await task


async def _print_interactive_output(dispatcher: AgentDispatcher, pending: dict) -> None:
    while True:
        outbound = await dispatcher.bus.consume_outbound()
        if outbound.event_type == "control":
            print(outbound.content)
        elif outbound.event_type == "tool_progress":
            print(f"Progress: {outbound.content}")
        else:
            print(f"Assistant: {outbound.content}")
        if outbound.terminal:
            pending["count"] = max(0, pending["count"] - 1)
            pending["changed"].set()


async def _wait_for_pending_output(pending: dict, timeout: float | None = 0.2) -> None:
    while pending["count"] > 0:
        pending["changed"].clear()
        if timeout is None:
            await pending["changed"].wait()
            continue
        try:
            await asyncio.wait_for(pending["changed"].wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return


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
