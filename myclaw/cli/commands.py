from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
from datetime import datetime
from pathlib import Path

from myclaw.agent import AgentConfig, AgentDispatcher, AgentLoop
from myclaw.bus import InboundMessage, MessageBus, OutboundMessage
from myclaw.config.env import load_env_file
from myclaw.providers import FakeProvider, OpenAICompatibleProvider
from myclaw.session import SessionManager
from myclaw.tools import build_default_tool_registry

EXIT_COMMANDS = {"exit", "quit"}
DEFAULT_CLI_SESSION_NAME = "direct"
CLI_SESSION_PREFIX = "cli:"


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
            AgentConfig(model=model, auto_title=True),
            session_manager=session_manager,
            tool_registry=tool_registry,
        )
    return AgentLoop(
        FakeProvider(),
        AgentConfig(model="fake", auto_title=True),
        session_manager=session_manager,
        tool_registry=tool_registry,
    )


def build_dispatcher() -> AgentDispatcher:
    return AgentDispatcher(MessageBus(), build_agent_loop())


def _cli_session_key(session_name: str) -> str:
    return f"{CLI_SESSION_PREFIX}{session_name}"


def _cli_session_name(session_key: str) -> str | None:
    if not session_key.startswith(CLI_SESSION_PREFIX):
        return None
    return session_key[len(CLI_SESSION_PREFIX):]


async def dispatch_text(
    dispatcher: AgentDispatcher,
    text: str,
    *,
    session_name: str = DEFAULT_CLI_SESSION_NAME,
) -> OutboundMessage:
    await dispatcher.bus.publish_inbound(
        InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id=session_name,
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


async def run_once(
    dispatcher: AgentDispatcher,
    text: str,
    *,
    session_name: str = DEFAULT_CLI_SESSION_NAME,
) -> None:
    outbound = await dispatch_text(dispatcher, text, session_name=session_name)
    print(outbound.content)


async def run_interactive(
    dispatcher: AgentDispatcher,
    *,
    session_name: str | None = None,
) -> None:
    if session_name is None:
        session_name = _create_new_cli_session(dispatcher)
    pending = {"count": 0, "changed": asyncio.Event()}
    dispatcher_task = asyncio.create_task(dispatcher.run())
    output_task = asyncio.create_task(_print_interactive_output(dispatcher, pending))
    try:
        while True:
            try:
                text = await asyncio.to_thread(input, f"You[{_session_label(dispatcher, session_name)}]: ")
            except EOFError:
                print()
                await _wait_for_pending_output(pending)
                return
            if text.strip().lower() in EXIT_COMMANDS:
                await _wait_for_pending_output(pending)
                return
            if not text.strip():
                continue
            resumed = _resume_target(text)
            if resumed is not None:
                session_name = _handle_resume(dispatcher, session_name, resumed, pending)
                continue
            if _new_requested(text):
                session_name = _handle_new_session(dispatcher, session_name, pending)
                continue
            pending["count"] += 1
            pending["changed"].clear()
            await dispatcher.bus.publish_inbound(
                InboundMessage(
                    channel="cli",
                    sender_id="user",
                    chat_id=session_name,
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


def _resume_target(text: str) -> str | None:
    stripped = text.strip()
    lowered = stripped.lower()
    if lowered == "/resume":
        return ""
    if lowered.startswith("/resume "):
        return stripped[len("/resume "):].strip()
    return None


def _handle_resume(
    dispatcher: AgentDispatcher,
    current_session_name: str,
    target: str,
    pending: dict,
) -> str:
    if pending["count"] > 0:
        print("Cannot switch sessions while a turn is running. Use /stop first.")
        return current_session_name

    manager = dispatcher.loop.session_manager
    if not target:
        _print_session_list(manager, current_session_name)
        return current_session_name

    session = manager.get_or_create(_cli_session_key(target))
    manager.save(session)
    print(f"Resumed session '{target}' - {_session_display_title(session)}")
    return target


def _new_requested(text: str) -> bool:
    return text.strip().lower() == "/new"


def _handle_new_session(
    dispatcher: AgentDispatcher,
    current_session_name: str,
    pending: dict,
) -> str:
    if pending["count"] > 0:
        print("Cannot start a new session while a turn is running. Use /stop first.")
        return current_session_name

    session_name = _create_new_cli_session(dispatcher)
    print(f"Started new session '{session_name}'")
    return session_name


def _create_new_cli_session(dispatcher: AgentDispatcher) -> str:
    manager = dispatcher.loop.session_manager
    session_name = new_cli_session_name(manager)
    session = manager.get_or_create(_cli_session_key(session_name))
    manager.save(session)
    return session_name


def new_cli_session_name(manager: SessionManager) -> str:
    base = datetime.now().strftime("chat-%Y%m%d-%H%M%S")
    existing = {
        name
        for session in manager.list_sessions()
        if (name := _cli_session_name(session.key)) is not None
    }
    if base not in existing:
        return base
    index = 2
    while f"{base}-{index}" in existing:
        index += 1
    return f"{base}-{index}"


def _print_session_list(manager: SessionManager, current_session_name: str) -> None:
    current = manager.get_or_create(_cli_session_key(current_session_name))
    manager.save(current)
    sessions = [
        session
        for session in manager.list_sessions()
        if _cli_session_name(session.key) is not None
    ]
    print("Sessions:")
    for session in sessions:
        name = _cli_session_name(session.key)
        assert name is not None
        marker = "*" if name == current_session_name else " "
        print(f"{marker} {name} - {_session_display_title(session)}")


def _session_display_title(session) -> str:
    title = session.metadata.get("title")
    return title if isinstance(title, str) and title.strip() else "Untitled"


def _session_label(dispatcher: AgentDispatcher, session_name: str) -> str:
    manager = dispatcher.loop.session_manager
    if manager is None:
        return session_name
    session = manager.get_or_create(_cli_session_key(session_name))
    title = session.metadata.get("title")
    return title if isinstance(title, str) and title.strip() else session_name


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Run the myclaw assistant MVP.")
    parser.add_argument(
        "--session",
        default=None,
        help="CLI session name to use.",
    )
    parser.add_argument("message", nargs="*", help="Message to send in one-shot mode.")
    args = parser.parse_args()

    dispatcher = build_dispatcher()
    if args.message:
        await run_once(
            dispatcher,
            " ".join(args.message),
            session_name=args.session or DEFAULT_CLI_SESSION_NAME,
        )
    else:
        await run_interactive(dispatcher, session_name=args.session)


def main() -> None:
    asyncio.run(async_main())
