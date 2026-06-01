from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import sys
from datetime import datetime
from pathlib import Path

from myclaw.agent import AgentConfig, AgentDispatcher, AgentLoop, DispatcherRuntime
from myclaw.bus import InboundMessage, MessageBus, OutboundMessage
from myclaw.config import (
    CLI_EXIT_COMMANDS,
    CLI_SESSION_PREFIX,
    DEFAULT_CLI_SESSION_NAME,
    DEFAULT_GATEWAY_HOST,
    DEFAULT_GATEWAY_PORT,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    FAKE_PROVIDER_MODEL,
    IDLE_COMPACT_AFTER_MINUTES_ENV_VAR,
    OPENAI_API_KEY_ENV_VAR,
    OPENAI_BASE_URL_ENV_VAR,
    OPENAI_MODEL_ENV_VAR,
    load_env_file,
)
from myclaw.gateway import run_gateway
from myclaw.providers import FakeProvider, OpenAICompatibleProvider
from myclaw.session import SessionManager
from myclaw.tools import build_default_tool_registry


def build_agent_loop() -> AgentLoop:
    load_env_file()
    session_manager = SessionManager()
    tool_registry = build_default_tool_registry(Path.cwd(), memory_workspace=session_manager.workspace)
    model = os.environ.get(OPENAI_MODEL_ENV_VAR, DEFAULT_OPENAI_MODEL)
    idle_compact_after_minutes = _env_int(IDLE_COMPACT_AFTER_MINUTES_ENV_VAR, default=0)
    api_key = os.environ.get(OPENAI_API_KEY_ENV_VAR)
    if api_key:
        provider = OpenAICompatibleProvider(
            api_key=api_key,
            base_url=os.environ.get(OPENAI_BASE_URL_ENV_VAR, DEFAULT_OPENAI_BASE_URL),
            model=model,
        )
        return AgentLoop(
            provider,
            AgentConfig(
                model=model,
                auto_title=True,
                idle_compact_after_minutes=idle_compact_after_minutes,
            ),
            session_manager=session_manager,
            tool_registry=tool_registry,
        )
    return AgentLoop(
        FakeProvider(),
        AgentConfig(
            model=FAKE_PROVIDER_MODEL,
            auto_title=True,
            idle_compact_after_minutes=idle_compact_after_minutes,
        ),
        session_manager=session_manager,
        tool_registry=tool_registry,
    )


def _env_int(name: str, *, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def build_dispatcher() -> AgentDispatcher:
    return AgentDispatcher(MessageBus(), build_agent_loop())


def _cli_session_key(session_name: str) -> str:
    return f"{CLI_SESSION_PREFIX}{session_name}"


def _cli_session_name(session_key: str) -> str | None:
    if not session_key.startswith(CLI_SESSION_PREFIX):
        return None
    return session_key[len(CLI_SESSION_PREFIX):]


async def dispatch_text(
    runtime: DispatcherRuntime,
    text: str,
    *,
    session_name: str = DEFAULT_CLI_SESSION_NAME,
) -> OutboundMessage:
    await runtime.dispatcher.bus.publish_inbound(
        InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id=session_name,
            content=text,
        )
    )
    while True:
        outbound = await runtime.dispatcher.bus.consume_outbound()
        if outbound.terminal:
            return outbound


async def run_once(
    dispatcher: AgentDispatcher,
    text: str,
    *,
    session_name: str = DEFAULT_CLI_SESSION_NAME,
) -> None:
    async with DispatcherRuntime(dispatcher) as runtime:
        outbound = await dispatch_text(runtime, text, session_name=session_name)
    print(outbound.content)


async def run_interactive(
    dispatcher: AgentDispatcher,
    *,
    session_name: str | None = None,
) -> None:
    if session_name is None:
        session_name = _create_new_cli_session(dispatcher)
    pending = {"count": 0, "changed": asyncio.Event()}
    async with DispatcherRuntime(dispatcher):
        output_task = asyncio.create_task(_print_interactive_output(dispatcher, pending))
        try:
            while True:
                try:
                    text = await asyncio.to_thread(input, f"You[{_session_label(dispatcher, session_name)}]: ")
                except EOFError:
                    print()
                    await _wait_for_pending_output(pending)
                    return
                if text.strip().lower() in CLI_EXIT_COMMANDS:
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
                        metadata={"stream": True},
                    )
                )
                await asyncio.sleep(0)
                await _wait_for_pending_output(pending, timeout=None)
        finally:
            output_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await output_task


async def _print_interactive_output(dispatcher: AgentDispatcher, pending: dict) -> None:
    streaming_assistant = False
    while True:
        outbound = await dispatcher.bus.consume_outbound()
        if outbound.event_type == "message_delta":
            if not streaming_assistant:
                print("Assistant: ", end="", flush=True)
                streaming_assistant = True
            print(outbound.content, end="", flush=True)
        elif outbound.event_type == "control":
            if streaming_assistant:
                print()
                streaming_assistant = False
            print(outbound.content)
        elif outbound.event_type == "tool_progress":
            if streaming_assistant:
                print()
                streaming_assistant = False
            print(f"Progress: {outbound.content}")
        elif streaming_assistant:
            print()
            streaming_assistant = False
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
    if len(sys.argv) > 1 and sys.argv[1] == "gateway":
        parser = argparse.ArgumentParser(description="Run the myclaw HTTP gateway.")
        parser.add_argument("--host", default=DEFAULT_GATEWAY_HOST, help="Host to bind.")
        parser.add_argument("--port", type=int, default=DEFAULT_GATEWAY_PORT, help="Port to bind.")
        args = parser.parse_args(sys.argv[2:])
        try:
            await run_gateway(build_dispatcher(), host=args.host, port=args.port)
        except OSError as exc:
            print(f"Error: could not start gateway on {args.host}:{args.port}: {exc}", file=sys.stderr)
            raise SystemExit(1) from None
        return

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
