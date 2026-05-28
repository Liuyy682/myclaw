from __future__ import annotations

import argparse
import asyncio
import os

from myclaw.agent import AgentConfig, AgentLoop
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


async def run_once(loop: AgentLoop, text: str) -> None:
    result = await loop.process(text)
    print(result.content)


async def run_interactive(loop: AgentLoop) -> None:
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
        result = await loop.process(text)
        print(f"Assistant: {result.content}")


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Run the myclaw assistant MVP.")
    parser.add_argument("message", nargs="*", help="Message to send in one-shot mode.")
    args = parser.parse_args()

    loop = build_agent_loop()
    if args.message:
        await run_once(loop, " ".join(args.message))
    else:
        await run_interactive(loop)


def main() -> None:
    asyncio.run(async_main())
