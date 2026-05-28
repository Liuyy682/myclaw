from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from myclaw import Agent, AgentConfig, FakeProvider, OpenAICompatibleProvider

EXIT_COMMANDS = {"exit", "quit"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE_VAR = "MYCLAW_ENV_FILE"


def load_env_file(path: Path | None = None) -> None:
    if path is None:
        configured = os.environ.get(ENV_FILE_VAR)
        path = Path(configured) if configured else PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def build_agent() -> Agent:
    load_env_file()
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        provider = OpenAICompatibleProvider(
            api_key=api_key,
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=model,
        )
        return Agent(provider, AgentConfig(model=model))
    return Agent(FakeProvider(), AgentConfig(model="fake"))


async def run_once(agent: Agent, text: str) -> None:
    result = await agent.run(text)
    print(result.content)


async def run_interactive(agent: Agent) -> None:
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
        result = await agent.run(text)
        print(f"Assistant: {result.content}")


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Run the myclaw assistant MVP.")
    parser.add_argument("message", nargs="*", help="Message to send in one-shot mode.")
    args = parser.parse_args()

    agent = build_agent()
    if args.message:
        await run_once(agent, " ".join(args.message))
    else:
        await run_interactive(agent)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
