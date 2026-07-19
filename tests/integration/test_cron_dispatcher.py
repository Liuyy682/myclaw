import asyncio
import contextlib
from datetime import datetime, timedelta

from myclaw import AgentDispatcher
from myclaw.bus import MessageBus
from myclaw.cron import CronStore


async def _stop(task):
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


class CronLoop:
    def __init__(self, workspace):
        self.cron_store = CronStore(workspace)
        self.auto_compact = None
        self.calls = []
        self.called = asyncio.Event()

    async def run(self, text, *, session_key, channel=None, chat_id=None, metadata=None, **kwargs):
        self.calls.append(
            {
                "text": text,
                "session_key": session_key,
                "channel": channel,
                "chat_id": chat_id,
                "metadata": metadata,
            }
        )
        self.called.set()
        return type("Result", (), {"content": "cron done"})()


def test_dispatcher_idle_tick_runs_due_cron_jobs(tmp_path):
    bus = MessageBus()
    loop = CronLoop(tmp_path)
    job = loop.cron_store.create(
        name="due",
        prompt="run scheduled check",
        at=datetime.now() - timedelta(seconds=1),
        next_run_at=datetime.now() - timedelta(seconds=1),
        session_key="cron:due",
    )
    dispatcher = AgentDispatcher(bus, loop)
    dispatcher._AUTO_COMPACT_IDLE_TICK_SECONDS = 0.01

    async def scenario():
        task = asyncio.create_task(dispatcher.run())
        await asyncio.wait_for(loop.called.wait(), timeout=0.5)
        outbound = await bus.consume_outbound()
        await _stop(task)
        return outbound

    outbound = asyncio.run(scenario())

    assert loop.calls == [
        {
            "text": "run scheduled check",
            "session_key": "cron:due",
            "channel": "cron",
            "chat_id": job["id"],
            "metadata": {"cron_job_id": job["id"], "cron_job_name": "due"},
        }
    ]
    assert outbound.channel == "cron"
    assert outbound.chat_id == job["id"]
    assert outbound.content == "cron done"
    assert outbound.event_type == "cron"
