import asyncio
import contextlib

from myclaw.agent import AgentConfig, AgentDispatcher, AgentLoop
from myclaw.bus import InboundMessage, MessageBus
from myclaw.providers import LLMResponse, ToolCallRequest
from myclaw.session import SessionManager
from myclaw.tasks import ProjectPlanStore
from myclaw.tools import build_default_tool_registry


class ScriptedPlanProvider:
    """Exercise real dispatch, schemas, approvals and storage without an API."""

    model = "plan-fixture"

    def __init__(self):
        self.steps = []
        self.schemas = []
        self.tool_results = []
        self.calls = 0

    async def complete(self, messages, *, tools=None, **kwargs):
        self.calls += 1
        self.schemas.append({entry["function"]["name"] for entry in tools or []})
        self.tool_results.extend(str(message["content"]) for message in messages if message["role"] == "tool")
        step = self.steps.pop(0)
        if isinstance(step, str):
            return LLMResponse(content=step)
        return LLMResponse(content="", final=False, stop_reason="tool_calls", tool_calls=[
            ToolCallRequest(id=f"plan-{self.calls}-{index}", name=name, arguments=args)
            for index, (name, args) in enumerate(step)
        ])


def test_plan_confirm_deviation_replan_complete_and_takeover(tmp_path):
    async def scenario():
        code = tmp_path / "code"
        code.mkdir()
        workspace = tmp_path / "state"
        provider = ScriptedPlanProvider()
        registry = build_default_tool_registry(code, state_workspace=workspace, memory_workspace=workspace)
        loop = AgentLoop(provider, AgentConfig(auto_title=False, observation_memory_enabled=False),
                         session_manager=SessionManager(workspace), tool_registry=registry)
        store = registry.get("task_list").plan_store
        bus = MessageBus()
        dispatcher = AgentDispatcher(bus, loop)
        running = asyncio.create_task(dispatcher.run())

        async def send(content, *, chat="one", expect_model=True):
            result = await dispatcher.submit(InboundMessage(
                channel="cli", sender_id="user", chat_id=chat, content=content,
            ))
            assert result.accepted
            while True:
                event = await asyncio.wait_for(bus.consume_outbound(), timeout=5)
                if event.event_type == "ask":
                    await dispatcher.submit(InboundMessage(
                        channel="cli", sender_id="user", chat_id=chat, content="allow",
                    ))
                elif event.terminal and (event.event_type != "control" or not expect_model):
                    # Let the completed turn release its dispatcher reservation.
                    await asyncio.sleep(0)
                    return event.content

        try:
            provider.steps = [[
                ("task_create", {"title": "Implement output", "approach": "Write a text file",
                                 "acceptance_criteria": "result.txt contains done"}),
                ("write_file", {"path": "forbidden.txt", "content": "must not happen"}),
            ], "Plan ready"]
            assert await send("/plan new Produce a verified output") == "Plan ready"
            plan = store.owned_plan("cli:one")
            task = store.list_tasks(plan["id"])[0]
            assert plan["pending_confirmation"]
            assert not (code / "forbidden.txt").exists()
            assert "write_file" not in provider.schemas[0]
            assert any("write_file" in text and "Error" in text for text in provider.tool_results)

            provider.steps = ["Waiting for /execute"]
            await send("开始吧")
            assert store.get_plan(plan["id"])["mode"] == "plan"

            provider.steps = [[
                ("task_progress", {"id": task["id"], "status": "in_progress"}),
                ("write_file", {"path": "result.txt", "content": "done"}),
                ("task_progress", {"id": task["id"], "status": "blocked",
                                   "progress": "Need an additional documentation task; waiting for replan"}),
                ("task_create", {"title": "Forbidden execution edit"}),
            ], "Need /plan to add documentation"]
            await send("/execute")
            assert (code / "result.txt").read_text() == "done"
            assert store.get_task(task["id"])["status"] == "blocked"
            assert len(store.list_tasks(plan["id"])) == 1
            assert "task_create" not in provider.schemas[-1]

            # A separate session can inspect but cannot take over the owned plan.
            listing = await send("/plans", chat="two", expect_model=False)
            assert plan["id"] in listing and "cli:one" in listing
            denied = await send(f"/plan {plan['id']}", chat="two", expect_model=False)
            assert "Error" in denied

            provider.steps = [[("task_create", {
                "title": "Document output", "acceptance_criteria": "docs.txt describes done",
                "depends_on": [task["id"]],
            })], "Revised plan ready"]
            await send(f"/plan {plan['id']}")
            assert store.get_plan(plan["id"])["pending_confirmation"]
            assert store.get_task(task["id"])["status"] == "blocked"
            doc = store.list_tasks(plan["id"])[1]

            provider.steps = [[
                ("task_progress", {"id": doc["id"], "status": "in_progress"}),
                ("task_progress", {"id": task["id"], "status": "in_progress"}),
                ("read_file", {"path": "result.txt"}),
            ], [
                ("task_progress", {"id": task["id"], "status": "completed", "progress": "Read result.txt: done"}),
                ("task_progress", {"id": doc["id"], "status": "in_progress"}),
                ("write_file", {"path": "docs.txt", "content": "The output is done"}),
                ("read_file", {"path": "docs.txt"}),
            ], [("task_progress", {"id": doc["id"], "status": "completed", "progress": "Read docs.txt: The output is done"})],
                "Both tasks checked and complete"]
            await send("/execute")
            reloaded = ProjectPlanStore(workspace, code)
            assert all(task["status"] == "completed" for task in reloaded.list_tasks(plan["id"]))
            assert reloaded.get_task(doc["id"])["progress"].startswith("Read docs.txt")
            assert any("dependency" in text and "Error" in text for text in provider.tool_results)
            assert reloaded.get_plan(plan["id"])["owner_session"] == "cli:one"

            await send("/exit-plan", expect_model=False)
            provider.steps = ["Existing completed work retained"]
            await send(f"/plan {plan['id']}", chat="two")
            assert store.get_plan(plan["id"])["owner_session"] == "cli:two"
            assert store.get_plan(plan["id"])["pending_confirmation"]
        finally:
            running.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await running

    asyncio.run(scenario())
