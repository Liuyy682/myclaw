import asyncio
import json

from myclaw.memory import ObservationMemoryStore
from myclaw.tools import RecallTool, ToolRuntimeContext, tool_context


def _seed(store):
    store.enqueue_turn(
        "cli:one",
        "turn-1",
        [{"event_id": "event-1", "role": "user", "content": "Keep compatibility."}],
    )
    observation = store.write_observation(
        "cli:one", "Keep compatibility.", ["event-1"], kind="constraint"
    )
    reflection = store.record_reflection(
        "cli:one", "Compatibility is required.", [observation["id"]], kind="constraint"
    )
    return observation, reflection


def test_recall_tool_enforces_session_visibility_and_returns_original_evidence(tmp_path):
    store = ObservationMemoryStore(tmp_path)
    observation, reflection = _seed(store)
    tool = RecallTool(store)

    with tool_context(ToolRuntimeContext(session_key="cli:one")):
        visible = asyncio.run(tool.execute(id=observation["id"]))
    with tool_context(ToolRuntimeContext(session_key="cli:two")):
        hidden = asyncio.run(tool.execute(id=observation["id"]))

    assert json.loads(visible)["source_entries"][0]["content"] == "Keep compatibility."
    assert hidden.startswith("Error: memory not found")

    store.promote_reflection(reflection["id"])
    with tool_context(ToolRuntimeContext(session_key="cli:two")):
        promoted = asyncio.run(tool.execute(id=reflection["id"]))
    assert json.loads(promoted)["content"] == "Compatibility is required."
