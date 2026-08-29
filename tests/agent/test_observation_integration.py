import asyncio
import json

from myclaw.agent import AgentConfig, AgentLoop
from myclaw.agent.observation import ObservationMemoryWorker
from myclaw.memory import ObservationMemoryStore
from myclaw.session import SessionManager
from myclaw.tools import ToolRegistry


class ExtractionProvider:
    model = "extractor"

    async def complete(self, messages, **kwargs):
        system = messages[0]["content"]
        payload = json.loads(messages[1]["content"].split("\n", 1)[1])
        if system.startswith("Extract durable observations"):
            event_id = payload["events"][0]["id"]
            return json.dumps(
                {
                    "observations": [
                        {
                            "content": "Keep the API stable.",
                            "kind": "constraint",
                            "importance": 0.9,
                            "confidence": 0.95,
                            "source_event_ids": [event_id],
                        }
                    ]
                }
            )
        observation_id = payload["active_unreflected_observations"][0]["id"]
        return json.dumps(
            {
                "reflections": [
                    {
                        "content": "The session requires API compatibility.",
                        "kind": "constraint",
                        "supports": [observation_id],
                        "supersedes": [],
                    }
                ]
            }
        )


def test_real_store_and_worker_complete_observation_reflection_and_recall(tmp_path):
    store = ObservationMemoryStore(tmp_path)
    store.enqueue_turn(
        "cli:one",
        "turn-1",
        [{"event_id": "event-1", "role": "user", "content": "Do not break the API."}],
    )
    worker = ObservationMemoryWorker(ExtractionProvider(), store, batch_size=1)

    result = asyncio.run(worker.process_once())

    assert result.outcome == "observed"
    assert result.reflection_count == 1
    reflection = store.active_reflections("cli:one")[0]
    recalled = store.recall(reflection["id"], session_id="cli:one")
    assert recalled["status"] == "found"
    assert recalled["source_entries"][0]["id"] == "event-1"
    assert store.turns_are_safe("cli:one", {"turn-1"}) is True


def test_worker_resumes_due_reflection_when_no_observation_job_remains(tmp_path):
    store = ObservationMemoryStore(tmp_path)
    store.enqueue_turn(
        "cli:one",
        "turn-1",
        [{"event_id": "event-1", "role": "user", "content": "Keep compatibility."}],
    )
    job = store.claim_observation_job()
    store.commit_observation_result(
        job["id"],
        [
            {
                "content": "Keep compatibility.",
                "kind": "constraint",
                "importance": 0.9,
                "confidence": 0.95,
                "source_event_ids": ["event-1"],
            }
        ],
    )
    worker = ObservationMemoryWorker(ExtractionProvider(), store, batch_size=1)

    result = asyncio.run(worker.process_once())

    assert result.outcome == "reflected"
    assert result.reflection_count == 1


class ConversationProvider:
    model = "conversation"

    def __init__(self):
        self.calls = []

    async def complete(self, messages, **kwargs):
        self.calls.append([dict(message) for message in messages])
        return "done"


def test_agent_loop_durably_enqueues_completed_turn_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "myclaw.agent.context.TokenEstimator._load_encoding", lambda *args: None
    )
    manager = SessionManager(tmp_path)
    registry = ToolRegistry()
    loop = AgentLoop(
        ConversationProvider(),
        AgentConfig(
            system_prompt="",
            auto_title=False,
            observation_memory_enabled=True,
        ),
        session_manager=manager,
        tool_registry=registry,
    )

    asyncio.run(loop.run("hello", session_key="cli:one"))

    status = loop.observation_store.status("cli:one")
    assert status["source_events"] == 2
    assert status["jobs"]["pending"] == 1
    assert loop.observation_worker._wake_event.is_set()
    assert registry.get("recall") is not None
    messages = manager.get_or_create("cli:one").messages
    assert len({message["turn_id"] for message in messages}) == 1
    assert all(message.get("event_id") for message in messages)


def test_agent_loop_leaves_observation_memory_absent_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "myclaw.agent.context.TokenEstimator._load_encoding", lambda *args: None
    )
    loop = AgentLoop(
        ConversationProvider(),
        AgentConfig(system_prompt="", auto_title=False),
        session_manager=SessionManager(tmp_path),
    )

    asyncio.run(loop.run("hello", session_key="cli:one"))

    assert loop.observation_store is None
    assert not (tmp_path / "memory" / "observation_memory.db").exists()


def test_failed_observation_enqueue_does_not_wake_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "myclaw.agent.context.TokenEstimator._load_encoding", lambda *args: None
    )
    loop = AgentLoop(
        ConversationProvider(),
        AgentConfig(system_prompt="", auto_title=False, observation_memory_enabled=True),
        session_manager=SessionManager(tmp_path),
    )
    wake_calls = []
    monkeypatch.setattr(loop.observation_worker, "wake", lambda: wake_calls.append(True))

    def fail_enqueue(*args, **kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(loop.observation_store, "enqueue_turn", fail_enqueue)

    asyncio.run(loop.run("hello", session_key="cli:one"))

    assert wake_calls == []


def test_safe_observation_watermark_uses_fast_compaction_without_summary_call(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "myclaw.agent.context.TokenEstimator._load_encoding", lambda *args: None
    )
    manager = SessionManager(tmp_path)
    provider = ConversationProvider()
    loop = AgentLoop(
        provider,
        AgentConfig(
            system_prompt="",
            auto_title=False,
            max_context_messages=3,
            max_context_tokens=100_000,
            observation_memory_enabled=True,
        ),
        session_manager=manager,
    )
    session = manager.get_or_create("cli:one")
    for number in (1, 2):
        turn_id = f"turn-{number}"
        events = []
        for role in ("user", "assistant"):
            event_id = f"event-{number}-{role}"
            session.add_message(
                role, f"{role}-{number}", turn_id=turn_id, event_id=event_id
            )
            events.append(
                {
                    "event_id": event_id,
                    "turn_id": turn_id,
                    "role": role,
                    "content": f"{role}-{number}",
                }
            )
        loop.observation_store.enqueue_turn("cli:one", turn_id, events)
        job = loop.observation_store.claim_observation_job()
        loop.observation_store.commit_observation_result(job["id"], [], no_facts=True)
    manager.save(session)

    asyncio.run(loop.run("next", session_key="cli:one"))

    assert len(provider.calls) == 1
    assert not any(
        "Summarize older conversation turns" in str(message.get("content", ""))
        for message in provider.calls[0]
    )
    summary = manager.get_or_create("cli:one").metadata["context_summary"]
    assert "no durable facts" in summary["content"]


def test_legacy_messages_fall_back_to_existing_model_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "myclaw.agent.context.TokenEstimator._load_encoding", lambda *args: None
    )
    manager = SessionManager(tmp_path)
    provider = ConversationProvider()
    loop = AgentLoop(
        provider,
        AgentConfig(
            system_prompt="",
            auto_title=False,
            max_context_messages=3,
            max_context_tokens=100_000,
            observation_memory_enabled=True,
        ),
        session_manager=manager,
    )
    session = manager.get_or_create("cli:one")
    for number in (1, 2):
        session.add_message("user", f"legacy-user-{number}")
        session.add_message("assistant", f"legacy-assistant-{number}")
    manager.save(session)

    asyncio.run(loop.run("next", session_key="cli:one"))

    assert any(
        "Summarize older conversation turns" in str(message.get("content", ""))
        for call in provider.calls
        for message in call
    )
