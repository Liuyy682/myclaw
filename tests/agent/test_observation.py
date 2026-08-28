import asyncio
import json

from myclaw.agent.observation import ObservationMemoryWorker


def _observation(index: int, *, source="event-1"):
    return {
        "id": f"obs-{index}",
        "content": f"fact {index}",
        "kind": "decision",
        "importance": 0.8,
        "confidence": 0.9,
        "source_event_ids": [source],
    }


class FakeProvider:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    async def complete(self, messages):
        self.calls.append(messages)
        return self.responses.pop(0)


class FakeStore:
    def __init__(self, jobs=(), active=()):
        self.jobs = list(jobs)
        self.active = {"s1": list(active)}
        self.observation_commits = []
        self.reflection_commits = []
        self.failures = []

    def claim_observation_job(self):
        return self.jobs.pop(0) if self.jobs else None

    def commit_observation_result(self, job_id, observations, *, no_facts=False):
        self.observation_commits.append((job_id, observations, no_facts))
        if observations:
            self.active.setdefault("s1", []).extend(observations)

    def fail_observation_job(self, job_id, error, *, retryable=True):
        self.failures.append(("observation", job_id, error, retryable))

    def list_unreflected_observations(self, session_id):
        return self.active.get(session_id, [])

    def commit_reflection_result(self, session_id, reflections):
        self.reflection_commits.append((session_id, reflections))
        supported = {support for item in reflections for support in item["supports"]}
        self.active[session_id] = [item for item in self.active.get(session_id, []) if item["id"] not in supported]

    def fail_reflection(self, session_id, error, *, retryable=True):
        self.failures.append(("reflection", session_id, error, retryable))


def run(coro):
    return asyncio.run(coro)


def test_observation_call_is_no_tool_and_commits_valid_json():
    provider = FakeProvider(json.dumps({"observations": [_observation(1)]}))
    store = FakeStore([{"id": "job-1", "session_id": "s1", "event_ids": ["event-1"]}])
    worker = ObservationMemoryWorker(provider, store)

    result = run(worker.process_once())

    assert result.outcome == "observed"
    assert len(provider.calls) == 1
    assert provider.calls[0][0]["role"] == "system"
    assert store.observation_commits[0][0] == "job-1"
    assert store.observation_commits[0][1][0]["source_event_ids"] == ["event-1"]


def test_empty_observation_array_is_committed_as_no_facts():
    provider = FakeProvider('{"observations": []}')
    store = FakeStore([{"id": "job-1", "session_id": "s1", "event_ids": ["event-1"]}])
    worker = ObservationMemoryWorker(provider, store)

    result = run(worker.process_once())

    assert result.outcome == "no_facts"
    assert store.observation_commits == [("job-1", [], True)]
    assert store.failures == []


def test_invalid_observation_is_failed_and_not_partially_committed():
    invalid = _observation(1, source="not-in-job")
    provider = FakeProvider(json.dumps({"observations": [invalid]}))
    store = FakeStore([{"id": "job-1", "session_id": "s1", "event_ids": ["event-1"]}])
    worker = ObservationMemoryWorker(provider, store)

    result = run(worker.process_once())

    assert result.outcome == "failed"
    assert store.observation_commits == []
    assert store.failures[0][0:2] == ("observation", "job-1")
    assert store.failures[0][3] is True


def test_five_active_observations_trigger_one_reflection_with_existing_supports():
    active = [_observation(index) for index in range(5)]
    supported = [f"obs-{i}" for i in range(4)] + ["obs-99"]
    reflection = {
        "id": "ref-1",
        "content": "The assistant follows the selected decision.",
        "kind": "decision",
        "supports": supported,
        "supersedes": [],
    }
    provider = FakeProvider(
        json.dumps({"observations": [_observation(99)]}),
        json.dumps({"reflections": [reflection]}),
    )
    # The sixth observation is appended by the observation commit; five are
    # enough to cross the threshold and the worker reflects the oldest five.
    store = FakeStore(
        [{"id": "job-1", "session_id": "s1", "event_ids": ["event-1"]}],
        active=active[:4],
    )
    worker = ObservationMemoryWorker(provider, store)

    result = run(worker.process_once())

    assert result.reflection_count == 1
    assert len(provider.calls) == 2
    assert len(store.reflection_commits) == 1
    assert store.reflection_commits[0][1][0]["supports"] == supported


def test_unknown_reflection_support_is_failed_without_commit():
    active = [_observation(index) for index in range(5)]
    bad_reflection = {
        "content": "unsupported",
        "kind": "decision",
        "supports": ["does-not-exist"],
        "supersedes": [],
    }
    provider = FakeProvider(
        json.dumps({"observations": []}),
    )
    store = FakeStore(
        [{"id": "job-1", "session_id": "s1", "event_ids": ["event-1"]}],
        active=active,
    )
    # Empty observation output does not trigger reflection in the same step;
    # this test exercises the parser's strict support check directly below.
    worker = ObservationMemoryWorker(provider, store)
    result = run(worker.process_once())
    assert result.outcome == "no_facts"

    from myclaw.agent.observation import ReflectionOutputError, parse_reflections

    try:
        parse_reflections(json.dumps({"reflections": [bad_reflection]}), [item["id"] for item in active])
    except ReflectionOutputError:
        pass
    else:
        raise AssertionError("unknown support id must be rejected")


def test_provider_failure_is_contained_and_retryable():
    class BrokenProvider:
        async def complete(self, messages):
            raise RuntimeError("temporary outage")

    store = FakeStore([{"id": "job-1", "session_id": "s1", "event_ids": ["event-1"]}])
    result = run(ObservationMemoryWorker(BrokenProvider(), store).process_once())

    assert result.outcome == "failed"
    assert store.observation_commits == []
    assert store.failures[0][0:2] == ("observation", "job-1")
