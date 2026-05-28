import asyncio
import json
import urllib.error
import urllib.request

import pytest

from myclaw.provider import OpenAICompatibleProvider


class FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_openai_compatible_provider_builds_chat_completion_request(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeHTTPResponse({"choices": [{"message": {"content": "hello back"}}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(
        api_key="secret",
        base_url="https://example.test/v1/",
        model="demo-model",
    )

    content = asyncio.run(provider.complete([{"role": "user", "content": "hello"}]))

    assert content == "hello back"
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["timeout"] == 120
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["headers"]["Content-type"] == "application/json"
    assert captured["body"] == {
        "model": "demo-model",
        "messages": [{"role": "user", "content": "hello"}],
    }


def test_openai_compatible_provider_raises_readable_http_errors(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            url=request.full_url,
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(api_key="bad", model="demo-model")

    with pytest.raises(RuntimeError, match="LLM request failed: HTTP 401 Unauthorized"):
        asyncio.run(provider.complete([{"role": "user", "content": "hello"}]))
