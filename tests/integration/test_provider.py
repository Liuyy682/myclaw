import asyncio
import json
import urllib.error
import urllib.request

import pytest

from myclaw import LLMResponse, ToolCallRequest
from myclaw.providers.openai_compat import OpenAICompatibleProvider


class FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class FakeHTTPStreamResponse:
    def __init__(self, lines):
        self.lines = [line.encode("utf-8") for line in lines]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        return iter(self.lines)


def _stream_line(payload):
    return f"data: {json.dumps(payload)}\n\n"


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


def test_openai_compatible_provider_includes_tools_when_supplied(monkeypatch):
    captured = {}
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two numbers",
                "parameters": {"type": "object"},
            },
        }
    ]

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeHTTPResponse({"choices": [{"message": {"content": "hello back"}}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(api_key="secret", model="demo-model")

    content = asyncio.run(provider.complete([{"role": "user", "content": "hello"}], tools=tools))

    assert content == "hello back"
    assert captured["body"]["tools"] == tools


def test_openai_compatible_provider_parses_tool_call_response(monkeypatch):
    def fake_urlopen(request, timeout):
        return FakeHTTPResponse(
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_add",
                                    "type": "function",
                                    "function": {
                                        "name": "add",
                                        "arguments": '{"a": 2, "b": 3}',
                                    },
                                }
                            ],
                        },
                    }
                ]
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(api_key="secret", model="demo-model")

    response = asyncio.run(provider.complete([{"role": "user", "content": "add"}], tools=[]))

    assert isinstance(response, LLMResponse)
    assert response.content == ""
    assert response.final is False
    assert response.stop_reason == "tool_calls"
    assert response.tool_calls == [
        ToolCallRequest(id="call_add", name="add", arguments={"a": 2, "b": 3}),
    ]


def test_openai_compatible_provider_rejects_invalid_tool_arguments(monkeypatch):
    def fake_urlopen(request, timeout):
        return FakeHTTPResponse(
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_bad",
                                    "type": "function",
                                    "function": {"name": "add", "arguments": "{bad json"},
                                }
                            ],
                        },
                    }
                ]
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(api_key="secret", model="demo-model")

    with pytest.raises(RuntimeError, match="tool call arguments were not valid JSON"):
        asyncio.run(provider.complete([{"role": "user", "content": "add"}], tools=[]))


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


def test_openai_compatible_provider_streams_text_deltas(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeHTTPStreamResponse(
            [
                _stream_line({"choices": [{"delta": {"content": "hello"}}]}),
                _stream_line({"choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}]}),
                "data: [DONE]\n\n",
            ]
        )

    async def record_delta(delta):
        deltas.append(delta)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(api_key="secret", model="demo-model")
    deltas = []

    content = asyncio.run(
        provider.stream_complete([{"role": "user", "content": "hello"}], delta_callback=record_delta)
    )

    assert content == "hello world"
    assert deltas == ["hello", " world"]
    assert captured["body"]["stream"] is True
    assert captured["body"]["stream_options"] == {"include_usage": True}


def test_openai_compatible_provider_parses_usage_from_regular_and_stream_responses(monkeypatch):
    responses = iter(
        [
            FakeHTTPResponse(
                {
                    "choices": [{"message": {"content": "hello"}}],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
                }
            ),
            FakeHTTPStreamResponse(
                [
                    _stream_line({"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}),
                    _stream_line({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}}),
                    "data: [DONE]\n\n",
                ]
            ),
        ]
    )

    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: next(responses))
    provider = OpenAICompatibleProvider(api_key="secret", model="demo-model")

    regular = asyncio.run(provider.complete([{"role": "user", "content": "hello"}]))
    streamed = asyncio.run(provider.stream_complete([{"role": "user", "content": "hi"}]))

    assert isinstance(regular, LLMResponse)
    assert regular.usage is not None and regular.usage.total_tokens == 6
    assert isinstance(streamed, LLMResponse)
    assert streamed.content == "hi"
    assert streamed.usage is not None and streamed.usage.prompt_tokens == 5


def test_openai_compatible_provider_streams_tool_call_deltas(monkeypatch):
    def fake_urlopen(request, timeout):
        return FakeHTTPStreamResponse(
            [
                _stream_line(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_add",
                                            "type": "function",
                                            "function": {"name": "add", "arguments": '{"a": '},
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                ),
                _stream_line(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {"arguments": '2, "b": 3}'},
                                        }
                                    ]
                                },
                                "finish_reason": "tool_calls",
                            }
                        ]
                    }
                ),
                "data: [DONE]\n\n",
            ]
        )

    async def record_delta(delta):
        deltas.append(delta)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(api_key="secret", model="demo-model")
    deltas = []

    response = asyncio.run(
        provider.stream_complete([{"role": "user", "content": "add"}], tools=[], delta_callback=record_delta)
    )

    assert deltas == []
    assert isinstance(response, LLMResponse)
    assert response.final is False
    assert response.stop_reason == "tool_calls"
    assert response.tool_calls == [
        ToolCallRequest(id="call_add", name="add", arguments={"a": 2, "b": 3}),
    ]
