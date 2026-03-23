import pathlib
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from src.api import unified_api
from src.core.gateway_config import GatewayConfig
from src.formats.converter_factory import (
    OPENAI_CHAT_COMPLETIONS_FORMAT,
    OPENAI_RESPONSES_FORMAT,
)


class _FakeGatewayStreamingResponse:
    def __init__(self, chunks):
        self.status_code = 200
        self.headers = {}
        self._chunks = chunks

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk.encode("utf-8") if isinstance(chunk, str) else chunk

    async def aiter_lines(self):
        for chunk in self._chunks:
            text = chunk.decode() if isinstance(chunk, (bytes, bytearray)) else chunk
            for line in text.splitlines():
                yield line



def _build_gateway_client(monkeypatch, captured):
    app = FastAPI()
    app.include_router(unified_api.router)

    monkeypatch.setattr(
        unified_api,
        "load_gateway_config",
        lambda: GatewayConfig(
            provider="openai",
            base_url="https://example-gateway.test/v1",
            timeout=30,
            max_retries=1,
            model_mapping={},
            enabled=True,
        ),
    )

    async def _fake_handle_streaming_response(
        response,
        channel,
        request_data,
        source_format,
        request_started_at=None,
        upstream_response_started_at=None,
        request_id=None,
        request_method=None,
        request_path=None,
        upstream_url=None,
        client_model=None,
        upstream_model=None,
        input_chars=None,
    ):
        captured.append(
            {
                "kind": "stream",
                "source_format": source_format,
                "request_data": request_data,
                "channel_provider": channel.provider,
            }
        )
        if source_format == OPENAI_RESPONSES_FORMAT:
            yield 'event: response.created\ndata: {"type":"response.created"}\n\n'
            yield 'event: response.completed\ndata: {"type":"response.completed"}\n\n'
        else:
            yield 'data: {"object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            yield 'data: [DONE]\n\n'

    monkeypatch.setattr(unified_api, "handle_streaming_response", _fake_handle_streaming_response)

    class _FakeAsyncClient:
        def __init__(self, timeout=None):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, json=None, headers=None):
            captured.append(
                {
                    "kind": "upstream_stream_request",
                    "method": method,
                    "url": url,
                    "json": json,
                    "headers": headers,
                }
            )
            chunks = ['data: {"dummy":true}\n\n']
            return _FakeStreamContext(_FakeGatewayStreamingResponse(chunks))

        async def request(self, method, url, json=None, headers=None):
            captured.append(
                {
                    "kind": "upstream_request",
                    "method": method,
                    "url": url,
                    "json": json,
                    "headers": headers,
                }
            )
            if url.endswith("/v1/responses"):
                return _FakeJSONResponse(
                    {
                        "id": "resp_gateway",
                        "object": "response",
                        "status": "completed",
                        "output": [],
                        "output_text": "gateway ok",
                    }
                )
            return _FakeJSONResponse(
                {
                    "id": "chatcmpl_gateway",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "gateway ok"},
                            "finish_reason": "stop",
                        }
                    ],
                }
            )

    monkeypatch.setattr(unified_api.httpx, "AsyncClient", _FakeAsyncClient)
    return TestClient(app)


class _FakeStreamContext:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeJSONResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.headers = {}
        self.content = b"{}"
        self.text = "{}"

    def json(self):
        return self._payload



def test_gateway_chat_path_normalizes_misdirected_responses_payload(monkeypatch):
    captured = []
    client = _build_gateway_client(monkeypatch, captured)

    payload = {
        "model": "gpt-4.1",
        "instructions": "你是代码助手",
        "input": "修复这个问题",
        "stream": False,
        "max_output_tokens": 128,
    }

    response = client.post(
        "/gateway/v1/chat/completions",
        headers={"Authorization": "Bearer user-key"},
        json=payload,
    )

    assert response.status_code == 200
    assert response.json()["object"] == "chat.completion"

    upstream = next(item for item in captured if item["kind"] == "upstream_request")
    assert upstream["url"].endswith("/v1/chat/completions")
    assert upstream["json"]["messages"][0] == {"role": "system", "content": "你是代码助手"}
    assert upstream["json"]["messages"][1] == {"role": "user", "content": "修复这个问题"}
    assert upstream["json"]["max_tokens"] == 128



def test_gateway_responses_path_routes_to_openai_responses_endpoint(monkeypatch):
    captured = []
    client = _build_gateway_client(monkeypatch, captured)

    payload = {
        "model": "gpt-4.1",
        "input": "hello gateway",
        "stream": False,
    }

    response = client.post(
        "/gateway/v1/responses",
        headers={"Authorization": "Bearer user-key"},
        json=payload,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"

    upstream = next(item for item in captured if item["kind"] == "upstream_request")
    assert upstream["url"].endswith("/v1/responses")
    assert upstream["json"]["input"] == "hello gateway"



def test_gateway_responses_streaming_returns_response_events(monkeypatch):
    captured = []
    client = _build_gateway_client(monkeypatch, captured)

    payload = {
        "model": "gpt-4.1",
        "input": "stream gateway responses",
        "stream": True,
    }

    with client.stream(
        "POST",
        "/gateway/v1/responses",
        headers={"Authorization": "Bearer user-key"},
        json=payload,
    ) as response:
        assert response.status_code == 200
        body = response.read().decode()

    assert "event: response.created" in body
    assert "event: response.completed" in body
    assert "data: [DONE]" not in body

    upstream = next(item for item in captured if item["kind"] == "upstream_stream_request")
    assert upstream["url"].endswith("/v1/responses")
    stream_call = next(item for item in captured if item["kind"] == "stream")
    assert stream_call["source_format"] == OPENAI_RESPONSES_FORMAT
    assert stream_call["request_data"]["stream"] is True



def test_gateway_chat_streaming_returns_chat_done_marker(monkeypatch):
    captured = []
    client = _build_gateway_client(monkeypatch, captured)

    payload = {
        "model": "gpt-4.1",
        "messages": [{"role": "user", "content": "hello gateway"}],
        "stream": True,
    }

    with client.stream(
        "POST",
        "/gateway/v1/chat/completions",
        headers={"Authorization": "Bearer user-key"},
        json=payload,
    ) as response:
        assert response.status_code == 200
        body = response.read().decode()

    assert "chat.completion.chunk" in body
    assert "data: [DONE]" in body

    upstream = next(item for item in captured if item["kind"] == "upstream_stream_request")
    assert upstream["url"].endswith("/v1/chat/completions")
    stream_call = next(item for item in captured if item["kind"] == "stream")
    assert stream_call["source_format"] == OPENAI_CHAT_COMPLETIONS_FORMAT
    assert stream_call["request_data"]["stream"] is True
