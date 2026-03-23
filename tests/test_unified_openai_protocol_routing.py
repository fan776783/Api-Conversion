import json
import pathlib
import sys
from dataclasses import asdict

from fastapi import FastAPI
from fastapi.testclient import TestClient

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from src.api import unified_api
from src.channels.channel_manager import ChannelInfo
from src.formats.converter_factory import (
    OPENAI_CHAT_COMPLETIONS_FORMAT,
    OPENAI_RESPONSES_FORMAT,
)


class _FakeStreamingResult:
    def __init__(self, body: str, status_code: int = 200):
        self.status_code = status_code
        self.body = body

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        yield self.body



def _build_test_client(monkeypatch, captured):
    app = FastAPI()
    app.include_router(unified_api.router)

    fake_channel = ChannelInfo(
        id="ch_test",
        name="test-openai-channel",
        provider="openai",
        base_url="https://example.test/v1",
        api_key="upstream-secret",
        custom_key="test-key",
        timeout=30,
        max_retries=3,
        enabled=True,
    )

    monkeypatch.setattr(
        unified_api.channel_manager,
        "get_channel_by_custom_key",
        lambda custom_key: fake_channel if custom_key == "test-key" else None,
    )

    async def _fake_forward_request_to_channel(
        channel,
        request_data,
        source_format,
        headers=None,
        request_started_at=None,
        request_id=None,
        request_method=None,
        request_path=None,
    ):
        captured.append(
            {
                "channel": asdict(channel),
                "request_data": request_data,
                "source_format": source_format,
                "request_method": request_method,
                "request_path": request_path,
            }
        )
        if request_data.get("stream"):
            if source_format == OPENAI_RESPONSES_FORMAT:
                return _FakeStreamingResult(
                    "event: response.created\ndata: {\"type\":\"response.created\"}\n\n"
                    "event: response.completed\ndata: {\"type\":\"response.completed\"}\n\n"
                )
            return _FakeStreamingResult(
                "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
                "data: [DONE]\n\n"
            )

        if source_format == OPENAI_RESPONSES_FORMAT:
            return {
                "id": "resp_test",
                "object": "response",
                "status": "completed",
                "output": [],
                "output_text": "ok",
            }

        return {
            "id": "chatcmpl_test",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }

    monkeypatch.setattr(unified_api, "forward_request_to_channel", _fake_forward_request_to_channel)
    return TestClient(app)



def test_chat_endpoint_normalizes_misdirected_responses_payload(monkeypatch):
    captured = []
    client = _build_test_client(monkeypatch, captured)

    payload = {
        "model": "gpt-4.1",
        "instructions": "你是代码助手",
        "input": "请修复这个 bug",
        "stream": False,
        "max_output_tokens": 128,
    }

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json=payload,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"

    assert len(captured) == 1
    forwarded = captured[0]
    assert forwarded["source_format"] == OPENAI_CHAT_COMPLETIONS_FORMAT
    assert forwarded["request_path"] == "/v1/chat/completions"
    assert forwarded["request_data"]["stream"] is False
    assert forwarded["request_data"]["max_tokens"] == 128
    assert forwarded["request_data"]["messages"][0] == {"role": "system", "content": "你是代码助手"}
    assert forwarded["request_data"]["messages"][1] == {"role": "user", "content": "请修复这个 bug"}



def test_responses_endpoint_keeps_responses_protocol_for_non_streaming(monkeypatch):
    captured = []
    client = _build_test_client(monkeypatch, captured)

    payload = {
        "model": "gpt-4.1",
        "input": "hello",
        "stream": False,
    }

    response = client.post(
        "/v1/responses",
        headers={"Authorization": "Bearer test-key"},
        json=payload,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"

    assert len(captured) == 1
    forwarded = captured[0]
    assert forwarded["source_format"] == OPENAI_RESPONSES_FORMAT
    assert forwarded["request_path"] == "/v1/responses"
    assert forwarded["request_data"]["input"] == "hello"



def test_responses_endpoint_streaming_returns_response_events(monkeypatch):
    captured = []
    client = _build_test_client(monkeypatch, captured)

    payload = {
        "model": "gpt-4.1",
        "input": "stream please",
        "stream": True,
    }

    with client.stream(
        "POST",
        "/v1/responses",
        headers={"Authorization": "Bearer test-key"},
        json=payload,
    ) as response:
        assert response.status_code == 200
        body = response.read().decode()

    assert "event: response.created" in body
    assert "event: response.completed" in body
    assert "chat.completion.chunk" not in body
    assert "data: [DONE]" not in body

    assert len(captured) == 1
    forwarded = captured[0]
    assert forwarded["source_format"] == OPENAI_RESPONSES_FORMAT
    assert forwarded["request_data"]["stream"] is True



def test_chat_endpoint_streaming_returns_chat_done_marker(monkeypatch):
    captured = []
    client = _build_test_client(monkeypatch, captured)

    payload = {
        "model": "gpt-4.1",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
    }

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json=payload,
    ) as response:
        assert response.status_code == 200
        body = response.read().decode()

    assert "chat.completion.chunk" in body
    assert "data: [DONE]" in body

    assert len(captured) == 1
    forwarded = captured[0]
    assert forwarded["source_format"] == OPENAI_CHAT_COMPLETIONS_FORMAT
    assert forwarded["request_data"]["stream"] is True
