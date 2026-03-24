import json
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

from src.api import conversion_api


class _FakeChannel:
    def __init__(self):
        self.name = "test-openai-channel"
        self.provider = "openai"
        self.base_url = "https://example.test/v1"
        self.api_key = "upstream-secret"
        self.custom_key = "test-key"
        self.timeout = 30
        self.max_retries = 3
        self.enabled = True
        self.models_mapping = {}


def _build_conversion_client(monkeypatch, captured):
    app = FastAPI()
    app.include_router(conversion_api.router)

    monkeypatch.setattr(conversion_api, "get_session_user", lambda request=None: True)
    monkeypatch.setattr(
        conversion_api.channel_manager,
        "get_channels_by_provider",
        lambda provider: [_FakeChannel()] if provider == "openai" else [],
    )

    async def _fake_forward_request(channel, converted_data, headers, target_format, method="POST"):
        captured.append(
            {
                "channel": channel,
                "request_data": converted_data,
                "headers": headers,
                "target_format": target_format,
                "method": method,
            }
        )
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

    monkeypatch.setattr(conversion_api, "forward_request", _fake_forward_request)
    return TestClient(app)


def test_conversion_chat_passthrough_rehydrates_edit_tool(monkeypatch):
    captured = []
    client = _build_conversion_client(monkeypatch, captured)

    payload = {
        "model": "gpt-4.1",
        "metadata": {"discovered_tools": ["Edit"]},
        "x-tool-schemas": {
            "Edit": {
                "type": "function",
                "function": {
                    "name": "Edit",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "old_string": {"type": "string"},
                            "new_string": {"type": "string"},
                        },
                        "required": ["old_string", "new_string"],
                    },
                },
            }
        },
        "messages": [
            {"role": "user", "content": "继续编辑"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_edit_9",
                        "type": "function",
                        "function": {
                            "name": "Edit",
                            "arguments": json.dumps({"old_string": "before", "new_string": "after"}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_edit_9", "content": "ok"},
        ],
    }

    response = client.post("/openai/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert response.json()["object"] == "chat.completion"
    forwarded = captured[0]
    assert forwarded["target_format"] == "openai_chat_completions"
    assert forwarded["request_data"]["tools"][0]["function"]["name"] == "Edit"
    assert forwarded["request_data"]["metadata"]["discovered_tools"] == ["Edit"]
    assert "Edit" in forwarded["request_data"]["x-tool-policy"]["rehydrated_tools"]
