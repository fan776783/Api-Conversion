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
from src.core.gateway_config import GatewayConfig
from src.formats.converter_factory import (
    OPENAI_CHAT_COMPLETIONS_FORMAT,
    OPENAI_RESPONSES_FORMAT,
)


def test_conversion_formats_exposes_openai_protocol_options(monkeypatch):
    app = FastAPI()
    app.include_router(conversion_api.router)

    monkeypatch.setattr(
        conversion_api,
        "get_session_user",
        lambda request=None: True,
    )

    client = TestClient(app)
    response = client.get("/conversion/formats")

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert OPENAI_CHAT_COMPLETIONS_FORMAT in body["formats"]
    assert OPENAI_RESPONSES_FORMAT in body["formats"]
    assert body["openai_protocol_options"] == [
        {
            "value": OPENAI_CHAT_COMPLETIONS_FORMAT,
            "label": "OpenAI Chat Completions",
        },
        {
            "value": OPENAI_RESPONSES_FORMAT,
            "label": "OpenAI Responses",
        },
    ]


def test_gateway_config_round_trip_preserves_openai_protocol_fields():
    config = GatewayConfig(
        provider="openai",
        base_url="https://api.openai.com/v1",
        timeout=45,
        max_retries=2,
        model_mapping={"claude-3-5-sonnet": "gpt-4.1"},
        default_target_format=OPENAI_RESPONSES_FORMAT,
        supported_formats=[OPENAI_CHAT_COMPLETIONS_FORMAT, OPENAI_RESPONSES_FORMAT],
        enabled=False,
    )

    payload = config.to_dict()
    restored = GatewayConfig.from_dict(payload)

    assert payload["default_target_format"] == OPENAI_RESPONSES_FORMAT
    assert payload["supported_formats"] == [
        OPENAI_CHAT_COMPLETIONS_FORMAT,
        OPENAI_RESPONSES_FORMAT,
    ]
    assert restored.default_target_format == OPENAI_RESPONSES_FORMAT
    assert restored.supported_formats == [
        OPENAI_CHAT_COMPLETIONS_FORMAT,
        OPENAI_RESPONSES_FORMAT,
    ]
    assert restored.model_mapping == {"claude-3-5-sonnet": "gpt-4.1"}
    assert restored.enabled is False


def test_gateway_config_from_dict_defaults_supported_formats_to_empty_list():
    restored = GatewayConfig.from_dict(
        {
            "provider": "openai",
            "base_url": "https://api.openai.com/v1",
            "timeout": 30,
            "max_retries": 1,
            "model_mapping": {},
            "default_target_format": None,
            "enabled": True,
        }
    )

    assert restored.default_target_format is None
    assert restored.supported_formats == []
