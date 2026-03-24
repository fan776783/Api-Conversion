import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from src.core.protocol_resolution import (
    OPENAI_PROVIDER,
    detect_client_format,
    normalize_request_for_client_format,
    provider_supports_format,
    resolve_openai_target_format,
    resolve_protocol_context,
    same_protocol,
)
from src.formats.converter_factory import (
    OPENAI_CHAT_COMPLETIONS_FORMAT,
    OPENAI_RESPONSES_FORMAT,
)


def test_detect_client_format_prefers_explicit_chat_path_over_responses_shape():
    payload = {
        "model": "gpt-4.1",
        "input": "hello",
        "instructions": "be helpful",
        "stream": True,
    }

    detected = detect_client_format(
        request_data=payload,
        request_path="/v1/chat/completions",
        explicit_client_format=OPENAI_CHAT_COMPLETIONS_FORMAT,
    )

    assert detected == OPENAI_CHAT_COMPLETIONS_FORMAT


def test_normalize_request_for_chat_adapts_responses_payload_but_keeps_stream():
    payload = {
        "model": "gpt-4.1",
        "instructions": "系统提示",
        "input": "你好",
        "stream": True,
        "max_output_tokens": 128,
    }

    normalized = normalize_request_for_client_format(OPENAI_CHAT_COMPLETIONS_FORMAT, payload)

    assert normalized["stream"] is True
    assert normalized["max_tokens"] == 128
    assert normalized["messages"][0] == {"role": "system", "content": "系统提示"}
    assert normalized["messages"][1] == {"role": "user", "content": "你好"}


def test_normalize_request_for_chat_applies_tool_policy_rehydration():
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
                        "id": "call_edit_1",
                        "type": "function",
                        "function": {
                            "name": "Edit",
                            "arguments": "{\"old_string\":\"before\",\"new_string\":\"after\"}",
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_edit_1", "content": "ok"},
        ],
    }

    normalized = normalize_request_for_client_format(OPENAI_CHAT_COMPLETIONS_FORMAT, payload)

    assert normalized["tools"][0]["function"]["name"] == "Edit"
    assert normalized["metadata"]["discovered_tools"] == ["Edit"]
    assert normalized["x-tool-policy"]["rehydrated_tools"] == ["Edit"]


def test_resolve_openai_target_format_uses_supported_formats_and_default():
    target = resolve_openai_target_format(
        client_format=OPENAI_RESPONSES_FORMAT,
        default_target_format=OPENAI_CHAT_COMPLETIONS_FORMAT,
        supported_formats=[OPENAI_CHAT_COMPLETIONS_FORMAT, OPENAI_RESPONSES_FORMAT],
    )
    assert target == OPENAI_RESPONSES_FORMAT

    fallback = resolve_openai_target_format(
        client_format="anthropic",
        default_target_format=OPENAI_RESPONSES_FORMAT,
        supported_formats=[OPENAI_RESPONSES_FORMAT],
    )
    assert fallback == OPENAI_RESPONSES_FORMAT


def test_provider_supports_format_respects_openai_declared_capabilities():
    assert provider_supports_format(
        OPENAI_PROVIDER,
        OPENAI_RESPONSES_FORMAT,
        default_target_format=OPENAI_CHAT_COMPLETIONS_FORMAT,
        supported_formats=[OPENAI_CHAT_COMPLETIONS_FORMAT, OPENAI_RESPONSES_FORMAT],
    ) is True

    assert provider_supports_format(
        OPENAI_PROVIDER,
        OPENAI_RESPONSES_FORMAT,
        default_target_format=OPENAI_CHAT_COMPLETIONS_FORMAT,
        supported_formats=[OPENAI_CHAT_COMPLETIONS_FORMAT],
    ) is False


def test_same_protocol_distinguishes_openai_subprotocols():
    assert same_protocol(OPENAI_CHAT_COMPLETIONS_FORMAT, OPENAI_CHAT_COMPLETIONS_FORMAT) is True
    assert same_protocol(OPENAI_RESPONSES_FORMAT, OPENAI_RESPONSES_FORMAT) is True
    assert same_protocol(OPENAI_CHAT_COMPLETIONS_FORMAT, OPENAI_RESPONSES_FORMAT) is False


def test_resolve_protocol_context_for_openai_channel_keeps_client_protocol():
    payload = {
        "model": "gpt-4.1",
        "input": "hello",
        "stream": False,
    }

    context = resolve_protocol_context(
        request_data=payload,
        request_path="/v1/responses",
        target_provider="openai",
        explicit_client_format=OPENAI_RESPONSES_FORMAT,
        default_target_format=OPENAI_CHAT_COMPLETIONS_FORMAT,
        supported_formats=[OPENAI_CHAT_COMPLETIONS_FORMAT, OPENAI_RESPONSES_FORMAT],
    )

    assert context.client_format == OPENAI_RESPONSES_FORMAT
    assert context.target_format == OPENAI_RESPONSES_FORMAT
    assert context.normalized_request_data["input"] == "hello"


def test_resolve_protocol_context_uses_default_target_for_generic_openai_provider():
    payload = {
        "model": "gpt-4.1",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }

    context = resolve_protocol_context(
        request_data=payload,
        request_path="/custom/openai",
        target_provider="openai",
        explicit_client_format=None,
        default_target_format=OPENAI_RESPONSES_FORMAT,
        supported_formats=[OPENAI_CHAT_COMPLETIONS_FORMAT, OPENAI_RESPONSES_FORMAT],
    )

    assert context.client_format == OPENAI_CHAT_COMPLETIONS_FORMAT
    assert context.target_format == OPENAI_CHAT_COMPLETIONS_FORMAT
