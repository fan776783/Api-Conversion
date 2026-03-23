import json
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from src.api.conversion_api import detect_request_format
from src.formats.converter_factory import (
    OPENAI_CHAT_COMPLETIONS_FORMAT,
    OPENAI_RESPONSES_FORMAT,
    canonical_format_name,
    convert_response,
    convert_streaming_chunk,
)
from src.formats.openai_responses_request_adapter import OpenAIResponsesRequestAdapter


def test_detect_request_format_recognizes_openai_responses_path_and_shape():
    payload = {
        "model": "gpt-4.1",
        "input": "hello",
        "instructions": "be helpful",
        "max_output_tokens": 128,
    }

    detected_by_shape = __import__("asyncio").run(detect_request_format(payload, "/v1/chat/completions"))
    assert detected_by_shape == OPENAI_RESPONSES_FORMAT

    detected_by_path = __import__("asyncio").run(detect_request_format({"model": "gpt-4.1"}, "/v1/responses"))
    assert detected_by_path == OPENAI_RESPONSES_FORMAT



def test_openai_responses_request_adapter_preserves_stream_and_maps_fields():
    payload = {
        "model": "gpt-4.1",
        "instructions": "system prompt",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        "max_output_tokens": 64,
        "reasoning": {"effort": "high"},
        "stream": True,
    }

    adapted = OpenAIResponsesRequestAdapter.adapt(payload)
    assert adapted["stream"] is True
    assert adapted["max_tokens"] == 64
    assert adapted["reasoning_effort"] == "high"
    assert adapted["messages"][0] == {"role": "system", "content": "system prompt"}
    assert adapted["messages"][1]["role"] == "user"



def test_chat_streaming_tool_arguments_are_only_emitted_on_content_block_stop():
    stream_id = "claude-3-5"
    start = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "tool_use",
            "id": "call_edit_1",
            "name": "Edit",
        },
    }
    delta = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {
            "type": "input_json_delta",
            "partial_json": '{"old_string":"before",',
        },
    }
    stop = {
        "type": "content_block_stop",
        "index": 0,
    }
    finish = {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use"},
    }

    start_result = convert_streaming_chunk("anthropic", OPENAI_CHAT_COMPLETIONS_FORMAT, start, stream_id)
    delta_result = convert_streaming_chunk("anthropic", OPENAI_CHAT_COMPLETIONS_FORMAT, delta, stream_id)
    stop_result = convert_streaming_chunk("anthropic", OPENAI_CHAT_COMPLETIONS_FORMAT, stop, stream_id)
    finish_result = convert_streaming_chunk("anthropic", OPENAI_CHAT_COMPLETIONS_FORMAT, finish, stream_id)

    start_tool_calls = start_result.data["choices"][0]["delta"]["tool_calls"]
    assert start_tool_calls[0]["function"]["name"] == "Edit"

    assert delta_result.data["choices"][0]["delta"] == {}

    stop_tool_calls = stop_result.data["choices"][0]["delta"]["tool_calls"]
    assert stop_tool_calls[0]["function"]["arguments"] == '{"old_string":"before",'

    assert finish_result.data["choices"][0]["finish_reason"] == "tool_calls"



def test_responses_streaming_emits_named_events_and_completed_payload():
    response_id_model = "claude-3-5"
    message_start = {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "model": response_id_model,
            "usage": {"input_tokens": 10, "output_tokens": 0},
        },
    }
    tool_start = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "tool_use",
            "id": "call_1",
            "name": "Edit",
        },
    }
    tool_delta = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {
            "type": "input_json_delta",
            "partial_json": '{"old_string":"before"}',
        },
    }
    tool_stop = {"type": "content_block_stop", "index": 0}
    message_stop = {"type": "message_stop"}

    created = convert_streaming_chunk("anthropic", OPENAI_RESPONSES_FORMAT, message_start, response_id_model)
    added = convert_streaming_chunk("anthropic", OPENAI_RESPONSES_FORMAT, tool_start, response_id_model)
    delta = convert_streaming_chunk("anthropic", OPENAI_RESPONSES_FORMAT, tool_delta, response_id_model)
    done = convert_streaming_chunk("anthropic", OPENAI_RESPONSES_FORMAT, tool_stop, response_id_model)
    completed = convert_streaming_chunk("anthropic", OPENAI_RESPONSES_FORMAT, message_stop, response_id_model)

    assert created.success is True
    assert any("event: response.created" in event for event in created.data)
    assert any("event: response.output_item.added" in event for event in added.data)
    assert any("event: response.function_call_arguments.delta" in event for event in delta.data)
    assert any("event: response.function_call_arguments.done" in event for event in done.data)
    assert any("event: response.completed" in event for event in completed.data)

    completed_payload = completed.data[-1]
    assert '"status": "completed"' in completed_payload
    assert '"arguments": "{\\"old_string\\":\\"before\\"}"' in completed_payload



def test_non_streaming_responses_conversion_builds_response_object():
    anthropic_response = {
        "id": "msg_123",
        "model": "claude-3-5",
        "content": [
            {"type": "text", "text": "hello"},
            {"type": "tool_use", "id": "call_1", "name": "Edit", "input": {"old_string": "before"}},
        ],
        "usage": {"input_tokens": 5, "output_tokens": 7},
    }

    result = convert_response("anthropic", OPENAI_RESPONSES_FORMAT, anthropic_response, original_model="gpt-4.1")
    assert result.success is True
    assert result.data["object"] == "response"
    assert result.data["status"] == "completed"
    assert result.data["output_text"] == "hello"
    function_call = next(item for item in result.data["output"] if item["type"] == "function_call")
    assert json.loads(function_call["arguments"]) == {"old_string": "before"}



def test_canonical_format_name_keeps_openai_alias_chat_default():
    assert canonical_format_name("openai") == OPENAI_CHAT_COMPLETIONS_FORMAT
    assert canonical_format_name("openai_chat_completions") == OPENAI_CHAT_COMPLETIONS_FORMAT
    assert canonical_format_name("openai_responses") == OPENAI_RESPONSES_FORMAT
