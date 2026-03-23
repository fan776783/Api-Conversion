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
from src.formats.anthropic_converter import AnthropicConverter
from src.formats.gemini_converter import GeminiConverter
from src.formats.converter_factory import (
    OPENAI_CHAT_COMPLETIONS_FORMAT,
    OPENAI_RESPONSES_FORMAT,
    canonical_format_name,
    convert_request,
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



def test_chat_streaming_prevents_old_new_string_intermediate_state_leakage():
    """回归用户问题：Anthropic->OpenAI Chat 流式时不应提前泄漏半截 Edit 参数。"""
    stream_id = "claude-3-5"
    events = [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "call_edit_1",
                "name": "Edit",
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "input_json_delta",
                "partial_json": '{"old_string":"same text",',
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "input_json_delta",
                "partial_json": '"new_string":"same text"}',
            },
        },
        {
            "type": "content_block_stop",
            "index": 0,
        },
    ]

    results = [
        convert_streaming_chunk("anthropic", OPENAI_CHAT_COMPLETIONS_FORMAT, event, stream_id)
        for event in events
    ]

    first_delta = results[1].data["choices"][0]["delta"]
    second_delta = results[2].data["choices"][0]["delta"]
    final_delta = results[3].data["choices"][0]["delta"]

    assert first_delta == {}
    assert second_delta == {}

    final_tool_call = final_delta["tool_calls"][0]
    arguments = final_tool_call["function"]["arguments"]
    assert json.loads(arguments) == {
        "old_string": "same text",
        "new_string": "same text",
    }
    assert final_tool_call["function"]["name"] == "Edit"



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



def test_responses_stream_request_keeps_stream_flag_when_misdirected_to_chat_endpoint():
    """回归用户问题：Responses 风格请求误投 chat 端点时，stream=true 不能丢。"""
    payload = {
        "model": "gpt-4.1",
        "instructions": "你是一个代码助手",
        "input": "修复这个问题",
        "stream": True,
        "max_output_tokens": 256,
    }

    detected = __import__("asyncio").run(detect_request_format(payload, "/v1/chat/completions"))
    adapted = OpenAIResponsesRequestAdapter.adapt(payload)

    assert detected == OPENAI_RESPONSES_FORMAT
    assert adapted["stream"] is True
    assert adapted["max_tokens"] == 256
    assert adapted["messages"][0]["role"] == "system"
    assert adapted["messages"][1] == {"role": "user", "content": "修复这个问题"}



def test_responses_streaming_uses_response_events_not_chat_completion_chunks():
    """回归用户问题：Responses 客户端不应再收到 chat.completion.chunk 风格事件。"""
    response_id_model = "claude-3-5"
    message_start = {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "model": response_id_model,
            "usage": {"input_tokens": 8, "output_tokens": 0},
        },
    }
    text_start = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "text",
            "text": "",
        },
    }
    text_delta = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {
            "type": "text_delta",
            "text": "hello",
        },
    }
    text_stop = {"type": "content_block_stop", "index": 0}
    message_stop = {"type": "message_stop"}

    outputs = []
    for event in (message_start, text_start, text_delta, text_stop, message_stop):
        result = convert_streaming_chunk("anthropic", OPENAI_RESPONSES_FORMAT, event, response_id_model)
        outputs.extend(result.data)

    joined = "\n".join(outputs)
    assert "event: response.created" in joined
    assert "event: response.output_text.delta" in joined
    assert "event: response.completed" in joined
    assert "chat.completion.chunk" not in joined
    assert "data: [DONE]" not in joined



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



def test_anthropic_converter_accepts_openai_chat_completions_target_alias():
    converter = AnthropicConverter()
    payload = {
        "model": "claude-3-5-sonnet",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "hello"}],
    }

    result = converter.convert_request(payload, OPENAI_CHAT_COMPLETIONS_FORMAT)

    assert result.success is True
    assert result.data["model"] == "claude-3-5-sonnet"
    assert result.data["messages"][0] == {"role": "user", "content": "hello"}



def test_gemini_converter_accepts_openai_chat_completions_target_alias():
    payload = {
        "model": "gemini-2.5-pro",
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "hello"}],
            }
        ],
    }

    result = convert_request("gemini", OPENAI_CHAT_COMPLETIONS_FORMAT, payload)

    assert result.success is True
    assert result.data["model"] == "gemini-2.5-pro"
    assert result.data["messages"][0] == {"role": "user", "content": "hello"}



def test_canonical_format_name_keeps_openai_alias_chat_default():
    assert canonical_format_name("openai") == OPENAI_CHAT_COMPLETIONS_FORMAT
    assert canonical_format_name("openai_chat_completions") == OPENAI_CHAT_COMPLETIONS_FORMAT
    assert canonical_format_name("openai_responses") == OPENAI_RESPONSES_FORMAT
