"""Integration tests for AnthropicConverter with Unified layer.

Covers text/tool/thinking/annotation round-trip and streaming conversion.
"""

import json
import pytest

from src.formats.anthropic_converter import AnthropicConverter
from src.formats.base_converter import ConversionError
from src.formats.unified import UnifiedChatRequest, UnifiedChatResponse, UnifiedContentType


def _make_converter(model: str = "claude-3-5") -> AnthropicConverter:
    converter = AnthropicConverter()
    converter.set_original_model(model)
    return converter


class TestRequestConversion:
    """Tests for Anthropic to OpenAI request conversion."""

    def test_text_only_round_trip(self):
        """Text-only request should convert correctly."""
        anthropic_request = {
            "model": "claude-3-5",
            "system": "sys",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            ],
            "max_tokens": 32,
            "temperature": 0.1,
            "top_p": 0.9,
            "stop_sequences": ["END"],
        }

        converter = _make_converter()
        result = converter.convert_request(anthropic_request, target_format="openai")
        assert result.success is True

        openai_req = result.data
        assert openai_req["messages"][0] == {"role": "system", "content": "sys"}
        assert openai_req["messages"][1]["role"] == "user"

        # Round-trip back to Anthropic via unified layer
        unified = UnifiedChatRequest.from_openai(openai_req)
        anthropic_round_trip = unified.to_anthropic()

        assert anthropic_round_trip["model"] == "claude-3-5"
        assert anthropic_round_trip["system"] == "sys"

    def test_thinking_maps_to_reasoning(self, monkeypatch: pytest.MonkeyPatch):
        """Anthropic thinking should map to OpenAI reasoning_effort."""
        anthropic_request = {
            "model": "claude-3-5",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "solve"}]},
            ],
            "thinking": {"type": "enabled", "budget_tokens": 150},
        }

        monkeypatch.setenv("ANTHROPIC_TO_OPENAI_LOW_REASONING_THRESHOLD", "100")
        monkeypatch.setenv("ANTHROPIC_TO_OPENAI_HIGH_REASONING_THRESHOLD", "200")
        monkeypatch.setenv("OPENAI_REASONING_MAX_TOKENS", "256")

        converter = _make_converter()
        result = converter.convert_request(anthropic_request, target_format="openai")
        assert result.success is True

        openai_req = result.data
        assert openai_req["reasoning_effort"] == "medium"
        assert openai_req["max_completion_tokens"] == 256
        assert "max_tokens" not in openai_req

    def test_cleans_unmatched_tool_calls(self):
        """Unmatched tool_calls should be pruned."""
        anthropic_request = {
            "model": "claude-3-5",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "call a tool"}]},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "do_something",
                            "input": {},
                        }
                    ],
                },
            ],
        }

        converter = _make_converter()
        warnings = []

        def _capture_warning(message, *args):
            warnings.append(message % args if args else message)

        converter.logger.warning = _capture_warning
        result = converter.convert_request(anthropic_request, target_format="openai")
        assert result.success is True

        assistant_msgs = [m for m in result.data["messages"] if m.get("role") == "assistant"]
        assert len(assistant_msgs) == 1
        assert "tool_calls" not in assistant_msgs[0]
        assert len(warnings) == 1
        assert "assistant_message_index=1" in warnings[0]
        assert "original_tool_call_count=1" in warnings[0]

    def test_repairs_tool_message_missing_tool_call_id_by_name(self):
        """Tool message missing tool_call_id should be repaired by tool name when unambiguous."""
        anthropic_request = {
            "model": "claude-3-5",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "call tool"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "call_1", "name": "Bash", "input": {"cmd": "pwd"}},
                    ],
                },
                {
                    "role": "tool",
                    "name": "Bash",
                    "content": "ok",
                },
            ],
        }

        converter = _make_converter()
        result = converter.convert_request(anthropic_request, target_format="openai")
        assert result.success is True

        tool_msgs = [m for m in result.data["messages"] if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "call_1"

        assistant_msg = next(m for m in result.data["messages"] if m.get("role") == "assistant")
        assert len(assistant_msg.get("tool_calls", [])) == 1

    def test_repairs_tool_result_without_tool_use_id_when_single_candidate(self):
        """单一候选时，缺失 tool_use_id 的 tool_result 应被唯一候选自动补齐。"""
        anthropic_request = {
            "model": "claude-3-5",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "run one tool"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "call_only", "name": "Edit", "input": {"old_text": "a"}},
                    ],
                },
                {
                    "role": "tool",
                    "content": "validation failed",
                },
            ],
        }

        converter = _make_converter()
        result = converter.convert_request(anthropic_request, target_format="openai")
        assert result.success is True

        tool_msg = next(m for m in result.data["messages"] if m.get("role") == "tool")
        assert tool_msg["tool_call_id"] == "call_only"

        assistant_msg = next(m for m in result.data["messages"] if m.get("role") == "assistant")
        assert len(assistant_msg.get("tool_calls", [])) == 1

    def test_does_not_bind_ambiguous_tool_result_without_tool_use_id(self):
        """多个候选时，缺失 tool_use_id 的 tool_result 不应误绑定，应触发清理。"""
        anthropic_request = {
            "model": "claude-3-5",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "run tools"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "call_1", "name": "Edit", "input": {"old_text": "a"}},
                        {"type": "tool_use", "id": "call_2", "name": "Edit", "input": {"old_text": "b"}},
                    ],
                },
                {
                    "role": "tool",
                    "content": "validation failed",
                },
            ],
        }

        converter = _make_converter()
        warnings = []

        def _capture_warning(message, *args):
            warnings.append(message % args if args else message)

        converter.logger.warning = _capture_warning
        result = converter.convert_request(anthropic_request, target_format="openai")
        assert result.success is True

        assistant_msg = next(m for m in result.data["messages"] if m.get("role") == "assistant")
        assert "tool_calls" not in assistant_msg
        assert assistant_msg["content"] == ""
        assert len(warnings) == 1
        assert "original_tool_call_count=2" in warnings[0]
    def test_regression_schema_alias_wrong_args_and_repaired_tool_response(self):
        """schema 别名回退后，错误参数名仍可透传，且缺失 tool_call_id 的工具响应可被修补。"""
        anthropic_request = {
            "model": "claude-3-5",
            "tools": [
                {
                    "name": "Edit",
                    "description": "edit text with schema alias",
                    "parametersJsonSchema": {
                        "type": "object",
                        "properties": {
                            "old_string": {"type": "string"},
                            "new_string": {"type": "string"},
                        },
                        "required": ["old_string", "new_string"],
                    },
                }
            ],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "patch this file"}]},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_edit_1",
                            "name": "Edit",
                            "input": {
                                "old_text": "before",
                                "new_text": "after",
                            },
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_edit_1",
                            "content": (
                                "InputValidationError: required old_string/new_string missing; "
                                "unexpected old_text/new_text"
                            ),
                        }
                    ],
                },
            ],
        }

        converter = _make_converter()
        result = converter.convert_request(anthropic_request, target_format="openai")
        assert result.success is True

        openai_req = result.data
        function_def = openai_req["tools"][0]["function"]
        assert function_def["parameters"]["required"] == ["old_string", "new_string"]
        assert function_def["parametersJsonSchema"]["required"] == ["old_string", "new_string"]

        assistant_msg = next(m for m in openai_req["messages"] if m.get("role") == "assistant")
        assert len(assistant_msg.get("tool_calls", [])) == 1
        assert json.loads(assistant_msg["tool_calls"][0]["function"]["arguments"]) == {
            "old_text": "before",
            "new_text": "after",
        }

        tool_msg = next(m for m in openai_req["messages"] if m.get("role") == "tool")
        assert tool_msg["tool_call_id"] == "call_edit_1"
        assert "old_string/new_string missing" in tool_msg["content"]



class TestResponseConversion:
    """Tests for OpenAI to Anthropic response conversion."""

    def test_text_and_tool_calls(self):
        """Response with text and tool_calls should convert correctly."""
        openai_response = {
            "id": "chatcmpl-123",
            "model": "gpt-4.1-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "here is the answer",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "do_something", "arguments": '{"a": 1}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 7},
        }

        converter = _make_converter("claude-3-5-sonnet")
        result = converter.convert_response(
            openai_response, source_format="openai", target_format="anthropic"
        )
        assert result.success is True

        anthropic = result.data
        assert anthropic["model"] == "claude-3-5-sonnet"
        types = [c["type"] for c in anthropic["content"]]
        assert "text" in types
        assert "tool_use" in types
        assert anthropic["stop_reason"] == "tool_use"
        assert anthropic["usage"]["input_tokens"] == 3

    def test_reasoning_content_and_annotations(self):
        """reasoning_content and annotations should be preserved."""
        openai_response = {
            "id": "chatcmpl-xyz",
            "model": "gpt-o1",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "visible answer",
                        "reasoning_content": "internal reasoning chain",
                        "annotations": [{"type": "thinking_utilization", "value": "high"}],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }

        converter = _make_converter("claude-3-5-opus")
        result = converter.convert_response(
            openai_response, source_format="openai", target_format="anthropic"
        )
        assert result.success is True

        anthropic = result.data
        types = [c["type"] for c in anthropic["content"]]
        assert types[0] == "thinking"
        assert types[1] == "text"

        thinking_block = anthropic["content"][0]
        text_block = anthropic["content"][1]
        assert thinking_block["thinking"] == "internal reasoning chain"
        assert text_block.get("annotations") is not None
        assert anthropic["stop_reason"] == "end_turn"

    def test_missing_original_model_returns_failure(self):
        """Missing original_model should return failure result."""
        converter = AnthropicConverter()  # No set_original_model call

        openai_response = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        }

        result = converter.convert_response(openai_response, source_format="openai", target_format="anthropic")
        assert result.success is False
        assert "Original model name is required" in result.error


class TestStreamingConversion:
    """Tests for OpenAI streaming to Anthropic SSE conversion."""

    def test_reasoning_content_streaming(self):
        """reasoning_content should stream as thinking_delta."""
        converter = _make_converter("claude-3-5-sonnet")

        chunk = {
            "id": "chatcmpl-stream",
            "model": "gpt-o3-mini",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "reasoning_content": "think-1 "},
                    "finish_reason": None,
                }
            ],
        }

        result = converter._convert_from_openai_streaming_chunk(chunk)
        assert result.success is True

        sse = result.data or ""
        assert "event: message_start" in sse
        assert '"type": "thinking_delta"' in sse

    def test_text_content_streaming(self):
        """Text content should stream as text_delta."""
        converter = _make_converter("claude-3-5-sonnet")

        # First chunk to initialize
        chunk1 = {
            "id": "chatcmpl-stream",
            "model": "gpt-4",
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        converter._convert_from_openai_streaming_chunk(chunk1)

        # Text chunk
        chunk2 = {
            "id": "chatcmpl-stream",
            "model": "gpt-4",
            "choices": [{"index": 0, "delta": {"content": "Hello world"}, "finish_reason": None}],
        }

        result = converter._convert_from_openai_streaming_chunk(chunk2)
        assert result.success is True
        assert '"type": "text_delta"' in (result.data or "")

    def test_tool_call_streaming(self):
        """Tool calls should stream as input_json_delta."""
        converter = _make_converter("claude-3-5-sonnet")

        # Initialize
        chunk1 = {
            "id": "chatcmpl-stream",
            "model": "gpt-4",
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        converter._convert_from_openai_streaming_chunk(chunk1)

        # Tool call chunk
        chunk2 = {
            "id": "chatcmpl-stream",
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "tool", "arguments": '{"x":'},
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        }

        result = converter._convert_from_openai_streaming_chunk(chunk2)
        assert result.success is True

        sse = result.data or ""
        assert '"type": "tool_use"' in sse
        assert '"type": "input_json_delta"' in sse

    def test_finish_reason_emits_stop_events(self):
        """finish_reason should emit content_block_stop and message_stop."""
        converter = _make_converter("claude-3-5-sonnet")

        # Initialize and send content
        chunks = [
            {"id": "chatcmpl-stream", "model": "gpt-4", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
            {"id": "chatcmpl-stream", "model": "gpt-4", "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}]},
        ]
        for chunk in chunks:
            converter._convert_from_openai_streaming_chunk(chunk)

        # Final chunk with finish_reason
        final_chunk = {
            "id": "chatcmpl-stream",
            "model": "gpt-4",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10},
        }

        result = converter._convert_from_openai_streaming_chunk(final_chunk)
        assert result.success is True

        sse = result.data or ""
        assert "event: content_block_stop" in sse
        assert "event: message_delta" in sse
        assert "event: message_stop" in sse

    def test_streaming_state_reset(self):
        """reset_streaming_state should set force_reset flag."""
        converter = _make_converter("claude-3-5")

        chunk = {
            "id": "chatcmpl-stream",
            "model": "gpt-4",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hi"}, "finish_reason": None}],
        }
        converter._convert_from_openai_streaming_chunk(chunk)

        assert hasattr(converter, "_openai_stream_state")

        converter.reset_streaming_state()

        # reset_streaming_state sets _force_reset flag for lazy cleanup
        assert getattr(converter, "_force_reset", False) is True


class TestThinkingTagExtraction:
    """Tests for <thinking> tag extraction from OpenAI text."""

    def test_extract_thinking_tags(self):
        """<thinking> tags should be extracted to thinking blocks."""
        converter = _make_converter()

        text = "<thinking>Let me think about this...</thinking>Here is the answer."
        result = converter._extract_thinking_from_openai_text(text)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["type"] == "thinking"
        assert result[0]["thinking"] == "Let me think about this..."
        assert result[1]["type"] == "text"
        assert result[1]["text"] == "Here is the answer."

    def test_no_thinking_tags_returns_string(self):
        """Text without thinking tags should return original string."""
        converter = _make_converter()

        text = "Just a normal response."
        result = converter._extract_thinking_from_openai_text(text)

        assert result == "Just a normal response."

    def test_multiple_thinking_tags(self):
        """Multiple thinking tags should all be extracted."""
        converter = _make_converter()

        text = "<thinking>First thought</thinking>Middle text<thinking>Second thought</thinking>Final answer"
        result = converter._extract_thinking_from_openai_text(text)

        assert isinstance(result, list)
        thinking_blocks = [b for b in result if b.get("type") == "thinking"]
        text_blocks = [b for b in result if b.get("type") == "text"]

        assert len(thinking_blocks) == 2
        assert len(text_blocks) == 2
