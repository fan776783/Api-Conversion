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
        result = converter.convert_request(anthropic_request, target_format="openai")
        assert result.success is True

        assistant_msgs = [m for m in result.data["messages"] if m.get("role") == "assistant"]
        assert len(assistant_msgs) == 1
        assert "tool_calls" not in assistant_msgs[0]

    def test_with_tool_result_pair(self):
        """Tool use + tool result pair should convert correctly."""
        anthropic_request = {
            "model": "claude-3-5",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "call"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "call_1", "name": "tool", "input": {"x": 1}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "call_1", "content": "result"},
                    ],
                },
            ],
        }

        converter = _make_converter()
        result = converter.convert_request(anthropic_request, target_format="openai")
        assert result.success is True

        messages = result.data["messages"]
        assistant_msg = next(m for m in messages if m.get("role") == "assistant")
        assert "tool_calls" in assistant_msg
        assert len(assistant_msg["tool_calls"]) == 1

        tool_msg = next(m for m in messages if m.get("role") == "tool")
        assert tool_msg["tool_call_id"] == "call_1"


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
