"""Unit tests for Unified intermediate layer types.

Covers text/tool/thinking/annotation round-trip scenarios.
"""

import json
import pytest

from src.formats.unified import (
    UnifiedContentType,
    UnifiedContent,
    UnifiedMessage,
    UnifiedChatRequest,
    UnifiedChatResponse,
    UnifiedUsage,
)


class TestUnifiedContent:
    """Tests for UnifiedContent type conversions."""

    def test_text_with_annotations_round_trip_anthropic(self):
        """Text content with annotations should preserve all fields."""
        annotations = [{"type": "meta", "label": "example"}]
        anthropic_block = {
            "type": "text",
            "text": "hello",
            "annotations": annotations,
            "cache_control": {"type": "ephemeral"},
        }

        unified = UnifiedContent.from_anthropic(anthropic_block)

        assert unified.type == UnifiedContentType.TEXT
        assert unified.text == "hello"
        assert unified.annotations == annotations
        assert unified.cache_control == {"type": "ephemeral"}

        result = unified.to_anthropic()
        assert result["type"] == "text"
        assert result["text"] == "hello"
        assert result["annotations"] == annotations
        assert result["cache_control"]["type"] == "ephemeral"

    def test_tool_use_round_trip_openai_anthropic(self):
        """Tool calls should convert correctly between formats."""
        openai_tool_call = {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": json.dumps({"location": "San Francisco"}),
            },
        }

        unified = UnifiedContent.from_openai(openai_tool_call, content_type="tool_call")

        assert unified.type == UnifiedContentType.TOOL_USE
        assert unified.tool_use_id == "call_1"
        assert unified.tool_name == "get_weather"
        assert unified.tool_input == {"location": "San Francisco"}

        back_to_openai = unified.to_openai()
        assert back_to_openai["id"] == "call_1"
        assert back_to_openai["type"] == "function"
        assert back_to_openai["function"]["name"] == "get_weather"
        assert json.loads(back_to_openai["function"]["arguments"]) == {"location": "San Francisco"}

        anthropic_block = unified.to_anthropic()
        assert anthropic_block["type"] == "tool_use"
        assert anthropic_block["id"] == "call_1"
        assert anthropic_block["name"] == "get_weather"
        assert anthropic_block["input"] == {"location": "San Francisco"}

    def test_thinking_content_from_anthropic(self):
        """Thinking blocks should parse correctly."""
        anthropic_block = {"type": "thinking", "thinking": "internal chain of thought"}

        unified = UnifiedContent.from_anthropic(anthropic_block)

        assert unified.type == UnifiedContentType.THINKING
        assert unified.text == "internal chain of thought"

        result = unified.to_anthropic()
        assert result["type"] == "thinking"
        assert result["thinking"] == "internal chain of thought"

    def test_signature_content_from_anthropic(self):
        """Signature blocks should parse correctly."""
        anthropic_block = {"type": "signature", "signature": "model-signature-xyz"}

        unified = UnifiedContent.from_anthropic(anthropic_block)

        assert unified.type == UnifiedContentType.SIGNATURE
        assert unified.text == "model-signature-xyz"

        result = unified.to_anthropic()
        assert result["type"] == "signature"
        assert result["signature"] == "model-signature-xyz"

    def test_tool_result_from_anthropic(self):
        """Tool results should parse correctly."""
        anthropic_block = {
            "type": "tool_result",
            "tool_use_id": "call_1",
            "content": "The weather is sunny",
            "is_error": False,
        }

        unified = UnifiedContent.from_anthropic(anthropic_block)

        assert unified.type == UnifiedContentType.TOOL_RESULT
        assert unified.tool_use_id_ref == "call_1"
        assert unified.tool_result_content == "The weather is sunny"
        assert unified.is_error is False

    def test_image_content_base64_from_anthropic(self):
        """Base64 image content should parse correctly."""
        anthropic_block = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "iVBORw0KGgo=",
            },
        }

        unified = UnifiedContent.from_anthropic(anthropic_block)

        assert unified.type == UnifiedContentType.IMAGE
        assert unified.image_media_type == "image/png"
        assert unified.image_data == "iVBORw0KGgo="

    def test_reasoning_content_from_openai(self):
        """OpenAI reasoning_content should map to THINKING type."""
        unified = UnifiedContent.from_openai("internal reasoning", content_type="reasoning")

        assert unified.type == UnifiedContentType.THINKING
        assert unified.text == "internal reasoning"


class TestUnifiedMessage:
    """Tests for UnifiedMessage type conversions."""

    def test_to_anthropic_content_ordering(self):
        """Content blocks should be ordered: thinking → signature → text → tool_use."""
        msg = UnifiedMessage(
            role="assistant",
            content=[
                UnifiedContent(type=UnifiedContentType.TEXT, text="visible answer"),
                UnifiedContent(
                    type=UnifiedContentType.TOOL_USE,
                    tool_use_id="call_1",
                    tool_name="tool",
                    tool_input={"x": 1},
                ),
                UnifiedContent(type=UnifiedContentType.THINKING, text="internal chain"),
                UnifiedContent(type=UnifiedContentType.SIGNATURE, text="signature-by-model"),
            ],
        )

        anthropic_msg = msg.to_anthropic()
        types_order = [block["type"] for block in anthropic_msg["content"]]

        assert types_order == ["thinking", "signature", "text", "tool_use"]

    def test_to_openai_reasoning_content_separate(self):
        """THINKING content should become reasoning_content in OpenAI format."""
        msg = UnifiedMessage(
            role="assistant",
            content=[
                UnifiedContent(type=UnifiedContentType.THINKING, text="internal reasoning"),
                UnifiedContent(type=UnifiedContentType.TEXT, text="final visible answer"),
            ],
        )

        openai_msg = msg.to_openai()

        assert openai_msg["role"] == "assistant"
        assert openai_msg["content"] == "final visible answer"
        assert openai_msg["reasoning_content"] == "internal reasoning"

    def test_from_openai_reasoning_and_tool_calls_ordering(self):
        """OpenAI message with reasoning and tool_calls should parse correctly."""
        openai_msg = {
            "role": "assistant",
            "content": "visible content",
            "reasoning_content": "internal reasoning",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "foo", "arguments": '{"x": 1}'},
                }
            ],
        }

        unified_msg = UnifiedMessage.from_openai(openai_msg)

        assert unified_msg.role == "assistant"
        types = [c.type for c in unified_msg.content]
        # reasoning_content inserted at index 0, then text, then tool_calls
        assert types[0] == UnifiedContentType.THINKING
        assert types[1] == UnifiedContentType.TEXT
        assert types[2] == UnifiedContentType.TOOL_USE

    def test_tool_role_message_conversion(self):
        """Tool role messages should convert correctly."""
        openai_msg = {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "Tool result",
        }

        unified_msg = UnifiedMessage.from_openai(openai_msg)

        assert unified_msg.role == "tool"
        assert unified_msg.tool_call_id == "call_1"
        assert len(unified_msg.content) == 1
        assert unified_msg.content[0].type == UnifiedContentType.TOOL_RESULT

        anthropic_msg = unified_msg.to_anthropic()
        assert anthropic_msg["role"] == "user"
        assert anthropic_msg["content"][0]["type"] == "tool_result"


class TestUnifiedChatRequest:
    """Tests for UnifiedChatRequest type conversions."""

    def test_from_anthropic_normalizes_tool_result_user_message(self):
        """Anthropic tool_result in user message should normalize to tool role."""
        anthropic_request = {
            "model": "claude-3-5",
            "system": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant.",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "What is the weather?"}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "call_1", "content": "Sunny"}
                    ],
                },
            ],
            "thinking": {"type": "enabled", "budget_tokens": 128},
        }

        unified = UnifiedChatRequest.from_anthropic(anthropic_request)

        assert unified.model == "claude-3-5"
        assert unified.system == "You are a helpful assistant."
        assert unified.system_cache_control == {"type": "ephemeral"}
        assert unified.thinking_enabled is True
        assert unified.thinking_budget_tokens == 128

        assert [m.role for m in unified.messages] == ["user", "tool"]
        tool_msg = unified.messages[1]
        assert tool_msg.tool_call_id == "call_1"
        assert tool_msg.content[0].type == UnifiedContentType.TOOL_RESULT

    def test_to_openai_with_system_tools_and_stop_sequences(self):
        """Request conversion should handle system, tools, and stop sequences."""
        unified = UnifiedChatRequest(
            model="gpt-4.1",
            system="sys-message",
            messages=[
                UnifiedMessage(
                    role="user",
                    content=[UnifiedContent(type=UnifiedContentType.TEXT, text="hi")],
                )
            ],
            max_tokens=128,
            temperature=0.5,
            top_p=0.9,
            stop_sequences=["END"],
            tools=[
                {
                    "name": "calc",
                    "description": "calculator",
                    "input_schema": {"type": "object", "properties": {"x": {"type": "number"}}},
                }
            ],
        )

        openai_req = unified.to_openai()

        assert openai_req["model"] == "gpt-4.1"
        assert openai_req["messages"][0] == {"role": "system", "content": "sys-message"}
        assert openai_req["messages"][1]["role"] == "user"
        assert openai_req["stop"] == ["END"]
        assert len(openai_req["tools"]) == 1
        assert openai_req["tools"][0]["type"] == "function"

    def test_thinking_mode_to_openai(self):
        """Thinking mode should map to reasoning_effort and max_completion_tokens."""
        unified = UnifiedChatRequest(
            model="gpt-o1",
            messages=[],
            thinking_enabled=True,
            reasoning_effort="medium",
            max_completion_tokens=1024,
        )

        openai_req = unified.to_openai()

        assert openai_req["reasoning_effort"] == "medium"
        assert openai_req["max_completion_tokens"] == 1024


class TestUnifiedChatResponse:
    """Tests for UnifiedChatResponse type conversions."""

    def test_from_openai_to_anthropic_preserves_order_and_stop_reason(self):
        """Response conversion should preserve content order and map stop_reason."""
        openai_response = {
            "id": "chatcmpl-xyz",
            "model": "gpt-o1",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "visible answer",
                        "reasoning_content": "internal reasoning",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "tool", "arguments": '{"a": 1}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        unified = UnifiedChatResponse.from_openai(openai_response, original_model="claude-3")

        assert unified.id == "xyz"
        assert unified.model == "gpt-o1"
        assert unified.original_model == "claude-3"
        assert unified.stop_reason == "tool_calls"
        assert isinstance(unified.usage, UnifiedUsage)
        assert unified.usage.input_tokens == 10
        assert unified.usage.output_tokens == 5

        anthropic = unified.to_anthropic()
        types = [c["type"] for c in anthropic["content"]]
        assert types == ["thinking", "text", "tool_use"]
        assert anthropic["stop_reason"] == "tool_use"

    def test_from_anthropic_round_trip(self):
        """Anthropic response should round-trip correctly."""
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet",
            "content": [
                {"type": "thinking", "thinking": "Let me think..."},
                {"type": "text", "text": "Here is the answer"},
            ],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 20,
            },
        }

        unified = UnifiedChatResponse.from_anthropic(anthropic_response)
        result = unified.to_anthropic()

        assert result["id"] == "msg_123"
        assert result["model"] == "claude-3-5-sonnet"
        assert result["stop_reason"] == "end_turn"
        assert len(result["content"]) == 2
        assert result["usage"]["input_tokens"] == 100
        assert result["usage"]["cache_read_input_tokens"] == 20
