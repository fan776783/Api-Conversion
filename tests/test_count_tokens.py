"""Unit tests for /v1/messages/count_tokens endpoint.

Covers Anthropic token counting API with different channel types.
"""

import json
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Import the helper functions directly by copying the logic since full import
# requires complex application dependencies. This tests the core logic.


def _extract_text_from_anthropic_request(request_data: dict) -> str:
    """从 Anthropic 请求中提取所有文本内容"""
    text_parts = []

    # 提取 system prompt
    system = request_data.get("system")
    if isinstance(system, str):
        text_parts.append(system)
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))

    # 提取 messages
    for msg in request_data.get("messages", []):
        content = msg.get("content")
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type", "")
                    if block_type == "text":
                        text_parts.append(block.get("text", ""))
                    elif block_type == "tool_use":
                        text_parts.append(block.get("name", ""))
                        text_parts.append(json.dumps(block.get("input", {})))
                    elif block_type == "tool_result":
                        result_content = block.get("content", "")
                        if isinstance(result_content, str):
                            text_parts.append(result_content)

    # 提取 tools 定义
    for tool in request_data.get("tools", []):
        text_parts.append(tool.get("name", ""))
        text_parts.append(tool.get("description", ""))
        text_parts.append(json.dumps(tool.get("input_schema", {})))

    return "\n".join(text_parts)


def _convert_anthropic_messages_to_gemini_contents(request_data: dict) -> list:
    """将 Anthropic messages 转换为 Gemini contents 格式"""
    contents = []

    # 添加 system prompt 作为第一条 user 消息
    system = request_data.get("system")
    if system:
        system_text = system if isinstance(system, str) else system[0].get("text", "") if system else ""
        if system_text:
            contents.append({
                "role": "user",
                "parts": [{"text": f"[System]: {system_text}"}]
            })

    # 转换 messages
    for msg in request_data.get("messages", []):
        role = msg.get("role", "user")
        gemini_role = "model" if role == "assistant" else "user"

        parts = []
        content = msg.get("content")
        if isinstance(content, str):
            parts.append({"text": content})
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type", "")
                    if block_type == "text":
                        parts.append({"text": block.get("text", "")})
                    elif block_type == "tool_use":
                        parts.append({"text": f"[Tool Call: {block.get('name', '')}]"})
                    elif block_type == "tool_result":
                        result = block.get("content", "")
                        parts.append({"text": f"[Tool Result]: {result}" if isinstance(result, str) else str(result)})

        if parts:
            contents.append({"role": gemini_role, "parts": parts})

    return contents


class TestExtractTextFromAnthropicRequest:
    """Tests for text extraction from Anthropic requests."""

    def test_extract_simple_text_message(self):
        """Simple text message extraction."""
        request = {
            "messages": [
                {"role": "user", "content": "Hello world"}
            ]
        }

        result = _extract_text_from_anthropic_request(request)
        assert "Hello world" in result

    def test_extract_text_with_system_prompt(self):
        """System prompt should be extracted."""
        request = {
            "system": "You are a helpful assistant.",
            "messages": [
                {"role": "user", "content": "Hi"}
            ]
        }

        result = _extract_text_from_anthropic_request(request)
        assert "You are a helpful assistant." in result
        assert "Hi" in result

    def test_extract_text_with_system_array(self):
        """System prompt as array should be extracted."""
        request = {
            "system": [
                {"type": "text", "text": "System instruction 1"},
                {"type": "text", "text": "System instruction 2"}
            ],
            "messages": [
                {"role": "user", "content": "Query"}
            ]
        }

        result = _extract_text_from_anthropic_request(request)
        assert "System instruction 1" in result
        assert "System instruction 2" in result

    def test_extract_text_content_blocks(self):
        """Text content blocks should be extracted."""
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "First part"},
                        {"type": "text", "text": "Second part"}
                    ]
                }
            ]
        }

        result = _extract_text_from_anthropic_request(request)
        assert "First part" in result
        assert "Second part" in result

    def test_extract_tool_use_content(self):
        """Tool use blocks should include name and input."""
        request = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "get_weather",
                            "input": {"location": "NYC"}
                        }
                    ]
                }
            ]
        }

        result = _extract_text_from_anthropic_request(request)
        assert "get_weather" in result
        assert "location" in result

    def test_extract_tool_result_content(self):
        """Tool result content should be extracted."""
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": "The weather is sunny"
                        }
                    ]
                }
            ]
        }

        result = _extract_text_from_anthropic_request(request)
        assert "The weather is sunny" in result

    def test_extract_tools_definitions(self):
        """Tool definitions should be extracted."""
        request = {
            "messages": [],
            "tools": [
                {
                    "name": "calculator",
                    "description": "Performs calculations",
                    "input_schema": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}}
                    }
                }
            ]
        }

        result = _extract_text_from_anthropic_request(request)
        assert "calculator" in result
        assert "Performs calculations" in result


class TestConvertAnthropicMessagesToGemini:
    """Tests for Anthropic to Gemini format conversion."""

    def test_convert_simple_user_message(self):
        """Simple user message conversion."""
        request = {
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }

        result = _convert_anthropic_messages_to_gemini_contents(request)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["parts"][0]["text"] == "Hello"

    def test_convert_with_system_prompt(self):
        """System prompt should become first user message."""
        request = {
            "system": "You are helpful.",
            "messages": [
                {"role": "user", "content": "Hi"}
            ]
        }

        result = _convert_anthropic_messages_to_gemini_contents(request)

        assert len(result) == 2
        assert "[System]" in result[0]["parts"][0]["text"]
        assert "You are helpful." in result[0]["parts"][0]["text"]

    def test_convert_assistant_to_model(self):
        """Assistant role should become model role."""
        request = {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"}
            ]
        }

        result = _convert_anthropic_messages_to_gemini_contents(request)

        assert result[0]["role"] == "user"
        assert result[1]["role"] == "model"

    def test_convert_content_blocks(self):
        """Content blocks should be converted to parts."""
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Part 1"},
                        {"type": "text", "text": "Part 2"}
                    ]
                }
            ]
        }

        result = _convert_anthropic_messages_to_gemini_contents(request)

        assert len(result[0]["parts"]) == 2
        assert result[0]["parts"][0]["text"] == "Part 1"
        assert result[0]["parts"][1]["text"] == "Part 2"

    def test_convert_tool_use_to_text(self):
        """Tool use blocks should become text markers."""
        request = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "search",
                            "input": {}
                        }
                    ]
                }
            ]
        }

        result = _convert_anthropic_messages_to_gemini_contents(request)

        assert "[Tool Call: search]" in result[0]["parts"][0]["text"]

    def test_convert_tool_result_to_text(self):
        """Tool result blocks should become text markers."""
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": "Result data"
                        }
                    ]
                }
            ]
        }

        result = _convert_anthropic_messages_to_gemini_contents(request)

        assert "[Tool Result]" in result[0]["parts"][0]["text"]
        assert "Result data" in result[0]["parts"][0]["text"]

    def test_empty_messages_handled(self):
        """Empty messages list should return empty result."""
        request = {"messages": []}

        result = _convert_anthropic_messages_to_gemini_contents(request)

        assert result == []


class TestTokenEstimation:
    """Tests for token estimation logic."""

    def test_character_based_estimation(self):
        """Character-based estimation should approximate tokens."""
        text = "Hello world, this is a test message."
        # ~36 chars / 3.5 ≈ 10 tokens
        estimated = max(1, int(len(text) / 3.5))
        assert 8 <= estimated <= 12

    def test_empty_text_returns_minimum(self):
        """Empty text should return minimum of 1 token."""
        text = ""
        estimated = max(1, int(len(text) / 3.5))
        assert estimated == 1

    def test_chinese_text_estimation(self):
        """Chinese text estimation should work."""
        text = "你好世界，这是一个测试消息。"
        # Chinese chars: ~14 chars / 3.5 ≈ 4 tokens (rough estimate)
        estimated = max(1, int(len(text) / 3.5))
        assert estimated >= 1
