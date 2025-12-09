"""
Lightweight stream event types for the unified layer.

These TypedDict-based events are designed for efficient internal streaming
parsing and conversion between provider-specific formats (Anthropic/OpenAI/Gemini).

Design principles:
- Minimal memory footprint compared to dataclass-based UnifiedContent
- Type-safe with Literal discriminators for pattern matching
- Compatible with both sync and async streaming pipelines
"""

from typing import Literal, Optional, TypedDict, Union


class TextEvent(TypedDict):
    """Text content chunk from a streaming response."""

    type: Literal["text"]
    index: int
    text: str


class ThinkingEvent(TypedDict):
    """Reasoning/thinking content chunk from a streaming response.

    Maps to:
    - Anthropic: thinking_delta
    - OpenAI: reasoning_content (o1/o3 models)
    - Gemini: thinkingContent
    """

    type: Literal["thinking"]
    index: int
    text: str


class ToolCallStartEvent(TypedDict):
    """Tool call start event.

    Emitted when a new tool/function invocation begins.
    """

    type: Literal["tool_call_start"]
    index: int
    tool_call_id: str
    name: str


class ToolCallDeltaEvent(TypedDict):
    """Incremental arguments for an in-flight tool call.

    Arguments are streamed as JSON string fragments that should be
    accumulated by the caller.
    """

    type: Literal["tool_call_delta"]
    index: int
    tool_call_id: str
    arguments_delta: str


class ToolCallEndEvent(TypedDict):
    """Tool call completion event."""

    type: Literal["tool_call_end"]
    index: int
    tool_call_id: str


class ContentBlockStartEvent(TypedDict):
    """Content block start marker.

    Used for Anthropic SSE compatibility.
    """

    type: Literal["content_block_start"]
    index: int
    block_type: str  # "text", "thinking", "tool_use"


class ContentBlockStopEvent(TypedDict):
    """Content block stop marker."""

    type: Literal["content_block_stop"]
    index: int


class MessageStartEvent(TypedDict):
    """Message start event with metadata."""

    type: Literal["message_start"]
    message_id: str
    model: str


class MessageDeltaEvent(TypedDict):
    """Message-level delta with stop reason and usage."""

    type: Literal["message_delta"]
    stop_reason: Optional[str]
    input_tokens: Optional[int]
    output_tokens: Optional[int]


class EndEvent(TypedDict):
    """Stream end marker."""

    type: Literal["end"]
    stop_reason: Optional[str]


# Union type for all stream events
StreamEvent = Union[
    TextEvent,
    ThinkingEvent,
    ToolCallStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    MessageStartEvent,
    MessageDeltaEvent,
    EndEvent,
]


def create_text_event(index: int, text: str) -> TextEvent:
    """Factory function for TextEvent."""
    return {"type": "text", "index": index, "text": text}


def create_thinking_event(index: int, text: str) -> ThinkingEvent:
    """Factory function for ThinkingEvent."""
    return {"type": "thinking", "index": index, "text": text}


def create_tool_call_start(
    index: int, tool_call_id: str, name: str
) -> ToolCallStartEvent:
    """Factory function for ToolCallStartEvent."""
    return {
        "type": "tool_call_start",
        "index": index,
        "tool_call_id": tool_call_id,
        "name": name,
    }


def create_tool_call_delta(
    index: int, tool_call_id: str, arguments_delta: str
) -> ToolCallDeltaEvent:
    """Factory function for ToolCallDeltaEvent."""
    return {
        "type": "tool_call_delta",
        "index": index,
        "tool_call_id": tool_call_id,
        "arguments_delta": arguments_delta,
    }


def create_end_event(stop_reason: Optional[str] = None) -> EndEvent:
    """Factory function for EndEvent."""
    return {"type": "end", "stop_reason": stop_reason}


__all__ = [
    # Event types
    "TextEvent",
    "ThinkingEvent",
    "ToolCallStartEvent",
    "ToolCallDeltaEvent",
    "ToolCallEndEvent",
    "ContentBlockStartEvent",
    "ContentBlockStopEvent",
    "MessageStartEvent",
    "MessageDeltaEvent",
    "EndEvent",
    "StreamEvent",
    # Factory functions
    "create_text_event",
    "create_thinking_event",
    "create_tool_call_start",
    "create_tool_call_delta",
    "create_end_event",
]
