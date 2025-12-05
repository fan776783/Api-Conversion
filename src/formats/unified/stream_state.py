"""
Stream State Management

Manages streaming conversion state for unified format transformations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import time


class StreamPhase(str, Enum):
    """Current phase of streaming response."""

    NOT_STARTED = "not_started"
    MESSAGE_STARTED = "message_started"
    CONTENT_STREAMING = "content_streaming"
    TOOL_STREAMING = "tool_streaming"
    FINISHED = "finished"


@dataclass
class ToolCallState:
    """State for a single tool call being streamed."""

    index: int
    tool_call_id: str
    name: str
    arguments_buffer: str = ""
    content_block_index: int = 0
    is_complete: bool = False


@dataclass
class StreamState:
    """Manages streaming conversion state.

    Replaces scattered instance attributes like:
    - _streaming_state
    - _gemini_sent_start
    - _gemini_stream_id
    - _gemini_text_started
    - _anthropic_tool_state
    """

    # Identifiers
    stream_id: str = field(default_factory=lambda: f"msg_{int(time.time() * 1000)}")
    model: str = ""
    original_model: str = ""

    # Phase tracking
    phase: StreamPhase = StreamPhase.NOT_STARTED

    # Content block tracking
    content_block_count: int = 0
    text_block_index: Optional[int] = None
    text_block_started: bool = False

    # Tool call tracking
    tool_calls: Dict[int, ToolCallState] = field(default_factory=dict)
    tool_call_to_content_index: Dict[int, int] = field(default_factory=dict)

    # Thinking block tracking
    thinking_block_index: Optional[int] = None
    thinking_block_started: bool = False
    thinking_buffer: str = ""

    # Signature block tracking
    signature_block_index: Optional[int] = None
    signature_block_started: bool = False

    # Usage accumulation (for streaming)
    input_tokens: int = 0
    output_tokens: int = 0

    # Format-specific flags
    sent_message_start: bool = False
    sent_message_stop: bool = False

    def reset(self) -> None:
        """Reset state for new stream."""
        self.stream_id = f"msg_{int(time.time() * 1000)}"
        self.phase = StreamPhase.NOT_STARTED
        self.content_block_count = 0
        self.text_block_index = None
        self.text_block_started = False
        self.tool_calls.clear()
        self.tool_call_to_content_index.clear()
        self.thinking_block_index = None
        self.thinking_block_started = False
        self.thinking_buffer = ""
        self.signature_block_index = None
        self.signature_block_started = False
        self.input_tokens = 0
        self.output_tokens = 0
        self.sent_message_start = False
        self.sent_message_stop = False

    def start_thinking_block(self) -> int:
        """Start a new thinking content block, return its index.

        Thinking blocks should be emitted first per content ordering rules.
        """
        if not self.thinking_block_started:
            self.thinking_block_index = self.content_block_count
            self.content_block_count += 1
            self.thinking_block_started = True
        return self.thinking_block_index  # type: ignore

    def start_signature_block(self) -> int:
        """Start a new signature content block, return its index.

        Signature blocks follow thinking blocks.
        """
        if not self.signature_block_started:
            self.signature_block_index = self.content_block_count
            self.content_block_count += 1
            self.signature_block_started = True
        return self.signature_block_index  # type: ignore

    def start_text_block(self) -> int:
        """Start a new text content block, return its index.

        Text blocks follow thinking and signature blocks.
        """
        if not self.text_block_started:
            self.text_block_index = self.content_block_count
            self.content_block_count += 1
            self.text_block_started = True
        return self.text_block_index  # type: ignore

    def start_tool_call(self, openai_index: int, tool_call_id: str, name: str) -> int:
        """Start a new tool call block, return its content block index.

        Tool call blocks are emitted last per content ordering rules.
        """
        if openai_index not in self.tool_calls:
            content_index = self.content_block_count
            self.tool_calls[openai_index] = ToolCallState(
                index=openai_index,
                tool_call_id=tool_call_id,
                name=name,
                content_block_index=content_index,
            )
            self.tool_call_to_content_index[openai_index] = content_index
            self.content_block_count += 1
        return self.tool_call_to_content_index[openai_index]

    def append_tool_arguments(self, openai_index: int, arguments_chunk: str) -> None:
        """Append arguments to an existing tool call."""
        if openai_index in self.tool_calls:
            self.tool_calls[openai_index].arguments_buffer += arguments_chunk

    def complete_tool_call(self, openai_index: int) -> None:
        """Mark a tool call as complete."""
        if openai_index in self.tool_calls:
            self.tool_calls[openai_index].is_complete = True

    def get_tool_call(self, openai_index: int) -> Optional[ToolCallState]:
        """Get tool call state by OpenAI index."""
        return self.tool_calls.get(openai_index)

    def get_all_content_block_indices(self) -> List[int]:
        """Get all active content block indices for closing.

        Returns indices in the order they should be closed:
        thinking, signature, text, tool_use blocks.
        """
        indices = []
        if self.thinking_block_started and self.thinking_block_index is not None:
            indices.append(self.thinking_block_index)
        if self.signature_block_started and self.signature_block_index is not None:
            indices.append(self.signature_block_index)
        if self.text_block_started and self.text_block_index is not None:
            indices.append(self.text_block_index)
        for tc in self.tool_calls.values():
            indices.append(tc.content_block_index)
        return sorted(indices)

    def is_any_block_started(self) -> bool:
        """Check if any content block has been started."""
        return (
            self.thinking_block_started
            or self.signature_block_started
            or self.text_block_started
            or len(self.tool_calls) > 0
        )

    def accumulate_usage(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Accumulate token usage from streaming chunks."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
