"""Unit tests for StreamState management.

Covers content block tracking, tool call lifecycle, and state reset.
"""

import pytest

from src.formats.unified import StreamState, StreamPhase


class TestStreamStateContentBlocks:
    """Tests for content block index management."""

    def test_content_block_indices_monotonic(self):
        """Content block indices should be monotonically increasing."""
        state = StreamState(model="gpt-o3", original_model="claude-3-5")

        thinking_idx = state.start_thinking_block()
        signature_idx = state.start_signature_block()
        text_idx = state.start_text_block()
        tool_idx_0 = state.start_tool_call(0, "call_0", "tool0")
        tool_idx_1 = state.start_tool_call(1, "call_1", "tool1")

        assert thinking_idx < signature_idx < text_idx < tool_idx_0 < tool_idx_1
        assert state.content_block_count == 5

    def test_get_all_content_block_indices_sorted(self):
        """All content block indices should be returned sorted."""
        state = StreamState(model="gpt-o3", original_model="claude-3-5")

        state.start_text_block()
        state.start_thinking_block()  # Will get higher index
        state.start_tool_call(0, "call_0", "tool0")

        indices = state.get_all_content_block_indices()
        assert indices == sorted(indices)
        assert len(indices) == 3

    def test_thinking_block_idempotent(self):
        """Starting thinking block multiple times should return same index."""
        state = StreamState(model="gpt-o3", original_model="claude-3-5")

        idx1 = state.start_thinking_block()
        idx2 = state.start_thinking_block()

        assert idx1 == idx2
        assert state.content_block_count == 1

    def test_text_block_idempotent(self):
        """Starting text block multiple times should return same index."""
        state = StreamState(model="gpt-o3", original_model="claude-3-5")

        idx1 = state.start_text_block()
        idx2 = state.start_text_block()

        assert idx1 == idx2
        assert state.content_block_count == 1

    def test_signature_block_idempotent(self):
        """Starting signature block multiple times should return same index."""
        state = StreamState(model="gpt-o3", original_model="claude-3-5")

        idx1 = state.start_signature_block()
        idx2 = state.start_signature_block()

        assert idx1 == idx2
        assert state.content_block_count == 1


class TestStreamStateToolCalls:
    """Tests for tool call state management."""

    def test_tool_call_lifecycle(self):
        """Tool call should track id, name, arguments, and completion."""
        state = StreamState(model="gpt-o3", original_model="claude-3-5")

        content_idx = state.start_tool_call(0, "call_0", "get_weather")
        assert content_idx in state.get_all_content_block_indices()

        state.append_tool_arguments(0, '{"location":')
        state.append_tool_arguments(0, ' "NYC"}')

        tool_state = state.get_tool_call(0)
        assert tool_state is not None
        assert tool_state.tool_call_id == "call_0"
        assert tool_state.name == "get_weather"
        assert tool_state.arguments_buffer == '{"location": "NYC"}'
        assert tool_state.content_block_index == content_idx
        assert tool_state.is_complete is False

        state.complete_tool_call(0)
        assert state.get_tool_call(0).is_complete is True

    def test_multiple_tool_calls(self):
        """Multiple tool calls should be tracked independently."""
        state = StreamState(model="gpt-o3", original_model="claude-3-5")

        idx0 = state.start_tool_call(0, "call_0", "tool0")
        idx1 = state.start_tool_call(1, "call_1", "tool1")

        state.append_tool_arguments(0, '{"a": 1}')
        state.append_tool_arguments(1, '{"b": 2}')

        assert state.get_tool_call(0).arguments_buffer == '{"a": 1}'
        assert state.get_tool_call(1).arguments_buffer == '{"b": 2}'
        assert idx0 != idx1

    def test_tool_call_idempotent(self):
        """Starting same tool call index twice should return same content index."""
        state = StreamState(model="gpt-o3", original_model="claude-3-5")

        idx1 = state.start_tool_call(0, "call_0", "tool0")
        idx2 = state.start_tool_call(0, "call_0", "tool0")

        assert idx1 == idx2
        assert state.content_block_count == 1

    def test_get_nonexistent_tool_call(self):
        """Getting nonexistent tool call should return None."""
        state = StreamState(model="gpt-o3", original_model="claude-3-5")

        assert state.get_tool_call(999) is None

    def test_append_to_nonexistent_tool_call(self):
        """Appending to nonexistent tool call should be a no-op."""
        state = StreamState(model="gpt-o3", original_model="claude-3-5")

        state.append_tool_arguments(999, "test")
        assert state.get_tool_call(999) is None


class TestStreamStateReset:
    """Tests for state reset behavior."""

    def test_reset_clears_state_preserves_model(self):
        """Reset should clear all state but preserve model info."""
        state = StreamState(model="gpt-o3", original_model="claude-3-5")

        state.start_thinking_block()
        state.start_text_block()
        state.start_tool_call(0, "call_0", "tool0")
        state.accumulate_usage(input_tokens=5, output_tokens=7)
        state.sent_message_start = True
        state.sent_message_stop = True
        state.phase = StreamPhase.CONTENT_STREAMING

        state.reset()

        # stream_id regenerated (starts with msg_)
        assert state.stream_id.startswith("msg_")
        assert state.phase == StreamPhase.NOT_STARTED
        assert state.content_block_count == 0
        assert state.text_block_index is None
        assert state.thinking_block_index is None
        assert state.signature_block_index is None
        assert state.tool_calls == {}
        assert state.tool_call_to_content_index == {}
        assert state.input_tokens == 0
        assert state.output_tokens == 0
        assert state.sent_message_start is False
        assert state.sent_message_stop is False

        # Model metadata should remain unchanged
        assert state.model == "gpt-o3"
        assert state.original_model == "claude-3-5"


class TestStreamStateHelpers:
    """Tests for helper methods."""

    def test_is_any_block_started_false_initially(self):
        """No blocks started initially."""
        state = StreamState(model="gpt-o3", original_model="claude-3-5")
        assert state.is_any_block_started() is False

    def test_is_any_block_started_after_thinking(self):
        """Thinking block should trigger is_any_block_started."""
        state = StreamState(model="gpt-o3", original_model="claude-3-5")
        state.start_thinking_block()
        assert state.is_any_block_started() is True

    def test_is_any_block_started_after_text(self):
        """Text block should trigger is_any_block_started."""
        state = StreamState(model="gpt-o3", original_model="claude-3-5")
        state.start_text_block()
        assert state.is_any_block_started() is True

    def test_is_any_block_started_after_tool(self):
        """Tool call should trigger is_any_block_started."""
        state = StreamState(model="gpt-o3", original_model="claude-3-5")
        state.start_tool_call(0, "call_0", "tool0")
        assert state.is_any_block_started() is True

    def test_is_any_block_started_after_reset(self):
        """Reset should clear is_any_block_started."""
        state = StreamState(model="gpt-o3", original_model="claude-3-5")
        state.start_text_block()
        assert state.is_any_block_started() is True
        state.reset()
        assert state.is_any_block_started() is False

    def test_accumulate_usage_adds_tokens(self):
        """Usage accumulation should add tokens correctly."""
        state = StreamState(model="gpt-o3", original_model="claude-3-5")

        state.accumulate_usage(input_tokens=3, output_tokens=5)
        state.accumulate_usage(input_tokens=2, output_tokens=4)

        assert state.input_tokens == 5
        assert state.output_tokens == 9


class TestStreamStatePhases:
    """Tests for phase tracking."""

    def test_initial_phase(self):
        """Initial phase should be NOT_STARTED."""
        state = StreamState(model="gpt-o3", original_model="claude-3-5")
        assert state.phase == StreamPhase.NOT_STARTED

    def test_phase_transitions(self):
        """Phase transitions should work correctly."""
        state = StreamState(model="gpt-o3", original_model="claude-3-5")

        state.phase = StreamPhase.MESSAGE_STARTED
        assert state.phase == StreamPhase.MESSAGE_STARTED

        state.phase = StreamPhase.CONTENT_STREAMING
        assert state.phase == StreamPhase.CONTENT_STREAMING

        state.phase = StreamPhase.TOOL_STREAMING
        assert state.phase == StreamPhase.TOOL_STREAMING

        state.phase = StreamPhase.FINISHED
        assert state.phase == StreamPhase.FINISHED
