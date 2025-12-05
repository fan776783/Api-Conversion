"""
Unified Chat Types for API Format Conversion

This module provides a unified intermediate layer for converting between
different chat API formats (Anthropic, OpenAI, Gemini).

Key Types:
- UnifiedContentType: Enum of all supported content block types
- UnifiedContent: Unified content block representation
- UnifiedMessage: Unified message representation
- UnifiedChatRequest: Unified chat completion request
- UnifiedChatResponse: Unified chat completion response
- UnifiedUsage: Token usage information
- StreamState: Streaming conversion state management

Usage:
    from src.formats.unified import (
        UnifiedContentType,
        UnifiedContent,
        UnifiedMessage,
        UnifiedChatRequest,
        UnifiedChatResponse,
        StreamState,
    )
"""

from .types import (
    UnifiedContentType,
    UnifiedContent,
    UnifiedMessage,
    UnifiedChatRequest,
    UnifiedChatResponse,
    UnifiedUsage,
)
from .stream_state import (
    StreamPhase,
    ToolCallState,
    StreamState,
)
from .exceptions import (
    UnifiedConversionError,
    InvalidContentTypeError,
    MissingRequiredFieldError,
    StreamStateError,
    ToolCallMismatchError,
)

__all__ = [
    # Types
    "UnifiedContentType",
    "UnifiedContent",
    "UnifiedMessage",
    "UnifiedChatRequest",
    "UnifiedChatResponse",
    "UnifiedUsage",
    # Stream State
    "StreamPhase",
    "ToolCallState",
    "StreamState",
    # Exceptions
    "UnifiedConversionError",
    "InvalidContentTypeError",
    "MissingRequiredFieldError",
    "StreamStateError",
    "ToolCallMismatchError",
]
