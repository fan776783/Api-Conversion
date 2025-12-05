"""
Base Format Adapter

Abstract base class for format-specific adapters.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generator

from ..types import UnifiedChatRequest, UnifiedChatResponse, UnifiedMessage
from ..stream_state import StreamState


class BaseFormatAdapter(ABC):
    """Abstract base class for format adapters.

    Each format (Anthropic, OpenAI, Gemini) should implement this interface
    to convert to/from the unified intermediate representation.
    """

    @abstractmethod
    def request_to_unified(self, data: Dict[str, Any]) -> UnifiedChatRequest:
        """Convert format-specific request to unified format."""
        pass

    @abstractmethod
    def unified_to_request(self, unified: UnifiedChatRequest) -> Dict[str, Any]:
        """Convert unified format to format-specific request."""
        pass

    @abstractmethod
    def response_to_unified(
        self, data: Dict[str, Any], original_model: str | None = None
    ) -> UnifiedChatResponse:
        """Convert format-specific response to unified format."""
        pass

    @abstractmethod
    def unified_to_response(self, unified: UnifiedChatResponse) -> Dict[str, Any]:
        """Convert unified format to format-specific response."""
        pass

    @abstractmethod
    def message_to_unified(self, data: Dict[str, Any]) -> UnifiedMessage:
        """Convert a single format-specific message to unified format."""
        pass

    @abstractmethod
    def unified_to_message(self, unified: UnifiedMessage) -> Dict[str, Any]:
        """Convert a unified message to format-specific format."""
        pass

    def streaming_chunk_to_unified(
        self, chunk: Dict[str, Any], state: StreamState
    ) -> Generator[Dict[str, Any], None, None]:
        """Convert a streaming chunk to unified SSE events.

        Default implementation raises NotImplementedError.
        Subclasses should override if streaming is supported.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support streaming conversion"
        )
