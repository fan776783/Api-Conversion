"""
Unified Conversion Exceptions

Exception hierarchy for unified format conversion errors.
"""

from typing import Optional, List


class UnifiedConversionError(Exception):
    """Base exception for unified conversion errors."""

    def __init__(
        self,
        message: str,
        source_format: Optional[str] = None,
        target_format: Optional[str] = None,
    ):
        super().__init__(message)
        self.source_format = source_format
        self.target_format = target_format


class InvalidContentTypeError(UnifiedConversionError):
    """Raised when content type is not recognized or invalid."""

    def __init__(self, content_type: str, **kwargs):
        super().__init__(f"Invalid content type: {content_type}", **kwargs)
        self.content_type = content_type


class MissingRequiredFieldError(UnifiedConversionError):
    """Raised when a required field is missing."""

    def __init__(self, field_name: str, **kwargs):
        super().__init__(f"Missing required field: {field_name}", **kwargs)
        self.field_name = field_name


class StreamStateError(UnifiedConversionError):
    """Raised when streaming state is corrupted or invalid."""

    pass


class ToolCallMismatchError(UnifiedConversionError):
    """Raised when tool_calls don't have matching tool responses."""

    def __init__(self, unmatched_ids: List[str], **kwargs):
        super().__init__(f"Unmatched tool call IDs: {unmatched_ids}", **kwargs)
        self.unmatched_ids = unmatched_ids
