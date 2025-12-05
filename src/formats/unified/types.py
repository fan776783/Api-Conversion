"""
Unified Chat Types

Core type definitions for the unified intermediate layer.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import json


class UnifiedContentType(str, Enum):
    """Unified content block types across all API formats."""

    TEXT = "text"
    THINKING = "thinking"  # Anthropic thinking / OpenAI reasoning_content
    SIGNATURE = "signature"  # Anthropic signature (thinking verification)
    TOOL_USE = "tool_use"  # Tool/function call request
    TOOL_RESULT = "tool_result"  # Tool/function call result
    IMAGE = "image"  # Image content (base64 or URL)
    ANNOTATION = "annotation"  # Anthropic annotations array
    WEB_SEARCH = "web_search"  # Web search tool results


@dataclass
class UnifiedContent:
    """Unified content block representation.

    Maps to:
    - Anthropic: content[].type
    - OpenAI: message.content / message.tool_calls / message.reasoning_content
    """

    type: UnifiedContentType

    # Text content (for TEXT, THINKING, SIGNATURE types)
    text: Optional[str] = None

    # Tool use fields (for TOOL_USE type)
    tool_use_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None

    # Tool result fields (for TOOL_RESULT type)
    tool_result_content: Optional[str] = None
    tool_use_id_ref: Optional[str] = None  # Reference to tool_use_id for result
    is_error: bool = False

    # Image fields (for IMAGE type)
    image_media_type: Optional[str] = None
    image_data: Optional[str] = None  # base64 encoded
    image_url: Optional[str] = None  # URL reference

    # Annotation fields (for ANNOTATION type)
    annotations: Optional[List[Dict[str, Any]]] = None

    # Web search result (for WEB_SEARCH type)
    web_search_results: Optional[List[Dict[str, Any]]] = None

    # Cache control passthrough (Anthropic prompt caching)
    cache_control: Optional[Dict[str, str]] = None

    # Raw/passthrough data for format-specific fields
    raw_data: Optional[Dict[str, Any]] = field(default_factory=dict)

    def to_anthropic(self) -> Dict[str, Any]:
        """Convert to Anthropic content block format."""
        result: Dict[str, Any] = {"type": self.type.value}

        if self.type == UnifiedContentType.TEXT:
            result["text"] = self.text or ""
            if self.annotations:
                result["annotations"] = self.annotations
        elif self.type == UnifiedContentType.THINKING:
            result["type"] = "thinking"
            result["thinking"] = self.text or ""
        elif self.type == UnifiedContentType.SIGNATURE:
            result["type"] = "signature"
            result["signature"] = self.text or ""
        elif self.type == UnifiedContentType.TOOL_USE:
            result["id"] = self.tool_use_id or ""
            result["name"] = self.tool_name or ""
            result["input"] = self.tool_input or {}
        elif self.type == UnifiedContentType.TOOL_RESULT:
            result["type"] = "tool_result"
            result["tool_use_id"] = self.tool_use_id_ref or ""
            result["content"] = self.tool_result_content or ""
            if self.is_error:
                result["is_error"] = True
        elif self.type == UnifiedContentType.IMAGE:
            result["type"] = "image"
            result["source"] = {
                "type": "base64",
                "media_type": self.image_media_type or "image/png",
                "data": self.image_data or "",
            }

        if self.cache_control:
            result["cache_control"] = self.cache_control

        return result

    def to_openai(self) -> Dict[str, Any]:
        """Convert to OpenAI message content format.

        Note: Different content types map to different OpenAI structures:
        - TEXT -> {"type": "text", "text": "..."}
        - IMAGE -> {"type": "image_url", "image_url": {"url": "..."}}
        - TOOL_USE -> tool_calls array item (handled at message level)
        - TOOL_RESULT -> role="tool" message (handled at message level)
        - THINKING -> reasoning_content field (handled at message level)
        """
        if self.type == UnifiedContentType.TEXT:
            return {"type": "text", "text": self.text or ""}
        elif self.type == UnifiedContentType.IMAGE:
            if self.image_url:
                url = self.image_url
            elif self.image_data:
                url = f"data:{self.image_media_type or 'image/png'};base64,{self.image_data}"
            else:
                url = ""
            return {"type": "image_url", "image_url": {"url": url}}
        elif self.type == UnifiedContentType.TOOL_USE:
            return {
                "id": self.tool_use_id or "",
                "type": "function",
                "function": {
                    "name": self.tool_name or "",
                    "arguments": json.dumps(self.tool_input or {}, ensure_ascii=False),
                },
            }
        # THINKING/SIGNATURE/TOOL_RESULT handled at message level
        return {"type": "text", "text": self.text or ""}

    @classmethod
    def from_anthropic(cls, data: Dict[str, Any]) -> "UnifiedContent":
        """Create from Anthropic content block."""
        block_type = data.get("type", "text")

        if block_type == "text":
            return cls(
                type=UnifiedContentType.TEXT,
                text=data.get("text", ""),
                annotations=data.get("annotations"),
                cache_control=data.get("cache_control"),
                raw_data=data,
            )
        elif block_type == "thinking":
            return cls(
                type=UnifiedContentType.THINKING,
                text=data.get("thinking", ""),
                raw_data=data,
            )
        elif block_type == "signature":
            return cls(
                type=UnifiedContentType.SIGNATURE,
                text=data.get("signature", ""),
                raw_data=data,
            )
        elif block_type == "tool_use":
            return cls(
                type=UnifiedContentType.TOOL_USE,
                tool_use_id=data.get("id", ""),
                tool_name=data.get("name", ""),
                tool_input=data.get("input", {}),
                raw_data=data,
            )
        elif block_type == "tool_result":
            return cls(
                type=UnifiedContentType.TOOL_RESULT,
                tool_use_id_ref=data.get("tool_use_id", ""),
                tool_result_content=data.get("content", ""),
                is_error=data.get("is_error", False),
                raw_data=data,
            )
        elif block_type == "image":
            source = data.get("source", {})
            return cls(
                type=UnifiedContentType.IMAGE,
                image_media_type=source.get("media_type"),
                image_data=source.get("data") if source.get("type") == "base64" else None,
                image_url=source.get("url") if source.get("type") == "url" else None,
                raw_data=data,
            )
        else:
            # Unknown type, preserve as raw
            return cls(
                type=UnifiedContentType.TEXT,
                text=str(data),
                raw_data=data,
            )

    @classmethod
    def from_openai(cls, data: Any, content_type: str = "text") -> "UnifiedContent":
        """Create from OpenAI message content."""
        is_dict = isinstance(data, dict)
        if content_type == "text" or (is_dict and data.get("type") == "text"):
            return cls(
                type=UnifiedContentType.TEXT,
                text=data.get("text", "") if is_dict else str(data),
                raw_data=data if is_dict else {},
            )
        elif content_type == "image_url" or (is_dict and data.get("type") == "image_url"):
            url = data.get("image_url", {}).get("url", "") if is_dict else ""
            if url.startswith("data:"):
                # Parse data URL
                try:
                    header, base64_data = url.split(";base64,", 1)
                    media_type = header.replace("data:", "")
                    return cls(
                        type=UnifiedContentType.IMAGE,
                        image_media_type=media_type,
                        image_data=base64_data,
                        raw_data=data if is_dict else {},
                    )
                except ValueError:
                    pass
            return cls(
                type=UnifiedContentType.IMAGE,
                image_url=url,
                raw_data=data if is_dict else {},
            )
        elif content_type == "tool_call" and is_dict:
            # From tool_calls array item
            func = data.get("function", {})
            args_str = func.get("arguments", "{}")
            try:
                args = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                args = {}
            return cls(
                type=UnifiedContentType.TOOL_USE,
                tool_use_id=data.get("id", ""),
                tool_name=func.get("name", ""),
                tool_input=args,
                raw_data=data,
            )
        elif content_type == "reasoning":
            return cls(
                type=UnifiedContentType.THINKING,
                text=data if isinstance(data, str) else str(data),
            )
        else:
            return cls(
                type=UnifiedContentType.TEXT,
                text=str(data),
            )


@dataclass
class UnifiedMessage:
    """Unified message representation.

    Maps to:
    - Anthropic: messages[].role, messages[].content
    - OpenAI: messages[].role, messages[].content, messages[].tool_calls
    """

    role: str  # "user", "assistant", "system", "tool"
    content: List[UnifiedContent] = field(default_factory=list)

    # OpenAI tool role specific fields
    tool_call_id: Optional[str] = None  # For role="tool" responses

    # Name field (OpenAI function calling)
    name: Optional[str] = None

    # Cache control at message level
    cache_control: Optional[Dict[str, str]] = None

    def to_anthropic(self) -> Dict[str, Any]:
        """Convert to Anthropic message format."""
        # Handle tool result messages -> user message with tool_result content
        if self.role == "tool":
            tool_result_blocks = [
                c.to_anthropic()
                for c in self.content
                if c.type == UnifiedContentType.TOOL_RESULT
            ]
            if not tool_result_blocks:
                # Create tool_result from message
                tool_result_blocks = [
                    {
                        "type": "tool_result",
                        "tool_use_id": self.tool_call_id or "",
                        "content": self.content[0].text if self.content else "",
                    }
                ]
            return {"role": "user", "content": tool_result_blocks}

        result: Dict[str, Any] = {"role": self.role}

        # Convert content blocks
        # Order: thinking -> signature -> text -> tool_use (per tech design)
        thinking_blocks = []
        signature_blocks = []
        text_blocks = []
        tool_blocks = []
        other_blocks = []

        for c in self.content:
            block = c.to_anthropic()
            if c.type == UnifiedContentType.THINKING:
                thinking_blocks.append(block)
            elif c.type == UnifiedContentType.SIGNATURE:
                signature_blocks.append(block)
            elif c.type == UnifiedContentType.TEXT:
                text_blocks.append(block)
            elif c.type == UnifiedContentType.TOOL_USE:
                tool_blocks.append(block)
            else:
                other_blocks.append(block)

        ordered_content = thinking_blocks + signature_blocks + text_blocks + tool_blocks + other_blocks
        result["content"] = ordered_content if ordered_content else [{"type": "text", "text": ""}]

        return result

    def to_openai(self) -> Dict[str, Any]:
        """Convert to OpenAI message format."""
        result: Dict[str, Any] = {"role": self.role}

        # Handle tool response
        if self.role == "tool":
            result["tool_call_id"] = self.tool_call_id or ""
            result["content"] = self.content[0].tool_result_content if self.content else ""
            return result

        # Collect different content types
        text_parts = []
        tool_calls = []
        reasoning_content = None

        for c in self.content:
            if c.type == UnifiedContentType.TEXT:
                text_parts.append(c.text or "")
            elif c.type == UnifiedContentType.THINKING:
                reasoning_content = c.text
            elif c.type == UnifiedContentType.TOOL_USE:
                tool_calls.append(c.to_openai())
            elif c.type == UnifiedContentType.IMAGE:
                # Keep multimodal content
                pass

        # Build content
        if len(self.content) == 1 and self.content[0].type == UnifiedContentType.TEXT:
            result["content"] = self.content[0].text or ""
        elif any(c.type == UnifiedContentType.IMAGE for c in self.content):
            # Multimodal
            result["content"] = [c.to_openai() for c in self.content if c.type in (UnifiedContentType.TEXT, UnifiedContentType.IMAGE)]
        else:
            result["content"] = " ".join(text_parts) if text_parts else None

        # Add tool_calls
        if tool_calls:
            result["tool_calls"] = tool_calls
            if result["content"] is None:
                result["content"] = None  # OpenAI allows null content with tool_calls

        # Add reasoning_content if present (for OpenAI reasoning models)
        if reasoning_content:
            result["reasoning_content"] = reasoning_content

        return result

    @classmethod
    def from_anthropic(cls, data: Dict[str, Any]) -> "UnifiedMessage":
        """Create from Anthropic message."""
        role = data.get("role", "user")
        raw_content = data.get("content", [])

        # Handle string content
        if isinstance(raw_content, str):
            return cls(
                role=role,
                content=[UnifiedContent(type=UnifiedContentType.TEXT, text=raw_content)],
                cache_control=data.get("cache_control"),
            )

        # Handle list content
        content_blocks = []
        for block in raw_content:
            if isinstance(block, str):
                content_blocks.append(UnifiedContent(type=UnifiedContentType.TEXT, text=block))
            elif isinstance(block, dict):
                content_blocks.append(UnifiedContent.from_anthropic(block))

        return cls(
            role=role,
            content=content_blocks,
            cache_control=data.get("cache_control"),
        )

    @classmethod
    def from_openai(cls, data: Dict[str, Any]) -> "UnifiedMessage":
        """Create from OpenAI message."""
        role = data.get("role", "user")

        # Handle tool message
        if role == "tool":
            return cls(
                role="tool",
                tool_call_id=data.get("tool_call_id"),
                content=[
                    UnifiedContent(
                        type=UnifiedContentType.TOOL_RESULT,
                        tool_use_id_ref=data.get("tool_call_id"),
                        tool_result_content=data.get("content", ""),
                    )
                ],
                name=data.get("name"),
            )

        content_blocks = []
        raw_content = data.get("content")

        # Handle string content
        if isinstance(raw_content, str):
            content_blocks.append(UnifiedContent(type=UnifiedContentType.TEXT, text=raw_content))
        elif isinstance(raw_content, list):
            # Multimodal content
            for item in raw_content:
                if isinstance(item, str):
                    content_blocks.append(UnifiedContent(type=UnifiedContentType.TEXT, text=item))
                elif isinstance(item, dict):
                    content_blocks.append(UnifiedContent.from_openai(item, item.get("type", "text")))

        # Handle reasoning_content
        if reasoning := data.get("reasoning_content"):
            content_blocks.insert(0, UnifiedContent(type=UnifiedContentType.THINKING, text=reasoning))

        # Handle tool_calls
        if tool_calls := data.get("tool_calls"):
            for tc in tool_calls:
                content_blocks.append(UnifiedContent.from_openai(tc, "tool_call"))

        return cls(
            role=role,
            content=content_blocks,
            name=data.get("name"),
        )


@dataclass
class UnifiedUsage:
    """Token usage information."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: Optional[int] = None  # Anthropic prompt caching
    cache_creation_input_tokens: Optional[int] = None


@dataclass
class UnifiedChatRequest:
    """Unified chat completion request.

    Contains all parameters needed to make a chat completion request
    in a format-agnostic way.
    """

    model: str
    messages: List[UnifiedMessage] = field(default_factory=list)

    # System message (Anthropic top-level, OpenAI in messages)
    system: Optional[str] = None
    system_cache_control: Optional[Dict[str, str]] = None

    # Generation parameters
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None  # Anthropic only, ignored for OpenAI
    stop_sequences: Optional[List[str]] = None

    # Streaming
    stream: bool = False

    # Tool definitions
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None  # "auto", "none", or specific tool

    # Thinking/Reasoning mode
    thinking_enabled: bool = False
    thinking_budget_tokens: Optional[int] = None

    # OpenAI specific
    reasoning_effort: Optional[str] = None  # "low", "medium", "high"
    max_completion_tokens: Optional[int] = None

    # Passthrough fields (format-specific, preserved as-is)
    anthropic_extra: Dict[str, Any] = field(default_factory=dict)
    openai_extra: Dict[str, Any] = field(default_factory=dict)

    def to_anthropic(self) -> Dict[str, Any]:
        """Convert to Anthropic API request format."""
        result: Dict[str, Any] = {"model": self.model}

        # System message
        if self.system:
            if self.system_cache_control:
                result["system"] = [
                    {"type": "text", "text": self.system, "cache_control": self.system_cache_control}
                ]
            else:
                result["system"] = self.system

        # Messages
        result["messages"] = [m.to_anthropic() for m in self.messages]

        # Parameters
        if self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            result["temperature"] = self.temperature
        if self.top_p is not None:
            result["top_p"] = self.top_p
        if self.top_k is not None:
            result["top_k"] = self.top_k
        if self.stop_sequences:
            result["stop_sequences"] = self.stop_sequences
        if self.stream:
            result["stream"] = True

        # Tools
        if self.tools:
            anthropic_tools = []
            for tool in self.tools:
                if tool.get("type") == "function" and "function" in tool:
                    func = tool["function"]
                    anthropic_tools.append({
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {}),
                    })
                else:
                    anthropic_tools.append(tool)
            result["tools"] = anthropic_tools

        # Thinking mode
        if self.thinking_enabled and self.thinking_budget_tokens:
            result["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget_tokens,
            }

        # Merge extra fields
        result.update(self.anthropic_extra)

        return result

    def to_openai(self) -> Dict[str, Any]:
        """Convert to OpenAI API request format."""
        result: Dict[str, Any] = {"model": self.model}

        # Build messages list
        messages = []

        # Add system message first
        if self.system:
            messages.append({"role": "system", "content": self.system})

        # Add other messages
        for m in self.messages:
            messages.append(m.to_openai())

        result["messages"] = messages

        # Parameters
        if self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            result["temperature"] = self.temperature
        if self.top_p is not None:
            result["top_p"] = self.top_p
        if self.stop_sequences:
            result["stop"] = self.stop_sequences
        if self.stream:
            result["stream"] = True

        # Tools
        if self.tools:
            openai_tools = []
            for tool in self.tools:
                if "type" not in tool:
                    # Anthropic format -> OpenAI format
                    openai_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.get("name", ""),
                            "description": tool.get("description", ""),
                            "parameters": tool.get("input_schema", {}),
                        },
                    })
                else:
                    openai_tools.append(tool)
            result["tools"] = openai_tools

        # Thinking/Reasoning mode
        if self.thinking_enabled:
            if self.reasoning_effort:
                result["reasoning_effort"] = self.reasoning_effort
            if self.max_completion_tokens:
                result["max_completion_tokens"] = self.max_completion_tokens

        # Merge extra fields
        result.update(self.openai_extra)

        return result

    @classmethod
    def from_anthropic(cls, data: Dict[str, Any]) -> "UnifiedChatRequest":
        """Create from Anthropic API request."""
        # Extract system
        system = None
        system_cache_control = None
        raw_system = data.get("system")
        if isinstance(raw_system, str):
            system = raw_system
        elif isinstance(raw_system, list) and raw_system:
            system = raw_system[0].get("text", "")
            system_cache_control = raw_system[0].get("cache_control")

        # Extract messages with special handling for tool_result
        messages: List[UnifiedMessage] = []
        for m in data.get("messages", []):
            if not isinstance(m, dict):
                messages.append(UnifiedMessage.from_anthropic(m))
                continue

            role = m.get("role", "user")
            content = m.get("content", [])

            # Special case: Anthropic encodes tool results as user message with tool_result blocks
            # Normalize to role="tool" messages for consistent OpenAI mapping
            if (
                role == "user"
                and isinstance(content, list)
                and content
                and all(isinstance(block, dict) and block.get("type") == "tool_result" for block in content)
            ):
                for block in content:
                    unified_content = UnifiedContent.from_anthropic(block)
                    messages.append(
                        UnifiedMessage(
                            role="tool",
                            tool_call_id=unified_content.tool_use_id_ref,
                            content=[unified_content],
                            cache_control=m.get("cache_control"),
                        )
                    )
            else:
                messages.append(UnifiedMessage.from_anthropic(m))

        # Extract thinking
        thinking = data.get("thinking", {})
        thinking_enabled = thinking.get("type") == "enabled" if thinking else False
        thinking_budget_tokens = thinking.get("budget_tokens") if thinking else None

        return cls(
            model=data.get("model", ""),
            messages=messages,
            system=system,
            system_cache_control=system_cache_control,
            max_tokens=data.get("max_tokens"),
            temperature=data.get("temperature"),
            top_p=data.get("top_p"),
            top_k=data.get("top_k"),
            stop_sequences=data.get("stop_sequences"),
            stream=data.get("stream", False),
            tools=data.get("tools"),
            tool_choice=data.get("tool_choice"),
            thinking_enabled=thinking_enabled,
            thinking_budget_tokens=thinking_budget_tokens,
        )

    @classmethod
    def from_openai(cls, data: Dict[str, Any]) -> "UnifiedChatRequest":
        """Create from OpenAI API request."""
        # Extract system message from messages
        system = None
        messages = []
        for m in data.get("messages", []):
            if m.get("role") == "system":
                system = m.get("content", "")
            else:
                messages.append(UnifiedMessage.from_openai(m))

        return cls(
            model=data.get("model", ""),
            messages=messages,
            system=system,
            max_tokens=data.get("max_tokens"),
            temperature=data.get("temperature"),
            top_p=data.get("top_p"),
            stop_sequences=data.get("stop") if isinstance(data.get("stop"), list) else ([data["stop"]] if data.get("stop") else None),
            stream=data.get("stream", False),
            tools=data.get("tools"),
            tool_choice=data.get("tool_choice"),
            reasoning_effort=data.get("reasoning_effort"),
            max_completion_tokens=data.get("max_completion_tokens"),
        )


@dataclass
class UnifiedChatResponse:
    """Unified chat completion response.

    Represents a complete (non-streaming) response.
    """

    id: str
    model: str
    content: List[UnifiedContent] = field(default_factory=list)
    stop_reason: Optional[str] = None
    usage: Optional[UnifiedUsage] = None

    # Original model (for response mapping)
    original_model: Optional[str] = None

    def to_anthropic(self) -> Dict[str, Any]:
        """Convert to Anthropic API response format."""
        # Build content blocks in correct order
        thinking_blocks = []
        signature_blocks = []
        text_blocks = []
        tool_blocks = []

        for c in self.content:
            block = c.to_anthropic()
            if c.type == UnifiedContentType.THINKING:
                thinking_blocks.append(block)
            elif c.type == UnifiedContentType.SIGNATURE:
                signature_blocks.append(block)
            elif c.type == UnifiedContentType.TEXT:
                text_blocks.append(block)
            elif c.type == UnifiedContentType.TOOL_USE:
                tool_blocks.append(block)

        ordered_content = thinking_blocks + signature_blocks + text_blocks + tool_blocks

        # Map stop_reason
        stop_reason_map = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "stop_sequence",
        }
        anthropic_stop = stop_reason_map.get(self.stop_reason, self.stop_reason) or "end_turn"

        result: Dict[str, Any] = {
            "id": self.id,
            "type": "message",
            "role": "assistant",
            "model": self.original_model or self.model,
            "content": ordered_content or [{"type": "text", "text": ""}],
            "stop_reason": anthropic_stop,
        }

        if self.usage:
            result["usage"] = {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
            }
            if self.usage.cache_read_input_tokens is not None:
                result["usage"]["cache_read_input_tokens"] = self.usage.cache_read_input_tokens
            if self.usage.cache_creation_input_tokens is not None:
                result["usage"]["cache_creation_input_tokens"] = self.usage.cache_creation_input_tokens

        return result

    def to_openai(self) -> Dict[str, Any]:
        """Convert to OpenAI API response format."""
        import time

        # Collect content
        text_parts = []
        tool_calls = []
        reasoning_content = None

        for c in self.content:
            if c.type == UnifiedContentType.TEXT:
                text_parts.append(c.text or "")
            elif c.type == UnifiedContentType.THINKING:
                reasoning_content = c.text
            elif c.type == UnifiedContentType.TOOL_USE:
                tool_calls.append(c.to_openai())

        message: Dict[str, Any] = {"role": "assistant"}
        message["content"] = " ".join(text_parts) if text_parts else None

        if tool_calls:
            message["tool_calls"] = tool_calls

        if reasoning_content:
            message["reasoning_content"] = reasoning_content

        # Map stop_reason
        stop_reason_map = {
            "end_turn": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
            "stop_sequence": "stop",
        }
        finish_reason = stop_reason_map.get(self.stop_reason, self.stop_reason) or "stop"
        if tool_calls:
            finish_reason = "tool_calls"

        result: Dict[str, Any] = {
            "id": f"chatcmpl-{self.id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.original_model or self.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
        }

        if self.usage:
            result["usage"] = {
                "prompt_tokens": self.usage.input_tokens,
                "completion_tokens": self.usage.output_tokens,
                "total_tokens": self.usage.input_tokens + self.usage.output_tokens,
            }

        return result

    @classmethod
    def from_anthropic(cls, data: Dict[str, Any], original_model: Optional[str] = None) -> "UnifiedChatResponse":
        """Create from Anthropic API response."""
        content_blocks = []
        for block in data.get("content", []):
            content_blocks.append(UnifiedContent.from_anthropic(block))

        usage = None
        if raw_usage := data.get("usage"):
            usage = UnifiedUsage(
                input_tokens=raw_usage.get("input_tokens", 0),
                output_tokens=raw_usage.get("output_tokens", 0),
                cache_read_input_tokens=raw_usage.get("cache_read_input_tokens"),
                cache_creation_input_tokens=raw_usage.get("cache_creation_input_tokens"),
            )

        return cls(
            id=data.get("id", ""),
            model=data.get("model", ""),
            content=content_blocks,
            stop_reason=data.get("stop_reason"),
            usage=usage,
            original_model=original_model or data.get("model"),
        )

    @classmethod
    def from_openai(cls, data: Dict[str, Any], original_model: Optional[str] = None) -> "UnifiedChatResponse":
        """Create from OpenAI API response."""
        content_blocks = []

        if choices := data.get("choices"):
            message = choices[0].get("message", {})

            # Handle reasoning_content first
            if reasoning := message.get("reasoning_content"):
                content_blocks.append(UnifiedContent(type=UnifiedContentType.THINKING, text=reasoning))

            # Handle text content
            if text := message.get("content"):
                content_blocks.append(UnifiedContent(type=UnifiedContentType.TEXT, text=text))

            # Handle tool_calls
            for tc in message.get("tool_calls", []):
                content_blocks.append(UnifiedContent.from_openai(tc, "tool_call"))

            finish_reason = choices[0].get("finish_reason", "stop")
        else:
            finish_reason = "stop"

        usage = None
        if raw_usage := data.get("usage"):
            usage = UnifiedUsage(
                input_tokens=raw_usage.get("prompt_tokens", 0),
                output_tokens=raw_usage.get("completion_tokens", 0),
            )

        return cls(
            id=data.get("id", "").replace("chatcmpl-", ""),
            model=data.get("model", ""),
            content=content_blocks,
            stop_reason=finish_reason,
            usage=usage,
            original_model=original_model or data.get("model"),
        )
