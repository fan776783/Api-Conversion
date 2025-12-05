# Technical Design: Anthropic-to-OpenAI Conversion Refactoring

## 1. Executive Summary

This document describes the architectural refactoring of the Anthropic-to-OpenAI format conversion pipeline in Api-Conversion project. The primary goal is to introduce a **Unified Intermediate Layer** that decouples source and target formats, enabling cleaner separation of concerns and easier support for new features.

### Key Changes

1. **Unified Intermediate Types**: `UnifiedContent`, `UnifiedMessage`, `UnifiedChatRequest`, `UnifiedChatResponse`
2. **StreamState Class**: Dedicated state management for streaming conversions
3. **Format Adapters**: Separate adapter modules for Anthropic, OpenAI (and Gemini)
4. **New Feature Support**: `thinking`/`signature`, `annotations`, `cache_control`, multiple `tool_use`

### Conversion Flow (New Architecture)

```
Anthropic Request  ->  AnthropicAdapter.to_unified()  ->  UnifiedChatRequest
                                                              |
                                                              v
OpenAI Response    <-  OpenAIAdapter.from_unified()   <-  UnifiedChatRequest
                                                              |
                                                              v (streaming)
                                                          StreamState
```

---

## 2. Type Definitions

### 2.1 UnifiedContentType Enum

```python
from enum import Enum

class UnifiedContentType(str, Enum):
    """Unified content block types across all API formats."""
    TEXT = "text"
    THINKING = "thinking"          # Anthropic thinking / OpenAI reasoning_content
    SIGNATURE = "signature"        # Anthropic signature (thinking verification)
    TOOL_USE = "tool_use"          # Tool/function call request
    TOOL_RESULT = "tool_result"    # Tool/function call result
    IMAGE = "image"                # Image content (base64 or URL)
    ANNOTATION = "annotation"      # Anthropic annotations array
    WEB_SEARCH = "web_search"      # Web search tool results (new Anthropic feature)
```

### 2.2 UnifiedContent Dataclass

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

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
    is_error: bool = False

    # Image fields (for IMAGE type)
    image_media_type: Optional[str] = None
    image_data: Optional[str] = None       # base64 encoded
    image_url: Optional[str] = None        # URL reference

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
        ...

    def to_openai(self) -> Dict[str, Any]:
        """Convert to OpenAI message content format."""
        ...

    @classmethod
    def from_anthropic(cls, data: Dict[str, Any]) -> "UnifiedContent":
        """Create from Anthropic content block."""
        ...

    @classmethod
    def from_openai(cls, data: Dict[str, Any], content_type: str = "text") -> "UnifiedContent":
        """Create from OpenAI message content."""
        ...
```

### 2.3 UnifiedMessage Dataclass

```python
from dataclasses import dataclass, field
from typing import Optional, List

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
        """Convert to Anthropic message format.

        Handles:
        - system -> top-level system field (extracted separately)
        - tool -> user with tool_result content blocks
        - Multiple tool_use blocks in assistant messages
        """
        ...

    def to_openai(self) -> Dict[str, Any]:
        """Convert to OpenAI message format.

        Handles:
        - thinking -> reasoning_content field (if supported)
        - tool_use -> tool_calls array
        - tool_result -> role="tool" message
        """
        ...

    @classmethod
    def from_anthropic(cls, data: Dict[str, Any]) -> "UnifiedMessage":
        """Create from Anthropic message."""
        ...

    @classmethod
    def from_openai(cls, data: Dict[str, Any]) -> "UnifiedMessage":
        """Create from OpenAI message."""
        ...
```

### 2.4 UnifiedChatRequest Dataclass

```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

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
    anthropic_extra: Optional[Dict[str, Any]] = field(default_factory=dict)
    openai_extra: Optional[Dict[str, Any]] = field(default_factory=dict)

    def to_anthropic(self) -> Dict[str, Any]:
        """Convert to Anthropic API request format."""
        ...

    def to_openai(self) -> Dict[str, Any]:
        """Convert to OpenAI API request format."""
        ...

    @classmethod
    def from_anthropic(cls, data: Dict[str, Any]) -> "UnifiedChatRequest":
        """Create from Anthropic API request."""
        ...

    @classmethod
    def from_openai(cls, data: Dict[str, Any]) -> "UnifiedChatRequest":
        """Create from OpenAI API request."""
        ...
```

### 2.5 UnifiedChatResponse Dataclass

```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class UnifiedUsage:
    """Token usage information."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: Optional[int] = None   # Anthropic prompt caching
    cache_creation_input_tokens: Optional[int] = None

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
        ...

    def to_openai(self) -> Dict[str, Any]:
        """Convert to OpenAI API response format."""
        ...

    @classmethod
    def from_anthropic(cls, data: Dict[str, Any]) -> "UnifiedChatResponse":
        """Create from Anthropic API response."""
        ...

    @classmethod
    def from_openai(cls, data: Dict[str, Any]) -> "UnifiedChatResponse":
        """Create from OpenAI API response."""
        ...
```

### 2.6 StreamState Class

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
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
        self.input_tokens = 0
        self.output_tokens = 0
        self.sent_message_start = False
        self.sent_message_stop = False

    def start_text_block(self) -> int:
        """Start a new text content block, return its index."""
        if not self.text_block_started:
            self.text_block_index = self.content_block_count
            self.content_block_count += 1
            self.text_block_started = True
        return self.text_block_index

    def start_thinking_block(self) -> int:
        """Start a new thinking content block, return its index."""
        if not self.thinking_block_started:
            self.thinking_block_index = self.content_block_count
            self.content_block_count += 1
            self.thinking_block_started = True
        return self.thinking_block_index

    def start_tool_call(self, openai_index: int, tool_call_id: str, name: str) -> int:
        """Start a new tool call block, return its content block index."""
        if openai_index not in self.tool_calls:
            content_index = self.content_block_count
            self.tool_calls[openai_index] = ToolCallState(
                index=openai_index,
                tool_call_id=tool_call_id,
                name=name,
                content_block_index=content_index
            )
            self.tool_call_to_content_index[openai_index] = content_index
            self.content_block_count += 1
        return self.tool_call_to_content_index[openai_index]

    def append_tool_arguments(self, openai_index: int, arguments_chunk: str) -> None:
        """Append arguments to an existing tool call."""
        if openai_index in self.tool_calls:
            self.tool_calls[openai_index].arguments_buffer += arguments_chunk

    def get_all_content_block_indices(self) -> List[int]:
        """Get all active content block indices for closing."""
        indices = []
        if self.thinking_block_started and self.thinking_block_index is not None:
            indices.append(self.thinking_block_index)
        if self.text_block_started and self.text_block_index is not None:
            indices.append(self.text_block_index)
        for tc in self.tool_calls.values():
            indices.append(tc.content_block_index)
        return sorted(indices)
```

---

## 3. Conversion Pipeline Architecture

### 3.1 Request Conversion: Anthropic -> Unified -> OpenAI

```
Anthropic Request
      |
      v
AnthropicAdapter.request_to_unified(data: Dict) -> UnifiedChatRequest
      |
      | - Extract top-level "system" -> unified.system
      | - Convert messages[] -> unified.messages[]
      |   - Each content block -> UnifiedContent
      |   - tool_use -> UnifiedContent(type=TOOL_USE)
      |   - tool_result -> UnifiedContent(type=TOOL_RESULT)
      | - Extract thinking.type/budget_tokens -> unified.thinking_*
      | - Extract tools[] -> unified.tools
      | - Preserve cache_control fields
      |
      v
UnifiedChatRequest
      |
      v
OpenAIAdapter.unified_to_request(unified: UnifiedChatRequest) -> Dict
      |
      | - unified.system -> messages[0] with role="system"
      | - unified.messages[] -> OpenAI messages[]
      |   - UnifiedContent(TOOL_USE) -> message.tool_calls[]
      |   - UnifiedContent(TOOL_RESULT) -> message with role="tool"
      | - thinking_enabled + budget_tokens -> reasoning_effort mapping
      | - unified.tools -> OpenAI tools[] format
      | - Validate tool_calls have matching tool responses
      |
      v
OpenAI Request (Dict)
```

### 3.2 Response Conversion: OpenAI -> Unified -> Anthropic

```
OpenAI Response
      |
      v
OpenAIAdapter.response_to_unified(data: Dict, original_model: str) -> UnifiedChatResponse
      |
      | - Extract choices[0].message.content -> UnifiedContent(TEXT)
      | - Extract choices[0].message.reasoning_content -> UnifiedContent(THINKING)
      | - Extract choices[0].message.tool_calls[] -> UnifiedContent(TOOL_USE)[]
      | - Map finish_reason -> stop_reason
      | - Extract usage -> UnifiedUsage
      |
      v
UnifiedChatResponse
      |
      v
AnthropicAdapter.unified_to_response(unified: UnifiedChatResponse) -> Dict
      |
      | - Build content[] array from unified.content
      |   - UnifiedContent(THINKING) -> {"type": "thinking", "thinking": ...}
      |   - UnifiedContent(TEXT) -> {"type": "text", "text": ...}
      |   - UnifiedContent(TOOL_USE) -> {"type": "tool_use", ...}
      | - Map stop_reason: "stop" -> "end_turn", "tool_calls" -> "tool_use"
      | - Format usage: prompt_tokens -> input_tokens, etc.
      | - Use original_model for model field
      |
      v
Anthropic Response (Dict)
```

### 3.3 Streaming Flow with StreamState

```
                    OpenAI SSE Stream
                          |
                          v
+--------------------------------------------------------------+
|                    StreamState Instance                       |
|  - stream_id: "msg_1733..."                                  |
|  - phase: NOT_STARTED -> MESSAGE_STARTED -> CONTENT_STREAMING |
|  - content_block_count: 0 -> 1 -> 2 -> ...                   |
|  - tool_calls: {0: ToolCallState(...)}                       |
+--------------------------------------------------------------+
                          |
    For each SSE chunk:   |
                          v
+--------------------------------------------------------------+
|  OpenAI chunk: {"choices":[{"delta":{"role":"assistant"}}]}  |
|        |                                                      |
|        v                                                      |
|  if phase == NOT_STARTED and has_role:                       |
|      -> emit event: message_start                            |
|      -> phase = MESSAGE_STARTED                              |
|      -> sent_message_start = True                            |
+--------------------------------------------------------------+
                          |
                          v
+--------------------------------------------------------------+
|  OpenAI chunk: {"choices":[{"delta":{"content":"Hello"}}]}   |
|        |                                                      |
|        v                                                      |
|  if not text_block_started:                                  |
|      -> emit event: content_block_start (index=0, type=text) |
|      -> text_block_started = True                            |
|  -> emit event: content_block_delta (text_delta)             |
+--------------------------------------------------------------+
                          |
                          v
+--------------------------------------------------------------+
|  OpenAI chunk: {"choices":[{"delta":{"tool_calls":[...]}}]}  |
|        |                                                      |
|        v                                                      |
|  if tool_call.index not in tool_calls:                       |
|      -> start_tool_call(index, id, name)                     |
|      -> emit event: content_block_start (type=tool_use)      |
|  -> append_tool_arguments(index, arguments_chunk)            |
|  -> emit event: content_block_delta (input_json_delta)       |
+--------------------------------------------------------------+
                          |
                          v
+--------------------------------------------------------------+
|  OpenAI chunk: {"choices":[{"finish_reason":"stop"}]}        |
|        |                                                      |
|        v                                                      |
|  -> emit content_block_stop for all active blocks            |
|  -> emit message_delta with stop_reason                      |
|  -> emit message_stop                                        |
|  -> phase = FINISHED                                         |
+--------------------------------------------------------------+
                          |
                          v
                   Anthropic SSE Events
```

---

## 4. New Feature Support

### 4.1 Thinking/Signature (reasoning_content)

**Current State**:
- Anthropic `thinking` blocks are extracted from `<thinking>` tags in OpenAI response text
- No native `reasoning_content` field support

**New Implementation**:

```python
# OpenAI Response with reasoning_content (new field)
{
    "choices": [{
        "message": {
            "role": "assistant",
            "content": "The answer is 4",
            "reasoning_content": "Let me think step by step...",  # New field
            "signature": "..."  # Verification signature
        }
    }]
}

# Conversion to Anthropic
def openai_reasoning_to_unified(message: Dict) -> List[UnifiedContent]:
    content_blocks = []

    # 1. Handle reasoning_content -> thinking block
    if reasoning := message.get("reasoning_content"):
        content_blocks.append(UnifiedContent(
            type=UnifiedContentType.THINKING,
            text=reasoning
        ))

    # 2. Handle signature -> signature block (if present)
    if signature := message.get("signature"):
        content_blocks.append(UnifiedContent(
            type=UnifiedContentType.SIGNATURE,
            text=signature
        ))

    # 3. Handle main content
    if content := message.get("content"):
        content_blocks.append(UnifiedContent(
            type=UnifiedContentType.TEXT,
            text=content
        ))

    return content_blocks

# Streaming: reasoning_content comes as separate delta field
def handle_openai_streaming_reasoning(chunk: Dict, state: StreamState) -> List[str]:
    events = []
    delta = chunk.get("choices", [{}])[0].get("delta", {})

    # New: Handle reasoning_content delta
    if reasoning_delta := delta.get("reasoning_content"):
        if not state.thinking_block_started:
            idx = state.start_thinking_block()
            events.append(format_content_block_start(idx, "thinking"))
        events.append(format_thinking_delta(state.thinking_block_index, reasoning_delta))

    return events
```

### 4.2 Annotations Array

**Anthropic Response with Annotations**:
```json
{
    "content": [
        {
            "type": "text",
            "text": "Here is some information...",
            "annotations": [
                {
                    "type": "citation",
                    "start_char_index": 0,
                    "end_char_index": 23,
                    "url": "https://example.com"
                }
            ]
        }
    ]
}
```

**Implementation**:

```python
@dataclass
class UnifiedContent:
    # ... existing fields ...
    annotations: Optional[List[Dict[str, Any]]] = None

def unified_to_anthropic_content(content: UnifiedContent) -> Dict:
    result = {"type": content.type.value}

    if content.type == UnifiedContentType.TEXT:
        result["text"] = content.text
        # Preserve annotations if present
        if content.annotations:
            result["annotations"] = content.annotations

    return result

def unified_to_openai_content(content: UnifiedContent) -> Dict:
    # OpenAI doesn't support annotations, store in metadata or drop
    result = {"type": "text", "text": content.text}
    # Optionally: store in raw_data for round-trip preservation
    return result
```

### 4.3 Cache Control Passthrough

**Anthropic Request with cache_control**:
```json
{
    "system": [
        {
            "type": "text",
            "text": "You are a helpful assistant.",
            "cache_control": {"type": "ephemeral"}
        }
    ],
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Hello",
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        }
    ]
}
```

**Implementation Strategy**:

```python
# Preserve cache_control in UnifiedContent
@dataclass
class UnifiedContent:
    cache_control: Optional[Dict[str, str]] = None

# Passthrough for Anthropic -> Anthropic
def unified_to_anthropic_content(content: UnifiedContent) -> Dict:
    result = {"type": content.type.value, "text": content.text}
    if content.cache_control:
        result["cache_control"] = content.cache_control
    return result

# For Anthropic -> OpenAI: cache_control is stripped (OpenAI doesn't support it)
# But preserved in anthropic_extra for potential round-trip

@dataclass
class UnifiedChatRequest:
    anthropic_extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_anthropic(cls, data: Dict) -> "UnifiedChatRequest":
        # Preserve cache_control settings for potential passthrough
        request = cls(...)
        if cache_settings := data.get("cache_control"):
            request.anthropic_extra["cache_control"] = cache_settings
        return request
```

### 4.4 Multiple Tool Use Blocks

**Anthropic allows multiple tool_use in one assistant message**:
```json
{
    "role": "assistant",
    "content": [
        {"type": "tool_use", "id": "tool_1", "name": "get_weather", "input": {"city": "Paris"}},
        {"type": "tool_use", "id": "tool_2", "name": "get_time", "input": {"timezone": "CET"}}
    ]
}
```

**Current Limitation**: Code only handles first tool_use block.

**New Implementation**:

```python
def anthropic_message_to_unified(msg: Dict) -> UnifiedMessage:
    content_blocks = []

    for item in msg.get("content", []):
        if item.get("type") == "tool_use":
            content_blocks.append(UnifiedContent(
                type=UnifiedContentType.TOOL_USE,
                tool_use_id=item.get("id"),
                tool_name=item.get("name"),
                tool_input=item.get("input", {})
            ))
        # ... handle other types

    return UnifiedMessage(role=msg["role"], content=content_blocks)

def unified_to_openai_message(msg: UnifiedMessage) -> Dict:
    tool_calls = []
    text_content = None

    for content in msg.content:
        if content.type == UnifiedContentType.TOOL_USE:
            tool_calls.append({
                "id": content.tool_use_id,
                "type": "function",
                "function": {
                    "name": content.tool_name,
                    "arguments": json.dumps(content.tool_input, ensure_ascii=False)
                }
            })
        elif content.type == UnifiedContentType.TEXT:
            text_content = content.text

    result = {"role": "assistant"}
    if tool_calls:
        result["tool_calls"] = tool_calls
        result["content"] = text_content  # Can be None
    else:
        result["content"] = text_content or ""

    return result
```

**Streaming Multiple Tool Calls**:

```python
def handle_openai_streaming_tool_calls(chunk: Dict, state: StreamState) -> List[str]:
    events = []
    tool_calls = chunk.get("choices", [{}])[0].get("delta", {}).get("tool_calls", [])

    for tc in tool_calls:
        if not tc:
            continue

        tc_index = tc.get("index", 0)

        # Check if this is a new tool call
        if tc_index not in state.tool_calls:
            tool_id = tc.get("id", f"call_{int(time.time())}_{tc_index}")
            tool_name = tc.get("function", {}).get("name", f"tool_{tc_index}")

            # Start new tool_use content block
            content_idx = state.start_tool_call(tc_index, tool_id, tool_name)
            events.append(format_tool_use_start(content_idx, tool_id, tool_name))

        # Append arguments if present
        if args := tc.get("function", {}).get("arguments"):
            state.append_tool_arguments(tc_index, args)
            content_idx = state.tool_call_to_content_index[tc_index]
            events.append(format_input_json_delta(content_idx, args))

    return events
```

---

## 5. File Structure

### New Files to Create

```
src/formats/unified/
    __init__.py
    types.py                    # UnifiedContentType, UnifiedContent, UnifiedMessage,
                               # UnifiedChatRequest, UnifiedChatResponse, UnifiedUsage
    stream_state.py            # StreamState, StreamPhase, ToolCallState

src/formats/unified/adapters/
    __init__.py
    base_adapter.py            # BaseFormatAdapter abstract class
    anthropic_adapter.py       # AnthropicAdapter (to/from unified)
    openai_adapter.py          # OpenAIAdapter (to/from unified)
    gemini_adapter.py          # GeminiAdapter (to/from unified) - future
```

### Files to Modify

```
src/formats/
    anthropic_converter.py     # Refactor to use unified types internally
    openai_converter.py        # Refactor to use unified types internally
    converter_factory.py       # Add USE_UNIFIED_PIPELINE environment switch
    base_converter.py          # Add unified type imports (optional)

src/api/
    unified_api.py             # Update streaming handler to use StreamState
```

### Complete File Tree After Refactoring

```
src/formats/
    __init__.py                # Updated exports
    base_converter.py          # Unchanged (backward compat)
    anthropic_converter.py     # Modified: uses unified internally
    openai_converter.py        # Modified: uses unified internally
    gemini_converter.py        # Modified: uses unified internally
    converter_factory.py       # Modified: USE_UNIFIED_PIPELINE switch
    unified/
        __init__.py            # Exports: UnifiedContent, UnifiedMessage, etc.
        types.py               # ~300 lines: All type definitions
        stream_state.py        # ~150 lines: StreamState class
        adapters/
            __init__.py        # Exports: get_adapter()
            base_adapter.py    # ~50 lines: Abstract base
            anthropic_adapter.py  # ~400 lines: Anthropic conversion
            openai_adapter.py     # ~400 lines: OpenAI conversion
            gemini_adapter.py     # ~300 lines: Gemini conversion (Phase 2)
```

---

## 6. Implementation Plan

### Phase 1: Foundation (Core Types)

| Task | Description | Dependencies | Est. Hours |
|------|-------------|--------------|------------|
| 1.1 | Create `src/formats/unified/` directory structure | None | 0.5 |
| 1.2 | Implement `types.py` - UnifiedContentType enum | None | 1 |
| 1.3 | Implement `types.py` - UnifiedContent dataclass | 1.2 | 2 |
| 1.4 | Implement `types.py` - UnifiedMessage dataclass | 1.3 | 2 |
| 1.5 | Implement `types.py` - UnifiedChatRequest dataclass | 1.4 | 2 |
| 1.6 | Implement `types.py` - UnifiedChatResponse dataclass | 1.4 | 1.5 |
| 1.7 | Implement `stream_state.py` - StreamState class | None | 2 |
| 1.8 | Unit tests for all types | 1.2-1.7 | 3 |

**Phase 1 Deliverable**: Unified types module with full test coverage.

### Phase 2: Adapters (Request/Response Conversion)

| Task | Description | Dependencies | Est. Hours |
|------|-------------|--------------|------------|
| 2.1 | Implement `base_adapter.py` - abstract interface | Phase 1 | 1 |
| 2.2 | Implement `anthropic_adapter.py` - request_to_unified | 2.1 | 4 |
| 2.3 | Implement `anthropic_adapter.py` - unified_to_response | 2.2 | 3 |
| 2.4 | Implement `openai_adapter.py` - unified_to_request | 2.1 | 4 |
| 2.5 | Implement `openai_adapter.py` - response_to_unified | 2.4 | 3 |
| 2.6 | Integration tests for Anthropic->Unified->OpenAI flow | 2.2, 2.4 | 3 |
| 2.7 | Integration tests for OpenAI->Unified->Anthropic flow | 2.3, 2.5 | 3 |

**Phase 2 Deliverable**: Working non-streaming conversion pipeline.

### Phase 3: Streaming Integration

| Task | Description | Dependencies | Est. Hours |
|------|-------------|--------------|------------|
| 3.1 | Add streaming methods to OpenAIAdapter | Phase 2 | 4 |
| 3.2 | Add streaming methods to AnthropicAdapter | Phase 2 | 4 |
| 3.3 | Integrate StreamState with anthropic_converter.py | 3.1, 3.2 | 3 |
| 3.4 | Integrate StreamState with converter_factory.py | 3.3 | 2 |
| 3.5 | Update unified_api.py streaming handler | 3.4 | 2 |
| 3.6 | End-to-end streaming tests | 3.5 | 4 |

**Phase 3 Deliverable**: Full streaming support with StreamState.

### Phase 4: New Features

| Task | Description | Dependencies | Est. Hours |
|------|-------------|--------------|------------|
| 4.1 | Add reasoning_content support to OpenAIAdapter | Phase 3 | 2 |
| 4.2 | Add thinking/signature streaming support | 4.1 | 3 |
| 4.3 | Add annotations passthrough | Phase 3 | 2 |
| 4.4 | Add cache_control passthrough | Phase 3 | 2 |
| 4.5 | Add multiple tool_use support | Phase 3 | 2 |
| 4.6 | Feature tests for all new capabilities | 4.1-4.5 | 4 |

**Phase 4 Deliverable**: All new features working.

### Phase 5: Migration & Cleanup

| Task | Description | Dependencies | Est. Hours |
|------|-------------|--------------|------------|
| 5.1 | Add USE_UNIFIED_PIPELINE env var to converter_factory | Phase 4 | 1 |
| 5.2 | Refactor anthropic_converter.py to use unified internally | 5.1 | 4 |
| 5.3 | Refactor openai_converter.py to use unified internally | 5.1 | 4 |
| 5.4 | Full regression test suite | 5.2, 5.3 | 4 |
| 5.5 | Documentation update | 5.4 | 2 |
| 5.6 | Performance benchmarking | 5.4 | 2 |

**Phase 5 Deliverable**: Production-ready unified pipeline.

### Timeline Summary

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 1: Foundation | ~14 hours | 14 hours |
| Phase 2: Adapters | ~21 hours | 35 hours |
| Phase 3: Streaming | ~19 hours | 54 hours |
| Phase 4: New Features | ~15 hours | 69 hours |
| Phase 5: Migration | ~17 hours | 86 hours |

**Total Estimated Effort**: ~86 hours (~2-3 weeks at moderate pace)

---

## 7. Risk Assessment

### High Risk

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Streaming state corruption | Data loss, malformed responses | Medium | StreamState is immutable except through explicit methods; add state validation |
| Backward compatibility break | Existing integrations fail | Medium | USE_UNIFIED_PIPELINE=0 as default; gradual rollout |
| Performance regression | Slower response times | Low-Medium | Benchmark before/after; optimize hot paths |

### Medium Risk

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Incomplete type coverage | Edge cases fail | Medium | Extensive test suite with real API payloads |
| Tool call state race conditions | Malformed tool calls in streaming | Low | Single-threaded streaming; mutex if needed |
| Cache control not preserved | Prompt caching fails | Low | Explicit passthrough in UnifiedChatRequest |

### Low Risk

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Enum value collisions | Type confusion | Very Low | Use string enums with explicit values |
| JSON serialization issues | Invalid API requests | Low | Add serialization tests for all types |

---

## 8. Acceptance Criteria

### Functional Requirements

- [ ] All existing Anthropic->OpenAI conversion tests pass unchanged
- [ ] All existing OpenAI->Anthropic conversion tests pass unchanged
- [ ] Streaming responses produce identical SSE events (modulo timing)
- [ ] Tool calls with multiple functions convert correctly
- [ ] Thinking content appears as proper content blocks
- [ ] Model name preserved correctly in responses
- [ ] Error handling matches existing behavior

### New Feature Requirements

- [ ] `reasoning_content` field converted to `thinking` block
- [ ] `signature` field converted to `signature` block (when present)
- [ ] `annotations` array preserved in Anthropic responses
- [ ] `cache_control` passed through for Anthropic->Anthropic
- [ ] Multiple `tool_use` blocks handled in single message

### Non-Functional Requirements

- [ ] USE_UNIFIED_PIPELINE=0 uses old code path (backward compat)
- [ ] USE_UNIFIED_PIPELINE=1 uses new unified pipeline
- [ ] No performance regression >10% on throughput
- [ ] Streaming latency to first token unchanged
- [ ] Memory usage increase <20% for typical requests

### Test Coverage Requirements

- [ ] Unit tests for UnifiedContent all types: >95% coverage
- [ ] Unit tests for UnifiedMessage: >95% coverage
- [ ] Unit tests for UnifiedChatRequest/Response: >95% coverage
- [ ] Unit tests for StreamState: >95% coverage
- [ ] Integration tests for each conversion direction
- [ ] Streaming integration tests with mock SSE
- [ ] End-to-end tests with real API (manual/CI)

### Documentation Requirements

- [ ] Type definitions documented with examples
- [ ] Conversion flow diagrams updated
- [ ] Migration guide for USE_UNIFIED_PIPELINE
- [ ] API differences noted (if any)

---

## Appendix A: Existing Code Analysis

### Current State Issues

1. **Scattered Streaming State**: Instance attributes like `_streaming_state`, `_gemini_sent_start`, `_gemini_text_started` are spread across converter classes, making state management error-prone.

2. **Tight Coupling**: Direct conversion between formats means N^2 conversion methods as formats grow. Currently: OpenAI<->Anthropic, OpenAI<->Gemini, Anthropic<->Gemini = 6 conversions.

3. **Limited New Feature Support**: `reasoning_content`, `annotations`, `cache_control` not handled.

4. **Single Tool Use**: Only first `tool_use` block is processed.

5. **Code Duplication**: Similar conversion logic repeated across `anthropic_converter.py` (1557 lines) and `openai_converter.py` (1085 lines).

### Reference: Current Streaming State Dictionary

```python
# anthropic_converter.py current implementation
self._streaming_state = {
    'message_id': "msg_<timestamp_ms>",
    'model': original_model,
    'has_started': False,
    'has_text_content_started': False,
    'has_finished': False,
    'content_index': 0,
    'text_content_index': None,
    'tool_calls': {},  # OpenAI tool_call_index -> {...}
    'tool_call_index_to_content_block_index': {},
    'is_closed': False
}
```

This will be replaced by the `StreamState` dataclass for type safety and cleaner state management.

---

## Appendix B: API Format Comparison

### Anthropic Content Block Types

| Type | Description | OpenAI Equivalent |
|------|-------------|-------------------|
| `text` | Plain text | `message.content` |
| `thinking` | Reasoning trace | `message.reasoning_content` |
| `signature` | Thinking signature | `message.signature` |
| `tool_use` | Tool call request | `message.tool_calls[].function` |
| `tool_result` | Tool call result | `role="tool"` message |
| `image` | Image data | `content[].image_url` |

### OpenAI Special Fields

| Field | Description | Anthropic Equivalent |
|-------|-------------|---------------------|
| `reasoning_content` | Extended thinking | `content[].type="thinking"` |
| `signature` | Verification | `content[].type="signature"` |
| `tool_calls` | Function calls | `content[].type="tool_use"` |
| `finish_reason` | Stop reason | `stop_reason` |

### Stop Reason Mapping

| OpenAI | Anthropic |
|--------|-----------|
| `stop` | `end_turn` |
| `length` | `max_tokens` |
| `content_filter` | `stop_sequence` |
| `tool_calls` | `tool_use` |

---

## Appendix C: Content Block Ordering (Codex Review Suggestion)

**IMPORTANT**: When converting OpenAI response to Anthropic format, content blocks MUST follow this order:

```
1. thinking block(s)     ← From reasoning_content
2. signature block       ← From signature field (if present)
3. text block(s)         ← From content field
4. tool_use block(s)     ← From tool_calls array
```

Example conversion:

```python
# OpenAI Response
{
    "choices": [{
        "message": {
            "reasoning_content": "Let me analyze this step by step...",
            "signature": "sig_abc123",
            "content": "The answer is 42.",
            "tool_calls": [...]
        }
    }]
}

# Converted Anthropic Response
{
    "content": [
        {"type": "thinking", "thinking": "Let me analyze this step by step..."},  # 1st
        {"type": "signature", "signature": "sig_abc123"},                           # 2nd
        {"type": "text", "text": "The answer is 42."},                             # 3rd
        {"type": "tool_use", "id": "...", "name": "...", "input": {...}}           # 4th
    ]
}
```

---

## Appendix D: Error Handling Design (Codex Review Suggestion)

### UnifiedConversionError Hierarchy

```python
class UnifiedConversionError(Exception):
    """Base exception for unified conversion errors."""
    def __init__(self, message: str, source_format: str = None, target_format: str = None):
        super().__init__(message)
        self.source_format = source_format
        self.target_format = target_format

class InvalidContentTypeError(UnifiedConversionError):
    """Raised when content type is not recognized or invalid."""
    pass

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
```

### Error Recovery Strategy

| Error Type | Recovery Action |
|------------|-----------------|
| `InvalidContentTypeError` | Log warning, skip unknown block, continue |
| `MissingRequiredFieldError` | Raise to caller (request validation) |
| `StreamStateError` | Reset state, emit error event, continue stream |
| `ToolCallMismatchError` | Remove orphan tool_calls, log warning |

---

## Appendix E: Key Test Scenarios (Codex Review Suggestion)

### E.1 Text Conversion Round-Trip

```python
def test_text_round_trip():
    """Test: Anthropic text -> Unified -> OpenAI -> Unified -> Anthropic"""
    anthropic_msg = {
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello, world!"}]
    }

    unified = AnthropicAdapter.message_to_unified(anthropic_msg)
    assert len(unified.content) == 1
    assert unified.content[0].type == UnifiedContentType.TEXT

    openai_msg = unified.to_openai()
    assert openai_msg["content"] == "Hello, world!"

    # Reverse
    unified2 = OpenAIAdapter.message_to_unified(openai_msg)
    anthropic_msg2 = unified2.to_anthropic()
    assert anthropic_msg2 == anthropic_msg
```

### E.2 Thinking + Text Streaming

```python
def test_thinking_text_streaming():
    """Test: OpenAI reasoning_content + content -> Anthropic thinking + text blocks"""
    state = StreamState()
    events = []

    # Chunk 1: role
    events.extend(openai_chunk_to_anthropic(
        {"choices": [{"delta": {"role": "assistant"}}]}, state
    ))
    assert "message_start" in events[0]

    # Chunk 2: reasoning_content
    events.extend(openai_chunk_to_anthropic(
        {"choices": [{"delta": {"reasoning_content": "Thinking..."}}]}, state
    ))
    assert "content_block_start" in events[-2]  # thinking block
    assert "thinking" in events[-2]

    # Chunk 3: content
    events.extend(openai_chunk_to_anthropic(
        {"choices": [{"delta": {"content": "Answer"}}]}, state
    ))
    assert "content_block_start" in events[-2]  # text block (index=1)

    # Verify order: thinking (index=0) < text (index=1)
    assert state.thinking_block_index == 0
    assert state.text_block_index == 1
```

### E.3 Multiple Tool Use

```python
def test_multiple_tool_use():
    """Test: Anthropic multiple tool_use -> OpenAI multiple tool_calls"""
    anthropic_msg = {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Paris"}},
            {"type": "tool_use", "id": "t2", "name": "get_time", "input": {"tz": "CET"}}
        ]
    }

    unified = AnthropicAdapter.message_to_unified(anthropic_msg)
    assert len(unified.content) == 2

    openai_msg = unified.to_openai()
    assert len(openai_msg["tool_calls"]) == 2
    assert openai_msg["tool_calls"][0]["function"]["name"] == "get_weather"
    assert openai_msg["tool_calls"][1]["function"]["name"] == "get_time"
```

### E.4 Cache Control Passthrough

```python
def test_cache_control_passthrough():
    """Test: Anthropic cache_control preserved in round-trip"""
    anthropic_req = {
        "system": [{"type": "text", "text": "You are helpful.", "cache_control": {"type": "ephemeral"}}],
        "messages": [...]
    }

    unified = UnifiedChatRequest.from_anthropic(anthropic_req)
    assert unified.system_cache_control == {"type": "ephemeral"}

    # When target is Anthropic, preserve cache_control
    anthropic_out = unified.to_anthropic()
    assert anthropic_out["system"][0]["cache_control"] == {"type": "ephemeral"}

    # When target is OpenAI, cache_control is stripped (not supported)
    openai_out = unified.to_openai()
    assert "cache_control" not in str(openai_out)
```

