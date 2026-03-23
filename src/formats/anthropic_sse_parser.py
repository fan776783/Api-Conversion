"""Anthropic SSE 事件解析工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
import json


@dataclass
class AnthropicSSEEvent:
    """标准化后的 Anthropic SSE 事件。"""

    event_type: str
    payload: Dict[str, Any]
    sse_event: Optional[str] = None

    @property
    def index(self) -> int:
        value = self.payload.get("index", 0)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0


class AnthropicSSEParseError(ValueError):
    """Anthropic SSE 事件解析失败。"""



def parse_anthropic_sse_event(data: Union[str, Dict[str, Any]]) -> AnthropicSSEEvent:
    """将字符串或字典解析为统一的 Anthropic SSE 事件对象。"""
    if isinstance(data, dict):
        payload = dict(data)
        sse_event = payload.get("_sse_event")
        event_type = sse_event or payload.get("type")
        if not event_type:
            raise AnthropicSSEParseError("Anthropic SSE event type is missing")
        return AnthropicSSEEvent(event_type=event_type, payload=payload, sse_event=sse_event)

    if not isinstance(data, str):
        raise AnthropicSSEParseError(f"Unsupported Anthropic SSE payload type: {type(data)!r}")

    event_type: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None

    for raw_line in data.strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("event: "):
            event_type = line[7:].strip()
            continue
        if line.startswith("data: "):
            body = line[6:]
            if body.strip() == "[DONE]":
                continue
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError as exc:
                raise AnthropicSSEParseError(f"Invalid Anthropic SSE JSON payload: {body}") from exc
            if isinstance(parsed, dict):
                payload = parsed
                if not event_type:
                    event_type = parsed.get("type")

    if not payload:
        raise AnthropicSSEParseError("Anthropic SSE payload is empty")
    if not event_type:
        event_type = payload.get("type")
    if not event_type:
        raise AnthropicSSEParseError("Anthropic SSE event type is missing")

    payload = dict(payload)
    if event_type and "_sse_event" not in payload:
        payload["_sse_event"] = event_type

    return AnthropicSSEEvent(event_type=event_type, payload=payload, sse_event=event_type)
