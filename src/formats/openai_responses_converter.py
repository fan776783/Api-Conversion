"""OpenAI Responses 协议转换器。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import copy
import json
import random
import string
import time

from .base_converter import BaseConverter, ConversionResult
from .anthropic_sse_parser import AnthropicSSEParseError, parse_anthropic_sse_event
from .openai_responses_request_adapter import OpenAIResponsesRequestAdapter


class OpenAIResponsesConverter(BaseConverter):
    """负责 OpenAI Responses 请求与响应转换。"""

    def __init__(self):
        super().__init__()
        self.original_model: Optional[str] = None

    def set_original_model(self, model: str):
        self.original_model = model

    def reset_streaming_state(self):
        for attr in (
            "_response_id",
            "_output_items",
            "_current_item_index",
            "_completed",
            "_message_text",
            "_reasoning_text",
            "_usage",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

    def get_supported_formats(self) -> List[str]:
        return ["openai_responses", "openai_chat_completions", "anthropic", "gemini", "openai"]

    def convert_request(
        self,
        data: Dict[str, Any],
        target_format: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> ConversionResult:
        if target_format in {"openai_responses", "openai"}:
            return ConversionResult(success=True, data=copy.deepcopy(data))
        if target_format in {"openai_chat_completions", "anthropic", "gemini"}:
            adapted = OpenAIResponsesRequestAdapter.adapt(data)
            if target_format == "openai_chat_completions":
                return ConversionResult(success=True, data=adapted)
            return ConversionResult(success=True, data=adapted)
        return ConversionResult(success=False, error=f"Unsupported target format: {target_format}")

    def convert_response(
        self,
        data: Dict[str, Any],
        source_format: str,
        target_format: str,
    ) -> ConversionResult:
        try:
            if source_format == "anthropic":
                return self._convert_from_anthropic_response(data)
            if source_format == "openai":
                return ConversionResult(success=True, data=copy.deepcopy(data))
            return ConversionResult(success=False, error=f"Unsupported source format: {source_format}")
        except Exception as exc:
            self.logger.error(f"Failed to convert {source_format} response to OpenAI Responses: {exc}")
            return ConversionResult(success=False, error=str(exc))

    def _convert_from_anthropic_response(self, data: Dict[str, Any]) -> ConversionResult:
        response_id = data.get("id") or self._ensure_response_id()
        output_items: List[Dict[str, Any]] = []
        output_text_parts: List[str] = []

        for content in data.get("content", []):
            content_type = content.get("type")
            if content_type == "text":
                text = content.get("text", "")
                output_text_parts.append(text)
                output_items.append(
                    {
                        "type": "message",
                        "id": f"msg_{len(output_items)}",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": text,
                            }
                        ],
                    }
                )
            elif content_type == "thinking":
                thinking_text = content.get("thinking", "")
                output_items.append(
                    {
                        "type": "reasoning",
                        "id": f"rs_{len(output_items)}",
                        "status": "completed",
                        "summary": [
                            {
                                "type": "summary_text",
                                "text": thinking_text,
                            }
                        ],
                    }
                )
            elif content_type == "tool_use":
                output_items.append(
                    {
                        "type": "function_call",
                        "id": content.get("id") or f"fc_{len(output_items)}",
                        "call_id": content.get("id") or f"call_{len(output_items)}",
                        "name": content.get("name", ""),
                        "arguments": json.dumps(content.get("input", {}), ensure_ascii=False),
                        "status": "completed",
                    }
                )

        usage = self._convert_usage(data.get("usage") or {})
        result = {
            "id": response_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed",
            "model": self.original_model or data.get("model") or "",
            "output": output_items,
            "output_text": "".join(output_text_parts),
            "usage": usage,
        }
        return ConversionResult(success=True, data=result)

    def _convert_from_anthropic_streaming_chunk(self, data: Dict[str, Any]) -> ConversionResult:
        try:
            parsed = parse_anthropic_sse_event(data)
        except AnthropicSSEParseError:
            return ConversionResult(success=True, data=[])

        response_id = self._ensure_response_id()
        payload = parsed.payload
        events: List[str] = []

        if not hasattr(self, "_output_items"):
            self._output_items = {}
        if not hasattr(self, "_message_text"):
            self._message_text = []
        if not hasattr(self, "_reasoning_text"):
            self._reasoning_text = {}
        if not hasattr(self, "_usage"):
            self._usage = {}

        if parsed.event_type == "message_start":
            message = payload.get("message", {})
            self._usage = self._convert_usage(message.get("usage") or {})
            events.append(
                self._sse(
                    "response.created",
                    {
                        "type": "response.created",
                        "response": {
                            "id": response_id,
                            "object": "response",
                            "created_at": int(time.time()),
                            "status": "in_progress",
                            "model": self.original_model or message.get("model") or "",
                            "output": [],
                        },
                    },
                )
            )

        elif parsed.event_type == "content_block_start":
            index = parsed.index
            block = payload.get("content_block", {})
            block_type = block.get("type")
            item = self._build_output_item(index, block)
            self._output_items[index] = item
            events.append(
                self._sse(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "response_id": response_id,
                        "output_index": index,
                        "item": item,
                    },
                )
            )
            if block_type == "text":
                events.append(
                    self._sse(
                        "response.content_part.added",
                        {
                            "type": "response.content_part.added",
                            "response_id": response_id,
                            "output_index": index,
                            "content_index": 0,
                            "part": {"type": "output_text", "text": ""},
                        },
                    )
                )

        elif parsed.event_type == "content_block_delta":
            index = parsed.index
            delta = payload.get("delta", {})
            delta_type = delta.get("type")
            item = self._output_items.setdefault(index, self._build_fallback_item(index, delta_type))

            if delta_type == "text_delta":
                text = delta.get("text", "")
                self._message_text.append(text)
                item.setdefault("content", [{"type": "output_text", "text": ""}])
                item["content"][0]["text"] += text
                events.append(
                    self._sse(
                        "response.output_text.delta",
                        {
                            "type": "response.output_text.delta",
                            "response_id": response_id,
                            "output_index": index,
                            "content_index": 0,
                            "delta": text,
                        },
                    )
                )
            elif delta_type == "thinking_delta":
                thinking_text = delta.get("thinking", "")
                summary = item.setdefault("summary", [{"type": "summary_text", "text": ""}])
                summary[0]["text"] += thinking_text
            elif delta_type == "input_json_delta":
                partial_json = delta.get("partial_json", "")
                item["arguments"] = item.get("arguments", "") + partial_json
                events.append(
                    self._sse(
                        "response.function_call_arguments.delta",
                        {
                            "type": "response.function_call_arguments.delta",
                            "response_id": response_id,
                            "output_index": index,
                            "item_id": item.get("id"),
                            "delta": partial_json,
                        },
                    )
                )

        elif parsed.event_type == "content_block_stop":
            index = parsed.index
            item = self._output_items.get(index)
            if item:
                item["status"] = "completed"
                if item.get("type") == "function_call":
                    events.append(
                        self._sse(
                            "response.function_call_arguments.done",
                            {
                                "type": "response.function_call_arguments.done",
                                "response_id": response_id,
                                "output_index": index,
                                "item_id": item.get("id"),
                                "arguments": item.get("arguments", ""),
                            },
                        )
                    )
                events.append(
                    self._sse(
                        "response.output_item.done",
                        {
                            "type": "response.output_item.done",
                            "response_id": response_id,
                            "output_index": index,
                            "item": item,
                        },
                    )
                )

        elif parsed.event_type == "message_delta":
            delta = payload.get("delta", {})
            stop_reason = delta.get("stop_reason")
            usage = payload.get("usage") or {}
            if usage:
                self._usage.update(self._convert_usage(usage))
            if stop_reason:
                events.append(
                    self._sse(
                        "response.in_progress",
                        {
                            "type": "response.in_progress",
                            "response": {
                                "id": response_id,
                                "status": "in_progress",
                                "stop_reason": stop_reason,
                            },
                        },
                    )
                )

        elif parsed.event_type == "message_stop":
            final_response = {
                "id": response_id,
                "object": "response",
                "created_at": int(time.time()),
                "status": "completed",
                "model": self.original_model or "",
                "output": [self._output_items[idx] for idx in sorted(self._output_items.keys())],
                "output_text": "".join(self._message_text),
                "usage": self._usage,
            }
            events.append(
                self._sse(
                    "response.completed",
                    {
                        "type": "response.completed",
                        "response": final_response,
                    },
                )
            )

        return ConversionResult(success=True, data=events)

    def _ensure_response_id(self) -> str:
        if not hasattr(self, "_response_id"):
            self._response_id = "resp_" + "".join(random.choices(string.ascii_letters + string.digits, k=24))
        return self._response_id

    @staticmethod
    def _sse(event_name: str, payload: Dict[str, Any]) -> str:
        return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    @staticmethod
    def _convert_usage(usage: Dict[str, Any]) -> Dict[str, Any]:
        input_tokens = usage.get("input_tokens") or 0
        output_tokens = usage.get("output_tokens") or 0
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": usage.get("total_tokens") or (input_tokens + output_tokens),
        }

    def _build_output_item(self, index: int, block: Dict[str, Any]) -> Dict[str, Any]:
        block_type = block.get("type")
        if block_type == "tool_use":
            return {
                "type": "function_call",
                "id": block.get("id") or f"fc_{index}",
                "call_id": block.get("id") or f"call_{index}",
                "name": block.get("name", ""),
                "arguments": "",
                "status": "in_progress",
            }
        if block_type == "thinking":
            return {
                "type": "reasoning",
                "id": f"rs_{index}",
                "status": "in_progress",
                "summary": [{"type": "summary_text", "text": block.get("thinking", "")}],
            }
        return {
            "type": "message",
            "id": f"msg_{index}",
            "status": "in_progress",
            "role": "assistant",
            "content": [{"type": "output_text", "text": block.get("text", "")}],
        }

    def _build_fallback_item(self, index: int, delta_type: Optional[str]) -> Dict[str, Any]:
        if delta_type == "input_json_delta":
            return {
                "type": "function_call",
                "id": f"fc_{index}",
                "call_id": f"call_{index}",
                "name": "",
                "arguments": "",
                "status": "in_progress",
            }
        return {
            "type": "message",
            "id": f"msg_{index}",
            "status": "in_progress",
            "role": "assistant",
            "content": [{"type": "output_text", "text": ""}],
        }
