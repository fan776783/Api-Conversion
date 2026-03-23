"""OpenAI Responses 请求归一化适配器。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import copy


_RESPONSES_REQUEST_KEYS = {
    "input",
    "instructions",
    "max_output_tokens",
    "previous_response_id",
    "parallel_tool_calls",
    "store",
    "truncation",
    "text",
    "reasoning",
}


class OpenAIResponsesRequestAdapter:
    """将 OpenAI Responses 风格请求归一化为 Chat Completions 风格请求。"""

    @classmethod
    def looks_like_responses_request(cls, payload: Dict[str, Any]) -> bool:
        if not isinstance(payload, dict):
            return False
        if "messages" in payload:
            return False
        return any(key in payload for key in _RESPONSES_REQUEST_KEYS)

    @classmethod
    def adapt(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise TypeError("Responses payload must be a dict")

        adapted = copy.deepcopy(payload)
        messages: List[Dict[str, Any]] = []

        instructions = adapted.pop("instructions", None)
        if instructions not in (None, ""):
            messages.append({"role": "system", "content": instructions})

        input_payload = adapted.pop("input", None)
        messages.extend(cls._convert_input_to_messages(input_payload))

        if messages:
            adapted["messages"] = messages
        elif "messages" not in adapted:
            adapted["messages"] = []

        if "max_output_tokens" in adapted and "max_tokens" not in adapted:
            adapted["max_tokens"] = adapted.pop("max_output_tokens")
        else:
            adapted.pop("max_output_tokens", None)

        reasoning = adapted.pop("reasoning", None)
        if isinstance(reasoning, dict):
            effort = reasoning.get("effort")
            if effort is not None and "reasoning_effort" not in adapted:
                adapted["reasoning_effort"] = effort

        adapted.pop("previous_response_id", None)
        adapted.pop("parallel_tool_calls", None)
        adapted.pop("store", None)
        adapted.pop("truncation", None)
        adapted.pop("text", None)

        if "stream" in payload:
            adapted["stream"] = payload["stream"]

        return adapted

    @classmethod
    def _convert_input_to_messages(cls, input_payload: Any) -> List[Dict[str, Any]]:
        if input_payload is None:
            return []
        if isinstance(input_payload, str):
            return [{"role": "user", "content": input_payload}]
        if isinstance(input_payload, list):
            messages: List[Dict[str, Any]] = []
            for item in input_payload:
                converted = cls._convert_input_item(item)
                if converted is None:
                    continue
                if isinstance(converted, list):
                    messages.extend(converted)
                else:
                    messages.append(converted)
            return messages
        if isinstance(input_payload, dict):
            converted = cls._convert_input_item(input_payload)
            if converted is None:
                return []
            return converted if isinstance(converted, list) else [converted]
        return [{"role": "user", "content": str(input_payload)}]

    @classmethod
    def _convert_input_item(cls, item: Any) -> Optional[Any]:
        if item is None:
            return None
        if isinstance(item, str):
            return {"role": "user", "content": item}
        if not isinstance(item, dict):
            return {"role": "user", "content": str(item)}

        role = item.get("role") or cls._map_item_type_to_role(item.get("type")) or "user"

        if role == "tool":
            tool_call_id = item.get("call_id") or item.get("tool_call_id") or ""
            content = cls._coerce_content(item.get("content"))
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
            }

        content = item.get("content")
        if content is None and item.get("text") is not None:
            content = item.get("text")
        if content is None and item.get("arguments") is not None:
            content = item.get("arguments")
        if content is None and item.get("input_text") is not None:
            content = item.get("input_text")

        if isinstance(content, list):
            normalized_content = []
            tool_calls = []
            for block in content:
                if isinstance(block, dict) and block.get("type") in {"function_call", "tool_call"}:
                    tool_calls.append(cls._convert_function_call_block(block))
                else:
                    converted_block = cls._convert_content_block(block)
                    if converted_block is not None:
                        normalized_content.append(converted_block)

            message: Dict[str, Any] = {"role": role}
            if tool_calls:
                message["tool_calls"] = tool_calls
                message["content"] = cls._coerce_openai_content(normalized_content)
            else:
                message["content"] = cls._coerce_openai_content(normalized_content)
            return message

        if item.get("type") in {"function_call", "tool_call"}:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [cls._convert_function_call_block(item)],
            }

        return {
            "role": role,
            "content": cls._coerce_content(content),
        }

    @staticmethod
    def _map_item_type_to_role(item_type: Optional[str]) -> Optional[str]:
        mapping = {
            "message": "user",
            "input_text": "user",
            "output_text": "assistant",
            "function_call": "assistant",
            "tool_call": "assistant",
            "function_call_output": "tool",
        }
        return mapping.get(item_type)

    @classmethod
    def _convert_content_block(cls, block: Any) -> Optional[Dict[str, Any]]:
        if block is None:
            return None
        if isinstance(block, str):
            return {"type": "text", "text": block}
        if not isinstance(block, dict):
            return {"type": "text", "text": str(block)}

        block_type = block.get("type")
        if block_type in {"input_text", "output_text", "text"}:
            return {"type": "text", "text": block.get("text") or block.get("content") or ""}
        if block_type == "input_image":
            image_url = block.get("image_url") or block.get("url")
            if image_url:
                return {"type": "image_url", "image_url": {"url": image_url}}
        if block_type in {"image_url", "image"} and block.get("image_url"):
            return {"type": "image_url", "image_url": block.get("image_url")}
        if block_type in {"function_call", "tool_call"}:
            return None
        if block_type and "text" in block:
            return {"type": "text", "text": block.get("text") or ""}
        if "content" in block and isinstance(block["content"], str):
            return {"type": "text", "text": block["content"]}
        return None

    @staticmethod
    def _convert_function_call_block(block: Dict[str, Any]) -> Dict[str, Any]:
        arguments = block.get("arguments")
        if arguments is None:
            arguments = block.get("input")
        if isinstance(arguments, dict):
            import json
            arguments = json.dumps(arguments, ensure_ascii=False)
        elif arguments is None:
            arguments = "{}"
        else:
            arguments = str(arguments)

        function_name = block.get("name") or block.get("function", {}).get("name") or ""
        tool_call_id = block.get("call_id") or block.get("id") or block.get("tool_call_id") or ""

        return {
            "id": tool_call_id,
            "type": "function",
            "function": {
                "name": function_name,
                "arguments": arguments,
            },
        }

    @staticmethod
    def _coerce_content(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("text") is not None:
                    text_parts.append(str(item["text"]))
                elif isinstance(item, str):
                    text_parts.append(item)
            return "".join(text_parts)
        return str(content)

    @staticmethod
    def _coerce_openai_content(content_blocks: List[Dict[str, Any]]) -> Any:
        if not content_blocks:
            return ""
        if len(content_blocks) == 1 and content_blocks[0].get("type") == "text":
            return content_blocks[0].get("text", "")
        return content_blocks
