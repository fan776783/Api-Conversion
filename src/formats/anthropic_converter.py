"""
Anthropic格式转换器
处理Anthropic API格式与其他格式之间的转换
"""
from typing import Dict, Any, Optional, List, Tuple, Pattern
import json
import copy
import os
import re

from .base_converter import BaseConverter, ConversionResult, ConversionError
from .unified import UnifiedChatRequest, UnifiedChatResponse, UnifiedContent, UnifiedContentType
from .unified.stream_state import StreamState, StreamPhase
from src.utils.logger import log_structured_error, ERROR_TYPE_CONVERSION

# 全局工具状态管理器
class ToolStateManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tool_mappings = {}
        return cls._instance
    
    def store_tool_mapping(self, func_name: str, tool_id: str):
        """存储工具名到ID的映射"""
        self._tool_mappings[func_name] = tool_id
    
    def get_tool_id(self, func_name: str) -> Optional[str]:
        """根据工具名获取ID"""
        return self._tool_mappings.get(func_name)
    
    def clear_mappings(self):
        """清除所有映射"""
        self._tool_mappings.clear()

# 全局工具状态管理器实例
tool_state_manager = ToolStateManager()


class AnthropicConverter(BaseConverter):
    """Anthropic格式转换器"""
    
    def __init__(self):
        super().__init__()
        self.original_model = None
        self._tool_id_mapping = {}  # 存储tool_use_id到function_name的映射
        
        # 使用统一的日志设置（继承自BaseConverter）
        # self.logger 已经在 BaseConverter.__init__() 中正确设置
    
    def set_original_model(self, model: str):
        """设置原始模型名称"""
        self.original_model = model
    
    def _determine_reasoning_effort_from_budget(self, budget_tokens: Optional[int]) -> str:
        """根据budget_tokens智能判断OpenAI reasoning_effort等级
        
        Args:
            budget_tokens: Anthropic thinking的budget_tokens值
            
        Returns:
            str: OpenAI reasoning_effort等级 ("low", "medium", "high")
        """
        import os
        
        # 如果没有提供budget_tokens，默认为high
        if budget_tokens is None:
            self.logger.info("No budget_tokens provided, defaulting to reasoning_effort='high'")
            return "high"
        
        # 从环境变量获取阈值配置（带默认值）
        low_threshold_str = os.environ.get("ANTHROPIC_TO_OPENAI_LOW_REASONING_THRESHOLD", "2048")
        high_threshold_str = os.environ.get("ANTHROPIC_TO_OPENAI_HIGH_REASONING_THRESHOLD", "16384")
        
        try:
            low_threshold = int(low_threshold_str)
            high_threshold = int(high_threshold_str)
            
            self.logger.debug(f"Threshold configuration: low <= {low_threshold}, medium <= {high_threshold}, high > {high_threshold}")
            
            if budget_tokens <= low_threshold:
                effort = "low"
            elif budget_tokens <= high_threshold:
                effort = "medium"
            else:
                effort = "high"
            
            self.logger.info(f"🎯 Budget tokens {budget_tokens} -> reasoning_effort '{effort}' (thresholds: low<={low_threshold}, high<={high_threshold})")
            return effort
            
        except ValueError as e:
            raise ConversionError(f"Invalid threshold values in environment variables: {e}. ANTHROPIC_TO_OPENAI_LOW_REASONING_THRESHOLD and ANTHROPIC_TO_OPENAI_HIGH_REASONING_THRESHOLD must be integers.")
    
    def reset_streaming_state(self):
        """重置所有流式相关的状态变量，避免状态污染"""
        self.logger.debug("reset_streaming_state() called - cleaning up streaming state")
        streaming_attrs = [
            '_streaming_state', '_gemini_sent_start', '_gemini_stream_id', 
            '_gemini_text_started'
        ]
        cleaned_attrs = []
        for attr in streaming_attrs:
            if hasattr(self, attr):
                cleaned_attrs.append(attr)
                delattr(self, attr)
        
        self.logger.debug(f"Cleaned streaming attributes: {cleaned_attrs}")
        
        # 强制重置，确保下次访问时重新初始化
        self._force_reset = True
    
    def get_supported_formats(self) -> List[str]:
        """获取支持的格式列表"""
        return ["openai", "anthropic", "gemini"]
    
    def convert_request(
        self,
        data: Dict[str, Any],
        target_format: str,
        headers: Optional[Dict[str, str]] = None
    ) -> ConversionResult:
        """转换Anthropic请求到目标格式"""
        try:
            if target_format == "anthropic":
                # Anthropic到Anthropic，格式与渠道相同，不需要转换思考参数
                return ConversionResult(success=True, data=data)
            elif target_format == "openai":
                return self._convert_to_openai_request(data)
            elif target_format == "gemini":
                return self._convert_to_gemini_request(data)
            else:
                return ConversionResult(
                    success=False,
                    error=f"Unsupported target format: {target_format}"
                )
        except Exception as e:
            log_structured_error(
                self.logger,
                error_type=ERROR_TYPE_CONVERSION,
                exc=e,
                request_body=data,
                extra={
                    "converter": self.__class__.__name__,
                    "source_format": "anthropic",
                    "target_format": target_format,
                    "stage": "convert_request",
                },
            )
            return ConversionResult(success=False, error=str(e))
    
    def convert_response(
        self,
        data: Dict[str, Any],
        source_format: str,
        target_format: str
    ) -> ConversionResult:
        """转换响应到Anthropic格式"""
        try:
            if source_format == "anthropic":
                return ConversionResult(success=True, data=data)
            elif source_format == "openai":
                return self._convert_from_openai_response(data)
            elif source_format == "gemini":
                return self._convert_from_gemini_response(data)
            else:
                return ConversionResult(
                    success=False,
                    error=f"Unsupported source format: {source_format}"
                )
        except Exception as e:
            log_structured_error(
                self.logger,
                error_type=ERROR_TYPE_CONVERSION,
                exc=e,
                response_body=data,
                extra={
                    "converter": self.__class__.__name__,
                    "source_format": source_format,
                    "target_format": target_format,
                    "stage": "convert_response",
                },
            )
            return ConversionResult(success=False, error=str(e))
    
    def _convert_to_openai_request(self, data: Dict[str, Any]) -> ConversionResult:
        """转换Anthropic请求到OpenAI格式（基于统一类型层）"""
        # 1) 使用 UnifiedChatRequest 解析 Anthropic 请求
        unified_request = UnifiedChatRequest.from_anthropic(data)

        # 1.1) 协议标签清理：对用户/系统输入做安全过滤，防止伪造协议标签注入
        if unified_request.system:
            unified_request.system = self._sanitize_protocol_tags(unified_request.system)

        for msg in unified_request.messages:
            # 只清理 user/system 侧内容，避免误改模型输出
            if msg.role in ("user", "system"):
                for content_block in msg.content:
                    if content_block.type in (
                        UnifiedContentType.TEXT,
                        UnifiedContentType.THINKING,
                    ):
                        if isinstance(content_block.text, str):
                            content_block.text = self._sanitize_protocol_tags(
                                content_block.text
                            )

        # 2) 工具定义清理（保持与 OpenAI 的 JSON Schema 兼容）
        if "tools" in data:
            cleaned_tools: List[Dict[str, Any]] = []
            for tool in data.get("tools", []):
                cleaned_tools.append({
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "input_schema": self._clean_json_schema_properties(tool.get("input_schema", {})),
                })
            unified_request.tools = cleaned_tools

        # 3) 思考模式到 OpenAI reasoning 模式的映射
        if unified_request.thinking_enabled:
            budget_tokens = unified_request.thinking_budget_tokens

            # 根据 budget_tokens 智能判断 reasoning_effort 等级
            reasoning_effort = self._determine_reasoning_effort_from_budget(budget_tokens)
            unified_request.reasoning_effort = reasoning_effort

            # 处理 max_completion_tokens 的优先级逻辑
            max_completion_tokens: Optional[int] = None

            # 优先级 1：客户端在 Anthropic 请求中传入的 max_tokens
            if "max_tokens" in data:
                max_completion_tokens = unified_request.max_tokens
                unified_request.max_tokens = None  # 避免同时下发 max_tokens 与 max_completion_tokens
                self.logger.info(
                    f"Using client max_tokens as max_completion_tokens: {max_completion_tokens}"
                )
            else:
                # 优先级 2：环境变量 OPENAI_REASONING_MAX_TOKENS
                env_max_tokens = os.environ.get("OPENAI_REASONING_MAX_TOKENS")
                if env_max_tokens:
                    try:
                        max_completion_tokens = int(env_max_tokens)
                        self.logger.info(
                            f"Using OPENAI_REASONING_MAX_TOKENS from environment: {max_completion_tokens}"
                        )
                    except ValueError:
                        self.logger.warning(
                            f"Invalid OPENAI_REASONING_MAX_TOKENS value '{env_max_tokens}', must be integer"
                        )
                        env_max_tokens = None

                if not env_max_tokens:
                    # 优先级 3：两者都缺失时抛出错误
                    raise ConversionError(
                        "For OpenAI reasoning models, max_completion_tokens is required. "
                        "Please specify max_tokens in the request or set OPENAI_REASONING_MAX_TOKENS environment variable."
                    )

            unified_request.max_completion_tokens = max_completion_tokens
            self.logger.info(
                f"Anthropic thinking enabled -> OpenAI reasoning_effort='{reasoning_effort}', "
                f"max_completion_tokens={max_completion_tokens}"
            )
            if budget_tokens:
                self.logger.info(
                    f"Budget tokens: {budget_tokens} -> reasoning_effort: '{reasoning_effort}'"
                )

        # 4) 统一请求对象转换为 OpenAI 请求
        result_data = unified_request.to_openai()

        # 5) tool_choice 兼容性处理
        if "tool_choice" in data:
            result_data["tool_choice"] = data["tool_choice"]
        elif "tools" in data and result_data.get("tools") and "tool_choice" not in result_data:
            result_data["tool_choice"] = "auto"

        # 6) OpenAI 兼容性校验：确保所有 assistant.tool_calls 均有匹配的 tool 响应
        messages = result_data.get("messages", [])
        validated_messages: List[Dict[str, Any]] = []
        for idx, msg in enumerate(messages):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                call_ids = [tc.get("id") for tc in msg.get("tool_calls", []) if tc.get("id")]
                unmatched = set(call_ids)

                # 在后续消息中查找对应的 tool 响应
                for later in messages[idx + 1:]:
                    if later.get("role") == "tool" and later.get("tool_call_id") in unmatched:
                        unmatched.discard(later["tool_call_id"])
                    if not unmatched:
                        break

                if unmatched:
                    self.logger.warning(
                        f"Unmatched tool_call IDs without tool responses, cleaning: {list(unmatched)}"
                    )
                    msg["tool_calls"] = [
                        tc for tc in msg.get("tool_calls", []) if tc.get("id") not in unmatched
                    ]
                    # 如果全部被移除，则降级为普通 assistant 文本消息
                    if not msg["tool_calls"]:
                        msg.pop("tool_calls", None)
                        if msg.get("content") is None:
                            msg["content"] = ""

            validated_messages.append(msg)

        result_data["messages"] = validated_messages

        # 7) 与旧实现保持一致：如果原请求未提供 model 字段，则不在输出中强行添加
        if "model" not in data:
            result_data.pop("model", None)

        return ConversionResult(success=True, data=result_data)
    
    def _convert_to_gemini_request(self, data: Dict[str, Any]) -> ConversionResult:
        """转换Anthropic请求到Gemini格式"""
        result_data = {}
        
        # 处理模型名称
        if "model" in data:
            # 直接使用原始模型名称，不进行映射
            result_data["model"] = data["model"]
        
        # 处理系统消息 - 基于2025年Gemini API文档格式
        if "system" in data:
            # 确保系统指令内容不为空且为字符串
            system_content = str(data["system"]).strip() if data["system"] else ""
            if system_content:
                # 对系统指令做协议标签过滤，防止伪造 <invoke>/<tool_result>/<thinking> 等标签
                system_content = self._sanitize_protocol_tags(system_content)
                result_data["system_instruction"] = {
                    "parts": [{"text": system_content}]
                }
        
        # 转换消息格式
        if "messages" in data:
            # 构建工具调用ID到函数名的映射
            tool_use_to_name = {}
            for msg in data["messages"]:
                if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
                    for item in msg["content"]:
                        if item.get("type") == "tool_use":
                            tool_use_to_name[item.get("id")] = item.get("name")
            
            # 设置映射供_build_function_response使用
            self._tool_use_mapping = tool_use_to_name
            
            gemini_contents = []
            for msg in data["messages"]:
                anthropic_role = msg.get("role", "assistant")
                # 传递 role 参数，以便对 user 消息做协议标签过滤
                parts_converted = self._convert_content_to_gemini(
                    msg.get("content", ""),
                    role=anthropic_role
                )

                # -------- 修正角色映射：tool 消息保持为 tool --------------
                if anthropic_role == "user":
                    role = "user"
                elif anthropic_role == "assistant":
                    role = "model"
                elif anthropic_role == "tool":
                    role = "tool"
                else:
                    role = "model"

                # 如果内容包含 functionResponse，强制设为 tool 角色，避免用户端 role 写错
                if any("functionResponse" in p for p in parts_converted):
                    role = "tool"
                
                # 确保 tool 角色的消息至少有一个有效的 part，避免 Gemini 500 错误
                if role == "tool" and not parts_converted:
                    parts_converted = [{"text": ""}]
                elif role == "tool" and all(not p for p in parts_converted):
                    parts_converted = [{"text": ""}]
                
                gemini_contents.append({
                    "role": role,
                    "parts": parts_converted
                })
            result_data["contents"] = gemini_contents
        
        # 处理生成配置
        generation_config = {}
        if "temperature" in data:
            generation_config["temperature"] = data["temperature"]
        if "top_p" in data:
            generation_config["topP"] = data["top_p"]
        if "top_k" in data:
            generation_config["topK"] = data["top_k"]
        if "max_tokens" in data:
            generation_config["maxOutputTokens"] = data["max_tokens"]
        if "stop_sequences" in data:
            generation_config["stopSequences"] = data["stop_sequences"]
        
        # 处理思考预算转换 (Anthropic thinkingBudget -> Gemini thinkingBudget)
        if "thinking" in data and data["thinking"].get("type") == "enabled":
            budget_tokens = data["thinking"].get("budget_tokens")
            if budget_tokens:
                generation_config["thinkingConfig"] = {
                    "thinkingBudget": budget_tokens
                }
                self.logger.info(f"Anthropic thinkingBudget {budget_tokens} -> Gemini thinkingBudget {budget_tokens}")
            elif "thinking" in data:
                # 如果没有设置budget_tokens，对应Gemini的-1（动态思考）
                generation_config["thinkingConfig"] = {
                    "thinkingBudget": -1
                }
                self.logger.info("Anthropic thinking enabled without budget -> Gemini thinkingBudget -1 (dynamic)")
        
        # 确保 generationConfig 永远存在，避免 Gemini 2.0+ 的 500 错误
        result_data["generationConfig"] = generation_config or {}
        
        # 处理工具调用
        if "tools" in data:
            function_declarations = []
            for tool in data["tools"]:
                function_declarations.append({
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": self._clean_json_schema_properties(tool.get("input_schema", {}))
                })
            
            if function_declarations:
                # Gemini官方规范使用 camelCase: functionDeclarations
                result_data["tools"] = [{"functionDeclarations": function_declarations}]
        
        # 应用深度清理，移除可能导致协议错误的字段
        cleaned_result_data = self._deep_clean_for_gemini(result_data)
        
        return ConversionResult(success=True, data=cleaned_result_data)
    
    def _convert_from_openai_response(self, data: Dict[str, Any]) -> ConversionResult:
        """转换OpenAI响应到Anthropic格式（基于统一中间层）"""
        # 必须有原始模型名称
        if not self.original_model:
            raise ValueError("Original model name is required for response conversion")

        # 1) 通过统一中间层解析 OpenAI 响应
        unified_response = UnifiedChatResponse.from_openai(data, self.original_model)

        # 2) 兼容旧实现：从文本内容中解析 <thinking> 标签
        normalized_content: List[UnifiedContent] = []
        for c in unified_response.content:
            if c.type == UnifiedContentType.TEXT and isinstance(c.text, str):
                extracted = self._extract_thinking_from_openai_text(c.text)
                if isinstance(extracted, list):
                    for block in extracted:
                        block_type = block.get("type")
                        if block_type == "thinking":
                            normalized_content.append(
                                UnifiedContent(
                                    type=UnifiedContentType.THINKING,
                                    text=block.get("thinking", ""),
                                )
                            )
                        else:
                            normalized_content.append(
                                UnifiedContent(
                                    type=UnifiedContentType.TEXT,
                                    text=block.get("text", ""),
                                )
                            )
                else:
                    c.text = extracted
                    normalized_content.append(c)
            else:
                normalized_content.append(c)
        unified_response.content = normalized_content

        # 3) 支持 annotations 透传
        annotations = self._extract_annotations_from_openai(data)
        if annotations:
            for c in unified_response.content:
                if c.type == UnifiedContentType.TEXT:
                    c.annotations = annotations

        # 4) 转换为 Anthropic 格式
        result_data = unified_response.to_anthropic()

        # 5) 保持向后兼容：id / stop_reason / usage
        result_data["id"] = data.get("id") or result_data.get("id") or "msg_openai"

        finish_reason = "stop"
        if data.get("choices") and data["choices"][0]:
            finish_reason = data["choices"][0].get("finish_reason", "stop")
        result_data["stop_reason"] = self._map_finish_reason(finish_reason, "openai", "anthropic")

        if "usage" not in result_data or not result_data["usage"]:
            if data.get("usage"):
                usage = data["usage"]
                result_data["usage"] = {
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                }
            else:
                result_data["usage"] = {}

        return ConversionResult(success=True, data=result_data)

    def _extract_annotations_from_openai(self, data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """从 OpenAI 响应中提取 annotations 数组"""
        try:
            choices = data.get("choices") or []
            if not choices:
                return data.get("annotations")
            first_choice = choices[0]
            if not isinstance(first_choice, dict):
                return None
            message = first_choice.get("message", {})
            return (
                message.get("annotations")
                or first_choice.get("annotations")
                or data.get("annotations")
            )
        except Exception:
            return None
    
    def _extract_thinking_from_openai_text(self, text: str) -> Any:
        """从OpenAI文本中提取thinking内容，返回Anthropic格式的content blocks"""
        import re
        
        # 匹配 <thinking>...</thinking> 标签
        thinking_pattern = r'<thinking>\s*(.*?)\s*</thinking>'
        matches = re.finditer(thinking_pattern, text, re.DOTALL)
        
        content_blocks = []
        last_end = 0
        
        for match in matches:
            # 添加thinking标签之前的文本（如果有）
            before_text = text[last_end:match.start()].strip()
            if before_text:
                content_blocks.append({
                    "type": "text",
                    "text": before_text
                })
            
            # 添加thinking内容
            thinking_text = match.group(1).strip()
            if thinking_text:
                content_blocks.append({
                    "type": "thinking",
                    "thinking": thinking_text
                })
            
            last_end = match.end()
        
        # 添加最后一个thinking标签之后的文本（如果有）
        after_text = text[last_end:].strip()
        if after_text:
            content_blocks.append({
                "type": "text",
                "text": after_text
            })
        
        # 如果没有找到thinking标签，返回原文本
        if not content_blocks:
            return text
        
        # 如果只有一个文本块，返回字符串
        if len(content_blocks) == 1 and content_blocks[0].get("type") == "text":
            return content_blocks[0]["text"]
        
        return content_blocks
    
    def _convert_from_gemini_response(self, data: Dict[str, Any]) -> ConversionResult:
        """转换Gemini响应到Anthropic格式"""
        # 必须有原始模型名称，否则报错
        if not self.original_model:
            raise ValueError("Original model name is required for response conversion")
            
        result_data = {
            "id": f"msg_gemini_{hash(str(data))}",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": self.original_model,  # 使用原始模型名称
            "stop_reason": "end_turn",
            "usage": {}
        }
        
        # 处理候选结果
        if "candidates" in data and data["candidates"] and data["candidates"][0]:
            candidate = data["candidates"][0]
            content_list = []
            if "content" in candidate and "parts" in candidate["content"]:
                for part in candidate["content"]["parts"]:
                    # 普通文本
                    if "text" in part:
                        content_list.append({
                            "type": "text",
                            "text": part["text"]
                        })
                    # 函数调用
                    elif "functionCall" in part:
                        fc = part["functionCall"]
                        content_list.append({
                            "type": "tool_use",
                            "id": f"call_{fc.get('name','tool')}_{abs(hash(str(fc)))}",
                            "name": fc.get("name", ""),
                            "input": fc.get("args", {})
                        })
            if content_list:
                result_data["content"] = content_list

            # 根据是否存在 functionCall 判断 stop_reason
            finish_reason = candidate.get("finishReason", "STOP")
            if content_list and any(c.get("type") == "tool_use" for c in content_list):
                result_data["stop_reason"] = "tool_use"
            else:
                result_data["stop_reason"] = self._map_finish_reason(finish_reason, "gemini", "anthropic")
        
        # 处理使用情况
        if "usageMetadata" in data:
            usage = data["usageMetadata"]
            result_data["usage"] = {
                "input_tokens": usage.get("promptTokenCount", 0),
                "output_tokens": usage.get("candidatesTokenCount", 0)
            }
        
        return ConversionResult(success=True, data=result_data)
    
    def _convert_from_openai_streaming_chunk(self, data: Dict[str, Any]) -> ConversionResult:
        """转换OpenAI流式响应chunk到Anthropic SSE格式（基于StreamState）"""
        import json
        import time

        # 1) 校验原始模型名
        if not self.original_model:
            raise ValueError("Original model name is required for streaming response conversion")

        # 2) 初始化 StreamState
        if not hasattr(self, "_openai_stream_state") or getattr(self, "_force_reset", False):
            for attr in ["_gemini_sent_start", "_gemini_stream_id", "_gemini_text_started", "_force_reset"]:
                if hasattr(self, attr):
                    delattr(self, attr)
            self._openai_stream_state = StreamState(
                model=self.original_model,
                original_model=self.original_model,
            )

        state: StreamState = self._openai_stream_state

        # 3) 解析 OpenAI chunk
        choices = data.get("choices") or []
        choice = choices[0] if choices else None
        if not choice:
            return ConversionResult(success=True, data="")

        delta = choice.get("delta", {}) or {}
        content = delta.get("content") or ""
        tool_calls = delta.get("tool_calls") or []
        reasoning_content = delta.get("reasoning_content") or ""
        finish_reason = choice.get("finish_reason")

        events: List[str] = []

        # 4) 发送 message_start（仅一次）
        if not state.sent_message_start and not state.sent_message_stop:
            has_meaningful_data = (
                delta.get("role")
                or content
                or reasoning_content
                or tool_calls
                or choice.get("index") is not None
            )
            if has_meaningful_data:
                state.sent_message_start = True
                state.phase = StreamPhase.MESSAGE_STARTED
                model_name = state.model or self.original_model or "unknown"
                message_start = {
                    "type": "message_start",
                    "message": {
                        "id": state.stream_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": model_name,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                }
                events.append(f"event: message_start\ndata: {json.dumps(message_start, ensure_ascii=False)}\n\n")

        # 5) 处理 reasoning_content → thinking 块（OpenAI o1/o3）
        if reasoning_content and not state.sent_message_stop:
            is_new_thinking = not state.thinking_block_started
            thinking_index = state.start_thinking_block()

            if is_new_thinking:
                content_block_start = {
                    "type": "content_block_start",
                    "index": thinking_index,
                    "content_block": {"type": "thinking", "thinking": ""},
                }
                events.append(f"event: content_block_start\ndata: {json.dumps(content_block_start, ensure_ascii=False)}\n\n")

            state.thinking_buffer += reasoning_content
            thinking_delta = {
                "type": "content_block_delta",
                "index": thinking_index,
                "delta": {"type": "thinking_delta", "thinking": reasoning_content},
            }
            events.append(f"event: content_block_delta\ndata: {json.dumps(thinking_delta, ensure_ascii=False)}\n\n")

            if state.phase in (StreamPhase.NOT_STARTED, StreamPhase.MESSAGE_STARTED):
                state.phase = StreamPhase.CONTENT_STREAMING

        # 6) 处理普通文本 content
        if content and not state.sent_message_stop:
            is_new_text = not state.text_block_started
            text_index = state.start_text_block()

            if is_new_text:
                content_block_start = {
                    "type": "content_block_start",
                    "index": text_index,
                    "content_block": {"type": "text", "text": ""},
                }
                events.append(f"event: content_block_start\ndata: {json.dumps(content_block_start, ensure_ascii=False)}\n\n")

            content_delta = {
                "type": "content_block_delta",
                "index": text_index,
                "delta": {"type": "text_delta", "text": content},
            }
            events.append(f"event: content_block_delta\ndata: {json.dumps(content_delta, ensure_ascii=False)}\n\n")

            if state.phase in (StreamPhase.NOT_STARTED, StreamPhase.MESSAGE_STARTED):
                state.phase = StreamPhase.CONTENT_STREAMING

        # 7) 处理工具调用 tool_calls → tool_use
        if tool_calls and not state.sent_message_stop:
            state.phase = StreamPhase.TOOL_STREAMING
            processed_in_this_chunk: set = set()

            for tool_call in tool_calls:
                if not tool_call:
                    continue

                tool_call_index = tool_call.get("index", 0)
                if tool_call_index in processed_in_this_chunk:
                    continue
                processed_in_this_chunk.add(tool_call_index)

                func = tool_call.get("function", {}) or {}
                tool_state = state.get_tool_call(tool_call_index)

                if not tool_state:
                    tool_call_id = tool_call.get("id", f"call_{int(time.time())}_{tool_call_index}")
                    tool_call_name = func.get("name", f"tool_{tool_call_index}")
                    tool_content_index = state.start_tool_call(tool_call_index, tool_call_id, tool_call_name)

                    content_block_start = {
                        "type": "content_block_start",
                        "index": tool_content_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tool_call_id,
                            "name": tool_call_name,
                            "input": {},
                        },
                    }
                    events.append(f"event: content_block_start\ndata: {json.dumps(content_block_start, ensure_ascii=False)}\n\n")
                else:
                    tool_content_index = tool_state.content_block_index

                arguments_fragment = func.get("arguments")
                if arguments_fragment:
                    state.append_tool_arguments(tool_call_index, arguments_fragment)
                    cleaned_fragment = self._clean_json_fragment(arguments_fragment)

                    if cleaned_fragment:
                        input_json_delta = {
                            "type": "content_block_delta",
                            "index": tool_content_index,
                            "delta": {"type": "input_json_delta", "partial_json": cleaned_fragment},
                        }
                        events.append(f"event: content_block_delta\ndata: {json.dumps(input_json_delta, ensure_ascii=False)}\n\n")

        # 8) 处理流结束
        if finish_reason and not state.sent_message_stop and state.sent_message_start:
            state.phase = StreamPhase.FINISHED

            # 按顺序关闭所有已开启的 content blocks
            for index in state.get_all_content_block_indices():
                content_block_stop = {"type": "content_block_stop", "index": index}
                events.append(f"event: content_block_stop\ndata: {json.dumps(content_block_stop, ensure_ascii=False)}\n\n")

            anthropic_stop_reason = self._map_finish_reason(finish_reason, "openai", "anthropic")

            message_delta: Dict[str, Any] = {
                "type": "message_delta",
                "delta": {"stop_reason": anthropic_stop_reason, "stop_sequence": None},
            }

            if data.get("usage") is not None:
                usage = data["usage"]
                state.accumulate_usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
            message_delta["usage"] = {
                "input_tokens": state.input_tokens or 0,
                "output_tokens": state.output_tokens or 0,
            }

            events.append(f"event: message_delta\ndata: {json.dumps(message_delta, ensure_ascii=False)}\n\n")
            events.append(f"event: message_stop\ndata: {{\"type\": \"message_stop\"}}\n\n")

            state.sent_message_stop = True

        # 9) 流结束后清理状态
        if finish_reason and hasattr(self, "_openai_stream_state"):
            delattr(self, "_openai_stream_state")

        # 10) 返回 SSE 串
        if not events:
            self.logger.debug(
                f"No events generated for chunk - content: {bool(content)}, "
                f"reasoning: {bool(reasoning_content)}, tool_calls: {bool(tool_calls)}, "
                f"sent_message_start: {state.sent_message_start}"
            )
            return ConversionResult(success=True, data="")

        result_data = "".join(events)
        self.logger.debug(f"Generated {len(events)} events, total data length: {len(result_data)}")
        return ConversionResult(success=True, data=result_data)
    
    def _clean_json_fragment(self, fragment: str) -> str:
        """清理JSON片段，避免不完整的Unicode字符或转义序列"""
        if not fragment:
            return fragment
        
        try:
            # 移除开头和结尾可能不完整的Unicode字符
            # 检查最后几个字符是否是不完整的转义序列
            cleaned = fragment
            
            # 处理可能被截断的转义序列
            if cleaned.endswith('\\') and not cleaned.endswith('\\\\'):
                cleaned = cleaned[:-1]  # 移除悬挂的反斜杠
            elif cleaned.endswith('\\u') or cleaned.endswith('\\u0') or cleaned.endswith('\\u00'):
                # 不完整的Unicode转义序列
                idx = cleaned.rfind('\\u')
                cleaned = cleaned[:idx]
            
            # 验证清理后的片段不会导致JSON解析错误
            if cleaned:
                # 简单测试：如果片段包含引号，确保它们是平衡的
                quote_count = cleaned.count('"') - cleaned.count('\\"')
                if quote_count % 2 == 1:
                    # 如果引号数量为奇数，可能在字符串中间被截断
                    pass  # 仍然发送，让接收端处理
            
            return cleaned
            
        except Exception as e:
            self.logger.warning(f"Error cleaning JSON fragment: {e}, returning original")
            return fragment
    
    
    def _convert_from_gemini_streaming_chunk(self, data: Dict[str, Any]) -> ConversionResult:
        """将 Gemini 流式 chunk 转为 Anthropic SSE 格式 - 简化版本"""
        import json, random, time
        
        self.logger.debug(f"Converting Gemini chunk: {str(data)[:200]}...")
        
        # 检查当前状态
        current_state = {
            '_gemini_stream_id': hasattr(self, '_gemini_stream_id'),
            '_gemini_sent_start': hasattr(self, '_gemini_sent_start'),
            '_gemini_text_started': hasattr(self, '_gemini_text_started'),
            '_streaming_state': hasattr(self, '_streaming_state'),
            '_force_reset': getattr(self, '_force_reset', False)
        }
        self.logger.debug(f"Current state before processing: {current_state}")
        
        # 每次开始新的流式转换时，重置所有相关状态变量，避免状态污染
        if not hasattr(self, '_gemini_stream_id') or getattr(self, '_force_reset', False):
            self.logger.debug("Initializing new Gemini stream")
            # 清理可能残留的状态
            for attr in ['_gemini_sent_start', '_gemini_text_started', '_streaming_state', '_force_reset']:
                if hasattr(self, attr):
                    delattr(self, attr)
            # 生成新的流ID
            self._gemini_stream_id = f"msg_{random.randint(100000, 999999)}"
            self.logger.debug(f"Generated stream ID: {self._gemini_stream_id}")

        # 保存模型名（必须已在 set_original_model 设置）
        if not self.original_model:
            raise ValueError("Original model name is required for streaming response conversion")

        # 提取本次 chunk 的 candidate、内容、结束标记
        candidate = None
        if data.get("candidates") and data["candidates"][0]:
            candidate = data["candidates"][0]

        content = ""
        function_calls = []
        if candidate and candidate.get("content") and candidate["content"].get("parts"):
            for part in candidate["content"]["parts"]:
                if "text" in part:
                    content += part["text"]
                elif "functionCall" in part:
                    # 处理函数调用
                    func_call = part["functionCall"]
                    function_calls.append({
                        "name": func_call.get("name", ""),
                        "args": func_call.get("args", {})
                    })

        is_end = bool(candidate and candidate.get("finishReason"))

        events: list[str] = []  # 保存 SSE 行

        # 第一次进入：发送 message_start
        if not hasattr(self, '_gemini_sent_start'):
            import logging
            unified_logger = logging.getLogger("unified_api")
            unified_logger.debug("ANTHROPIC_CONVERTER: Sending message_start for new Gemini stream")
            self.logger.debug("Sending message_start for new Gemini stream")
            self._gemini_sent_start = True

            # 防御性检查：确保model和role属性始终有效
            model_name = self.original_model or 'unknown'
            message_start = {
                "type": "message_start",
                "message": {
                    "id": self._gemini_stream_id,
                    "type": "message",
                    "role": "assistant",  # 始终确保role属性存在
                    "content": [],
                    "model": model_name,  # 使用防御性检查后的模型名称
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            }
            events += [
                "event: message_start",
                f"data: {json.dumps(message_start, ensure_ascii=False)}",
                "",
            ]

        # 处理文本内容
        if content:
            # 如果还没有发送过文本 content_block_start，先发送
            if not hasattr(self, '_gemini_text_started'):
                self._gemini_text_started = True
                content_block_start = {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                }
                events += [
                    "event: content_block_start",
                    f"data: {json.dumps(content_block_start, ensure_ascii=False)}",
                    "",
                ]

            # 发送文本增量
            content_block_delta = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": content},
            }
            events += [
                "event: content_block_delta",
                f"data: {json.dumps(content_block_delta, ensure_ascii=False)}",
                "",
            ]

        # 处理函数调用
        if function_calls:
            for i, func_call in enumerate(function_calls):
                # 确定索引：如果有文本内容，工具调用从索引1开始；否则从索引0开始
                # 由于我们上面可能添加了解释文本，所以_gemini_text_started应该已经设置
                tool_index = 1 if hasattr(self, '_gemini_text_started') else 0
                # 如果有多个工具调用，后续工具的索引需要递增
                tool_index += i
                
                # 发送 tool_use content_block_start
                tool_block_start = {
                    "type": "content_block_start",
                    "index": tool_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": f"toolu_{random.randint(100000, 999999)}",
                        "name": func_call["name"],
                        "input": {}
                    }
                }
                events += [
                    "event: content_block_start",
                    f"data: {json.dumps(tool_block_start, ensure_ascii=False)}",
                    "",
                ]

                # 发送工具调用参数
                if func_call["args"]:
                    tool_delta = {
                        "type": "content_block_delta",
                        "index": tool_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(func_call["args"], ensure_ascii=False)
                        }
                    }
                    events += [
                        "event: content_block_delta",
                        f"data: {json.dumps(tool_delta, ensure_ascii=False)}",
                        "",
                    ]

                # 发送 content_block_stop
                tool_block_stop = {"type": "content_block_stop", "index": tool_index}
                events += [
                    "event: content_block_stop",
                    f"data: {json.dumps(tool_block_stop, ensure_ascii=False)}",
                    "",
                ]

        # 如果本 chunk 携带 finishReason，说明对话结束，补充收尾事件
        if is_end:
            self.logger.debug(f"Stream ending with finishReason: {candidate.get('finishReason') if candidate else 'None'}")
            # 如果有文本内容块还未结束，发送 content_block_stop
            if hasattr(self, '_gemini_text_started'):
                content_block_stop = {"type": "content_block_stop", "index": 0}
                events += [
                    "event: content_block_stop",
                    f"data: {json.dumps(content_block_stop, ensure_ascii=False)}",
                    "",
                ]

            # message_delta（包含 stop_reason 与 usage）
            # 对于Gemini工具调用的特殊处理：
            # - 如果检测到函数调用，stop_reason应该是tool_use（无论Gemini的finishReason是什么）
            # - 如果没有函数调用，使用正常的finish_reason映射
            if function_calls:
                stop_reason = "tool_use"
                self.logger.info(f"Setting stop_reason to 'tool_use' due to detected function calls: {[fc.get('name') for fc in function_calls]}")
            else:
                stop_reason = self._map_finish_reason(candidate.get("finishReason", ""), "gemini", "anthropic")
                self.logger.debug(f"Mapped finish_reason '{candidate.get('finishReason', '')}' to '{stop_reason}'")
            
            message_delta = {
                "type": "message_delta",
                "delta": {
                    "stop_reason": stop_reason,
                    "stop_sequence": None,
                },
            }
            
            # 总是提供 usage 信息，即使 Gemini 没有 usageMetadata
            if data.get("usageMetadata"):
                usage = data["usageMetadata"]
                message_delta["usage"] = {
                    "input_tokens": usage.get("promptTokenCount", 0),
                    "output_tokens": usage.get("candidatesTokenCount", 0)
                }
            else:
                # 如果没有 usage 信息，提供默认值以避免前端错误
                message_delta["usage"] = {
                    "input_tokens": 0,
                    "output_tokens": 0
                }

            events += [
                "event: message_delta",
                f"data: {json.dumps(message_delta, ensure_ascii=False)}",
                "",
                "event: message_stop",
                "data: {\"type\": \"message_stop\"}",
                "",
            ]
            import logging
            unified_logger = logging.getLogger("unified_api")
            unified_logger.debug("ANTHROPIC_CONVERTER: Sent message_stop for Gemini stream")
            self.logger.debug("Sent message_stop for Gemini stream")

            # 结束当前流后清理状态，避免影响下一次请求
            self.logger.debug("Cleaning up Gemini streaming state after stream end")
            cleaned_attrs = []
            if hasattr(self, '_gemini_sent_start'):
                cleaned_attrs.append('_gemini_sent_start')
                delattr(self, '_gemini_sent_start')
            if hasattr(self, '_gemini_stream_id'):
                cleaned_attrs.append('_gemini_stream_id')
                delattr(self, '_gemini_stream_id')
            if hasattr(self, '_gemini_text_started'):
                cleaned_attrs.append('_gemini_text_started')
                delattr(self, '_gemini_text_started')
            self.logger.debug(f"Cleaned up attributes after stream end: {cleaned_attrs}")

        # 若没有任何事件需要发送，则返回空字符串（上层会忽略）
        if not events:
            return ConversionResult(success=True, data="")

        # 将事件按 "\n\n" 分组，每个完整事件作为列表的一个元素
        complete_events = []
        i = 0
        while i < len(events):
            if events[i].startswith("event:") or events[i].startswith("data:"):
                # 找到一个完整事件的结束（下一个空行）
                event_lines = []
                while i < len(events) and events[i] != "":
                    event_lines.append(events[i])
                    i += 1
                # 添加结束的空行
                if i < len(events) and events[i] == "":
                    event_lines.append("")
                    i += 1
                # 将完整事件拼接成字符串
                complete_events.append("\n".join(event_lines) + "\n")
            else:
                i += 1

        self.logger.debug(f"Successfully converted Gemini chunk to {len(complete_events)} events")
        return ConversionResult(success=True, data=complete_events)
        
    
    def _parse_anthropic_sse_event(self, sse_data: str) -> ConversionResult:
        """解析Anthropic SSE事件数据，提取事件类型和数据
        
        """
        import re
        import json
        
        # 使用与claude-to-chatgpt项目相同的正则表达式
        # /event:\s*.*?\s*\ndata:\s*(.*?)(?=\n\n|\s*$)/gs
        pattern = r'event:\s*([^\n]*)\s*\ndata:\s*([^\n]*)'
        matches = re.findall(pattern, sse_data)
        
        parsed_events = []
        for event_type, data_content in matches:
            event_type = event_type.strip()
            data_content = data_content.strip()
            
            # 尝试解析JSON数据
            try:
                if data_content:
                    # 检查是否是结束标记
                    if data_content.strip() == "[DONE]":
                        break
                    parsed_data = json.loads(data_content)
                    parsed_events.append({
                        'event': event_type,
                        'data': parsed_data
                    })
            except json.JSONDecodeError:
                # 如果不是JSON，跳过或记录警告
                self.logger.warning(f"Failed to parse SSE data as JSON: {data_content}")
                continue
        
        return ConversionResult(success=True, data=parsed_events)
    
    def _convert_content_from_anthropic(self, content: Any) -> Any:
        """转换Anthropic内容到通用格式"""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # 处理多模态内容
            converted_content = []
            for item in content:
                if item.get("type") == "text":
                    converted_content.append({
                        "type": "text",
                        "text": item.get("text", "")
                    })
                elif item.get("type") == "image":
                    # 转换图像格式
                    source = item.get("source", {})
                    if source.get("type") == "base64":
                        media_type = source.get("media_type", "image/jpeg")
                        data_part = source.get("data", "")
                        converted_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{data_part}"
                            }
                        })
            return converted_content if len(converted_content) > 1 else converted_content[0].get("text", "") if converted_content else ""
        return content
    
    def _convert_content_to_gemini(
        self, content: Any, role: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """将 Anthropic 的 content 转为 Gemini parts 结构。

        Args:
            content: Anthropic 格式的消息内容
            role: 消息角色，当为 "user" 时会对文本执行协议标签过滤

        Returns:
            Gemini parts 列表
        """
        is_user = role == "user"

        # 1. 纯文本
        if isinstance(content, str):
            text_val = self._sanitize_protocol_tags(content) if is_user else content
            return [{"text": text_val}]

        # 2. 列表（可能混杂多模态 / tool 消息）
        if isinstance(content, list):
            gemini_parts: List[Dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    continue

                item_type = item.get("type")

                # 2.1 普通文本
                if item_type == "text":
                    text_content = item.get("text", "")
                    if is_user and isinstance(text_content, str):
                        text_content = self._sanitize_protocol_tags(text_content)
                    if text_content:  # 只添加非空文本
                        gemini_parts.append({"text": text_content})

                # 2.2 图像（base64）
                elif item_type == "image":
                    source = item.get("source", {})
                    if source.get("type") == "base64":
                        gemini_parts.append({
                            "inlineData": {
                                "mimeType": source.get("media_type", "image/jpeg"),
                                "data": source.get("data", "")
                            }
                        })

                # 2.3 tool_use → functionCall
                elif item_type == "tool_use":
                    tool_name = item.get("name", "")
                    tool_id = item.get("id", "")
                    
                    # 存储tool_id到function_name的映射，用于后续tool_result转换
                    if tool_id and tool_name:
                        self._tool_id_mapping[tool_name] = tool_id
                        # 同时存储到全局工具状态管理器中
                        tool_state_manager.store_tool_mapping(tool_name, tool_id)
                        
                    gemini_parts.append({
                        "functionCall": {
                            "name": tool_name,
                            "args": item.get("input", {})
                        }
                    })

                # 2.4 tool_result → functionResponse
                elif item_type == "tool_result":
                    fr = self._build_function_response(item)
                    if fr:
                        gemini_parts.append(fr)

            # 如果没有有效的 parts，返回空文本而不是空数组
            if not gemini_parts:
                return [{"text": ""}]
            
            return gemini_parts

        # 3. 单个 dict（可能就是 tool_result）
        if isinstance(content, dict):
            fr = self._build_function_response(content)
            if fr:
                return [fr]
            # 如果不是工具结果，转为文本
            content_text = content.get("text") or json.dumps(content, ensure_ascii=False)
            if is_user and isinstance(content_text, str):
                content_text = self._sanitize_protocol_tags(content_text)
            return [{"text": content_text}]

        # 4. 其它类型统一转字符串
        text_val = str(content) if content else ""
        if is_user and text_val:
            text_val = self._sanitize_protocol_tags(text_val)
        return [{"text": text_val}]

    # --------- 辅助：构造 functionResponse part ---------
    def _build_function_response(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """根据 tool_result 字段构造 Gemini functionResponse"""
        if not isinstance(item, dict):
            return None

        # 判定是否为工具结果
        is_result = (
            item.get("type") == "tool_result"
            or "tool_use_id" in item
            or "tool_output" in item
            or "result" in item
            or "content" in item
        )
        if not is_result:
            return None

        # 提取函数名
        func_name = None
        
        # 方法1：从映射表中获取（Anthropic格式）
        tool_use_id = item.get("tool_use_id") or item.get("id")
        if tool_use_id and hasattr(self, '_tool_use_mapping'):
            func_name = self._tool_use_mapping.get(tool_use_id)
        
        # 方法1.5：使用全局工具状态管理器
        if not func_name and tool_use_id:
            # 先尝试从ID中提取可能的函数名
            potential_func_name = None
            if str(tool_use_id).startswith("call_"):
                name_and_hash = tool_use_id[len("call_"):]
                potential_func_name = name_and_hash.rsplit("_", 1)[0]
            
            # 检查全局管理器中是否有对应的映射
            if potential_func_name:
                stored_id = tool_state_manager.get_tool_id(potential_func_name)
                if stored_id == tool_use_id:
                    func_name = potential_func_name
        
        # 方法2：从 tool_use_id 中提取（OpenAI格式）
        if not func_name and tool_use_id and str(tool_use_id).startswith("call_"):
            # 格式: call_<function_name>_<hash> ，函数名可能包含多个下划线
            name_and_hash = tool_use_id[len("call_"):]
            func_name = name_and_hash.rsplit("_", 1)[0]  # 去掉最后一个 hash 段
        
        # 方法3：直接从字段获取
        if not func_name:
            func_name = (
                item.get("tool_name")
                or item.get("name")
                or item.get("function_name")
            )

        if not func_name:
            return None

        # 提取结果内容
        func_response = None
        
        # 尝试多个可能的结果字段
        for key in ["content", "tool_output", "output", "response", "result"]:
            if key in item:
                func_response = item[key]
                break
        
        # 如果 content 是列表，尝试提取文本
        if isinstance(func_response, list) and func_response:
            text_parts = [p.get("text", "") for p in func_response if isinstance(p, dict) and p.get("type") == "text"]
            if text_parts:
                func_response = "".join(text_parts)
        
        # 确保有响应内容
        if func_response is None:
            func_response = ""

        # Gemini 要求 response 为 JSON 对象，若为原始字符串则包装
        if not isinstance(func_response, (dict, list)):
            func_response = {"content": str(func_response)}

        return {
            "functionResponse": {
                "name": func_name,
                "response": func_response
            }
        }
    
    def _map_finish_reason(self, reason: str, source_format: str, target_format: str) -> str:
        """映射结束原因"""
        reason_mappings = {
            "openai": {
                "anthropic": {
                    "stop": "end_turn",
                    "length": "max_tokens",
                    "content_filter": "stop_sequence",
                    "tool_calls": "tool_use"
                }
            },
            "gemini": {
                "anthropic": {
                    # 旧版本大写格式
                    "STOP": "end_turn",
                    "MAX_TOKENS": "max_tokens",
                    "SAFETY": "stop_sequence",
                    "RECITATION": "stop_sequence",
                    # 新版本小写格式（v1beta/v1 API）
                    "stop": "end_turn",
                    "length": "max_tokens",
                    "safety": "stop_sequence",
                    "recitation": "stop_sequence",
                    "other": "end_turn"
                }
            }
        }
        
        try:
            return reason_mappings[source_format][target_format].get(reason, "end_turn")
        except KeyError:
            return "end_turn"

    # ==================== 协议标签安全过滤 ====================

    # 预编译正则表达式，避免运行时重复编译
    _DANGEROUS_BLOCK_PATTERN = None
    _THINKING_TAG_PATTERN = None
    _GENERIC_XML_TAG_PATTERN = None

    @classmethod
    def _get_sanitize_patterns(
        cls,
    ) -> Tuple[Pattern[str], Pattern[str], Pattern[str]]:
        """延迟初始化正则表达式模式（线程安全的单例模式）"""
        if cls._DANGEROUS_BLOCK_PATTERN is None:
            # 高风险协议块：完整删除（包括内部内容）
            cls._DANGEROUS_BLOCK_PATTERN = re.compile(
                r'<\s*(invoke|tool_result)\b[^>]*>.*?</\s*\1\s*>',
                re.IGNORECASE | re.DOTALL
            )
            # thinking 标签：保留内部内容，仅去除标签壳
            cls._THINKING_TAG_PATTERN = re.compile(
                r'<\s*thinking\b[^>]*>(.*?)</\s*thinking\s*>',
                re.IGNORECASE | re.DOTALL
            )
            # 通用 XML 标签：去除标签壳
            cls._GENERIC_XML_TAG_PATTERN = re.compile(
                r'</?[a-zA-Z][a-zA-Z0-9_\-:]*[^>]*>'
            )
        return (
            cls._DANGEROUS_BLOCK_PATTERN,
            cls._THINKING_TAG_PATTERN,
            cls._GENERIC_XML_TAG_PATTERN,
        )

    def _sanitize_protocol_tags(self, text: str) -> str:
        """过滤潜在协议标签，防止用户通过伪造 XML 标签进行提示注入攻击。

        处理策略：
        1. <invoke>...</invoke>、<tool_result>...</tool_result>：整块删除
        2. <thinking>...</thinking>：保留内部自然语言内容，仅移除标签
        3. 其他 XML 样式标签：移除标签壳，保留内部文本

        注意：此方法仅用于用户/系统侧输入文本，不应用于模型输出内容。
        """
        if not isinstance(text, str) or not text:
            return text

        dangerous_pattern, thinking_pattern, generic_pattern = self._get_sanitize_patterns()

        # 1) 移除高风险协议块（包括内部内容）
        cleaned = dangerous_pattern.sub('', text)

        # 2) 去掉 <thinking> 标签外壳，保留内部自然语言内容
        cleaned = thinking_pattern.sub(r'\1', cleaned)

        # 3) 移除剩余的 XML 样式标签（仅标签本身，保留内部文本）
        cleaned = generic_pattern.sub('', cleaned)

        return cleaned

    # ==================== Schema 清理方法 ====================

    def _sanitize_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """递归移除Gemini不支持的JSON Schema关键字"""
        if not isinstance(schema, dict):
            return schema

        allowed_keys = {"type", "description", "properties", "required", "enum", "items"}
        sanitized = {k: v for k, v in schema.items() if k in allowed_keys}

        if "properties" in sanitized and isinstance(sanitized["properties"], dict):
            sanitized["properties"] = {
                prop_name: self._sanitize_schema(prop_schema)
                for prop_name, prop_schema in sanitized["properties"].items()
            }

        if "items" in sanitized:
            sanitized["items"] = self._sanitize_schema(sanitized["items"])

        return sanitized

    def _clean_json_schema_properties(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """递归清理Gemini不支持的JSON Schema属性"""
        if not isinstance(schema, dict):
            return schema

        # 移除所有非标准属性
        sanitized = {k: v for k, v in schema.items() if k in {"type", "description", "properties", "required", "enum", "items"}}

        if "properties" in sanitized and isinstance(sanitized["properties"], dict):
            sanitized["properties"] = {
                prop_name: self._clean_json_schema_properties(prop_schema)
                for prop_name, prop_schema in sanitized["properties"].items()
            }

        if "items" in sanitized:
            sanitized["items"] = self._clean_json_schema_properties(sanitized["items"])

        return sanitized

    def _deep_clean_for_gemini(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """深度清理Gemini请求数据，移除可能引起协议错误的字段和格式问题"""
        if not isinstance(data, dict):
            return data
        
        cleaned = {}
        
        for key, value in data.items():
            # 处理 system_instruction
            if key == "system_instruction" and isinstance(value, dict):
                cleaned_si = {}
                if "parts" in value and isinstance(value["parts"], list):
                    clean_parts = []
                    for part in value["parts"]:
                        if isinstance(part, dict) and "text" in part:
                            # 确保text字段是纯字符串，无特殊字符或编码问题
                            text_content = str(part["text"]).strip()
                            if text_content:  # 只添加非空文本
                                clean_parts.append({"text": text_content})
                    if clean_parts:
                        cleaned_si["parts"] = clean_parts
                        cleaned[key] = cleaned_si
            
            # 处理 contents
            elif key == "contents" and isinstance(value, list):
                clean_contents = []
                for content in value:
                    if isinstance(content, dict):
                        clean_content = {}
                        # 确保role字段正确
                        if "role" in content:
                            clean_content["role"] = str(content["role"])
                        # 清理parts
                        if "parts" in content and isinstance(content["parts"], list):
                            clean_parts = []
                            for part in content["parts"]:
                                if isinstance(part, dict):
                                    clean_part = {}
                                    # 只保留支持的字段
                                    if "text" in part:
                                        text_val = str(part["text"]).strip() if part["text"] else ""
                                        if text_val:
                                            clean_part["text"] = text_val
                                    elif "functionCall" in part:
                                        clean_part["functionCall"] = part["functionCall"]
                                    elif "functionResponse" in part:
                                        clean_part["functionResponse"] = part["functionResponse"]
                                    elif "inlineData" in part:
                                        clean_part["inlineData"] = part["inlineData"]
                                    
                                    if clean_part:  # 只添加非空part
                                        clean_parts.append(clean_part)
                            
                            if clean_parts:
                                clean_content["parts"] = clean_parts
                                clean_contents.append(clean_content)
                
                if clean_contents:
                    cleaned[key] = clean_contents
            
            # 处理 generationConfig
            elif key == "generationConfig" and isinstance(value, dict):
                clean_gen_config = {}
                # 只保留Gemini支持的生成配置字段
                allowed_gen_keys = {"temperature", "topP", "topK", "maxOutputTokens", "stopSequences", "thinkingConfig"}
                for gen_key, gen_value in value.items():
                    if gen_key in allowed_gen_keys and gen_value is not None:
                        clean_gen_config[gen_key] = gen_value
                cleaned[key] = clean_gen_config
            
            # 处理 tools
            elif key == "tools" and isinstance(value, list):
                clean_tools = []
                for tool in value:
                    if isinstance(tool, dict) and "functionDeclarations" in tool:
                        clean_func_decls = []
                        for func_decl in tool["functionDeclarations"]:
                            if isinstance(func_decl, dict):
                                clean_decl = {}
                                if "name" in func_decl:
                                    clean_decl["name"] = str(func_decl["name"])
                                if "description" in func_decl:
                                    clean_decl["description"] = str(func_decl["description"])
                                if "parameters" in func_decl:
                                    # 应用现有的schema清理
                                    clean_decl["parameters"] = self._clean_json_schema_properties(func_decl["parameters"])
                                clean_func_decls.append(clean_decl)
                        
                        if clean_func_decls:
                            clean_tools.append({"functionDeclarations": clean_func_decls})
                
                if clean_tools:
                    cleaned[key] = clean_tools
            
            # 其他字段直接保留
            else:
                cleaned[key] = value
        
        return cleaned