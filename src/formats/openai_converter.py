"""
OpenAI Chat Completions 格式转换器
处理 OpenAI Chat Completions 与其他格式之间的转换
"""
from typing import Dict, Any, Optional, List
import json
import copy

from src.utils.logger import log_diagnose_event

from .base_converter import BaseConverter, ConversionResult, ConversionError
from .anthropic_sse_parser import AnthropicSSEParseError, parse_anthropic_sse_event


class OpenAIConverter(BaseConverter):
    """OpenAI格式转换器"""
    
    def __init__(self):
        super().__init__()
        self.original_model = None
    
    def set_original_model(self, model: str):
        """设置原始模型名称"""
        self.original_model = model
    
    def reset_streaming_state(self):
        """重置所有流式相关的状态变量，避免状态污染"""
        streaming_attrs = [
            '_anthropic_stream_id',
            '_openai_sent_start',
            '_openai_text_started',
            '_stream_id',
            '_anthropic_tool_state',
            '_anthropic_finish_reason',
        ]
        for attr in streaming_attrs:
            if hasattr(self, attr):
                delattr(self, attr)

    def get_supported_formats(self) -> List[str]:
        """获取支持的格式列表"""
        return ["openai", "openai_chat_completions", "anthropic", "gemini"]
    
    def convert_request(
        self,
        data: Dict[str, Any],
        target_format: str,
        headers: Optional[Dict[str, str]] = None
    ) -> ConversionResult:
        """转换OpenAI请求到目标格式"""
        try:
            if target_format in {"openai", "openai_chat_completions"}:
                # OpenAI Chat Completions 到 OpenAI Chat Completions，直接透传
                return ConversionResult(success=True, data=data)
            elif target_format == "anthropic":
                return self._convert_to_anthropic_request(data)
            elif target_format == "gemini":
                return self._convert_to_gemini_request(data)
            else:
                return ConversionResult(
                    success=False,
                    error=f"Unsupported target format: {target_format}"
                )
        except Exception as e:
            self.logger.error(f"Failed to convert OpenAI request to {target_format}: {e}")
            return ConversionResult(success=False, error=str(e))
    
    def convert_response(
        self,
        data: Dict[str, Any],
        source_format: str,
        target_format: str
    ) -> ConversionResult:
        """转换响应到OpenAI格式"""
        try:
            if source_format == "openai":
                return ConversionResult(success=True, data=data)
            elif source_format == "anthropic":
                return self._convert_from_anthropic_response(data)
            elif source_format == "gemini":
                return self._convert_from_gemini_response(data)
            else:
                return ConversionResult(
                    success=False,
                    error=f"Unsupported source format: {source_format}"
                )
        except Exception as e:
            self.logger.error(f"Failed to convert {source_format} response to OpenAI: {e}")
            return ConversionResult(success=False, error=str(e))
    
    def _convert_to_anthropic_request(self, data: Dict[str, Any]) -> ConversionResult:
        """转换OpenAI请求到Anthropic格式"""
        result_data = {}
        
        # 处理模型 - Anthropic 要求必须有 model，不做映射直接传递
        if "model" not in data:
            raise ConversionError("model parameter is required for Anthropic API")
        result_data["model"] = data["model"]
        
        # 处理消息和系统消息
        if "messages" in data:
            system_message, filtered_messages = self._extract_system_message(data["messages"])
            
            if system_message:
                result_data["system"] = system_message
            
            # 转换消息格式
            anthropic_messages = []
            for msg in filtered_messages:
                role = msg.get("role")
                
                if role == "user":
                    # 用户消息，正常转换内容
                    anthropic_messages.append({
                        "role": "user",
                        "content": self._convert_content_to_anthropic(msg.get("content", ""))
                    })
                    
                elif role == "assistant":
                    # 助手消息，检查是否包含tool_calls
                    if msg.get("tool_calls"):
                        # 转换tool_calls为tool_use content blocks
                        content_blocks = []
                        for tc in msg["tool_calls"]:
                            if tc and tc.get("type") == "function" and "function" in tc:
                                func = tc["function"]
                                # 解析arguments JSON字符串
                                args_str = func.get("arguments", "{}")
                                try:
                                    args_obj = json.loads(args_str) if args_str else {}
                                except json.JSONDecodeError:
                                    args_obj = {}
                                
                                content_blocks.append({
                                    "type": "tool_use",
                                    "id": tc.get("id", ""),
                                    "name": func.get("name", ""),
                                    "input": args_obj
                                })
                        
                        anthropic_messages.append({
                            "role": "assistant",
                            "content": content_blocks
                        })
                    else:
                        # 普通助手消息
                        anthropic_messages.append({
                            "role": "assistant",
                            "content": self._convert_content_to_anthropic(msg.get("content", ""))
                        })
                        
                elif role == "tool":
                    # 工具消息，转换为用户消息中的tool_result
                    tool_call_id = msg.get("tool_call_id", "")
                    content = str(msg.get("content", ""))
                    
                    anthropic_messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_call_id,
                                "content": content
                            }
                        ]
                    })
            
            result_data["messages"] = anthropic_messages
        
        # 处理其他参数
        # Anthropic 要求必须有 max_tokens，按优先级处理：
        # 1. 传入的max_tokens（最高优先级）
        # 2. 环境变量ANTHROPIC_MAX_TOKENS
        # 3. 都没有则报错
        if "max_tokens" in data:
            # 优先级1：使用传入的max_tokens
            result_data["max_tokens"] = data["max_tokens"]
        else:
            # 优先级2：检查环境变量ANTHROPIC_MAX_TOKENS
            import os
            env_max_tokens = os.environ.get("ANTHROPIC_MAX_TOKENS")
            if env_max_tokens:
                try:
                    max_tokens = int(env_max_tokens)
                    self.logger.info(f"Using ANTHROPIC_MAX_TOKENS from environment: {max_tokens}")
                    result_data["max_tokens"] = max_tokens
                except ValueError:
                    self.logger.warning(f"Invalid ANTHROPIC_MAX_TOKENS value '{env_max_tokens}', must be integer")
                    env_max_tokens = None
            
            if not env_max_tokens:
                # 优先级3：都没有则报错，要求用户明确指定
                raise ValueError(f"max_tokens is required for Anthropic API. Please specify max_tokens in the request or set ANTHROPIC_MAX_TOKENS environment variable.")
        
        if "temperature" in data:
            result_data["temperature"] = data["temperature"]
        if "top_p" in data:
            result_data["top_p"] = data["top_p"]
        if "stop" in data:
            result_data["stop_sequences"] = data["stop"] if isinstance(data["stop"], list) else [data["stop"]]
        if "stream" in data:
            result_data["stream"] = data["stream"]
        
        # 处理工具调用
        if "tools" in data:
            anthropic_tools = []
            for tool in data["tools"]:
                if tool.get("type") == "function" and "function" in tool:
                    func = tool["function"]
                    anthropic_tools.append({
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {})
                    })
            result_data["tools"] = anthropic_tools
        
        # 处理思考预算转换 (OpenAI max_completion_tokens + reasoning_effort -> Anthropic thinkingBudget)
        # 通过max_completion_tokens判断是否为思考模式
        if "max_completion_tokens" in data:
            log_diagnose_event(self.logger, "OpenAI max_completion_tokens detected, enabling Anthropic thinking")
            
            # 确定reasoning_effort：如果没传则默认为medium
            reasoning_effort = data.get("reasoning_effort", "medium")
            if "reasoning_effort" not in data:
                self.logger.info(f"🧠 [THINKING BUDGET] 未指定reasoning_effort，默认设为: '{reasoning_effort}'")
            else:
                self.logger.info(f"🧠 [THINKING BUDGET] 使用指定的reasoning_effort参数: '{reasoning_effort}'")
            
            # 根据环境变量映射reasoning_effort到具体的token数值
            import os
            thinking_budget = None
            env_key = None
            
            if reasoning_effort == "low":
                env_key = "OPENAI_LOW_TO_ANTHROPIC_TOKENS"
                env_value = os.environ.get(env_key)
            elif reasoning_effort == "medium":
                env_key = "OPENAI_MEDIUM_TO_ANTHROPIC_TOKENS"
                env_value = os.environ.get(env_key)
            elif reasoning_effort == "high":
                env_key = "OPENAI_HIGH_TO_ANTHROPIC_TOKENS"
                env_value = os.environ.get(env_key)
            
            self.logger.info(f"🔍 [THINKING BUDGET] 查找环境变量: {env_key}")
            
            if not env_value:
                self.logger.error(f"❌ [THINKING BUDGET] 环境变量未配置: {env_key}")
                raise ConversionError(f"环境变量 {env_key} 未配置。请在.env文件中设置该参数以支持OpenAI reasoning_effort到Anthropic thinkingBudget的转换。")
            
            self.logger.info(f"✅ [THINKING BUDGET] 环境变量获取成功: {env_key} = {env_value}")
            
            try:
                thinking_budget = int(env_value)
                self.logger.info(f"✅ [THINKING BUDGET] token数值解析成功: {thinking_budget}")
            except ValueError:
                self.logger.error(f"❌ [THINKING BUDGET] token数值解析失败: '{env_value}' 不是有效整数")
                raise ConversionError(f"环境变量 {env_key} 的值 '{env_value}' 不是有效的整数。")
            
            if thinking_budget:
                result_data["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget
                }
                self.logger.info(f"🎯 [THINKING BUDGET] 转换完成: OpenAI reasoning_effort '{reasoning_effort}' -> Anthropic thinkingBudget {thinking_budget}")
                self.logger.info(f"📤 [THINKING BUDGET] Anthropic请求将包含: {{\"thinking\": {{\"type\": \"enabled\", \"budget_tokens\": {thinking_budget}}}}}")
        
        return ConversionResult(success=True, data=result_data)
    
    def _convert_to_gemini_request(self, data: Dict[str, Any]) -> ConversionResult:
        """转换OpenAI请求到Gemini格式"""
        result_data = {}
        # 透传模型字段，保持原样传递给 Gemini
        if "model" in data:
            result_data["model"] = data["model"]

        def _sanitize_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
            """递归移除Gemini不支持的JSON Schema关键字"""
            if not isinstance(schema, dict):
                return schema

            allowed_keys = {"type", "description", "properties", "required", "enum", "items"}
            sanitized = {k: v for k, v in schema.items() if k in allowed_keys}

            # 递归处理子属性
            if "properties" in sanitized and isinstance(sanitized["properties"], dict):
                sanitized["properties"] = {
                    prop_name: _sanitize_schema(prop_schema)
                    for prop_name, prop_schema in sanitized["properties"].items()
                }

            # 处理 items
            if "items" in sanitized:
                sanitized["items"] = _sanitize_schema(sanitized["items"])

            return sanitized
        
        # 处理消息和系统消息
        if "messages" in data:
            system_message, filtered_messages = self._extract_system_message(data["messages"])
            
            if system_message:
                # 确保系统指令格式正确
                system_content = str(system_message).strip() if system_message else ""
                if system_content:
                    result_data["system_instruction"] = {
                        "parts": [{"text": system_content}]
                    }
            
            # 转换消息格式
            gemini_contents = []
            for msg in filtered_messages:
                msg_role = msg.get("role")

                # 1) 普通 user 消息
                if msg_role == "user":
                    gemini_contents.append({
                        "role": "user",
                        "parts": self._convert_content_to_gemini(msg.get("content", ""))
                    })

                # 2) assistant 带 tool_calls（functionCall）
                elif msg_role == "assistant" and msg.get("tool_calls"):
                    parts = []
                    for tc in msg["tool_calls"]:
                        if tc and "function" in tc:
                            fn_name = tc["function"].get("name")
                            # OpenAI 规定 arguments 为 JSON 字符串
                            arg_str = tc["function"].get("arguments", "{}")
                        else:
                            continue
                        try:
                            arg_obj = json.loads(arg_str) if isinstance(arg_str, str) else arg_str
                        except Exception:
                            arg_obj = {}
                        parts.append({
                            "functionCall": {
                                "name": fn_name,
                                "args": arg_obj
                            }
                        })
                    gemini_contents.append({
                        "role": "model",
                        "parts": parts if parts else [{"text": ""}]
                    })

                # 3) tool 结果 -> functionResponse
                elif msg_role == "tool":
                    tool_call_id = msg.get("tool_call_id", "")
                    # 从 call_<name>_<hash> 提取 name
                    fn_name = ""
                    if tool_call_id.startswith("call_"):
                        fn_name = "_".join(tool_call_id.split("_")[1:-1])  # 保留中间含下划线的函数名
                    response_content = msg.get("content")
                    # Gemini 要求 response 为对象
                    if not isinstance(response_content, dict):
                        response_content = {"content": response_content}
                    gemini_contents.append({
                        "role": "tool",
                        "parts": [{
                            "functionResponse": {
                                "name": fn_name,
                                "response": response_content
                            }
                        }]
                    })

                # 4) assistant 普通文本
                else:
                    gemini_contents.append({
                        "role": "model",
                        "parts": self._convert_content_to_gemini(msg.get("content", ""))
                    })
            
            result_data["contents"] = gemini_contents
        
        # 处理生成配置
        generation_config = {}
        if "temperature" in data:
            generation_config["temperature"] = data["temperature"]
        if "top_p" in data:
            generation_config["topP"] = data["top_p"]
        
        # 处理maxOutputTokens（Gemini的max_tokens等价字段）
        # 优先级：客户端max_tokens > 环境变量ANTHROPIC_MAX_TOKENS > 不设置（让Gemini使用默认值）
        if "max_tokens" in data:
            # 优先使用客户端传入的max_tokens
            generation_config["maxOutputTokens"] = data["max_tokens"]
        else:
            # 如果客户端没有传max_tokens，检查环境变量ANTHROPIC_MAX_TOKENS
            import os
            env_max_tokens = os.environ.get("ANTHROPIC_MAX_TOKENS")
            if env_max_tokens:
                try:
                    max_tokens = int(env_max_tokens)
                    generation_config["maxOutputTokens"] = max_tokens
                    self.logger.info(f"Using ANTHROPIC_MAX_TOKENS for Gemini maxOutputTokens: {max_tokens}")
                except ValueError:
                    self.logger.warning(f"Invalid ANTHROPIC_MAX_TOKENS value '{env_max_tokens}', must be integer")
                    # 不设置maxOutputTokens，让Gemini使用默认值
        
        if "stop" in data:
            generation_config["stopSequences"] = data["stop"] if isinstance(data["stop"], list) else [data["stop"]]
        
        # Gemini 2.x 要求 generationConfig 字段始终存在，即使为空对象
        result_data["generationConfig"] = generation_config
        
        # 处理工具调用
        if "tools" in data:
            function_declarations = []
            for tool in data["tools"]:
                if tool.get("type") == "function" and "function" in tool:
                    func = tool["function"]
                    function_declarations.append({
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "parameters": _sanitize_schema(func.get("parameters", {}))
                    })
            
            if function_declarations:
                # Gemini官方规范使用 camelCase: functionDeclarations
                result_data["tools"] = [{
                    "functionDeclarations": function_declarations
                }]

        # 处理结构化输出（无论是否包含工具调用）
        if "response_format" in data and data["response_format"].get("type") == "json_schema":
            generation_config["response_mime_type"] = "application/json"
            if "json_schema" in data["response_format"]:
                generation_config["response_schema"] = data["response_format"]["json_schema"].get("schema", {})
            result_data["generationConfig"] = generation_config
        
        # 处理思考预算转换 (OpenAI max_completion_tokens + reasoning_effort -> Gemini thinkingBudget)
        # 通过max_completion_tokens判断是否为思考模式
        if "max_completion_tokens" in data:
            log_diagnose_event(self.logger, "OpenAI max_completion_tokens detected, enabling Anthropic thinking")
            
            # 确定reasoning_effort：如果没传则默认为medium
            reasoning_effort = data.get("reasoning_effort", "medium")
            if "reasoning_effort" not in data:
                self.logger.info(f"🧠 [THINKING BUDGET] 未指定reasoning_effort，默认设为: '{reasoning_effort}'")
            else:
                self.logger.info(f"🧠 [THINKING BUDGET] 使用指定的reasoning_effort参数: '{reasoning_effort}'")
            
            # 根据环境变量映射reasoning_effort到具体的token数值
            import os
            thinking_budget = None
            env_key = None
            
            if reasoning_effort == "low":
                env_key = "OPENAI_LOW_TO_GEMINI_TOKENS"
                env_value = os.environ.get(env_key)
            elif reasoning_effort == "medium":
                env_key = "OPENAI_MEDIUM_TO_GEMINI_TOKENS"
                env_value = os.environ.get(env_key)
            elif reasoning_effort == "high":
                env_key = "OPENAI_HIGH_TO_GEMINI_TOKENS"
                env_value = os.environ.get(env_key)
            
            self.logger.info(f"🔍 [THINKING BUDGET] 查找环境变量: {env_key}")
            
            if not env_value:
                self.logger.error(f"❌ [THINKING BUDGET] 环境变量未配置: {env_key}")
                raise ConversionError(f"环境变量 {env_key} 未配置。请在.env文件中设置该参数以支持OpenAI reasoning_effort到Gemini thinkingBudget的转换。")
            
            self.logger.info(f"✅ [THINKING BUDGET] 环境变量获取成功: {env_key} = {env_value}")
            
            try:
                thinking_budget = int(env_value)
                self.logger.info(f"✅ [THINKING BUDGET] token数值解析成功: {thinking_budget}")
            except ValueError:
                self.logger.error(f"❌ [THINKING BUDGET] token数值解析失败: '{env_value}' 不是有效整数")
                raise ConversionError(f"环境变量 {env_key} 的值 '{env_value}' 不是有效的整数。")
            
            if thinking_budget:
                generation_config["thinkingConfig"] = {
                    "thinkingBudget": thinking_budget
                }
                result_data["generationConfig"] = generation_config
                self.logger.info(f"🎯 [THINKING BUDGET] 转换完成: OpenAI reasoning_effort '{reasoning_effort}' -> Gemini thinkingBudget {thinking_budget}")
                self.logger.info(f"📤 [THINKING BUDGET] Gemini请求将包含: {{\"generationConfig\": {{\"thinkingConfig\": {{\"thinkingBudget\": {thinking_budget}}}}}}}")
        
        return ConversionResult(success=True, data=result_data)
    
    def _convert_from_anthropic_response(self, data: Dict[str, Any]) -> ConversionResult:
        """转换Anthropic响应到OpenAI格式"""
        # 必须有原始模型名称，否则报错
        if not self.original_model:
            raise ValueError("Original model name is required for response conversion")
            
        import time
        result_data = {
            "id": f"chatcmpl-{data.get('id', 'anthropic')}",
            "object": "chat.completion",
            "created": int(time.time()),  # 使用当前时间戳
            "model": self.original_model,  # 使用原始模型名称
            "choices": [],
            "usage": {}
        }
        
        # 处理内容、工具调用和思考内容
        content = ""
        tool_calls = []
        thinking_content = ""
        
        if "content" in data and isinstance(data["content"], list):
            for item in data["content"]:
                if item.get("type") == "text":
                    content += item.get("text", "")
                elif item.get("type") == "thinking":
                    # 收集thinking内容，OpenAI Chat Completions API不直接支持reasoning格式
                    # 我们将thinking内容作为特殊标记包含在响应中
                    thinking_text = item.get("thinking", "")
                    if thinking_text.strip():
                        thinking_content += thinking_text
                elif item.get("type") == "tool_use":
                    # 转换 Anthropic tool_use 为 OpenAI tool_calls
                    tool_calls.append({
                        "id": item.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": json.dumps(item.get("input", {}), ensure_ascii=False)
                        }
                    })
        
        # 如果有thinking内容，将其作为前缀添加到content中
        # 使用特殊格式标记，便于客户端识别和处理
        if thinking_content.strip():
            content = f"<thinking>\n{thinking_content.strip()}\n</thinking>\n\n{content}"
        
        # 构建消息，根据是否有工具调用决定结构
        message = {"role": "assistant"}
        
        if tool_calls:
            # 有工具调用时，content 可以为 None（OpenAI 规范）
            message["content"] = content if content else None
            message["tool_calls"] = tool_calls
            finish_reason = "tool_calls"
        else:
            # 没有工具调用时，只有文本内容
            message["content"] = content
            finish_reason = self._map_finish_reason(data.get("stop_reason", ""), "anthropic", "openai")
        
        result_data["choices"] = [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason
        }]
        
        # 处理使用情况
        if "usage" in data and data["usage"] is not None:
            usage = data["usage"]
            result_data["usage"] = {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            }
        
        return ConversionResult(success=True, data=result_data)
    
    def _convert_from_gemini_response(self, data: Dict[str, Any]) -> ConversionResult:
        """转换Gemini响应到OpenAI格式"""
        import time
        import random
        import string
        # 生成类似OpenAI的ID格式
        random_id = ''.join(random.choices(string.ascii_letters + string.digits, k=29))
        # 必须有原始模型名称，否则报错
        if not self.original_model:
            raise ValueError("Original model name is required for response conversion")
            
        result_data = {
            "id": f"chatcmpl-{random_id}",
            "object": "chat.completion",
            "created": int(time.time()),  # 使用当前时间戳
            "model": self.original_model,  # 必须使用原始模型名称
            "usage": {},
            "choices": []
        }
        
        # 处理候选结果
        if "candidates" in data and data["candidates"] and data["candidates"][0]:
            candidate = data["candidates"][0]

            content_text = ""
            tool_calls = []

            if "content" in candidate and "parts" in candidate["content"]:
                for part in candidate["content"]["parts"]:
                    # 普通文本
                    if "text" in part:
                        content_text += part["text"]
                    # 函数调用
                    elif "functionCall" in part:
                        fc = part["functionCall"]
                        fc_name = fc.get("name", "")
                        fc_args = fc.get("args", {})
                        # 生成 tool_call id，遵循 "call_<name>_<hash>" 规则
                        random_hash = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
                        tool_calls.append({
                            "id": f"call_{fc_name}_{random_hash}",
                            "type": "function",
                            "function": {
                                "name": fc_name,
                                "arguments": json.dumps(fc_args, ensure_ascii=False)
                            }
                        })

            # 构造 message，根据是否存在工具调用决定结构
            if tool_calls:
                message_dict = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls
                }
                finish_reason_val = "tool_calls"
            else:
                message_dict = {
                    "role": "assistant",
                    "content": content_text
                }
                finish_reason_val = self._map_finish_reason(candidate.get("finishReason", ""), "gemini", "openai")

            result_data["choices"] = [{
                "message": message_dict,
                "finish_reason": finish_reason_val,
                "index": 0
            }]
        
        # 处理使用情况
        if "usageMetadata" in data:
            usage = data["usageMetadata"]
            result_data["usage"] = {
                "prompt_tokens": usage.get("promptTokenCount", 0),
                "completion_tokens": usage.get("candidatesTokenCount", 0),
                "total_tokens": usage.get("totalTokenCount", 0)
            }
        
        return ConversionResult(success=True, data=result_data)
    
    def _convert_from_gemini_streaming_chunk(self, data: Dict[str, Any]) -> ConversionResult:
        """转换Gemini流式响应chunk到OpenAI格式"""
        import time
        import random
        import string
        
        # 生成一致的随机ID（在同一次对话中保持一致）
        if not hasattr(self, '_stream_id'):
            self._stream_id = ''.join(random.choices(string.ascii_letters + string.digits, k=29))
        
        # 检查是否是完整的流式响应结束
        if "candidates" in data and data["candidates"] and data["candidates"][0] and data["candidates"][0].get("finishReason"):
            # 这是最后的chunk，包含finishReason
            # 提取内容（Gemini 最后一个 chunk 仍可能带文本或工具调用）
            final_content = ""
            tool_calls = []
            candidate = data["candidates"][0]
            if "content" in candidate and candidate["content"].get("parts"):
                for part in candidate["content"]["parts"]:
                    if "text" in part:
                        final_content += part["text"]
                    elif "functionCall" in part:
                        # 处理工具调用
                        fc = part["functionCall"]
                        fc_name = fc.get("name", "")
                        fc_args = fc.get("args", {})
                        
                        # 生成 tool_call id
                        if not hasattr(self, '_tool_call_counter'):
                            self._tool_call_counter = 0
                        self._tool_call_counter += 1
                        
                        tool_calls.append({
                            "id": f"call_{fc_name}_{self._tool_call_counter:04d}",
                            "type": "function",
                            "function": {
                                "name": fc_name,
                                "arguments": json.dumps(fc_args, ensure_ascii=False)
                            }
                        })
            
            if not self.original_model:
                raise ValueError("Model name is required for streaming response")
            
            # 构建delta内容
            delta = {}
            if final_content:
                delta["content"] = final_content
            if tool_calls:
                delta["tool_calls"] = tool_calls
            
            # 确定finish_reason - 如果有工具调用，应该是tool_calls
            finish_reason = data["candidates"][0].get("finishReason", "")
            if tool_calls:
                mapped_finish_reason = "tool_calls"
            else:
                mapped_finish_reason = self._map_finish_reason(finish_reason, "gemini", "openai")
            
            result_data = {
                "id": f"chatcmpl-{self._stream_id}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": self.original_model,
                "choices": [{
                    "index": 0,
                    "delta": delta,
                    "finish_reason": mapped_finish_reason
                }]
            }
            
            # 添加usage信息（如果有）
            if "usageMetadata" in data:
                usage = data["usageMetadata"]
                result_data["usage"] = {
                    "prompt_tokens": usage.get("promptTokenCount", 0),
                    "completion_tokens": usage.get("candidatesTokenCount", 0),
                    "total_tokens": usage.get("totalTokenCount", 0)
                }
            
            return ConversionResult(success=True, data=result_data)
            
        # 检查是否有增量内容
        elif "candidates" in data and data["candidates"] and data["candidates"][0]:
            candidate = data["candidates"][0]
            content = ""
            tool_calls = []
            
            if "content" in candidate and "parts" in candidate["content"]:
                for part in candidate["content"]["parts"]:
                    if "text" in part:
                        content += part["text"]
                    elif "functionCall" in part:
                        # 处理工具调用
                        fc = part["functionCall"]
                        fc_name = fc.get("name", "")
                        fc_args = fc.get("args", {})
                        
                        # 生成 tool_call id
                        if not hasattr(self, '_tool_call_counter'):
                            self._tool_call_counter = 0
                        self._tool_call_counter += 1
                        
                        tool_calls.append({
                            "id": f"call_{fc_name}_{self._tool_call_counter:04d}",
                            "type": "function",
                            "function": {
                                "name": fc_name,
                                "arguments": json.dumps(fc_args, ensure_ascii=False)
                            }
                        })
            
            # 始终创建chunk，即使内容为空（这对流式很重要）
            if not self.original_model:
                raise ValueError("Model name is required for streaming response")
            
            # 构建delta内容
            delta = {}
            if content:
                delta["content"] = content
            if tool_calls:
                delta["tool_calls"] = tool_calls
            
            result_data = {
                "id": f"chatcmpl-{self._stream_id}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": self.original_model,
                "choices": [{
                    "index": 0,
                    "delta": delta,
                    "finish_reason": None
                }]
            }
            
            return ConversionResult(success=True, data=result_data)
        
        # 如果没有candidates，可能是其他类型的数据（如开头的metadata）
        # 返回空的delta保持流式连接
        if not self.original_model:
            raise ValueError("Model name is required for streaming response")
        
        result_data = {
            "id": f"chatcmpl-{self._stream_id}",
            "object": "chat.completion.chunk", 
            "created": int(time.time()),
            "model": self.original_model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": None
            }]
        }
        
        return ConversionResult(success=True, data=result_data)
    
    def _convert_from_anthropic_streaming_chunk(self, data: Dict[str, Any]) -> ConversionResult:
        """转换Anthropic流式响应chunk到OpenAI Chat Completions格式"""
        import time
        import random
        import string

        if not hasattr(self, '_stream_id'):
            self._stream_id = ''.join(random.choices(string.ascii_letters + string.digits, k=29))

        if not self.original_model:
            raise ValueError("Original model name is required for streaming response conversion")

        try:
            parsed_event = parse_anthropic_sse_event(data)
        except AnthropicSSEParseError:
            result_data = {
                "id": f"chatcmpl-{self._stream_id}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": self.original_model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": None
                }]
            }
            return ConversionResult(success=True, data=result_data)

        event_data = parsed_event.payload
        event_type = parsed_event.event_type
        result_data = {
            "id": f"chatcmpl-{self._stream_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.original_model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": None
            }]
        }

        if not hasattr(self, '_anthropic_tool_state'):
            self._anthropic_tool_state = {}
        if not hasattr(self, '_anthropic_finish_reason'):
            self._anthropic_finish_reason = None

        if event_type == "content_block_start":
            content_block = event_data.get("content_block", {})
            if content_block.get("type") == "tool_use":
                tool_index = event_data.get("index", 0)
                self._anthropic_tool_state[tool_index] = {
                    "id": content_block.get("id", ""),
                    "name": content_block.get("name", ""),
                    "arguments": ""
                }
                result_data["choices"][0]["delta"]["tool_calls"] = [{
                    "index": tool_index,
                    "id": content_block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": content_block.get("name", "")
                    }
                }]

        elif event_type == "content_block_delta":
            delta = event_data.get("delta", {})
            index = event_data.get("index", 0)

            if delta.get("type") == "text_delta":
                result_data["choices"][0]["delta"]["content"] = delta.get("text", "")
            elif delta.get("type") == "input_json_delta":
                if index in self._anthropic_tool_state:
                    partial_json = delta.get("partial_json", "")
                    self._anthropic_tool_state[index]["arguments"] += partial_json
                    result_data["choices"][0]["delta"] = {}

        elif event_type == "content_block_stop":
            index = event_data.get("index", 0)
            tool_state = self._anthropic_tool_state.get(index)
            if tool_state:
                result_data["choices"][0]["delta"]["tool_calls"] = [{
                    "index": index,
                    "id": tool_state.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tool_state.get("name", ""),
                        "arguments": tool_state.get("arguments", "")
                    }
                }]

        elif event_type == "message_delta":
            delta = event_data.get("delta", {})
            stop_reason = delta.get("stop_reason")
            if stop_reason:
                self._anthropic_finish_reason = self._map_finish_reason(stop_reason, "anthropic", "openai")
                result_data["choices"][0]["finish_reason"] = self._anthropic_finish_reason

        elif event_type == "message_stop":
            result_data["choices"][0]["finish_reason"] = self._anthropic_finish_reason or "stop"
            if hasattr(self, '_anthropic_tool_state'):
                delattr(self, '_anthropic_tool_state')
            if hasattr(self, '_anthropic_finish_reason'):
                delattr(self, '_anthropic_finish_reason')

        return ConversionResult(success=True, data=result_data)
    
    def _convert_content_to_anthropic(self, content: Any) -> Any:
        """转换内容到Anthropic格式"""
        if isinstance(content, str):
            # 检查是否包含thinking标签
            return self._extract_thinking_from_text(content)
        elif isinstance(content, list):
            # 处理多模态内容 - 应用Anthropic最佳实践：图片在文本之前
            text_items = []
            image_items = []
            other_items = []
            
            # 1. 首先分类所有内容项
            for item in content:
                if item.get("type") == "text":
                    text_items.append(item)
                elif item.get("type") == "image_url":
                    image_items.append(item)
                else:
                    other_items.append(item)
            
            anthropic_content = []
            
            # 2. 首先处理图片（Anthropic最佳实践）
            for item in image_items:
                image_url = item.get("image_url", {}).get("url", "")
                if image_url.startswith("data:"):
                    # 处理base64图像
                    try:
                        media_type, data_part = image_url.split(";base64,", 1)
                        media_type = media_type.replace("data:", "")
                        anthropic_content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": data_part
                            }
                        })
                        self.logger.info(f"✅ OpenAI->Anthropic: Image processed FIRST (best practice): {media_type}")
                    except ValueError as e:
                        self.logger.error(f"Failed to parse base64 image URL: {e}")
            
            # 3. 然后处理文本内容
            for item in text_items:
                text_content = item.get("text", "")
                # 检查文本是否包含thinking标签
                extracted = self._extract_thinking_from_text(text_content)
                if isinstance(extracted, list):
                    # 有thinking内容，直接添加到content列表
                    anthropic_content.extend(extracted)
                else:
                    # 普通文本
                    anthropic_content.append({
                        "type": "text",
                        "text": extracted
                    })
                self.logger.info(f"✅ OpenAI->Anthropic: Text processed AFTER image (best practice): {text_content[:30]}...")
            
            # 4. 处理其他类型的内容
            for item in other_items:
                anthropic_content.append(item)
            
            return anthropic_content
        return content
    
    def _extract_thinking_from_text(self, text: str) -> Any:
        """从文本中提取thinking内容，返回Anthropic格式的content blocks"""
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
    
    def _convert_content_to_gemini(self, content: Any) -> List[Dict[str, Any]]:
        """转换内容到Gemini格式"""
        if isinstance(content, str):
            return [{"text": content}]
        elif isinstance(content, list):
            # 处理多模态内容
            gemini_parts = []
            for item in content:
                if item.get("type") == "text":
                    gemini_parts.append({"text": item.get("text", "")})
                elif item.get("type") == "image_url":
                    # 转换图像格式
                    image_url = item.get("image_url", {}).get("url", "")
                    if image_url.startswith("data:"):
                        # 处理base64图像
                        media_type, data_part = image_url.split(";base64,")
                        media_type = media_type.replace("data:", "")
                        gemini_parts.append({
                            "inlineData": {
                                "mimeType": media_type,
                                "data": data_part
                            }
                        })
            return gemini_parts
        return [{"text": str(content)}]
    
    def _map_finish_reason(self, reason: str, source_format: str, target_format: str) -> str:
        """映射结束原因"""
        reason_mappings = {
            "anthropic": {
                "openai": {
                    "end_turn": "stop",
                    "max_tokens": "length",
                    "stop_sequence": "stop",
                    "tool_use": "tool_calls"
                }
            },
            "gemini": {
                "openai": {
                    "STOP": "stop",
                    "MAX_TOKENS": "length",
                    "SAFETY": "content_filter",
                    "RECITATION": "content_filter"
                }
            }
        }
        
        try:
            return reason_mappings[source_format][target_format].get(reason, "stop")
        except KeyError:
            return "stop"