"""
Gemini格式转换器
处理Google Gemini API格式与其他格式之间的转换
"""
from typing import Dict, Any, Optional, List
import json
import copy

from .base_converter import BaseConverter, ConversionResult, ConversionError


class GeminiConverter(BaseConverter):
    """Gemini格式转换器"""

    def __init__(self):
        super().__init__()
        self.original_model = None
        # 用于跨 OpenAI 往返保留 Gemini thoughtSignature 的映射
        # key: tool_call_id, value: thoughtSignature
        self._thought_signatures_by_tool_call_id: Dict[str, str] = {}
        # 用于保留 OpenRouter reasoning_details（按 assistant 消息的首个 tool_call_id 索引）
        # key: first_tool_call_id, value: reasoning_details array
        # 多轮工具调用时，每个 assistant 消息的 reasoning_details 都需要保留
        self._reasoning_details_cache: Dict[str, List[Dict[str, Any]]] = {}
    
    def set_original_model(self, model: str):
        """设置原始模型名称"""
        self.original_model = model
    
    def _determine_reasoning_effort_from_budget(self, thinking_budget: Optional[int]) -> str:
        """根据thinkingBudget判断OpenAI reasoning_effort等级
        
        Args:
            thinking_budget: Gemini thinking的thinkingBudget值
            
        Returns:
            str: OpenAI reasoning_effort等级 ("low", "medium", "high")
        """
        import os
        
        # 如果没有提供thinking_budget或为-1（动态思考），默认为high
        if thinking_budget is None or thinking_budget == -1:
            reason = "dynamic thinking (-1)" if thinking_budget == -1 else "no budget provided"
            self.logger.info(f"No valid thinkingBudget ({reason}), defaulting to reasoning_effort='high'")
            return "high"
        
        # 从环境变量获取阈值配置
        low_threshold_str = os.environ.get("GEMINI_TO_OPENAI_LOW_REASONING_THRESHOLD")
        high_threshold_str = os.environ.get("GEMINI_TO_OPENAI_HIGH_REASONING_THRESHOLD")
        
        # 检查必需的环境变量
        if low_threshold_str is None:
            raise ConversionError("GEMINI_TO_OPENAI_LOW_REASONING_THRESHOLD environment variable is required for intelligent reasoning_effort determination")
        
        if high_threshold_str is None:
            raise ConversionError("GEMINI_TO_OPENAI_HIGH_REASONING_THRESHOLD environment variable is required for intelligent reasoning_effort determination")
        
        try:
            low_threshold = int(low_threshold_str)
            high_threshold = int(high_threshold_str)
            
            self.logger.debug(f"Threshold configuration: low <= {low_threshold}, medium <= {high_threshold}, high > {high_threshold}")
            
            if thinking_budget <= low_threshold:
                effort = "low"
            elif thinking_budget <= high_threshold:
                effort = "medium"
            else:
                effort = "high"
            
            self.logger.info(f"🎯 Thinking budget {thinking_budget} -> reasoning_effort '{effort}' (thresholds: low<={low_threshold}, high<={high_threshold})")
            return effort
            
        except ValueError as e:
            raise ConversionError(f"Invalid threshold values in environment variables: {e}. GEMINI_TO_OPENAI_LOW_REASONING_THRESHOLD and GEMINI_TO_OPENAI_HIGH_REASONING_THRESHOLD must be integers.")
    
    def reset_streaming_state(self):
        """重置所有流式相关的状态变量，避免状态污染"""
        streaming_attrs = [
            '_anthropic_stream_id', '_openai_sent_start', '_gemini_text_started',
            '_anthropic_to_gemini_state', '_streaming_tool_calls'
        ]
        for attr in streaming_attrs:
            if hasattr(self, attr):
                delattr(self, attr)
        # 清空 thoughtSignature 映射
        self._thought_signatures_by_tool_call_id.clear()
        # 注意：不清空 _reasoning_details_cache，因为它需要跨请求保留以支持多轮工具调用
    
    def get_supported_formats(self) -> List[str]:
        """获取支持的格式列表"""
        return ["openai", "anthropic", "gemini"]
    
    def convert_request(
        self,
        data: Dict[str, Any],
        target_format: str,
        headers: Optional[Dict[str, str]] = None
    ) -> ConversionResult:
        """转换Gemini请求到目标格式"""
        try:
            if target_format == "gemini":
                # Gemini到Gemini，格式与渠道相同，不需要转换思考参数
                # 但需要移除内部处理用的stream字段，因为Gemini API不接受此字段
                cleaned_data = data.copy()
                if "stream" in cleaned_data:
                    cleaned_data.pop("stream")
                    self.logger.debug("Removed internal stream flag for Gemini API")
                return ConversionResult(success=True, data=cleaned_data)
            elif target_format == "openai":
                return self._convert_to_openai_request(data)
            elif target_format == "anthropic":
                return self._convert_to_anthropic_request(data)
            else:
                return ConversionResult(
                    success=False,
                    error=f"Unsupported target format: {target_format}"
                )
        except Exception as e:
            self.logger.error(f"Failed to convert Gemini request to {target_format}: {e}")
            return ConversionResult(success=False, error=str(e))
    
    def convert_response(
        self,
        data: Dict[str, Any],
        source_format: str,
        target_format: str
    ) -> ConversionResult:
        """转换响应到Gemini格式"""
        try:
            if source_format == "gemini":
                return ConversionResult(success=True, data=data)
            elif source_format == "openai":
                return self._convert_from_openai_response(data)
            elif source_format == "anthropic":
                return self._convert_from_anthropic_response(data)
            else:
                return ConversionResult(
                    success=False,
                    error=f"Unsupported source format: {source_format}"
                )
        except Exception as e:
            self.logger.error(f"Failed to convert {source_format} response to Gemini: {e}")
            return ConversionResult(success=False, error=str(e))
    
    def _convert_to_openai_request(self, data: Dict[str, Any]) -> ConversionResult:
        """转换Gemini请求到OpenAI格式"""
        result_data = {}

        # 清空 thoughtSignature 映射，避免跨请求污染
        self._thought_signatures_by_tool_call_id.clear()
        # 注意：不清空 _reasoning_details_cache，因为它需要跨请求保留

        # 必须有原始模型名称，否则报错
        if not self.original_model:
            raise ValueError("Original model name is required for request conversion")

        result_data["model"] = self.original_model  # 使用原始模型名称

        # 初始化函数调用ID映射表，用于保持工具调用和工具结果的ID一致性
        # 先扫描整个对话历史，为每个functionCall和functionResponse建立映射关系
        self._function_call_mapping = self._build_function_call_mapping(data.get("contents", []))
        
        # 处理消息和系统消息
        messages = []
        # 用于去重：记录已处理的 functionResponse ID，避免 gemini-cli 发送重复消息导致 400 错误
        _processed_function_response_ids: set = set()

        # 🔍 调试日志：记录收到的 contents 结构
        contents = data.get("contents", [])
        print(f"📥 [GEMINI->OPENAI] Received {len(contents)} content items")
        for idx, content in enumerate(contents):
            role = content.get("role", "user")
            parts = content.get("parts", [])
            # 检查是否有 functionResponse
            fr_ids = []
            for part in parts:
                if "functionResponse" in part:
                    fr_id = part["functionResponse"].get("id", "NO_ID")
                    fr_ids.append(fr_id)
            if fr_ids:
                print(f"📥 [GEMINI->OPENAI] Content[{idx}] role={role}, functionResponse IDs: {fr_ids}")

        # 添加系统消息 - 支持两种格式
        system_instruction_data = data.get("systemInstruction") or data.get("system_instruction")
        if system_instruction_data:
            system_parts = system_instruction_data.get("parts", [])
            system_text = ""
            for part in system_parts:
                if "text" in part:
                    system_text += part["text"]
            if system_text:
                messages.append(self._create_system_message(system_text))
        
        # 转换内容
        if "contents" in data:
            for content in data["contents"]:
                gemini_role = content.get("role", "user")
                parts = content.get("parts", [])
                
                # 处理不同角色的消息
                if gemini_role == "user":
                    # 检查是否包含 functionResponse（工具结果）
                    has_function_response = any("functionResponse" in part for part in parts)
                    if has_function_response:
                        # 转换为 OpenAI 的 tool 消息
                        for part in parts:
                            if "functionResponse" in part:
                                fr = part["functionResponse"]
                                func_name = fr.get("name", "")
                                response_content = fr.get("response", {})

                                # 从响应内容中提取文本
                                if isinstance(response_content, dict):
                                    tool_result = response_content.get("content") or response_content.get("output") or json.dumps(response_content, ensure_ascii=False)
                                else:
                                    tool_result = str(response_content)

                                # 优先使用 functionResponse 自带的 id 字段
                                tool_call_id = fr.get("id")
                                if not tool_call_id:
                                    # 备选：使用映射
                                    if not hasattr(self, '_current_response_sequence'):
                                        self._current_response_sequence = {}
                                    sequence = self._current_response_sequence.get(func_name, 0) + 1
                                    self._current_response_sequence[func_name] = sequence
                                    tool_call_id = self._function_call_mapping.get(f"response_{func_name}_{sequence}")
                                    if not tool_call_id:
                                        tool_call_id = self._function_call_mapping.get(f"{func_name}_{sequence}")
                                        if not tool_call_id:
                                            tool_call_id = f"call_{func_name}_{sequence:04d}"

                                # 去重：跳过已处理的 functionResponse（基于 tool_call_id）
                                if tool_call_id in _processed_function_response_ids:
                                    print(f"🔄 [GEMINI->OPENAI] Skipping duplicate functionResponse: {tool_call_id}")
                                    continue
                                _processed_function_response_ids.add(tool_call_id)
                                print(f"✅ [GEMINI->OPENAI] Processing functionResponse: {tool_call_id}")

                                # 某些 OpenAI 兼容 API 要求 tool 消息包含 name 字段
                                tool_msg = {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "content": tool_result
                                }
                                if func_name:
                                    tool_msg["name"] = func_name
                                messages.append(tool_msg)
                    else:
                        # 普通用户消息
                        message_content = self._convert_content_from_gemini(parts)
                        messages.append({
                            "role": "user",
                            "content": message_content
                        })
                        
                elif gemini_role == "model":
                    # 助手消息，可能包含工具调用
                    message_content = self._convert_content_from_gemini(parts)

                    if isinstance(message_content, dict) and message_content.get("type") == "tool_calls":
                        # 有工具调用的助手消息
                        # 某些 OpenAI 兼容 API 要求 content 字段必须存在（即使为空字符串）
                        tool_call_content = message_content.get("content") or ""
                        tool_calls = message_content["tool_calls"]
                        message: Dict[str, Any] = {
                            "role": "assistant",
                            "content": tool_call_content,
                            "tool_calls": tool_calls
                        }
                        # 从 cache 中查找对应的 reasoning_details（按首个 tool_call_id 索引）
                        if tool_calls:
                            first_tool_call_id = tool_calls[0].get("id")
                            if first_tool_call_id and first_tool_call_id in self._reasoning_details_cache:
                                message["reasoning_details"] = self._reasoning_details_cache[first_tool_call_id]
                                print(f"🧠 [REASONING_DETAILS] Attached {len(message['reasoning_details'])} reasoning_details to assistant message (key={first_tool_call_id})")
                        messages.append(message)
                    else:
                        # 普通助手消息（无工具调用）
                        msg: Dict[str, Any] = {
                            "role": "assistant",
                            "content": message_content
                        }
                        messages.append(msg)
                        
                elif gemini_role == "tool":
                    # 工具角色的消息，处理functionResponse
                    for part in parts:
                        if "functionResponse" in part:
                            fr = part["functionResponse"]
                            func_name = fr.get("name", "")
                            response_content = fr.get("response", {})

                            # 从响应内容中提取文本
                            if isinstance(response_content, dict):
                                tool_result = response_content.get("content") or response_content.get("output") or json.dumps(response_content, ensure_ascii=False)
                            else:
                                tool_result = str(response_content)

                            # 优先使用 functionResponse 自带的 id 字段
                            tool_call_id = fr.get("id")
                            if not tool_call_id:
                                # 备选：使用映射
                                if not hasattr(self, '_current_response_sequence'):
                                    self._current_response_sequence = {}
                                sequence = self._current_response_sequence.get(func_name, 0) + 1
                                self._current_response_sequence[func_name] = sequence
                                tool_call_id = self._function_call_mapping.get(f"response_{func_name}_{sequence}")
                                if not tool_call_id:
                                    tool_call_id = self._function_call_mapping.get(f"{func_name}_{sequence}")
                                    if not tool_call_id:
                                        tool_call_id = f"call_{func_name}_{sequence:04d}"

                            # 去重：跳过已处理的 functionResponse（基于 tool_call_id）
                            if tool_call_id in _processed_function_response_ids:
                                print(f"🔄 [GEMINI->OPENAI] Skipping duplicate functionResponse (tool role): {tool_call_id}")
                                continue
                            _processed_function_response_ids.add(tool_call_id)
                            print(f"✅ [GEMINI->OPENAI] Processing functionResponse (tool role): {tool_call_id}")

                            # 某些 OpenAI 兼容 API 要求 tool 消息包含 name 字段
                            tool_msg = {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": tool_result
                            }
                            if func_name:
                                tool_msg["name"] = func_name
                            messages.append(tool_msg)
                else:
                    # 其他角色，默认转为assistant
                    message_content = self._convert_content_from_gemini(parts)
                    messages.append({
                        "role": "assistant",
                        "content": message_content
                    })

        # 合并连续的同角色消息（Gemini API 不允许连续同角色消息）
        # 这个问题在 OpenAI → Gemini 转换时会导致 400 错误
        messages = self._merge_consecutive_messages(messages)

        # 🔍 调试日志：记录转换后的 messages 结构
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        tool_call_ids = [m.get("tool_call_id") for m in tool_msgs]
        print(f"📤 [GEMINI->OPENAI] Generated {len(messages)} messages, {len(tool_msgs)} tool messages")
        if tool_call_ids:
            print(f"📤 [GEMINI->OPENAI] Tool message IDs: {tool_call_ids}")
        print(f"📤 [GEMINI->OPENAI] Dedup stats: processed {len(_processed_function_response_ids)} unique functionResponse IDs")

        result_data["messages"] = messages

        # P2-14: safetySettings 透传到 metadata
        if "safetySettings" in data:
            metadata = result_data.get("metadata") or {}
            metadata["gemini_safety_settings"] = data["safetySettings"]
            result_data["metadata"] = metadata

        # 处理生成配置
        if "generationConfig" in data:
            config = data["generationConfig"]
            if "temperature" in config:
                result_data["temperature"] = config["temperature"]
            if "topP" in config:
                result_data["top_p"] = config["topP"]
            if "maxOutputTokens" in config:
                result_data["max_tokens"] = config["maxOutputTokens"]
            if "stopSequences" in config:
                result_data["stop"] = config["stopSequences"]
            # P1: presencePenalty/frequencyPenalty/candidateCount 参数转换
            if "presencePenalty" in config:
                result_data["presence_penalty"] = config["presencePenalty"]
            if "frequencyPenalty" in config:
                result_data["frequency_penalty"] = config["frequencyPenalty"]
            if "candidateCount" in config:
                result_data["n"] = config["candidateCount"]

            # 处理结构化输出
            if config.get("response_mime_type") == "application/json":
                result_data["response_format"] = {"type": "json_object"}
                if "response_schema" in config:
                    result_data["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "response",
                            "strict": True,
                            "schema": config["response_schema"]
                        }
                    }

        # 确保有 max_tokens（某些 OpenAI 兼容 API 要求必须提供）
        # 优先级：1. generationConfig.maxOutputTokens 2. 环境变量 3. 默认值
        if "max_tokens" not in result_data:
            import os
            env_max_tokens = os.environ.get("OPENAI_DEFAULT_MAX_TOKENS")
            if env_max_tokens:
                try:
                    result_data["max_tokens"] = int(env_max_tokens)
                    self.logger.debug(f"Using OPENAI_DEFAULT_MAX_TOKENS: {result_data['max_tokens']}")
                except ValueError:
                    self.logger.warning(f"Invalid OPENAI_DEFAULT_MAX_TOKENS: {env_max_tokens}")
            else:
                # 使用合理的默认值
                result_data["max_tokens"] = 8192
                self.logger.debug("Using default max_tokens: 8192")
        
        # 处理工具调用
        if "tools" in data:
            openai_tools = []
            unsupported_tools = []
            has_code_execution = False
            has_google_search = False

            self.logger.info(f"🔧 [TOOLS] Processing {len(data['tools'])} tools")

            for tool in data["tools"]:
                self.logger.debug(f"🔧 [TOOLS] Tool structure keys: {list(tool.keys())}")

                # Gemini官方使用 snake_case: function_declarations
                func_key = None
                if "function_declarations" in tool:
                    func_key = "function_declarations"
                elif "functionDeclarations" in tool:  # 兼容旧写法
                    func_key = "functionDeclarations"

                if func_key:
                    self.logger.info(f"🔧 [TOOLS] Found {len(tool[func_key])} function declarations")
                    for func_decl in tool[func_key]:
                        func_name = func_decl.get("name", "")
                        function_def = {
                            "name": func_name,
                            "description": func_decl.get("description", "")
                        }

                        # 处理 parameters - 某些 OpenAI 兼容 API 要求必须有此字段
                        if "parameters" in func_decl:
                            raw_params = func_decl["parameters"]
                            self.logger.debug(f"🔧 [TOOLS] Function '{func_name}' raw parameters: {raw_params}")
                            function_def["parameters"] = self._sanitize_schema_for_openai(raw_params)
                        else:
                            # Gemini 没有提供 parameters，添加空的 parameters 以兼容第三方 API
                            function_def["parameters"] = {"type": "object", "properties": {}}
                            self.logger.debug(f"🔧 [TOOLS] Function '{func_name}' has no parameters, using empty schema")

                        openai_tools.append({
                            "type": "function",
                            "function": function_def
                        })

                # 检测非函数工具类型（与 func_key 独立处理）
                if "code_execution" in tool:
                    has_code_execution = True
                if "google_search" in tool:
                    has_google_search = True
                if "google_search_retrieval" in tool:
                    unsupported_tools.append("google_search_retrieval")
                if "retrieval" in tool:
                    unsupported_tools.append("retrieval")

            # code_execution → 虚拟函数工具
            if has_code_execution:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": "code_execution",
                        "description": "Execute Python code. Caller must implement the execution handler.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "code": {"type": "string", "description": "Python code to execute"}
                            },
                            "required": ["code"]
                        }
                    }
                })
                self.logger.warning(
                    "Gemini code_execution mapped to OpenAI function tool. "
                    "Caller must implement the execution handler."
                )

            # P2-15: google_search → 虚拟函数工具
            if has_google_search:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": "google_search",
                        "description": "Web search using Google. Caller must implement the search handler.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Search query"}
                            },
                            "required": ["query"]
                        }
                    }
                })
                self.logger.warning(
                    "Gemini google_search mapped to OpenAI function tool. "
                    "Caller must implement the search handler."
                )

            # 记录不支持的工具警告（去重）
            if unsupported_tools:
                self.logger.warning(
                    f"Gemini tools not supported for OpenAI conversion: {list(set(unsupported_tools))}. "
                    "These tools will be ignored in the converted request."
                )

            if openai_tools:
                result_data["tools"] = openai_tools
                result_data["tool_choice"] = "auto"
        
        # 处理思考预算转换 (Gemini thinkingConfig -> OpenAI reasoning_effort + max_completion_tokens)
        if "generationConfig" in data and "thinkingConfig" in data["generationConfig"]:
            thinking_config = data["generationConfig"]["thinkingConfig"]
            thinking_budget = thinking_config.get("thinkingBudget")
            
            if thinking_budget is not None and thinking_budget != 0:
                # 检测到思考参数，设置为OpenAI思考模型格式
                reasoning_effort = self._determine_reasoning_effort_from_budget(thinking_budget)
                result_data["reasoning_effort"] = reasoning_effort
                
                # 处理max_completion_tokens的优先级逻辑
                max_completion_tokens = None
                
                # 优先级1：客户端传入的maxOutputTokens
                if "generationConfig" in data and "maxOutputTokens" in data["generationConfig"]:
                    max_completion_tokens = data["generationConfig"]["maxOutputTokens"]
                    # 移除max_tokens，使用max_completion_tokens
                    if "max_tokens" in result_data:
                        result_data.pop("max_tokens", None)
                    self.logger.info(f"Using client maxOutputTokens as max_completion_tokens: {max_completion_tokens}")
                else:
                    # 优先级2：环境变量OPENAI_REASONING_MAX_TOKENS
                    import os
                    env_max_tokens = os.environ.get("OPENAI_REASONING_MAX_TOKENS")
                    if env_max_tokens:
                        try:
                            max_completion_tokens = int(env_max_tokens)
                            self.logger.info(f"Using OPENAI_REASONING_MAX_TOKENS from environment: {max_completion_tokens}")
                        except ValueError:
                            self.logger.warning(f"Invalid OPENAI_REASONING_MAX_TOKENS value '{env_max_tokens}', must be integer")
                            env_max_tokens = None
                    
                    if not env_max_tokens:
                        # 优先级3：都没有则报错
                        raise ConversionError("For OpenAI reasoning models, max_completion_tokens is required. Please specify maxOutputTokens in generationConfig or set OPENAI_REASONING_MAX_TOKENS environment variable.")
                
                result_data["max_completion_tokens"] = max_completion_tokens
                self.logger.info(f"Gemini thinkingBudget {thinking_budget} -> OpenAI reasoning_effort='{reasoning_effort}', max_completion_tokens={max_completion_tokens}")

        # 处理流式参数 - 关键修复！
        if "stream" in data:
            result_data["stream"] = data["stream"]

        # 汇总 thoughtSignature 捕获情况
        if self._thought_signatures_by_tool_call_id:
            print(f"🧠 [THOUGHT_SIGNATURE] Request conversion complete. Captured {len(self._thought_signatures_by_tool_call_id)} signatures: {list(self._thought_signatures_by_tool_call_id.keys())}")

        return ConversionResult(success=True, data=result_data)
    
    def _convert_to_anthropic_request(self, data: Dict[str, Any]) -> ConversionResult:
        """转换Gemini请求到Anthropic格式"""
        result_data = {}

        # 清空 thoughtSignature 映射，避免跨请求污染
        self._thought_signatures_by_tool_call_id.clear()

        # 必须有原始模型名称，否则报错
        if not self.original_model:
            raise ValueError("Original model name is required for request conversion")

        result_data["model"] = self.original_model  # 使用原始模型名称
        
        # 处理系统消息 - 支持两种格式
        system_instruction_data = data.get("systemInstruction") or data.get("system_instruction")
        if system_instruction_data:
            system_parts = system_instruction_data.get("parts", [])
            system_text = ""
            for part in system_parts:
                if "text" in part:
                    system_text += part["text"]
            if system_text:
                result_data["system"] = system_text
        
        # 转换消息格式
        if "contents" in data:
            # 建立工具调用ID映射表
            self._function_call_mapping = self._build_function_call_mapping(data["contents"])
            
            anthropic_messages = []
            for content in data["contents"]:
                role = content.get("role", "user")
                if role == "model":
                    role = "assistant"
                elif role == "tool":
                    # Gemini的tool角色（functionResponse）对应Anthropic的user角色
                    role = "user"
                
                message_content = self._convert_content_to_anthropic(content.get("parts", []))
                
                # 跳过空内容的消息，Anthropic不允许空内容
                if not message_content or (isinstance(message_content, str) and not message_content.strip()):
                    self.logger.warning(f"Skipping message with empty content for role '{role}'")
                    continue
                
                anthropic_messages.append({
                    "role": role,
                    "content": message_content
                })
            result_data["messages"] = anthropic_messages
        
        # 处理生成配置
        if "generationConfig" in data:
            config = data["generationConfig"]
            if "temperature" in config:
                result_data["temperature"] = config["temperature"]
            if "topP" in config:
                result_data["top_p"] = config["topP"]
            if "topK" in config:
                result_data["top_k"] = config["topK"]
            if "maxOutputTokens" in config:
                result_data["max_tokens"] = config["maxOutputTokens"]
            if "stopSequences" in config:
                result_data["stop_sequences"] = config["stopSequences"]
        
        # Anthropic 要求必须有 max_tokens，按优先级处理：
        # 1. Gemini generationConfig中的maxOutputTokens（最高优先级）
        # 2. 环境变量ANTHROPIC_MAX_TOKENS
        # 3. 都没有则报错
        if "max_tokens" not in result_data:
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
                raise ValueError(f"max_tokens is required for Anthropic API. Please specify max_tokens in generationConfig.maxOutputTokens or set ANTHROPIC_MAX_TOKENS environment variable.")
        
        # P2-14: safetySettings 透传到 metadata
        if "safetySettings" in data:
            metadata = result_data.get("metadata") or {}
            metadata["gemini_safety_settings"] = data["safetySettings"]
            result_data["metadata"] = metadata

        # 处理工具调用
        if "tools" in data:
            anthropic_tools = []
            unsupported_tools = []
            has_code_execution = False
            has_google_search = False

            for tool in data["tools"]:
                func_key = None
                if "function_declarations" in tool:
                    func_key = "function_declarations"
                elif "functionDeclarations" in tool:  # 兼容旧写法
                    func_key = "functionDeclarations"

                if func_key:
                    for func_decl in tool[func_key]:
                        anthropic_tools.append({
                            "name": func_decl.get("name", ""),
                            "description": func_decl.get("description", ""),
                            "input_schema": self._convert_schema_for_anthropic(func_decl.get("parameters", {}))
                        })

                # 检测非函数工具类型（与 func_key 独立处理）
                if "code_execution" in tool:
                    has_code_execution = True
                if "google_search" in tool:
                    has_google_search = True
                if "google_search_retrieval" in tool:
                    unsupported_tools.append("google_search_retrieval")
                if "retrieval" in tool:
                    unsupported_tools.append("retrieval")

            # code_execution → 虚拟工具
            if has_code_execution:
                anthropic_tools.append({
                    "name": "code_execution",
                    "description": "Execute Python code. Caller must implement the execution handler.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Python code to execute"}
                        },
                        "required": ["code"]
                    }
                })
                self.logger.warning(
                    "Gemini code_execution mapped to Anthropic tool. "
                    "Caller must implement the execution handler."
                )

            # P2-15: google_search → 虚拟工具
            if has_google_search:
                anthropic_tools.append({
                    "name": "google_search",
                    "description": "Web search using Google. Caller must implement the search handler.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"]
                    }
                })
                self.logger.warning(
                    "Gemini google_search mapped to Anthropic tool. "
                    "Caller must implement the search handler."
                )

            # 记录不支持的工具警告（去重）
            if unsupported_tools:
                self.logger.warning(
                    f"Gemini tools not supported for Anthropic conversion: {list(set(unsupported_tools))}. "
                    "These tools will be ignored in the converted request."
                )

            if anthropic_tools:
                result_data["tools"] = anthropic_tools
        
        # 处理思考预算转换 (Gemini thinkingBudget -> Anthropic thinkingBudget)
        if "generationConfig" in data and "thinkingConfig" in data["generationConfig"]:
            thinking_config = data["generationConfig"]["thinkingConfig"]
            thinking_budget = thinking_config.get("thinkingBudget")
            
            if thinking_budget is not None:
                if thinking_budget == -1:
                    # 动态思考，启用但不设置具体token数
                    result_data["thinking"] = {
                        "type": "enabled"
                    }
                    self.logger.info("Gemini thinkingBudget -1 (dynamic) -> Anthropic thinking enabled without budget")
                elif thinking_budget == 0:
                    # 不启用思考
                    pass
                else:
                    # 数值型思考预算，直接转换
                    result_data["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": thinking_budget
                    }
                    self.logger.info(f"Gemini thinkingBudget {thinking_budget} -> Anthropic thinkingBudget {thinking_budget}")
        
        # 处理流式参数 - 关键修复！
        if "stream" in data:
            result_data["stream"] = data["stream"]
        
        return ConversionResult(success=True, data=result_data)
    
    def _convert_from_openai_response(self, data: Dict[str, Any]) -> ConversionResult:
        """转换OpenAI响应到Gemini格式"""
        result_data = {
            "candidates": [],
            "usageMetadata": {}
        }

        # 处理选择
        if "choices" in data and data["choices"] and data["choices"][0]:
            choice = data["choices"][0]
            message = choice.get("message", {})
            content = message.get("content", "")
            tool_calls = message.get("tool_calls", [])

            # 捕获 reasoning_details（OpenRouter 返回的推理详情）
            # 按首个 tool_call_id 存入 cache，用于多轮工具调用时回传
            reasoning_details = message.get("reasoning_details")
            if reasoning_details and tool_calls:
                first_tool_call_id = tool_calls[0].get("id") if tool_calls else None
                if first_tool_call_id:
                    self._reasoning_details_cache[first_tool_call_id] = reasoning_details
                    print(f"🧠 [REASONING_DETAILS] Cached {len(reasoning_details)} reasoning_details (key={first_tool_call_id})")

            # 构建 parts
            parts = []

            # 添加文本内容（如果有）
            if content:
                parts.append({"text": content})
            
            # 添加工具调用（如果有）
            if tool_calls:
                for tool_call in tool_calls:
                    if tool_call and tool_call.get("type") == "function" and "function" in tool_call:
                        func = tool_call["function"]
                        func_name = func.get("name", "")
                        # OpenAI 的 arguments 是 JSON 字符串，需要解析
                        args_str = func.get("arguments", "{}")
                        try:
                            func_args = json.loads(args_str) if args_str else {}
                        except json.JSONDecodeError:
                            func_args = {}

                        func_args = self._adapt_tool_call_params(func_name, func_args)
                        function_call = {
                            "name": func_name,
                            "args": func_args
                        }
                        # 保留原始 tool_call ID，确保 functionResponse 可以使用相同的 ID
                        tool_call_id = tool_call.get("id")
                        if tool_call_id:
                            function_call["id"] = tool_call_id

                        # 构建 part，包含 functionCall
                        part: Dict[str, Any] = {"functionCall": function_call}

                        # 回填 thoughtSignature（如果之前捕获过）
                        if tool_call_id and tool_call_id in self._thought_signatures_by_tool_call_id:
                            thought_signature = self._thought_signatures_by_tool_call_id[tool_call_id]
                            part["thoughtSignature"] = thought_signature
                            print(f"🧠 [THOUGHT_SIGNATURE] Restored: tool_call_id={tool_call_id}, signature={thought_signature[:50]}...")

                        parts.append(part)
            
            # 如果没有任何内容，添加空文本
            if not parts:
                parts = [{"text": ""}]
            
            candidate = {
                "content": {
                    "parts": parts,
                    "role": "model"
                },
                "finishReason": self._map_finish_reason(choice.get("finish_reason", "stop"), "openai", "gemini"),
                "index": 0
            }
            result_data["candidates"] = [candidate]
        
        # 处理使用情况
        if "usage" in data and data["usage"] is not None:
            usage = data["usage"]
            result_data["usageMetadata"] = {
                "promptTokenCount": usage.get("prompt_tokens", 0),
                "candidatesTokenCount": usage.get("completion_tokens", 0),
                "totalTokenCount": usage.get("total_tokens", 0)
            }
        
        return ConversionResult(success=True, data=result_data)
    
    def _convert_from_anthropic_response(self, data: Dict[str, Any]) -> ConversionResult:
        """转换Anthropic响应到Gemini格式"""
        result_data = {
            "candidates": [],
            "usageMetadata": {}
        }
        
        # 处理内容，包括文本、工具调用和思考内容
        parts = []
        if "content" in data and isinstance(data["content"], list):
            for item in data["content"]:
                item_type = item.get("type")
                
                # 处理文本内容
                if item_type == "text":
                    text_content = item.get("text", "")
                    if text_content.strip():  # 只添加非空文本
                        parts.append({"text": text_content})
                
                # 处理思考内容 (thinking → text with thought: true)
                elif item_type == "thinking":
                    thinking_content = item.get("thinking", "")
                    if thinking_content.strip():
                        parts.append({
                            "text": thinking_content,
                            "thought": True  # Gemini 2025格式的thinking标识
                        })
                
                # 处理工具调用 (tool_use → functionCall)
                elif item_type == "tool_use":
                    func_name = item.get("name", "")
                    func_args = self._adapt_tool_call_params(func_name, item.get("input", {}))
                    function_call = {
                        "name": func_name,
                        "args": func_args
                    }
                    # 保留 Anthropic tool_use 的 ID
                    tool_use_id = item.get("id")
                    if tool_use_id:
                        function_call["id"] = tool_use_id

                    # 构建 part，包含 functionCall
                    part: Dict[str, Any] = {"functionCall": function_call}

                    # 回填 thoughtSignature（如果之前捕获过）
                    if tool_use_id and tool_use_id in self._thought_signatures_by_tool_call_id:
                        thought_signature = self._thought_signatures_by_tool_call_id[tool_use_id]
                        part["thoughtSignature"] = thought_signature
                        print(f"🧠 [THOUGHT_SIGNATURE] Restored (Anthropic): tool_use_id={tool_use_id}, signature={thought_signature[:50]}...")

                    parts.append(part)
        
        # 如果没有任何内容，添加空文本避免空parts数组
        if not parts:
            parts = [{"text": ""}]
        
        candidate = {
            "content": {
                "parts": parts,
                "role": "model"
            },
            "finishReason": self._map_finish_reason(data.get("stop_reason", "end_turn"), "anthropic", "gemini"),
            "index": 0
        }
        result_data["candidates"] = [candidate]
        
        # 处理使用情况
        if "usage" in data and data["usage"] is not None:
            usage = data["usage"]
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            result_data["usageMetadata"] = {
                "promptTokenCount": input_tokens,
                "candidatesTokenCount": output_tokens,
                "totalTokenCount": input_tokens + output_tokens
            }
        
        return ConversionResult(success=True, data=result_data)
    
    def _convert_from_openai_streaming_chunk(self, data: Dict[str, Any]) -> ConversionResult:
        """转换OpenAI流式响应chunk到Gemini格式"""
        self.logger.info(f"OPENAI->GEMINI CHUNK: {data}")  # 记录输入数据

        # 为流式工具调用维护状态
        if not hasattr(self, '_streaming_tool_calls'):
            self._streaming_tool_calls = {}

        # 为流式 reasoning_details 维护临时状态（在 finish 时存入 cache）
        if not hasattr(self, '_streaming_reasoning_details'):
            self._streaming_reasoning_details = None

        # 捕获流式响应中的 reasoning_details
        if "choices" in data and data["choices"] and data["choices"][0]:
            choice = data["choices"][0]
            delta = choice.get("delta", {})

            # reasoning_details 可能在 delta 或顶层 message 中
            reasoning_details = delta.get("reasoning_details") or choice.get("message", {}).get("reasoning_details")
            if reasoning_details:
                self._streaming_reasoning_details = reasoning_details
                print(f"🧠 [REASONING_DETAILS] Captured {len(reasoning_details)} reasoning_details from streaming chunk")

        # 先处理增量内容和工具调用（收集状态）
        if "choices" in data and data["choices"] and data["choices"][0]:
            choice = data["choices"][0]
            delta = choice.get("delta", {})
            
            # 收集流式工具调用信息
            if "tool_calls" in delta:
                for tool_call in delta["tool_calls"]:
                    call_index = tool_call.get("index", 0)
                    call_id = tool_call.get("id", "")
                    call_type = tool_call.get("type", "function")
                    
                    # 初始化工具调用状态
                    if call_index not in self._streaming_tool_calls:
                        self._streaming_tool_calls[call_index] = {
                            "id": call_id,
                            "type": call_type,
                            "function": {
                                "name": "",
                                "arguments": ""
                            }
                        }
                    
                    # 更新工具调用信息
                    if "function" in tool_call:
                        func = tool_call["function"]
                        if "name" in func:
                            self._streaming_tool_calls[call_index]["function"]["name"] = func["name"]
                        if "arguments" in func:
                            self._streaming_tool_calls[call_index]["function"]["arguments"] += func["arguments"]
                    
                    self.logger.debug(f"Updated tool call {call_index}: {self._streaming_tool_calls[call_index]}")
        
        # 检查是否是完整的流式响应结束
        if "choices" in data and data["choices"] and data["choices"][0] and data["choices"][0].get("finish_reason"):
            choice = data["choices"][0]
            delta = choice.get("delta", {})
            content = delta.get("content", "")
            
            # 构建parts数组，可能包含内容和工具调用
            parts = []
            
            # 处理文本内容
            if content:
                parts.append({"text": content})
            
            # 处理收集到的工具调用
            if self._streaming_tool_calls:
                self.logger.debug(f"FINISH: Processing collected tool calls: {self._streaming_tool_calls}")

                # 在流式结束时，将 reasoning_details 存入 cache（按首个 tool_call_id 索引）
                if hasattr(self, '_streaming_reasoning_details') and self._streaming_reasoning_details:
                    # 获取首个 tool_call_id
                    first_index = min(self._streaming_tool_calls.keys())
                    first_tool_call_id = self._streaming_tool_calls[first_index].get("id")
                    if first_tool_call_id:
                        self._reasoning_details_cache[first_tool_call_id] = self._streaming_reasoning_details
                        print(f"🧠 [REASONING_DETAILS] Cached {len(self._streaming_reasoning_details)} reasoning_details from streaming (key={first_tool_call_id})")
                    self._streaming_reasoning_details = None

                for call_index, tool_call in self._streaming_tool_calls.items():
                    func = tool_call.get("function", {})
                    func_name = func.get("name", "")
                    func_args = func.get("arguments", "{}")
                    self.logger.debug(f"FINISH TOOL CALL - name: {func_name}, args: '{func_args}'")

                    # OpenAI 的 arguments 字段是 JSON 字符串，需要解析
                    if func_args.strip() == "[DONE]":
                        self.logger.warning(f"Found [DONE] in tool call arguments, skipping")
                        continue
                    try:
                        func_args_json = json.loads(func_args) if isinstance(func_args, str) else func_args
                    except json.JSONDecodeError as e:
                        self.logger.error(f"JSON decode error in tool args: {e}, func_args: '{func_args}'")
                        func_args_json = {}

                    func_args_json = self._adapt_tool_call_params(func_name, func_args_json)
                    function_call = {
                        "name": func_name,
                        "args": func_args_json
                    }
                    # 保留原始 tool_call ID
                    tool_call_id = tool_call.get("id")
                    if tool_call_id:
                        function_call["id"] = tool_call_id

                    # 构建 part，包含 functionCall
                    part: Dict[str, Any] = {"functionCall": function_call}

                    # 回填 thoughtSignature（如果之前捕获过）
                    if tool_call_id and tool_call_id in self._thought_signatures_by_tool_call_id:
                        thought_signature = self._thought_signatures_by_tool_call_id[tool_call_id]
                        part["thoughtSignature"] = thought_signature
                        print(f"🧠 [THOUGHT_SIGNATURE] Restored (streaming): tool_call_id={tool_call_id}, signature={thought_signature[:50]}...")

                    parts.append(part)
                
                # 清理工具调用状态，为下次请求做准备
                self._streaming_tool_calls = {}
            
            # 这是最后的chunk，包含finish_reason
            result_data = {
                "candidates": [{
                    "content": {
                        "parts": parts,
                        "role": "model"
                    },
                    "finishReason": self._map_finish_reason(choice.get("finish_reason", "stop"), "openai", "gemini"),
                    "index": 0
                }]
            }
            
            # 添加usage信息（如果有且不为None）
            if "usage" in data and data["usage"] is not None:
                usage = data["usage"]
                result_data["usageMetadata"] = {
                    "promptTokenCount": usage.get("prompt_tokens", 0),
                    "candidatesTokenCount": usage.get("completion_tokens", 0),
                    "totalTokenCount": usage.get("total_tokens", 0)
                }
            
            return ConversionResult(success=True, data=result_data)
            
        # 检查是否有增量内容（非finish chunk）
        elif "choices" in data and data["choices"] and data["choices"][0]:
            choice = data["choices"][0]
            delta = choice.get("delta", {})
            content = delta.get("content", "")
            tool_calls = delta.get("tool_calls", [])
            
            parts = []
            
            # 处理文本内容
            if content:
                parts.append({"text": content})
            
            # 对于工具调用chunks，我们已经在上面收集了，这里不需要再处理
            # 只有当有文本内容时才发送chunk给客户端
            if tool_calls:
                self.logger.debug(f"Skipping tool call chunk (already collected): {tool_calls}")
            
            # 只有在有文本内容时才创建chunk
            if parts:
                result_data = {
                    "candidates": [{
                        "content": {
                            "parts": parts,
                            "role": "model"
                        },
                        "index": 0
                    }]
                }
                
                return ConversionResult(success=True, data=result_data)
        
        # 如果没有内容也没有工具调用，则返回空 candidates，保持 SSE 连接
        result_data = {
            "candidates": [{
                "content": {
                    "parts": [],
                    "role": "model"
                },
                "index": 0
            }]
        }
        
        return ConversionResult(success=True, data=result_data)
    
    def _convert_from_anthropic_streaming_chunk(self, data: Dict[str, Any]) -> ConversionResult:
        """转换Anthropic流式响应chunk到Gemini格式"""
        import json
        import random
        
        # 初始化流状态变量
        if not hasattr(self, '_anthropic_to_gemini_state'):
            self._anthropic_to_gemini_state = {
                'current_text': '',
                'current_tool_calls': {},  # index -> tool_call_info
                'has_started': False
            }
        
        state = self._anthropic_to_gemini_state
        
        # 检查是否是message_delta类型的结束
        if data.get("type") == "message_delta" and "delta" in data and "stop_reason" in data["delta"]:
            # 这是最后的chunk，只包含工具调用和结束信息，不重复发送文本
            parts = []
            
            # 只添加工具调用（文本内容已经在之前的text_delta中发送过了）
            for tool_info in state['current_tool_calls'].values():
                tool_name = tool_info.get('name')
                tool_use_id = tool_info.get('id')
                if tool_name and tool_info.get('complete_args'):
                    try:
                        args_obj = json.loads(tool_info['complete_args'])
                        args_obj = self._adapt_tool_call_params(tool_name, args_obj)
                        function_call = {
                            "name": tool_name,
                            "args": args_obj
                        }
                        # 保留 Anthropic tool_use 的 ID
                        if tool_use_id:
                            function_call["id"] = tool_use_id

                        # 构建 part，包含 functionCall
                        part: Dict[str, Any] = {"functionCall": function_call}

                        # 回填 thoughtSignature（如果之前捕获过）
                        if tool_use_id and tool_use_id in self._thought_signatures_by_tool_call_id:
                            thought_signature = self._thought_signatures_by_tool_call_id[tool_use_id]
                            part["thoughtSignature"] = thought_signature
                            print(f"🧠 [THOUGHT_SIGNATURE] Restored (Anthropic streaming): tool_use_id={tool_use_id}, signature={thought_signature[:50]}...")

                        parts.append(part)
                    except json.JSONDecodeError:
                        function_call = {
                            "name": tool_name,
                            "args": {}
                        }
                        if tool_use_id:
                            function_call["id"] = tool_use_id

                        # 构建 part，包含 functionCall
                        part_err: Dict[str, Any] = {"functionCall": function_call}

                        # 回填 thoughtSignature（如果之前捕获过）
                        if tool_use_id and tool_use_id in self._thought_signatures_by_tool_call_id:
                            thought_signature = self._thought_signatures_by_tool_call_id[tool_use_id]
                            part_err["thoughtSignature"] = thought_signature
                            print(f"🧠 [THOUGHT_SIGNATURE] Restored (Anthropic streaming, decode error): tool_use_id={tool_use_id}, signature={thought_signature[:50]}...")

                        parts.append(part_err)
            
            # 确定finish reason
            stop_reason = data["delta"]["stop_reason"]
            if stop_reason == "tool_use":
                finish_reason = "STOP"  # Gemini中工具调用也用STOP
            else:
                finish_reason = self._map_finish_reason(stop_reason, "anthropic", "gemini")
            
            result_data = {
                "candidates": [{
                    "content": {
                        "parts": parts,
                        "role": "model"
                    },
                    "finishReason": finish_reason,
                    "index": 0
                }]
            }
            
            # 添加usage信息（如果有且不为None）
            if "usage" in data and data["usage"] is not None:
                usage = data["usage"]
                result_data["usageMetadata"] = {
                    "promptTokenCount": usage.get("input_tokens", 0),
                    "candidatesTokenCount": usage.get("output_tokens", 0),
                    "totalTokenCount": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                }
            
            # 清理状态
            if hasattr(self, '_anthropic_to_gemini_state'):
                delattr(self, '_anthropic_to_gemini_state')
            
            return ConversionResult(success=True, data=result_data)
        
        # 处理content_block_start - 工具调用开始
        elif data.get("type") == "content_block_start" and "content_block" in data:
            content_block = data["content_block"]
            index = data.get("index", 0)
            
            if content_block.get("type") == "tool_use":
                # 记录工具调用信息
                state['current_tool_calls'][index] = {
                    'id': content_block.get("id", ""),
                    'name': content_block.get("name", ""),
                    'complete_args': ''
                }
            
            # 对于content_block_start，不返回任何内容
            return ConversionResult(success=True, data={})
        
        # 处理content_block_delta - 增量内容
        elif data.get("type") == "content_block_delta" and "delta" in data:
            delta = data["delta"]
            index = data.get("index", 0)
            
            # 文本增量
            if delta.get("type") == "text_delta":
                text_content = delta.get("text", "")
                if text_content:
                    state['current_text'] += text_content
                    
                    # 实时返回文本增量
                    result_data = {
                        "candidates": [{
                            "content": {
                                "parts": [{"text": text_content}],
                                "role": "model"
                            },
                            "index": 0
                        }]
                    }
                    return ConversionResult(success=True, data=result_data)
            
            # 工具调用参数增量
            elif delta.get("type") == "input_json_delta":
                partial_json = delta.get("partial_json", "")
                if index in state['current_tool_calls']:
                    state['current_tool_calls'][index]['complete_args'] += partial_json
                
                # 对于工具参数增量，不返回实时内容
                return ConversionResult(success=True, data={})
        
        # 处理content_block_stop
        elif data.get("type") == "content_block_stop":
            # 对于content_block_stop，不返回任何内容
            return ConversionResult(success=True, data={})
        
        # 处理message_start
        elif data.get("type") == "message_start":
            # 强制重置状态，确保新流开始时状态干净
            self._anthropic_to_gemini_state = {
                'current_text': '',
                'current_tool_calls': {},  # index -> tool_call_info
                'has_started': True
            }
            # 对于message_start，不返回任何内容
            return ConversionResult(success=True, data={})
        
        # 处理message_stop
        elif data.get("type") == "message_stop":
            # 清理状态
            if hasattr(self, '_anthropic_to_gemini_state'):
                delattr(self, '_anthropic_to_gemini_state')
            return ConversionResult(success=True, data={})
        
        # 其他类型的事件，返回空结构
        return ConversionResult(success=True, data={})
    
    def _convert_from_gemini_streaming_chunk(self, data: Dict[str, Any]) -> ConversionResult:
        """转换Gemini流式响应chunk到目标格式"""
        # Gemini流式响应通常包含candidates数组
        if "candidates" in data and data["candidates"]:
            candidate = data["candidates"][0]
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            finish_reason = candidate.get("finishReason")
            
            # 检查是否包含工具调用
            has_function_call = any("functionCall" in part for part in parts)
            has_text = any("text" in part for part in parts)
            
            result_data = {
                "candidates": [{
                    "content": {
                        "parts": parts,
                        "role": "model"
                    },
                    "index": 0
                }]
            }
            
            # 添加finishReason（如果有）
            if finish_reason:
                result_data["candidates"][0]["finishReason"] = finish_reason
            
            # 处理usage信息
            if "usageMetadata" in data:
                result_data["usageMetadata"] = data["usageMetadata"]
            
            return ConversionResult(success=True, data=result_data)
        
        # 如果没有candidates，返回空的结构
        result_data = {
            "candidates": [{
                "content": {
                    "parts": [],
                    "role": "model"
                },
                "index": 0
            }]
        }
        
        return ConversionResult(success=True, data=result_data)
    
    def _convert_content_from_gemini(self, parts: List[Dict[str, Any]]) -> Any:
        """转换Gemini内容到通用格式"""
        if len(parts) == 1 and "text" in parts[0]:
            return parts[0]["text"]
        
        # 处理多模态内容和工具调用
        converted_content = []
        has_tool_calls = False
        tool_calls = []
        text_content = ""
        
        for part in parts:
            if "text" in part:
                text_content += part["text"]
                converted_content.append({
                    "type": "text",
                    "text": part["text"]
                })
            elif "inlineData" in part:
                inline_data = part["inlineData"]
                mime_type = inline_data.get("mimeType", "image/jpeg")
                data_part = inline_data.get("data", "")

                if isinstance(mime_type, str) and mime_type.startswith("image/"):
                    # 图像内容 → OpenAI image_url
                    converted_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{data_part}"
                        }
                    })
                elif isinstance(mime_type, str) and mime_type.startswith("audio/"):
                    # 音频内容 → OpenAI input_audio
                    audio_format = self._get_audio_format_from_mime(mime_type)
                    converted_content.append({
                        "type": "input_audio",
                        "input_audio": {
                            "data": data_part,
                            "format": audio_format
                        }
                    })
                elif isinstance(mime_type, str) and mime_type.startswith("video/"):
                    # 视频内容：OpenAI不直接支持，记录警告
                    self.logger.warning(f"Video content (mimeType={mime_type}) not supported for OpenAI, skipping")
                else:
                    # 未知类型或非字符串，记录警告避免静默失败
                    self.logger.warning(f"Unsupported inlineData mimeType: {mime_type}, skipping")
            elif "fileData" in part:
                # P2-16: fileData 内容（GCS 文件引用）→ 文本占位符
                file_data = part.get("fileData") or {}
                mime_type = file_data.get("mimeType", "")
                file_uri = file_data.get("fileUri", "")
                placeholder = f"[fileData: mimeType={mime_type or 'unknown'}, uri={file_uri or 'unknown'}]"
                text_content += placeholder
                converted_content.append({"type": "text", "text": placeholder})
                self.logger.warning(f"Gemini fileData converted to text placeholder (uri={file_uri})")
            elif "executableCode" in part or "executable_code" in part:
                # Gemini 代码执行工具返回的可执行代码片段
                code_info = part.get("executableCode") or part.get("executable_code") or {}
                language = code_info.get("language", "python")
                code = code_info.get("code", "")

                # 转换为 Markdown fenced code block
                fence_lang = language if isinstance(language, str) else "python"
                code_block = f"```{fence_lang}\n{code}\n```"
                text_content += code_block
                converted_content.append({"type": "text", "text": code_block})
                self.logger.warning(
                    "Gemini executableCode converted to fenced code block. "
                    "OpenAI/Anthropic will not execute this code automatically."
                )
            elif "codeExecutionResult" in part or "code_execution_result" in part:
                # Gemini 代码执行结果
                result_info = part.get("codeExecutionResult") or part.get("code_execution_result") or {}
                outcome = result_info.get("outcome")
                output = result_info.get("output", "")

                lines = ["[code_execution_result]"]
                if outcome:
                    lines.append(f"outcome: {outcome}")
                if output:
                    lines.append(f"output:\n{output}")

                result_text = "\n".join(lines)
                text_content += result_text
                converted_content.append({"type": "text", "text": result_text})
                self.logger.warning(
                    "Gemini codeExecutionResult converted to plain text."
                )
            elif "functionCall" in part:
                # 转换函数调用
                fc = part["functionCall"]
                func_name = fc.get("name", "")
                func_args = fc.get("args", {})

                # 优先使用 functionCall 自带的 id 字段（与 functionResponse 中的 id 保持一致）
                tool_call_id = fc.get("id")
                if not tool_call_id:
                    # 备选：使用预先建立的ID映射
                    if not hasattr(self, '_current_call_sequence'):
                        self._current_call_sequence = {}

                    sequence = self._current_call_sequence.get(func_name, 0) + 1
                    self._current_call_sequence[func_name] = sequence

                    tool_call_id = self._function_call_mapping.get(f"{func_name}_{sequence}")
                    if not tool_call_id:
                        # 如果映射中没有找到，生成新的ID
                        tool_call_id = f"call_{func_name}_{sequence:04d}"

                # 提取并保存 thoughtSignature（用于后续回传）
                thought_signature = part.get("thoughtSignature")
                if thought_signature:
                    self._thought_signatures_by_tool_call_id[tool_call_id] = thought_signature
                    print(f"🧠 [THOUGHT_SIGNATURE] Captured: tool_call_id={tool_call_id}, signature={thought_signature[:50]}...")

                tool_calls.append({
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": json.dumps(func_args, ensure_ascii=False)
                    }
                })
                has_tool_calls = True
            elif "functionResponse" in part:
                # 函数响应在这里标记，但实际处理需要在消息级别
                converted_content.append({
                    "type": "function_response",
                    "function_response": part["functionResponse"]
                })
        
        # 如果有工具调用，返回特殊格式标识
        if has_tool_calls:
            return {
                "type": "tool_calls",
                "content": text_content if text_content else None,
                "tool_calls": tool_calls
            }
        
        # 如果只有一个文本且没有其他内容，直接返回文本
        if len(converted_content) == 1 and converted_content[0].get("type") == "text":
            return converted_content[0]["text"]
        
        return converted_content if converted_content else ""
    
    def _convert_content_to_anthropic(self, parts: List[Dict[str, Any]]) -> Any:
        """转换Gemini内容到Anthropic格式"""
        # 处理多模态和工具调用内容
        anthropic_content = []
        
        for part in parts:
            if "text" in part:
                text_content = part["text"]
                if text_content.strip():  # 只添加非空文本
                    # 检查是否是thinking内容
                    if part.get("thought", False):
                        # Gemini thinking → Anthropic thinking
                        anthropic_content.append({
                            "type": "thinking",
                            "thinking": text_content
                        })
                    else:
                        # 普通文本内容
                        anthropic_content.append({
                            "type": "text",
                            "text": text_content
                        })
            elif "inlineData" in part:
                inline_data = part["inlineData"]
                mime_type = inline_data.get("mimeType", "image/jpeg")
                data_part = inline_data.get("data", "")

                if isinstance(mime_type, str) and mime_type.startswith("image/"):
                    # 图像内容 → Anthropic image
                    anthropic_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": data_part
                        }
                    })
                elif isinstance(mime_type, str) and mime_type.startswith("audio/"):
                    # 音频内容：Anthropic不直接支持，记录警告
                    self.logger.warning(f"Audio content (mimeType={mime_type}) not supported for Anthropic, skipping")
                elif isinstance(mime_type, str) and mime_type.startswith("video/"):
                    # 视频内容：Anthropic不直接支持，记录警告
                    self.logger.warning(f"Video content (mimeType={mime_type}) not supported for Anthropic, skipping")
                else:
                    # 未知类型，记录警告
                    self.logger.warning(f"Unsupported inlineData mimeType: {mime_type}, skipping")
            elif "fileData" in part:
                # P2-16: fileData 内容（GCS 文件引用）→ 文本占位符
                file_data = part.get("fileData") or {}
                mime_type = file_data.get("mimeType", "")
                file_uri = file_data.get("fileUri", "")
                placeholder = f"[fileData: mimeType={mime_type or 'unknown'}, uri={file_uri or 'unknown'}]"
                anthropic_content.append({"type": "text", "text": placeholder})
                self.logger.warning(f"Gemini fileData converted to text placeholder for Anthropic (uri={file_uri})")
            elif "executableCode" in part or "executable_code" in part:
                # Gemini 代码执行工具返回的可执行代码片段 → 转为文本
                code_info = part.get("executableCode") or part.get("executable_code") or {}
                language = code_info.get("language", "python")
                code = code_info.get("code", "")
                fence_lang = language if isinstance(language, str) else "python"
                code_block = f"```{fence_lang}\n{code}\n```"
                anthropic_content.append({"type": "text", "text": code_block})
                self.logger.warning(
                    "Gemini executableCode converted to text for Anthropic."
                )
            elif "codeExecutionResult" in part or "code_execution_result" in part:
                # Gemini 代码执行结果 → 转为文本
                result_info = part.get("codeExecutionResult") or part.get("code_execution_result") or {}
                outcome = result_info.get("outcome")
                output = result_info.get("output", "")
                lines = ["[code_execution_result]"]
                if outcome:
                    lines.append(f"outcome: {outcome}")
                if output:
                    lines.append(f"output:\n{output}")
                result_text = "\n".join(lines)
                anthropic_content.append({"type": "text", "text": result_text})
                self.logger.warning(
                    "Gemini codeExecutionResult converted to text for Anthropic."
                )
            elif "functionCall" in part:
                # 转换工具调用 (functionCall → tool_use)
                func_call = part["functionCall"]
                func_name = func_call.get("name", "")
                
                # 使用映射表获取一致的ID
                tool_id = None
                if hasattr(self, '_function_call_mapping') and self._function_call_mapping:
                    # 查找对应的ID
                    for key, mapped_id in self._function_call_mapping.items():
                        if key.startswith(func_name) and not key.startswith("response_"):
                            tool_id = mapped_id
                            break
                
                # 如果没有找到映射ID，生成一个
                if not tool_id:
                    tool_id = f"call_{func_name}_{hash(str(func_call.get('args', {}))) % 1000000000}"
                
                anthropic_content.append({
                    "type": "tool_use",
                    "id": tool_id,
                    "name": func_name,
                    "input": func_call.get("args", {})
                })
            elif "functionResponse" in part:
                # 转换工具响应 (functionResponse → tool_result)
                func_response = part["functionResponse"]
                func_name = func_response.get("name", "")
                
                # 使用映射表获取对应的tool_use ID
                tool_id = None
                if hasattr(self, '_function_call_mapping') and self._function_call_mapping:
                    for key, mapped_id in self._function_call_mapping.items():
                        if key.startswith(f"response_{func_name}"):
                            tool_id = mapped_id
                            break
                
                # 如果没有找到映射ID，尝试从functionResponse中获取或生成
                if not tool_id:
                    # 检查functionResponse是否包含原始的tool_use_id
                    response_data = func_response.get("response", {})
                    if isinstance(response_data, dict) and "_tool_use_id" in response_data:
                        tool_id = response_data["_tool_use_id"]
                    else:
                        # 生成一个基于函数名的一致性ID
                        # 使用函数名作为种子来生成一致的hash
                        import hashlib
                        seed = f"{func_name}_seed"
                        hash_value = int(hashlib.md5(seed.encode()).hexdigest()[:8], 16) % 1000000000
                        tool_id = f"call_{func_name}_{hash_value}"
                
                # 提取实际的工具结果内容
                result_content = func_response.get("response", {})
                if isinstance(result_content, dict):
                    # 如果包含_tool_use_id，移除它
                    result_content = {k: v for k, v in result_content.items() if k != "_tool_use_id"}
                    # 提取实际内容
                    actual_content = result_content.get("content", result_content)
                else:
                    actual_content = result_content
                    
                anthropic_content.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": str(actual_content)
                })
        
        # 如果没有任何有效内容，返回空字符串（会在调用处处理）
        if not anthropic_content:
            return ""
        
        # 如果只有一个文本内容，直接返回字符串
        if len(anthropic_content) == 1 and anthropic_content[0].get("type") == "text":
            return anthropic_content[0]["text"]
        
        # 返回内容块数组
        return anthropic_content

    def _get_audio_format_from_mime(self, mime_type: str) -> str:
        """从音频MIME类型中提取格式字符串

        Args:
            mime_type: 音频MIME类型，如 'audio/wav', 'audio/mpeg'

        Returns:
            OpenAI支持的音频格式字符串，如 'wav', 'mp3'
        """
        if not mime_type or "/" not in mime_type:
            return "wav"

        subtype = mime_type.split("/")[1].lower()

        # 去掉常见前缀（如 x-wav）
        if subtype.startswith("x-"):
            subtype = subtype[2:]

        # MIME子类型到OpenAI格式的映射
        format_mapping = {
            "mpeg": "mp3",
            "mp3": "mp3",
            "wav": "wav",
            "wave": "wav",
            "webm": "webm",
            "ogg": "ogg",
            "flac": "flac",
        }

        return format_mapping.get(subtype, subtype or "wav")

    def _merge_consecutive_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """合并连续的同角色消息

        Gemini API 不允许连续的同角色消息（如连续两条 user 消息），会返回 400 错误。
        此方法将连续的同角色文本消息合并为一条，以避免该限制。

        注意：只合并纯文本消息，不合并包含 tool_calls 或 tool_call_id 的消息。
        """
        if not messages or len(messages) < 2:
            return messages

        merged = []
        i = 0
        while i < len(messages):
            current = messages[i]
            role = current.get("role")
            content = current.get("content")

            # 只对纯文本消息进行合并（不含 tool_calls / tool_call_id）
            if (role in ("user", "assistant")
                and "tool_calls" not in current
                and "tool_call_id" not in current
                and isinstance(content, str)):

                # 收集所有连续的同角色纯文本消息
                texts = [content]
                j = i + 1
                while j < len(messages):
                    next_msg = messages[j]
                    next_role = next_msg.get("role")
                    next_content = next_msg.get("content")
                    if (next_role == role
                        and "tool_calls" not in next_msg
                        and "tool_call_id" not in next_msg
                        and isinstance(next_content, str)):
                        texts.append(next_content)
                        j += 1
                    else:
                        break

                if len(texts) > 1:
                    # 合并为单条消息
                    merged.append({
                        "role": role,
                        "content": "\n\n".join(texts)
                    })
                    self.logger.debug(f"Merged {len(texts)} consecutive {role} messages")
                else:
                    merged.append(current)
                i = j
            else:
                merged.append(current)
                i += 1

        return merged

    def _sanitize_schema_for_openai(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """将Gemini格式的JSON Schema转换为OpenAI兼容的格式"""
        if not isinstance(schema, dict) or not schema:
            # OpenAI 要求 parameters 必须是有效的 JSON Schema，至少需要 type 字段
            return {"type": "object", "properties": {}}

        # 复制schema避免修改原始数据
        sanitized = copy.deepcopy(schema)

        # 确保有 type 字段
        if "type" not in sanitized:
            sanitized["type"] = "object"
        if sanitized.get("type") == "object" and "properties" not in sanitized:
            sanitized["properties"] = {}
        
        # Gemini到OpenAI的类型映射
        type_mapping = {
            "STRING": "string",
            "NUMBER": "number", 
            "INTEGER": "integer",
            "BOOLEAN": "boolean",
            "ARRAY": "array",
            "OBJECT": "object"
        }
        
        def convert_types(obj):
            """递归转换schema中的类型"""
            if isinstance(obj, dict):
                # 转换type字段
                if "type" in obj and isinstance(obj["type"], str):
                    obj["type"] = type_mapping.get(obj["type"].upper(), obj["type"].lower())
                
                # 转换需要整数值的字段（将字符串转换为整数）
                integer_fields = ["minItems", "maxItems", "minimum", "maximum", "minLength", "maxLength"]
                for field in integer_fields:
                    if field in obj and isinstance(obj[field], str) and obj[field].isdigit():
                        obj[field] = int(obj[field])
                
                # 递归处理所有字段（跳过已经处理过的标量字段）
                for key, value in obj.items():
                    if key not in ["type"] + integer_fields:  # 避免重复处理已转换的字段
                        obj[key] = convert_types(value)
                    
            elif isinstance(obj, list):
                # 处理数组中的每个元素
                return [convert_types(item) for item in obj]
            
            return obj
        
        return convert_types(sanitized)
    
    def _build_function_call_mapping(self, contents: List[Dict[str, Any]]) -> Dict[str, str]:
        """扫描整个对话历史，为functionCall和functionResponse建立ID映射"""
        mapping = {}
        function_call_sequence = {}  # {func_name: sequence_number}

        for content in contents:
            parts = content.get("parts", [])
            for part in parts:
                if "functionCall" in part:
                    fc = part["functionCall"]
                    func_name = fc.get("name", "")
                    if func_name:
                        # 为每个函数调用生成唯一的sequence number
                        sequence = function_call_sequence.get(func_name, 0) + 1
                        function_call_sequence[func_name] = sequence

                        # 优先使用上游提供的 id，保证与 functionResponse 一致
                        tool_call_id = fc.get("id") or f"call_{func_name}_{sequence:04d}"
                        mapping[f"{func_name}_{sequence}"] = tool_call_id
                        
                elif "functionResponse" in part:
                    func_name = part["functionResponse"].get("name", "")
                    if func_name:
                        # 为functionResponse分配最近的functionCall的ID
                        current_sequence = function_call_sequence.get(func_name, 0)
                        if current_sequence > 0:
                            mapping[f"response_{func_name}_{current_sequence}"] = mapping.get(f"{func_name}_{current_sequence}")
        
        return mapping
    
    def _convert_schema_for_anthropic(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """将Gemini格式的JSON Schema转换为Anthropic兼容的格式"""
        if not isinstance(schema, dict):
            return schema
        
        # 复制schema避免修改原始数据
        converted = copy.deepcopy(schema)
        
        # Gemini到Anthropic的类型映射
        type_mapping = {
            "STRING": "string",
            "NUMBER": "number", 
            "INTEGER": "integer",
            "BOOLEAN": "boolean",
            "ARRAY": "array",
            "OBJECT": "object"
        }
        
        def convert_types(obj):
            """递归转换schema中的类型"""
            if isinstance(obj, dict):
                # 转换type字段
                if "type" in obj and isinstance(obj["type"], str):
                    obj["type"] = type_mapping.get(obj["type"].upper(), obj["type"].lower())
                
                # 转换需要整数值的字段（将字符串转换为整数）
                integer_fields = ["minItems", "maxItems", "minimum", "maximum", "minLength", "maxLength"]
                for field in integer_fields:
                    if field in obj and isinstance(obj[field], str) and obj[field].isdigit():
                        obj[field] = int(obj[field])
                
                # 递归处理所有字段（跳过已经处理过的标量字段）
                for key, value in obj.items():
                    if key not in ["type"] + integer_fields:  # 避免重复处理已转换的字段
                        obj[key] = convert_types(value)
                    
            elif isinstance(obj, list):
                # 处理数组中的每个元素
                return [convert_types(item) for item in obj]
            
            return obj
        
        return convert_types(converted)

    def _adapt_tool_call_params(self, tool_name: str, args: Any) -> Any:
        """适配工具调用参数到预期的schema

        处理已知的不兼容情况，例如：
        - ReadFile: paths/path -> file_path
        """
        if not isinstance(args, dict):
            return args

        adapted = dict(args)

        # gemini-cli 内置 read_file/ReadFile 工具：paths/path -> file_path
        if tool_name.lower() in ("readfile", "read_file") and "file_path" not in adapted:
            # 优先处理 paths（复数）
            paths = adapted.get("paths")
            if isinstance(paths, list) and paths:
                adapted["file_path"] = paths[0]
                del adapted["paths"]
                if len(paths) > 1:
                    self.logger.warning(f"ReadFile: only first path used, {len(paths)-1} paths ignored")
            elif isinstance(paths, str) and paths:
                adapted["file_path"] = paths
                del adapted["paths"]
            # 兼容 Gemini 模型使用的单数 path 字段
            elif "path" in adapted:
                single_path = adapted["path"]
                if isinstance(single_path, str) and single_path:
                    adapted["file_path"] = single_path
                    del adapted["path"]
                elif isinstance(single_path, list) and single_path:
                    adapted["file_path"] = single_path[0]
                    del adapted["path"]
                    if len(single_path) > 1:
                        self.logger.warning(f"ReadFile: only first path used from 'path', {len(single_path)-1} ignored")

        return adapted

    def _map_finish_reason(self, reason: str, source_format: str, target_format: str) -> str:
        """映射结束原因"""
        reason_mappings = {
            "openai": {
                "gemini": {
                    "stop": "STOP",
                    "length": "MAX_TOKENS",
                    "content_filter": "SAFETY",
                    "tool_calls": "MODEL_REQUESTED_TOOL"
                }
            },
            "anthropic": {
                "gemini": {
                    "end_turn": "STOP",
                    "max_tokens": "MAX_TOKENS",
                    "stop_sequence": "STOP",
                    "tool_use": "STOP"
                }
            }
        }
        
        try:
            return reason_mappings[source_format][target_format].get(reason, "STOP")
        except KeyError:
            return "STOP"