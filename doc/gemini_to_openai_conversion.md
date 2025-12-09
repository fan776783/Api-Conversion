# Gemini 格式转 OpenAI 格式转换逻辑文档

## 目录

1. [概述](#概述)
2. [整体架构](#整体架构)
3. [请求转换流程](#请求转换流程)
4. [响应转换流程](#响应转换流程)
5. [流式响应转换](#流式响应转换)
6. [字段映射详解](#字段映射详解)
7. [特殊处理逻辑](#特殊处理逻辑)
8. [错误处理](#错误处理)
9. [配置项说明](#配置项说明)
10. [完整示例](#完整示例)

---

## 概述

本项目实现了 Google Gemini API 格式与 OpenAI Chat Completions API 格式之间的双向转换。当客户端使用 Gemini 格式发送请求，而目标渠道是 OpenAI 时，系统会自动进行格式转换。

### 支持的转换方向

- **请求转换**: Gemini → OpenAI（客户端发送 Gemini 格式，转发到 OpenAI API）
- **响应转换**: OpenAI → Gemini（接收 OpenAI 响应，返回 Gemini 格式给客户端）

### 核心文件

| 文件 | 说明 |
|------|------|
| `src/formats/gemini_converter.py` | Gemini 格式转换器，负责 Gemini → OpenAI 请求转换和 OpenAI → Gemini 响应转换 |
| `src/formats/openai_converter.py` | OpenAI 格式转换器 |
| `src/formats/converter_factory.py` | 转换器工厂，统一管理转换器实例 |
| `src/api/unified_api.py` | 统一 API 层，处理请求路由和转换调用 |

---

## 整体架构

### 请求流程图

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────┐
│   客户端    │────▶│  unified_api.py  │────▶│ GeminiConverter │────▶│  OpenAI API  │
│ (Gemini格式) │     │   请求接收       │     │  请求转换       │     │              │
└─────────────┘     └──────────────────┘     └─────────────────┘     └──────────────┘
                                                                            │
                                                                            ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────┐
│   客户端    │◀────│  unified_api.py  │◀────│ GeminiConverter │◀────│  OpenAI响应  │
│ (Gemini格式) │     │   响应返回       │     │  响应转换       │     │              │
└─────────────┘     └──────────────────┘     └─────────────────┘     └──────────────┘
```

### Gemini API 端点

系统支持以下 Gemini 格式的 API 端点：

| 端点 | 说明 |
|------|------|
| `POST /v1beta/models/{model_id}:generateContent` | 非流式生成 |
| `POST /v1beta/models/{model_id}:streamGenerateContent` | 流式生成 |

### 转换器工厂模式

```python
# converter_factory.py
class ConverterFactory:
    @classmethod
    def get_converter(cls, format_name: str) -> Optional[BaseConverter]:
        """获取指定格式的转换器"""
        converters = {
            "openai": OpenAIConverter,
            "anthropic": AnthropicConverter,
            "gemini": GeminiConverter
        }
        return converters.get(format_name)()
```

---

## 请求转换流程

### 入口函数

请求转换的入口位于 `src/formats/gemini_converter.py` 的 `_convert_to_openai_request` 方法（第 136-371 行）。

### 转换步骤

#### 1. 模型字段处理

```python
# 必须有原始模型名称，否则报错
if not self.original_model:
    raise ValueError("Original model name is required for request conversion")

result_data["model"] = self.original_model  # 使用原始模型名称
```

#### 2. 函数调用 ID 映射预扫描

在转换消息之前，先扫描整个对话历史，为 `functionCall` 和 `functionResponse` 建立一致的 ID 映射：

```python
# 初始化函数调用ID映射表，用于保持工具调用和工具结果的ID一致性
# 先扫描整个对话历史，为每个functionCall和functionResponse建立映射关系
self._function_call_mapping = self._build_function_call_mapping(data.get("contents", []))
```

**映射构建逻辑**（第 1210-1237 行）：

```python
def _build_function_call_mapping(self, contents: List[Dict[str, Any]]) -> Dict[str, str]:
    """扫描整个对话历史，为functionCall和functionResponse建立ID映射"""
    mapping = {}
    function_call_sequence = {}  # {func_name: sequence_number}

    for content in contents:
        parts = content.get("parts", [])
        for part in parts:
            if "functionCall" in part:
                func_name = part["functionCall"].get("name", "")
                if func_name:
                    # 为每个函数调用生成唯一的sequence number
                    sequence = function_call_sequence.get(func_name, 0) + 1
                    function_call_sequence[func_name] = sequence

                    # 生成一致的ID: call_{func_name}_{序号}
                    tool_call_id = f"call_{func_name}_{sequence:04d}"
                    mapping[f"{func_name}_{sequence}"] = tool_call_id

            elif "functionResponse" in part:
                func_name = part["functionResponse"].get("name", "")
                if func_name:
                    # 为functionResponse分配最近的functionCall的ID
                    current_sequence = function_call_sequence.get(func_name, 0)
                    if current_sequence > 0:
                        mapping[f"response_{func_name}_{current_sequence}"] = mapping.get(f"{func_name}_{current_sequence}")

    return mapping
```

#### 3. 系统消息转换

Gemini 的 `systemInstruction` 或 `system_instruction` 字段转换为 OpenAI 的 `system` 角色消息：

```python
# Gemini 格式（支持两种写法）
{
    "systemInstruction": {
        "parts": [{"text": "You are a helpful assistant."}]
    }
}
# 或
{
    "system_instruction": {
        "parts": [{"text": "You are a helpful assistant."}]
    }
}

# 转换为 OpenAI 格式
{
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
}
```

**代码实现**（第 152-162 行）：

```python
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
```

#### 4. 消息内容转换

Gemini 的 `contents` 数组转换为 OpenAI 的 `messages` 数组：

| Gemini 角色 | OpenAI 角色 | 说明 |
|-------------|-------------|------|
| `user` | `user` | 用户消息 |
| `model` | `assistant` | 助手消息 |
| `user` (含 functionResponse) | `tool` | 工具响应 |
| `tool` | `tool` | 工具响应（角色为 tool） |

**用户消息处理**（第 171-214 行）：

```python
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
                    tool_result = response_content.get("content", json.dumps(response_content, ensure_ascii=False))
                else:
                    tool_result = str(response_content)

                # 使用预先建立的映射获取对应的tool_call_id
                tool_call_id = self._function_call_mapping.get(f"response_{func_name}_{sequence}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": tool_result
                })
    else:
        # 普通用户消息
        message_content = self._convert_content_from_gemini(parts)
        messages.append({
            "role": "user",
            "content": message_content
        })
```

**助手消息处理（含工具调用）**（第 216-233 行）：

```python
elif gemini_role == "model":
    # 助手消息，可能包含工具调用
    message_content = self._convert_content_from_gemini(parts)

    if isinstance(message_content, dict) and message_content.get("type") == "tool_calls":
        # 有工具调用的助手消息
        message = {
            "role": "assistant",
            "content": message_content.get("content"),
            "tool_calls": message_content["tool_calls"]
        }
        messages.append(message)
    else:
        # 普通助手消息
        messages.append({
            "role": "assistant",
            "content": message_content
        })
```

#### 5. 生成配置转换

Gemini 的 `generationConfig` 转换为 OpenAI 的顶层参数：

| Gemini 参数 | OpenAI 参数 | 说明 |
|-------------|-------------|------|
| `temperature` | `temperature` | 温度参数 |
| `topP` | `top_p` | Top-P 采样 |
| `maxOutputTokens` | `max_tokens` | 最大输出 token 数 |
| `stopSequences` | `stop` | 停止序列 |
| `response_mime_type` | `response_format.type` | 响应格式 |
| `response_schema` | `response_format.json_schema` | JSON Schema |

**代码实现**（第 279-302 行）：

```python
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
```

#### 6. 工具调用转换

Gemini 的 `tools[].function_declarations` 转换为 OpenAI 的 `tools[].function`：

**Gemini 格式**（支持 snake_case 和 camelCase）：
```json
{
    "tools": [{
        "function_declarations": [{
            "name": "get_weather",
            "description": "Get weather info",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "location": {"type": "STRING"}
                }
            }
        }]
    }]
}
```

**OpenAI 格式**：
```json
{
    "tools": [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather info",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }
        }
    }],
    "tool_choice": "auto"
}
```

**代码实现**（第 304-326 行）：

```python
if "tools" in data:
    openai_tools = []
    for tool in data["tools"]:
        # Gemini官方使用 snake_case: function_declarations
        func_key = None
        if "function_declarations" in tool:
            func_key = "function_declarations"
        elif "functionDeclarations" in tool:  # 兼容旧写法
            func_key = "functionDeclarations"
        if func_key:
            for func_decl in tool[func_key]:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": func_decl.get("name", ""),
                        "description": func_decl.get("description", ""),
                        "parameters": self._sanitize_schema_for_openai(func_decl.get("parameters", {}))
                    }
                })
    if openai_tools:
        result_data["tools"] = openai_tools
        result_data["tool_choice"] = "auto"
```

**JSON Schema 类型转换**（第 1166-1208 行）：

Gemini 使用大写类型名，OpenAI 使用小写：

```python
def _sanitize_schema_for_openai(self, schema: Dict[str, Any]) -> Dict[str, Any]:
    """将Gemini格式的JSON Schema转换为OpenAI兼容的格式"""
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
        if isinstance(obj, dict):
            # 转换type字段
            if "type" in obj and isinstance(obj["type"], str):
                obj["type"] = type_mapping.get(obj["type"].upper(), obj["type"].lower())

            # 转换需要整数值的字段（将字符串转换为整数）
            integer_fields = ["minItems", "maxItems", "minimum", "maximum", "minLength", "maxLength"]
            for field in integer_fields:
                if field in obj and isinstance(obj[field], str) and obj[field].isdigit():
                    obj[field] = int(obj[field])

            # 递归处理所有字段
            for key, value in obj.items():
                obj[key] = convert_types(value)

        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]

        return obj

    return convert_types(copy.deepcopy(schema))
```

#### 6.1 非函数工具转换（虚拟工具映射）

Gemini 特有的非函数工具会被转换为虚拟函数工具，由调用方实现：

| Gemini 工具 | OpenAI 映射 | 说明 |
|------------|-------------|------|
| `code_execution` | `function: code_execution` | 代码执行工具，参数: `{code: string}` |
| `google_search` | `function: google_search` | Google 搜索工具，参数: `{query: string}` |
| `google_search_retrieval` | 警告并忽略 | 不支持 |
| `retrieval` | 警告并忽略 | 不支持 |

**代码实现示例**：
```python
# code_execution → 虚拟函数工具
if has_code_execution:
    openai_tools.append({
        "type": "function",
        "function": {
            "name": "code_execution",
            "description": "Execute Python code. Caller must implement the execution handler.",
            "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}
        }
    })
```

#### 6.2 代码执行响应转换

Gemini 的 `executableCode` 和 `codeExecutionResult` 响应部分转换为文本：

| Gemini 部分 | 转换结果 |
|------------|---------|
| `executableCode` | Markdown fenced code block: `` ```python\n{code}\n``` `` |
| `codeExecutionResult` | 文本: `[code_execution_result]\noutcome: {outcome}\noutput:\n{output}` |

#### 6.3 多模态内容处理

| Gemini 内容类型 | OpenAI 转换 | 说明 |
|----------------|-------------|------|
| `inlineData` (image/*) | `image_url` | Base64 图像 |
| `inlineData` (audio/*) | `input_audio` | 音频数据 |
| `inlineData` (video/*) | 警告并跳过 | OpenAI 不支持 |
| `fileData` | 文本占位符 | `[fileData: mimeType=..., uri=...]` |

#### 6.4 生成配置参数扩展

新增支持的 `generationConfig` 参数：

| Gemini 参数 | OpenAI 映射 |
|------------|------------|
| `presencePenalty` | `presence_penalty` |
| `frequencyPenalty` | `frequency_penalty` |
| `candidateCount` | `n` |

#### 6.5 安全设置透传

Gemini 的 `safetySettings` 会被保存到请求的 `metadata.gemini_safety_settings` 中，供下游中间件或日志使用。

#### 7. 思考模式转换（重点）

Gemini 的 `thinkingConfig.thinkingBudget` 转换为 OpenAI 的 `reasoning_effort` + `max_completion_tokens`：

**转换逻辑**（第 328-365 行）：

```python
if "generationConfig" in data and "thinkingConfig" in data["generationConfig"]:
    thinking_config = data["generationConfig"]["thinkingConfig"]
    thinking_budget = thinking_config.get("thinkingBudget")

    if thinking_budget is not None and thinking_budget != 0:
        # 根据thinking_budget判断reasoning_effort等级
        reasoning_effort = self._determine_reasoning_effort_from_budget(thinking_budget)
        result_data["reasoning_effort"] = reasoning_effort

        # 处理max_completion_tokens的优先级逻辑
        max_completion_tokens = None

        # 优先级1：客户端传入的maxOutputTokens
        if "generationConfig" in data and "maxOutputTokens" in data["generationConfig"]:
            max_completion_tokens = data["generationConfig"]["maxOutputTokens"]
            # 移除max_tokens，使用max_completion_tokens
            result_data.pop("max_tokens", None)
        else:
            # 优先级2：环境变量OPENAI_REASONING_MAX_TOKENS
            env_max_tokens = os.environ.get("OPENAI_REASONING_MAX_TOKENS")
            if env_max_tokens:
                max_completion_tokens = int(env_max_tokens)
            else:
                # 优先级3：都没有则报错
                raise ConversionError("For OpenAI reasoning models, max_completion_tokens is required.")

        result_data["max_completion_tokens"] = max_completion_tokens
```

**reasoning_effort 等级判断**（第 23-68 行）：

```python
def _determine_reasoning_effort_from_budget(self, thinking_budget: Optional[int]) -> str:
    """根据thinkingBudget判断OpenAI reasoning_effort等级"""
    # 如果没有提供thinking_budget或为-1（动态思考），默认为high
    if thinking_budget is None or thinking_budget == -1:
        return "high"

    # 从环境变量获取阈值配置（必需）
    low_threshold = int(os.environ.get("GEMINI_TO_OPENAI_LOW_REASONING_THRESHOLD"))
    high_threshold = int(os.environ.get("GEMINI_TO_OPENAI_HIGH_REASONING_THRESHOLD"))

    if thinking_budget <= low_threshold:
        return "low"
    elif thinking_budget <= high_threshold:
        return "medium"
    else:
        return "high"
```

**环境变量配置示例**：
```bash
GEMINI_TO_OPENAI_LOW_REASONING_THRESHOLD=4096
GEMINI_TO_OPENAI_HIGH_REASONING_THRESHOLD=16384
```

| thinkingBudget | reasoning_effort |
|----------------|------------------|
| `-1`（动态） | `high` |
| `<= 4096` | `low` |
| `4097 - 16384` | `medium` |
| `> 16384` | `high` |

---

## 响应转换流程

### 入口函数

响应转换的入口位于 `src/formats/gemini_converter.py` 的 `_convert_from_openai_response` 方法（第 504-568 行）。

### 转换步骤

#### 1. 基础结构构建

```python
result_data = {
    "candidates": [],
    "usageMetadata": {}
}
```

#### 2. 内容提取与转换

从 OpenAI 的 `choices[0].message` 中提取内容，转换为 Gemini 的 `candidates[0].content.parts`：

```python
if "choices" in data and data["choices"] and data["choices"][0]:
    choice = data["choices"][0]
    message = choice.get("message", {})
    content = message.get("content", "")
    tool_calls = message.get("tool_calls", [])

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
                func_args = json.loads(args_str) if args_str else {}

                parts.append({
                    "functionCall": {
                        "name": func_name,
                        "args": func_args
                    }
                })
```

#### 3. 候选项构建

```python
candidate = {
    "content": {
        "parts": parts if parts else [{"text": ""}],
        "role": "model"
    },
    "finishReason": self._map_finish_reason(choice.get("finish_reason", "stop"), "openai", "gemini"),
    "index": 0
}
result_data["candidates"] = [candidate]
```

#### 4. 使用量信息转换

```python
if "usage" in data and data["usage"] is not None:
    usage = data["usage"]
    result_data["usageMetadata"] = {
        "promptTokenCount": usage.get("prompt_tokens", 0),
        "candidatesTokenCount": usage.get("completion_tokens", 0),
        "totalTokenCount": usage.get("total_tokens", 0)
    }
```

#### 5. 结束原因映射

```python
reason_mappings = {
    "openai": {
        "gemini": {
            "stop": "STOP",
            "length": "MAX_TOKENS",
            "content_filter": "SAFETY",
            "tool_calls": "MODEL_REQUESTED_TOOL"
        }
    }
}
```

---

## 流式响应转换

### 流式 chunk 转换

流式 chunk 转换位于 `src/formats/gemini_converter.py` 的 `_convert_from_openai_streaming_chunk` 方法（第 634-782 行）。

### 关键设计：工具调用状态累积

OpenAI 流式响应中，工具调用的参数是逐步发送的，需要累积后在结束时一次性输出：

```python
def _convert_from_openai_streaming_chunk(self, data: Dict[str, Any]) -> ConversionResult:
    # 为流式工具调用维护状态
    if not hasattr(self, '_streaming_tool_calls'):
        self._streaming_tool_calls = {}

    # 收集流式工具调用信息
    if "choices" in data and data["choices"] and data["choices"][0]:
        choice = data["choices"][0]
        delta = choice.get("delta", {})

        if "tool_calls" in delta:
            for tool_call in delta["tool_calls"]:
                call_index = tool_call.get("index", 0)

                # 初始化工具调用状态
                if call_index not in self._streaming_tool_calls:
                    self._streaming_tool_calls[call_index] = {
                        "id": tool_call.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": "",
                            "arguments": ""
                        }
                    }

                # 累积函数名和参数
                if "function" in tool_call:
                    func = tool_call["function"]
                    if "name" in func:
                        self._streaming_tool_calls[call_index]["function"]["name"] = func["name"]
                    if "arguments" in func:
                        self._streaming_tool_calls[call_index]["function"]["arguments"] += func["arguments"]
```

### 结束 chunk 处理

当 OpenAI 响应中包含 `finish_reason` 时，输出完整的工具调用：

```python
if data["choices"][0].get("finish_reason"):
    parts = []

    # 处理文本内容
    content = delta.get("content", "")
    if content:
        parts.append({"text": content})

    # 处理收集到的工具调用
    if self._streaming_tool_calls:
        for call_index, tool_call in self._streaming_tool_calls.items():
            func = tool_call.get("function", {})
            func_name = func.get("name", "")
            func_args = func.get("arguments", "{}")

            # 解析JSON参数
            func_args_json = json.loads(func_args) if func_args else {}

            parts.append({
                "functionCall": {
                    "name": func_name,
                    "args": func_args_json
                }
            })

        # 清理工具调用状态
        self._streaming_tool_calls = {}

    result_data = {
        "candidates": [{
            "content": {
                "parts": parts,
                "role": "model"
            },
            "finishReason": self._map_finish_reason(finish_reason, "openai", "gemini"),
            "index": 0
        }]
    }

    # 添加usage信息（如果有）
    if "usage" in data and data["usage"] is not None:
        result_data["usageMetadata"] = {...}

    return ConversionResult(success=True, data=result_data)
```

### 普通文本 chunk 处理

对于纯文本的流式内容，实时转发：

```python
elif "choices" in data and data["choices"] and data["choices"][0]:
    delta = choice.get("delta", {})
    content = delta.get("content", "")

    if content:
        result_data = {
            "candidates": [{
                "content": {
                    "parts": [{"text": content}],
                    "role": "model"
                },
                "index": 0
            }]
        }
        return ConversionResult(success=True, data=result_data)
```

### 流式响应格式对比

**OpenAI 流式响应格式**：
```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":"stop"}],"usage":{...}}
data: [DONE]
```

**转换后的 Gemini 流式响应格式**：
```
data: {"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"},"index":0}]}
data: {"candidates":[{"content":{"parts":[{"text":" world"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{...}}
```

---

## 字段映射详解

### 请求字段映射表

| Gemini 字段 | OpenAI 字段 | 转换说明 |
|-------------|-------------|----------|
| `model` | `model` | 使用原始请求的模型名 |
| `systemInstruction.parts[].text` | `messages[role="system"].content` | 系统消息 |
| `system_instruction.parts[].text` | `messages[role="system"].content` | 系统消息（snake_case 写法） |
| `contents[role="user"]` | `messages[role="user"]` | 用户消息 |
| `contents[role="model"]` | `messages[role="assistant"]` | 助手消息 |
| `contents[].parts[].functionCall` | `messages[].tool_calls[]` | 工具调用 |
| `contents[].parts[].functionResponse` | `messages[role="tool"]` | 工具响应 |
| `generationConfig.temperature` | `temperature` | 温度 |
| `generationConfig.topP` | `top_p` | Top-P |
| `generationConfig.maxOutputTokens` | `max_tokens` | 最大输出（普通模式） |
| `generationConfig.maxOutputTokens` | `max_completion_tokens` | 最大输出（思考模式） |
| `generationConfig.stopSequences` | `stop` | 停止序列 |
| `tools[].function_declarations[]` | `tools[].function` | 工具定义 |
| `tools[].functionDeclarations[]` | `tools[].function` | 工具定义（camelCase 写法） |
| `generationConfig.thinkingConfig.thinkingBudget` | `reasoning_effort` | 思考等级映射 |
| `generationConfig.response_mime_type` | `response_format.type` | 响应格式 |
| `generationConfig.response_schema` | `response_format.json_schema` | JSON Schema |

### 响应字段映射表

| OpenAI 字段 | Gemini 字段 | 转换说明 |
|-------------|-------------|----------|
| `choices[0].message.content` | `candidates[0].content.parts[].text` | 文本内容 |
| `choices[0].message.tool_calls[]` | `candidates[0].content.parts[].functionCall` | 工具调用 |
| `choices[0].finish_reason` | `candidates[0].finishReason` | 结束原因映射 |
| `usage.prompt_tokens` | `usageMetadata.promptTokenCount` | 输入 token |
| `usage.completion_tokens` | `usageMetadata.candidatesTokenCount` | 输出 token |
| `usage.total_tokens` | `usageMetadata.totalTokenCount` | 总 token |

### 结束原因映射表

| OpenAI finish_reason | Gemini finishReason |
|----------------------|---------------------|
| `stop` | `STOP` |
| `length` | `MAX_TOKENS` |
| `content_filter` | `SAFETY` |
| `tool_calls` | `MODEL_REQUESTED_TOOL` |

### 类型映射表（JSON Schema）

| Gemini 类型 | OpenAI 类型 |
|-------------|-------------|
| `STRING` | `string` |
| `NUMBER` | `number` |
| `INTEGER` | `integer` |
| `BOOLEAN` | `boolean` |
| `ARRAY` | `array` |
| `OBJECT` | `object` |

---

## 特殊处理逻辑

### 1. 工具调用 ID 一致性

Gemini 的 `functionCall` 和 `functionResponse` 没有 ID 字段，但 OpenAI 需要 `tool_call_id` 来关联工具调用和响应。解决方案是预扫描对话历史，为每个调用生成一致的 ID：

```python
# 生成规则：call_{function_name}_{序号:04d}
# 例如：call_get_weather_0001, call_search_0001, call_get_weather_0002
tool_call_id = f"call_{func_name}_{sequence:04d}"
```

### 2. 多模态内容转换

**Gemini 格式**：
```python
{
    "parts": [{
        "inlineData": {
            "mimeType": "image/jpeg",
            "data": "base64_encoded_data"
        }
    }]
}
```

**转换为 OpenAI 格式**：
```python
{
    "type": "image_url",
    "image_url": {
        "url": "data:image/jpeg;base64,base64_encoded_data"
    }
}
```

**代码实现**（第 977-1005 行）：

```python
def _convert_content_from_gemini(self, parts: List[Dict[str, Any]]) -> Any:
    for part in parts:
        if "text" in part:
            text_content += part["text"]
        elif "inlineData" in part:
            inline_data = part["inlineData"]
            mime_type = inline_data.get("mimeType", "image/jpeg")
            data_part = inline_data.get("data", "")
            converted_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{data_part}"
                }
            })
```

### 3. 函数参数格式转换

Gemini 的 `args` 是对象，OpenAI 的 `arguments` 是 JSON 字符串：

```python
# Gemini → OpenAI：对象 → JSON字符串
"arguments": json.dumps(fc.get("args", {}), ensure_ascii=False)

# OpenAI → Gemini：JSON字符串 → 对象
func_args = json.loads(args_str) if args_str else {}
```

### 4. 思考预算动态值处理

当 `thinkingBudget` 为 `-1` 时，表示动态思考模式：

```python
if thinking_budget is None or thinking_budget == -1:
    # 动态思考，默认为 high
    return "high"
```

### 5. 空内容处理

Gemini 不允许空的 parts 数组：

```python
if not parts:
    parts = [{"text": ""}]
```

### 6. 流式工具调用参数累积

OpenAI 流式响应中，工具调用的参数是逐步发送的 JSON 片段，需要累积：

```python
# 累积 arguments
self._streaming_tool_calls[call_index]["function"]["arguments"] += func["arguments"]

# 最终解析
func_args_json = json.loads(func_args) if func_args else {}
```

### 7. max_completion_tokens 优先级

思考模式下的 token 限制优先级：

1. 客户端传入的 `maxOutputTokens`（最高优先级）
2. 环境变量 `OPENAI_REASONING_MAX_TOKENS`
3. 都没有则报错

```python
if "maxOutputTokens" in data["generationConfig"]:
    max_completion_tokens = data["generationConfig"]["maxOutputTokens"]
    result_data.pop("max_tokens", None)  # 移除普通模式的 max_tokens
else:
    env_max_tokens = os.environ.get("OPENAI_REASONING_MAX_TOKENS")
    if env_max_tokens:
        max_completion_tokens = int(env_max_tokens)
    else:
        raise ConversionError("max_completion_tokens is required for reasoning models")
```

---

## 错误处理

### 转换错误

```python
# convert_request 方法
try:
    return self._convert_to_openai_request(data)
except Exception as e:
    self.logger.error(f"Failed to convert Gemini request to openai: {e}")
    return ConversionResult(success=False, error=str(e))
```

### 必需环境变量缺失

思考模式转换需要特定环境变量：

```python
if low_threshold_str is None:
    raise ConversionError("GEMINI_TO_OPENAI_LOW_REASONING_THRESHOLD environment variable is required")

if high_threshold_str is None:
    raise ConversionError("GEMINI_TO_OPENAI_HIGH_REASONING_THRESHOLD environment variable is required")
```

### JSON 解析错误

工具调用参数解析失败时的处理：

```python
try:
    func_args = json.loads(args_str) if args_str else {}
except json.JSONDecodeError:
    func_args = {}
```

### 流式响应中的 [DONE] 标记

```python
if func_args.strip() == "[DONE]":
    self.logger.warning(f"Found [DONE] in tool call arguments, skipping")
    continue
```

---

## 配置项说明

### 环境变量

| 环境变量 | 必需 | 说明 | 示例值 |
|----------|------|------|--------|
| `GEMINI_TO_OPENAI_LOW_REASONING_THRESHOLD` | 是（思考模式） | low 等级的 thinking_budget 上限 | `4096` |
| `GEMINI_TO_OPENAI_HIGH_REASONING_THRESHOLD` | 是（思考模式） | medium 等级的 thinking_budget 上限 | `16384` |
| `OPENAI_REASONING_MAX_TOKENS` | 否 | 思考模式的默认 max_completion_tokens | `32768` |
| `ANTHROPIC_MAX_TOKENS` | 否 | 转 Anthropic 时的默认 max_tokens | `4096` |

### 阈值配置说明

```
thinkingBudget ≤ LOW_THRESHOLD  →  reasoning_effort = "low"
LOW_THRESHOLD < thinkingBudget ≤ HIGH_THRESHOLD  →  reasoning_effort = "medium"
thinkingBudget > HIGH_THRESHOLD  →  reasoning_effort = "high"
thinkingBudget = -1 (动态)  →  reasoning_effort = "high"
```

### 渠道配置

渠道配置中可以设置模型映射：

```python
if channel.models_mapping and isinstance(request_data, dict):
    original_model = request_data.get("model")
    if original_model:
        mapped_model = channel.models_mapping.get(original_model)
        if mapped_model:
            logger.info(f"Applying model mapping: {original_model} -> {mapped_model}")
```

---

## 完整示例

### 示例 1：基础对话请求

**Gemini 请求**：
```json
{
    "systemInstruction": {
        "parts": [{"text": "You are a helpful assistant."}]
    },
    "contents": [
        {
            "role": "user",
            "parts": [{"text": "What is the capital of France?"}]
        }
    ],
    "generationConfig": {
        "temperature": 0.7,
        "maxOutputTokens": 1000
    }
}
```

**转换后的 OpenAI 请求**：
```json
{
    "model": "gpt-4",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.7,
    "max_tokens": 1000
}
```

### 示例 2：工具调用请求

**Gemini 请求**：
```json
{
    "contents": [
        {
            "role": "user",
            "parts": [{"text": "What's the weather in Beijing?"}]
        }
    ],
    "tools": [{
        "function_declarations": [{
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "location": {"type": "STRING", "description": "City name"}
                },
                "required": ["location"]
            }
        }]
    }],
    "generationConfig": {
        "temperature": 0.7
    }
}
```

**转换后的 OpenAI 请求**：
```json
{
    "model": "gpt-4",
    "messages": [
        {"role": "user", "content": "What's the weather in Beijing?"}
    ],
    "tools": [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    }],
    "tool_choice": "auto",
    "temperature": 0.7
}
```

### 示例 3：工具响应流程

**Gemini 请求（包含工具调用结果）**：
```json
{
    "contents": [
        {
            "role": "user",
            "parts": [{"text": "What's the weather in Beijing?"}]
        },
        {
            "role": "model",
            "parts": [{
                "functionCall": {
                    "name": "get_weather",
                    "args": {"location": "Beijing"}
                }
            }]
        },
        {
            "role": "user",
            "parts": [{
                "functionResponse": {
                    "name": "get_weather",
                    "response": {"content": "Sunny, 25°C"}
                }
            }]
        }
    ]
}
```

**转换后的 OpenAI 请求**：
```json
{
    "model": "gpt-4",
    "messages": [
        {"role": "user", "content": "What's the weather in Beijing?"},
        {
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_get_weather_0001",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"location\": \"Beijing\"}"
                }
            }]
        },
        {
            "role": "tool",
            "tool_call_id": "call_get_weather_0001",
            "content": "Sunny, 25°C"
        }
    ]
}
```

### 示例 4：思考模式请求

**Gemini 请求**：
```json
{
    "contents": [
        {
            "role": "user",
            "parts": [{"text": "Solve this complex math problem..."}]
        }
    ],
    "generationConfig": {
        "thinkingConfig": {
            "thinkingBudget": 10000
        },
        "maxOutputTokens": 4096
    }
}
```

**转换后的 OpenAI 请求**（假设 LOW=4096, HIGH=16384）：
```json
{
    "model": "o1",
    "messages": [
        {"role": "user", "content": "Solve this complex math problem..."}
    ],
    "reasoning_effort": "medium",
    "max_completion_tokens": 4096
}
```

### 示例 5：OpenAI 响应转 Gemini

**OpenAI 响应**：
```json
{
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "gpt-4",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_xyz",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"location\": \"Beijing\"}"
                }
            }]
        },
        "finish_reason": "tool_calls"
    }],
    "usage": {
        "prompt_tokens": 50,
        "completion_tokens": 20,
        "total_tokens": 70
    }
}
```

**转换后的 Gemini 响应**：
```json
{
    "candidates": [{
        "content": {
            "parts": [{
                "functionCall": {
                    "name": "get_weather",
                    "args": {"location": "Beijing"}
                }
            }],
            "role": "model"
        },
        "finishReason": "MODEL_REQUESTED_TOOL",
        "index": 0
    }],
    "usageMetadata": {
        "promptTokenCount": 50,
        "candidatesTokenCount": 20,
        "totalTokenCount": 70
    }
}
```

---

## 总结

Gemini 到 OpenAI 的格式转换涉及以下核心点：

1. **消息结构差异**：Gemini 使用 `contents` 数组和单独的 `systemInstruction`，OpenAI 使用统一的 `messages` 数组
2. **角色映射**：`model` → `assistant`，系统消息需要合并到 messages 数组
3. **工具调用格式**：Gemini 的 `functionDeclarations/functionCall` 转换为 OpenAI 的 `tools/tool_calls`
4. **参数格式**：函数参数在 Gemini 中是对象，在 OpenAI 中是 JSON 字符串
5. **类型名称**：Gemini 使用大写类型名（STRING），OpenAI 使用小写（string）
6. **工具调用 ID**：需要预扫描生成一致的 ID 映射
7. **思考模式**：`thinkingBudget` 根据阈值映射为 `reasoning_effort` 等级
8. **流式响应**：需要累积工具调用参数，在结束时一次性输出完整的 functionCall
