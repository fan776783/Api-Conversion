# OpenAI 格式转 Gemini 格式转换逻辑文档

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

---

## 概述

本项目实现了 OpenAI Chat Completions API 格式与 Google Gemini API 格式之间的双向转换。当客户端使用 OpenAI 格式发送请求，而目标渠道是 Gemini 时，系统会自动进行格式转换。

### 支持的转换方向

- **请求转换**: OpenAI → Gemini（客户端发送 OpenAI 格式，转发到 Gemini API）
- **响应转换**: Gemini → OpenAI（接收 Gemini 响应，返回 OpenAI 格式给客户端）

### 核心文件

| 文件 | 说明 |
|------|------|
| `src/formats/openai_converter.py` | OpenAI 格式转换器，负责 OpenAI → Gemini 请求转换 |
| `src/formats/gemini_converter.py` | Gemini 格式转换器，负责 Gemini → OpenAI 响应转换 |
| `src/formats/converter_factory.py` | 转换器工厂，统一管理转换器实例 |
| `src/api/unified_api.py` | 统一 API 层，处理请求路由和转换调用 |

---

## 整体架构

### 请求流程图

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────┐
│   客户端    │────▶│  unified_api.py  │────▶│ OpenAIConverter │────▶│  Gemini API  │
│ (OpenAI格式) │     │   请求接收       │     │  请求转换       │     │              │
└─────────────┘     └──────────────────┘     └─────────────────┘     └──────────────┘
                                                                            │
                                                                            ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────┐
│   客户端    │◀────│  unified_api.py  │◀────│ OpenAIConverter │◀────│  Gemini响应  │
│ (OpenAI格式) │     │   响应返回       │     │  响应转换       │     │              │
└─────────────┘     └──────────────────┘     └─────────────────┘     └──────────────┘
```

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

请求转换的入口位于 `src/formats/openai_converter.py` 的 `_convert_to_gemini_request` 方法（第 259-472 行）。

### 转换步骤

#### 1. 模型字段透传

```python
# 透传模型字段，保持原样传递给 Gemini
if "model" in data:
    result_data["model"] = data["model"]
```

#### 2. 系统消息转换

OpenAI 的 `messages` 数组中 `role: "system"` 的消息需要转换为 Gemini 的 `system_instruction` 字段：

```python
# OpenAI 格式
{
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"}
    ]
}

# 转换为 Gemini 格式
{
    "system_instruction": {
        "parts": [{"text": "You are a helpful assistant."}]
    },
    "contents": [
        {"role": "user", "parts": [{"text": "Hello"}]}
    ]
}
```

**代码实现**（第 288-297 行）：

```python
if "messages" in data:
    system_message, filtered_messages = self._extract_system_message(data["messages"])

    if system_message:
        system_content = str(system_message).strip() if system_message else ""
        if system_content:
            result_data["system_instruction"] = {
                "parts": [{"text": system_content}]
            }
```

#### 3. 消息内容转换

OpenAI 的 `messages` 数组转换为 Gemini 的 `contents` 数组：

| OpenAI 角色 | Gemini 角色 | 说明 |
|-------------|-------------|------|
| `user` | `user` | 用户消息 |
| `assistant` | `model` | 助手消息 |
| `tool` | `tool` | 工具响应（functionResponse） |

**普通用户消息转换**（第 305-309 行）：

```python
if msg_role == "user":
    gemini_contents.append({
        "role": "user",
        "parts": self._convert_content_to_gemini(msg.get("content", ""))
    })
```

**助手带工具调用的消息转换**（第 312-334 行）：

```python
elif msg_role == "assistant" and msg.get("tool_calls"):
    parts = []
    for tc in msg["tool_calls"]:
        if tc and "function" in tc:
            fn_name = tc["function"].get("name")
            arg_str = tc["function"].get("arguments", "{}")
            arg_obj = json.loads(arg_str) if isinstance(arg_str, str) else arg_str
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
```

**工具响应转换**（第 337-355 行）：

```python
elif msg_role == "tool":
    tool_call_id = msg.get("tool_call_id", "")
    # 从 call_<name>_<hash> 提取 name
    fn_name = ""
    if tool_call_id.startswith("call_"):
        fn_name = "_".join(tool_call_id.split("_")[1:-1])
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
```

#### 4. 生成配置转换

OpenAI 的生成参数转换为 Gemini 的 `generationConfig`：

| OpenAI 参数 | Gemini 参数 | 说明 |
|-------------|-------------|------|
| `temperature` | `temperature` | 温度参数 |
| `top_p` | `topP` | Top-P 采样 |
| `max_tokens` | `maxOutputTokens` | 最大输出 token 数 |
| `stop` | `stopSequences` | 停止序列 |

**代码实现**（第 366-392 行）：

```python
generation_config = {}
if "temperature" in data:
    generation_config["temperature"] = data["temperature"]
if "top_p" in data:
    generation_config["topP"] = data["top_p"]
if "max_tokens" in data:
    generation_config["maxOutputTokens"] = data["max_tokens"]
if "stop" in data:
    generation_config["stopSequences"] = data["stop"] if isinstance(data["stop"], list) else [data["stop"]]

# Gemini 2.x 要求 generationConfig 字段始终存在
result_data["generationConfig"] = generation_config
```

#### 5. 工具调用转换

OpenAI 的 `tools` 数组转换为 Gemini 的 `functionDeclarations`：

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
    }]
}
```

**Gemini 格式**：
```json
{
    "tools": [{
        "functionDeclarations": [{
            "name": "get_weather",
            "description": "Get weather info",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }
        }]
    }]
}
```

**代码实现**（第 398-413 行）：

```python
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
        result_data["tools"] = [{
            "functionDeclarations": function_declarations
        }]
```

**JSON Schema 清理**（第 266-285 行）：

Gemini 对 JSON Schema 的支持有限，需要移除不支持的关键字：

```python
def _sanitize_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """递归移除Gemini不支持的JSON Schema关键字"""
    allowed_keys = {"type", "description", "properties", "required", "enum", "items"}
    sanitized = {k: v for k, v in schema.items() if k in allowed_keys}

    # 递归处理子属性
    if "properties" in sanitized:
        sanitized["properties"] = {
            prop_name: _sanitize_schema(prop_schema)
            for prop_name, prop_schema in sanitized["properties"].items()
        }
    if "items" in sanitized:
        sanitized["items"] = _sanitize_schema(sanitized["items"])

    return sanitized
```

#### 6. 结构化输出转换

OpenAI 的 `response_format` 转换为 Gemini 的结构化输出配置：

```python
if "response_format" in data and data["response_format"].get("type") == "json_schema":
    generation_config["response_mime_type"] = "application/json"
    if "json_schema" in data["response_format"]:
        generation_config["response_schema"] = data["response_format"]["json_schema"].get("schema", {})
```

#### 7. 思考模式转换

OpenAI 的 `max_completion_tokens` + `reasoning_effort` 转换为 Gemini 的 `thinkingConfig`：

**代码实现**（第 422-470 行）：

```python
if "max_completion_tokens" in data:
    # 确定reasoning_effort：如果没传则默认为medium
    reasoning_effort = data.get("reasoning_effort", "medium")

    # 根据环境变量映射reasoning_effort到具体的token数值
    env_key = None
    if reasoning_effort == "low":
        env_key = "OPENAI_LOW_TO_GEMINI_TOKENS"
    elif reasoning_effort == "medium":
        env_key = "OPENAI_MEDIUM_TO_GEMINI_TOKENS"
    elif reasoning_effort == "high":
        env_key = "OPENAI_HIGH_TO_GEMINI_TOKENS"

    env_value = os.environ.get(env_key)
    thinking_budget = int(env_value)

    if thinking_budget:
        generation_config["thinkingConfig"] = {
            "thinkingBudget": thinking_budget
        }
```

---

## 响应转换流程

### 入口函数

响应转换的入口位于 `src/formats/openai_converter.py` 的 `_convert_from_gemini_response` 方法（第 551-629 行）。

### 转换步骤

#### 1. 基础结构构建

```python
result_data = {
    "id": f"chatcmpl-{random_id}",  # 生成 OpenAI 风格的 ID
    "object": "chat.completion",
    "created": int(time.time()),
    "model": self.original_model,  # 使用原始请求中的模型名
    "usage": {},
    "choices": []
}
```

#### 2. 内容和工具调用提取

从 Gemini 的 `candidates[0].content.parts` 中提取文本和工具调用：

```python
if "candidates" in data and data["candidates"]:
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
                tool_calls.append({
                    "id": f"call_{fc_name}_{random_hash}",
                    "type": "function",
                    "function": {
                        "name": fc.get("name", ""),
                        "arguments": json.dumps(fc.get("args", {}), ensure_ascii=False)
                    }
                })
```

#### 3. 消息结构构建

根据是否有工具调用决定响应结构：

```python
if tool_calls:
    message_dict = {
        "role": "assistant",
        "content": None,  # 有工具调用时 content 为 null
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
```

#### 4. 使用量信息转换

```python
if "usageMetadata" in data:
    usage = data["usageMetadata"]
    result_data["usage"] = {
        "prompt_tokens": usage.get("promptTokenCount", 0),
        "completion_tokens": usage.get("candidatesTokenCount", 0),
        "total_tokens": usage.get("totalTokenCount", 0)
    }
```

#### 5. 结束原因映射

```python
reason_mappings = {
    "gemini": {
        "openai": {
            "STOP": "stop",
            "MAX_TOKENS": "length",
            "SAFETY": "content_filter",
            "RECITATION": "content_filter"
        }
    }
}
```

---

## 流式响应转换

### 流式请求处理

流式请求通过 `unified_api.py` 中的 `handle_streaming_response` 函数处理（第 495-760 行）。

### 流式 chunk 转换

流式 chunk 转换位于 `src/formats/openai_converter.py` 的 `_convert_from_gemini_streaming_chunk` 方法（第 631-784 行）。

#### 1. 状态维护

```python
# 生成一致的随机ID（在同一次对话中保持一致）
if not hasattr(self, '_stream_id'):
    self._stream_id = ''.join(random.choices(string.ascii_letters + string.digits, k=29))
```

#### 2. 普通 chunk 处理

```python
if "candidates" in data and data["candidates"]:
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
                tool_calls.append({
                    "id": f"call_{fc_name}_{counter:04d}",
                    "type": "function",
                    "function": {
                        "name": fc_name,
                        "arguments": json.dumps(fc_args, ensure_ascii=False)
                    }
                })

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
```

#### 3. 结束 chunk 处理

当 Gemini 响应中包含 `finishReason` 时，表示流式响应结束：

```python
if data["candidates"][0].get("finishReason"):
    # 确定finish_reason
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
        result_data["usage"] = {...}
```

### 流式响应格式对比

**Gemini 流式响应格式**：
```
data: {"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"},"index":0}]}
data: {"candidates":[{"content":{"parts":[{"text":" world"}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{...}}
```

**转换后的 OpenAI 流式响应格式**：
```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":"stop"}],"usage":{...}}
data: [DONE]
```

---

## 字段映射详解

### 请求字段映射表

| OpenAI 字段 | Gemini 字段 | 转换说明 |
|-------------|-------------|----------|
| `model` | `model` | 直接透传 |
| `messages[role="system"]` | `system_instruction.parts[].text` | 系统消息提取 |
| `messages[role="user"]` | `contents[role="user"].parts` | 用户消息 |
| `messages[role="assistant"]` | `contents[role="model"].parts` | 助手消息 |
| `messages[role="tool"]` | `contents[role="tool"].parts[].functionResponse` | 工具响应 |
| `temperature` | `generationConfig.temperature` | 温度 |
| `top_p` | `generationConfig.topP` | Top-P |
| `max_tokens` | `generationConfig.maxOutputTokens` | 最大输出 |
| `stop` | `generationConfig.stopSequences` | 停止序列 |
| `tools[].function` | `tools[].functionDeclarations[]` | 工具定义 |
| `tool_calls[].function.arguments` | `parts[].functionCall.args` | JSON字符串 → 对象 |
| `response_format.json_schema` | `generationConfig.response_schema` | 结构化输出 |
| `max_completion_tokens` | `generationConfig.thinkingConfig.thinkingBudget` | 思考预算 |

### 响应字段映射表

| Gemini 字段 | OpenAI 字段 | 转换说明 |
|-------------|-------------|----------|
| - | `id` | 生成 `chatcmpl-{random}` |
| - | `object` | 固定为 `chat.completion` |
| - | `created` | 当前时间戳 |
| `model` | `model` | 使用原始请求的模型名 |
| `candidates[0].content.parts[].text` | `choices[0].message.content` | 文本内容 |
| `candidates[0].content.parts[].functionCall` | `choices[0].message.tool_calls[]` | 工具调用 |
| `candidates[0].finishReason` | `choices[0].finish_reason` | 结束原因映射 |
| `usageMetadata.promptTokenCount` | `usage.prompt_tokens` | 输入 token |
| `usageMetadata.candidatesTokenCount` | `usage.completion_tokens` | 输出 token |
| `usageMetadata.totalTokenCount` | `usage.total_tokens` | 总 token |

### 结束原因映射表

| Gemini finishReason | OpenAI finish_reason |
|---------------------|----------------------|
| `STOP` | `stop` |
| `MAX_TOKENS` | `length` |
| `SAFETY` | `content_filter` |
| `RECITATION` | `content_filter` |
| 有工具调用时 | `tool_calls` |

---

## 特殊处理逻辑

### 1. 多模态内容处理

OpenAI 的图像内容格式转换为 Gemini 的 `inlineData` 格式：

```python
# _convert_content_to_gemini 方法 (第 1035-1059 行)
def _convert_content_to_gemini(self, content: Any) -> List[Dict[str, Any]]:
    if isinstance(content, list):
        gemini_parts = []
        for item in content:
            if item.get("type") == "text":
                gemini_parts.append({"text": item.get("text", "")})
            elif item.get("type") == "image_url":
                image_url = item.get("image_url", {}).get("url", "")
                if image_url.startswith("data:"):
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
```

### 2. 工具调用 ID 生成

OpenAI 需要唯一的 tool_call_id，Gemini 不提供此字段，需要生成：

```python
# 生成规则: call_{function_name}_{random_hash}
random_hash = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
tool_call_id = f"call_{fc_name}_{random_hash}"
```

### 3. 函数参数格式转换

OpenAI 的 `arguments` 是 JSON 字符串，Gemini 的 `args` 是对象：

```python
# OpenAI → Gemini：JSON字符串 → 对象
arg_obj = json.loads(arg_str) if isinstance(arg_str, str) else arg_str

# Gemini → OpenAI：对象 → JSON字符串
"arguments": json.dumps(fc.get("args", {}), ensure_ascii=False)
```

### 4. 空内容处理

Gemini 不允许空的 parts 数组，需要添加空文本：

```python
gemini_contents.append({
    "role": "model",
    "parts": parts if parts else [{"text": ""}]
})
```

### 5. max_tokens 优先级处理

```python
# 优先级：
# 1. 客户端传入的 max_tokens（最高优先级）
# 2. 环境变量 ANTHROPIC_MAX_TOKENS
# 3. 不设置（让 Gemini 使用默认值）

if "max_tokens" in data:
    generation_config["maxOutputTokens"] = data["max_tokens"]
else:
    env_max_tokens = os.environ.get("ANTHROPIC_MAX_TOKENS")
    if env_max_tokens:
        generation_config["maxOutputTokens"] = int(env_max_tokens)
```

---

## 错误处理

### 转换错误

```python
# 在 convert_request 中
try:
    return self._convert_to_gemini_request(data)
except Exception as e:
    self.logger.error(f"Failed to convert OpenAI request to gemini: {e}")
    return ConversionResult(success=False, error=str(e))
```

### 流式转换错误

```python
# 在 handle_streaming_response 中
try:
    response_conversion = convert_streaming_chunk(channel.provider, source_format, chunk_data, original_model)
except Exception as e:
    logger.error(f"Error in convert_streaming_chunk: {e}")
    # 发送原始数据作为后备
    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
    continue
```

### JSON 解析错误

```python
except json.JSONDecodeError as e:
    logger.error(f"JSON decode error in streaming response: {e}")
    # 特殊处理 [DONE] 标记
    if "[DONE]" in data_content:
        yield end_marker
        break
    # 尝试透传
    yield f"data: {data_content}\n\n"
```

---

## 配置项说明

### 环境变量

| 环境变量 | 说明 | 示例值 |
|----------|------|--------|
| `ANTHROPIC_MAX_TOKENS` | 默认最大 token 数（用作 maxOutputTokens 备选） | `4096` |
| `OPENAI_LOW_TO_GEMINI_TOKENS` | low reasoning_effort 对应的思考预算 | `1024` |
| `OPENAI_MEDIUM_TO_GEMINI_TOKENS` | medium reasoning_effort 对应的思考预算 | `8192` |
| `OPENAI_HIGH_TO_GEMINI_TOKENS` | high reasoning_effort 对应的思考预算 | `32768` |

### 渠道配置

渠道配置中可以设置模型映射，在发送到 Gemini 前自动映射模型名：

```python
# unified_api.py 第 393-409 行
if channel.models_mapping and isinstance(request_data, dict):
    original_model = request_data.get("model")
    if original_model:
        mapped_model = channel.models_mapping.get(original_model)
        if mapped_model:
            logger.info(f"Applying model mapping: {original_model} -> {mapped_model}")
            conversion_result.data = {**conversion_result.data, "model": mapped_model}
```

---

## 示例

### 完整请求转换示例

**OpenAI 请求**：
```json
{
    "model": "gpt-4",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
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
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }],
    "temperature": 0.7,
    "max_tokens": 1000,
    "stream": false
}
```

**转换后的 Gemini 请求**：
```json
{
    "model": "gpt-4",
    "system_instruction": {
        "parts": [{"text": "You are a helpful assistant."}]
    },
    "contents": [
        {
            "role": "user",
            "parts": [{"text": "What's the weather in Beijing?"}]
        }
    ],
    "tools": [{
        "functionDeclarations": [{
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }]
    }],
    "generationConfig": {
        "temperature": 0.7,
        "maxOutputTokens": 1000
    }
}
```

### 完整响应转换示例

**Gemini 响应**：
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
        "finishReason": "STOP",
        "index": 0
    }],
    "usageMetadata": {
        "promptTokenCount": 50,
        "candidatesTokenCount": 20,
        "totalTokenCount": 70
    }
}
```

**转换后的 OpenAI 响应**：
```json
{
    "id": "chatcmpl-abc123xyz",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "gpt-4",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_get_weather_a1b2c3d4",
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

---

## 总结

OpenAI 到 Gemini 的格式转换涉及以下核心点：

1. **消息结构差异**：OpenAI 使用 `messages` 数组，Gemini 使用 `contents` 数组和单独的 `system_instruction`
2. **角色映射**：`assistant` → `model`，`system` → `system_instruction`
3. **工具调用格式**：OpenAI 的 `tools/tool_calls` 转换为 Gemini 的 `functionDeclarations/functionCall`
4. **参数格式**：函数参数在 OpenAI 中是 JSON 字符串，在 Gemini 中是对象
5. **生成配置**：参数名称从 snake_case 转换为 camelCase
6. **流式响应**：需要维护状态并正确生成 OpenAI 风格的 chunk 格式
7. **思考模式**：`reasoning_effort` 映射为 `thinkingBudget`
