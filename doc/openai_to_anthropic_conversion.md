# OpenAI格式转Anthropic格式完整逻辑分析文档

## 一、整体架构概览

本项目采用**多格式代理转换架构**，通过工厂模式创建格式转换器，支持OpenAI、Anthropic和Gemini三种API格式的互相转换。

### 转换流程总览

```
客户端请求(OpenAI格式)
        ↓
extract_openai_api_key → 提取Authorization Bearer token
        ↓
handle_unified_request(source_format="openai")
        ↓
channel_manager.get_channel_by_custom_key() → 获取目标Anthropic渠道
        ↓
forward_request_to_channel()
        ├→ convert_request("openai" → "anthropic")
        ├→ apply_model_mapping (可选)
        ├→ POST /v1/messages (发送到Anthropic)
        └→ handle_streaming_response() / handle_non_streaming_response()
        ↓
convert_response(anthropic_response → openai_format)
        ↓
返回OpenAI格式响应给客户端
```

### 核心文件

| 文件 | 职责 |
|-----|-----|
| `src/api/unified_api.py` | API入口，请求路由与转发 |
| `src/formats/openai_converter.py` | OpenAI格式转换器 |
| `src/formats/anthropic_converter.py` | Anthropic格式转换器 |
| `src/formats/converter_factory.py` | 转换器工厂 |
| `src/channels/channel_manager.py` | 渠道管理与模型映射 |

---

## 二、请求转换流程详解

### 2.1 入口点

**文件**: `src/api/unified_api.py`

```python
@router.post("/v1/chat/completions")
async def unified_openai_format_endpoint(
    request: Request,
    api_key: str = Depends(extract_openai_api_key)
):
    """OpenAI格式统一端点（使用标准OpenAI认证）"""
    return await handle_unified_request(request, api_key, source_format="openai")
```

**认证方式**: `Authorization: Bearer <custom_key>`

### 2.2 请求转换核心函数

**文件**: `src/formats/openai_converter.py`

```python
def _convert_to_anthropic_request(self, data: Dict[str, Any]) -> ConversionResult:
```

**转换步骤**:

1. **模型处理**: 直接透传模型名（映射由渠道配置处理）
2. **消息转换**: 分离系统消息，转换消息格式
3. **参数映射**: 转换各类参数
4. **工具转换**: 转换tools定义
5. **Thinking处理**: 处理reasoning模式

---

## 三、消息格式转换

### 3.1 消息角色转换矩阵

| OpenAI角色 | 转换方式 | Anthropic结果 |
|-----------|--------|-------------|
| `system` | 提取为顶层`system`字段 | `{"system": "..."}` |
| `user` | 标准转换 | `{"role": "user", "content": "..."}` |
| `assistant` (无tool_calls) | 标准转换 | `{"role": "assistant", "content": "..."}` |
| `assistant` (有tool_calls) | 转为tool_use blocks | `{"role": "assistant", "content": [{"type": "tool_use", ...}]}` |
| `tool` | 转为user消息中的tool_result | `{"role": "user", "content": [{"type": "tool_result", ...}]}` |

### 3.2 System消息处理

OpenAI将system消息放在messages数组中，Anthropic需要分离为独立的system字段：

```python
def _extract_system_message(self, messages: List[Dict]) -> Tuple[str, List[Dict]]:
    """从消息列表中提取系统消息"""
    system_content = ""
    filtered_messages = []

    for msg in messages:
        if msg.get("role") == "system":
            system_content += msg.get("content", "") + "\n"
        else:
            filtered_messages.append(msg)

    return system_content.strip(), filtered_messages
```

### 3.3 工具调用消息转换

**OpenAI格式**（assistant带tool_calls）:
```json
{
  "role": "assistant",
  "tool_calls": [{
    "id": "call_abc123",
    "type": "function",
    "function": {
      "name": "search",
      "arguments": "{\"q\": \"weather\"}"
    }
  }]
}
```

**转换为Anthropic格式**:
```json
{
  "role": "assistant",
  "content": [{
    "type": "tool_use",
    "id": "call_abc123",
    "name": "search",
    "input": {"q": "weather"}
  }]
}
```

### 3.4 工具结果消息转换

**OpenAI格式**（tool角色）:
```json
{
  "role": "tool",
  "tool_call_id": "call_abc123",
  "content": "{\"result\": \"sunny\"}"
}
```

**转换为Anthropic格式**:
```json
{
  "role": "user",
  "content": [{
    "type": "tool_result",
    "tool_use_id": "call_abc123",
    "content": "{\"result\": \"sunny\"}"
  }]
}
```

---

## 四、参数映射表

### 4.1 基础参数映射

| OpenAI参数 | Anthropic参数 | 转换逻辑 |
|-----------|-------------|--------|
| `model` | `model` | 透传（可配置映射） |
| `messages` | `messages` + `system` | 消息和系统提示分离 |
| `temperature` | `temperature` | 直接映射 |
| `top_p` | `top_p` | 直接映射 |
| `stop` | `stop_sequences` | 列表转换 |
| `max_tokens` | `max_tokens` | **必需！** |
| `stream` | `stream` | 直接映射 |
| `tools` | `tools` | 函数定义转换 |

### 4.2 不支持的参数

以下OpenAI参数在Anthropic中不支持，会被忽略：

- `top_k`
- `presence_penalty`
- `frequency_penalty`
- `logprobs`
- `response_format`

### 4.3 max_tokens必需参数处理

Anthropic API要求必须提供max_tokens，按优先级处理：

```python
# 优先级：
# 1. 请求中传入的max_tokens（最高优先级）
# 2. 环境变量ANTHROPIC_MAX_TOKENS
# 3. 都没有则报错

if "max_tokens" in data:
    result_data["max_tokens"] = data["max_tokens"]
else:
    env_max_tokens = os.environ.get("ANTHROPIC_MAX_TOKENS")
    if env_max_tokens:
        result_data["max_tokens"] = int(env_max_tokens)
    else:
        raise ValueError("max_tokens is required for Anthropic API")
```

---

## 五、工具/函数调用转换

### 5.1 工具定义转换

**OpenAI工具格式**:
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

**转换为Anthropic格式**:
```json
{
  "tools": [{
    "name": "get_weather",
    "description": "Get weather info",
    "input_schema": {
      "type": "object",
      "properties": {
        "location": {"type": "string"}
      }
    }
  }]
}
```

### 5.2 关键差异

| 特性 | OpenAI | Anthropic |
|-----|--------|----------|
| 工具包装 | `type: "function"` 嵌套 | 扁平结构 |
| 参数schema | `parameters` | `input_schema` |
| arguments格式 | JSON字符串 | JSON对象 |

---

## 六、Thinking/Reasoning模式转换

### 6.1 转换流程

OpenAI o1/o3的推理模式转换为Anthropic thinking模式：

```
OpenAI格式:
{
  "max_completion_tokens": 8000,
  "reasoning_effort": "medium"  // low|medium|high
}
        ↓
根据环境变量映射：
- OPENAI_LOW_TO_ANTHROPIC_TOKENS      (e.g. 2000)
- OPENAI_MEDIUM_TO_ANTHROPIC_TOKENS   (e.g. 5000)
- OPENAI_HIGH_TO_ANTHROPIC_TOKENS     (e.g. 10000)
        ↓
Anthropic格式:
{
  "thinking": {
    "type": "enabled",
    "budget_tokens": 5000
  }
}
```

### 6.2 实现代码

```python
if "max_completion_tokens" in data:
    reasoning_effort = data.get("reasoning_effort", "medium")

    # 根据reasoning_effort映射token预算
    if reasoning_effort == "low":
        env_key = "OPENAI_LOW_TO_ANTHROPIC_TOKENS"
    elif reasoning_effort == "medium":
        env_key = "OPENAI_MEDIUM_TO_ANTHROPIC_TOKENS"
    elif reasoning_effort == "high":
        env_key = "OPENAI_HIGH_TO_ANTHROPIC_TOKENS"

    thinking_budget = int(os.environ.get(env_key))
    result_data["thinking"] = {
        "type": "enabled",
        "budget_tokens": thinking_budget
    }
```

---

## 七、多模态内容处理

### 7.1 图片内容转换

**OpenAI格式**:
```json
{
  "role": "user",
  "content": [{
    "type": "image_url",
    "image_url": {"url": "data:image/png;base64,iVBORw0KG..."}
  }]
}
```

**Anthropic格式**:
```json
{
  "role": "user",
  "content": [{
    "type": "image",
    "source": {
      "type": "base64",
      "media_type": "image/png",
      "data": "iVBORw0KG..."
    }
  }]
}
```

### 7.2 转换逻辑

按Anthropic最佳实践，**图片优先于文本**排序：

```python
def _convert_content_to_anthropic(self, content: Any) -> Any:
    # 1. 首先分类所有内容项
    text_items = []
    image_items = []
    other_items = []

    # 2. 首先处理图片（Anthropic最佳实践）
    for item in image_items:
        if image_url.startswith("data:"):
            media_type, data_part = image_url.split(";base64,", 1)
            anthropic_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data_part
                }
            })

    # 3. 然后处理文本内容
    for item in text_items:
        anthropic_content.append({
            "type": "text",
            "text": text_content
        })
```

---

## 八、响应转换流程

### 8.1 非流式响应转换

**Anthropic响应格式**:
```json
{
  "id": "msg_12345",
  "type": "message",
  "role": "assistant",
  "content": [
    {"type": "text", "text": "..."},
    {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
  ],
  "stop_reason": "tool_use",
  "usage": {
    "input_tokens": 100,
    "output_tokens": 50
  }
}
```

**转换为OpenAI响应格式**:
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "<original_model>",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "...",
      "tool_calls": [{
        "id": "...",
        "type": "function",
        "function": {
          "name": "...",
          "arguments": "{...}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }],
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_tokens": 150
  }
}
```

### 8.2 结束原因映射

| Anthropic stop_reason | OpenAI finish_reason |
|--------------------|--------------------|
| `end_turn` | `stop` |
| `max_tokens` | `length` |
| `stop_sequence` | `stop` |
| `tool_use` | `tool_calls` |

### 8.3 Thinking内容处理

Anthropic响应中的thinking块使用XML标签包装返回：

```python
if thinking_content.strip():
    content = f"<thinking>\n{thinking_content.strip()}\n</thinking>\n\n{content}"
```

---

## 九、流式响应转换

### 9.1 SSE流式处理架构

Anthropic使用**Server-Sent Events (SSE)** 格式，OpenAI使用**newline-delimited JSON**。

```
Anthropic SSE流
        ↓
parse_sse_events()
        ↓
_convert_from_anthropic_streaming_chunk()
        ↓
生成OpenAI格式chunks
        ↓
yield "data: {...}\n\n"
```

### 9.2 Anthropic SSE事件类型

| SSE事件类型 | 用途 | OpenAI转换 |
|-----------|------|----------|
| `message_start` | 流开始，包含初始消息结构 | 忽略 |
| `content_block_start` | 内容块开始（文本或工具调用） | 用于跟踪状态 |
| `content_block_delta` | 内容增量 | 转为delta对象 |
| `content_block_stop` | 内容块结束 | 标记完成 |
| `message_delta` | 消息级别更新 | 转为finish_reason |
| `message_stop` | 流完全结束 | 生成[DONE]标记 |

### 9.3 流式工具调用状态跟踪

```python
# 初始化工具状态管理
if not hasattr(self, '_anthropic_tool_state'):
    self._anthropic_tool_state = {}

# 事件: content_block_start (工具调用开始)
if event_type == "content_block_start":
    content_block = event_data.get("content_block", {})
    if content_block.get("type") == "tool_use":
        tool_index = event_data.get("index", 0)
        self._anthropic_tool_state[tool_index] = {
            "id": content_block.get("id"),
            "name": content_block.get("name"),
            "arguments": ""  # 累积参数
        }

# 事件: content_block_delta (参数增量)
elif event_type == "content_block_delta":
    delta = event_data.get("delta", {})
    if delta.get("type") == "input_json_delta":
        partial_json = delta.get("partial_json", "")
        self._anthropic_tool_state[index]["arguments"] += partial_json
```

### 9.4 流式响应示例

**Anthropic SSE流**:
```
event: message_start
data: {"type":"message_start","message":{...}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"这是"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"响应"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}

event: message_stop
data: {"type":"message_stop"}
```

**转换为OpenAI chunks**:
```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1704067200,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"这是"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1704067200,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"响应"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1704067200,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

---

## 十、模型映射机制

### 10.1 映射配置

在渠道配置中设置`models_mapping`：

```python
{
  "gpt-4": "claude-3-opus-20240229",
  "gpt-4-turbo": "claude-3-5-sonnet-20241022",
  "gpt-3.5-turbo": "claude-3-haiku-20240307"
}
```

### 10.2 映射应用

```python
# 在构建URL与发送前应用模型映射（仅影响下游请求）
mapped_model = None
if channel.models_mapping and isinstance(request_data, dict):
    original_model = request_data.get("model")
    if original_model:
        mapped_model = channel.models_mapping.get(original_model)
        if mapped_model:
            logger.info(f"Applying model mapping: {original_model} -> {mapped_model}")
            conversion_result.data["model"] = mapped_model
```

**关键特性**:
- 映射仅影响发送到下游的请求
- 响应中保留原始模型名
- 客户端收到的响应模型名保持原始值

---

## 十一、完整请求示例

### 11.1 简单文本对话

**客户端请求** (OpenAI格式):
```json
{
  "model": "gpt-4",
  "messages": [
    {"role": "system", "content": "你是一个助手"},
    {"role": "user", "content": "什么是Python?"}
  ],
  "temperature": 0.7,
  "max_tokens": 1000
}
```

**转换后** (Anthropic格式):
```json
{
  "model": "gpt-4",
  "system": "你是一个助手",
  "messages": [
    {"role": "user", "content": "什么是Python?"}
  ],
  "temperature": 0.7,
  "max_tokens": 1000
}
```

### 11.2 函数调用

**客户端请求** (OpenAI格式):
```json
{
  "model": "gpt-4",
  "messages": [
    {"role": "user", "content": "查询纽约天气"}
  ],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "获取天气信息",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {"type": "string"}
        },
        "required": ["location"]
      }
    }
  }],
  "max_tokens": 1000
}
```

**转换后** (Anthropic格式):
```json
{
  "model": "gpt-4",
  "messages": [
    {"role": "user", "content": "查询纽约天气"}
  ],
  "tools": [{
    "name": "get_weather",
    "description": "获取天气信息",
    "input_schema": {
      "type": "object",
      "properties": {
        "location": {"type": "string"}
      },
      "required": ["location"]
    }
  }],
  "max_tokens": 1000
}
```

### 11.3 Thinking模式

**客户端请求** (OpenAI格式):
```json
{
  "model": "o1-mini",
  "messages": [
    {"role": "user", "content": "解决数学问题: 2x + 5 = 13"}
  ],
  "max_completion_tokens": 8000,
  "reasoning_effort": "high"
}
```

**转换后** (Anthropic格式):
```json
{
  "model": "o1-mini",
  "messages": [
    {"role": "user", "content": "解决数学问题: 2x + 5 = 13"}
  ],
  "max_tokens": 8000,
  "thinking": {
    "type": "enabled",
    "budget_tokens": 10000
  }
}
```

---

## 十二、环境变量配置

### 必需配置

```bash
# Max tokens配置
ANTHROPIC_MAX_TOKENS=4096

# Thinking预算映射 (OpenAI → Anthropic)
OPENAI_LOW_TO_ANTHROPIC_TOKENS=2000
OPENAI_MEDIUM_TO_ANTHROPIC_TOKENS=5000
OPENAI_HIGH_TO_ANTHROPIC_TOKENS=10000
```

---

## 十三、错误处理

### 13.1 必需参数验证

| 参数 | 验证规则 |
|-----|--------|
| `model` | 必需 |
| `messages` | 必需，且不能为空 |
| `max_tokens` | 必需（Anthropic要求） |

### 13.2 流式处理容错

JSON片段清理，避免不完整的Unicode字符：

```python
def _clean_json_fragment(self, fragment: str) -> str:
    """清理JSON片段，避免不完整的Unicode字符"""
    cleaned = fragment

    # 处理可能被截断的转义序列
    if cleaned.endswith('\\') and not cleaned.endswith('\\\\'):
        cleaned = cleaned[:-1]
    elif cleaned.endswith('\\u') or cleaned.endswith('\\u0'):
        idx = cleaned.rfind('\\u')
        cleaned = cleaned[:idx]

    return cleaned
```

---

## 十四、完整流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    客户端发送OpenAI格式请求                          │
│  POST /v1/chat/completions                                         │
│  Authorization: Bearer <custom_key>                                 │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│          unified_openai_format_endpoint                             │
│  - extract_openai_api_key() 获取custom_key                          │
│  - call handle_unified_request(source_format="openai")              │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│          handle_unified_request                                      │
│  - channel_manager.get_channel_by_custom_key(api_key)               │
│  - 获取目标渠道（Anthropic Provider）                                │
│  - 检查stream标志，选择流式/非流式处理                              │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
    (stream)                     (非stream)
        │                             │
        ▼                             ▼
┌─────────────────────────────┐  ┌──────────────────────────┐
│  convert_request()          │  │  convert_request()       │
│  OpenAI -> Anthropic        │  │  OpenAI -> Anthropic     │
│  - 分离system消息           │  │  - 转换消息              │
│  - 转换消息格式             │  │  - 转换参数              │
│  - 转换参数                 │  │  - 处理thinking          │
│  - 处理thinking             │  │                          │
│  - 转换tools定义            │  │                          │
│  - 应用模型映射             │  │                          │
└────────────┬────────────────┘  └────────┬─────────────────┘
             │                            │
             ▼                            ▼
┌─────────────────────────────┐  ┌──────────────────────────┐
│  POST /v1/messages (流式)   │  │  POST /v1/messages       │
│  async stream_generator()   │  │  httpx.post()            │
│  - 接收SSE流                │  │  - 获取完整响应          │
│  - parse_sse_events()       │  │                          │
└────────────┬────────────────┘  └────────┬─────────────────┘
             │                            │
             ▼                            ▼
┌─────────────────────────────┐  ┌──────────────────────────┐
│  handle_streaming_response()│  │  handle_non_streaming()  │
│  - 逐个处理SSE事件          │  │  - 调用convert_response()│
│  - 转换为OpenAI chunks      │  │    Anthropic -> OpenAI   │
└────────────┬────────────────┘  └────────┬─────────────────┘
             │                            │
             ▼                            ▼
┌─────────────────────────────────────────────────────────────┐
│      StreamingResponse / JSONResponse                        │
│      返回转换后的OpenAI格式响应给客户端                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 十五、总结

### 核心特性

1. **完整的格式互转**: OpenAI ↔ Anthropic格式全面支持
2. **流式和非流式**: 两种模式都完整支持
3. **工具/函数调用**: 完整的tools转换
4. **Thinking/Reasoning**: 支持推理模式映射
5. **模型映射**: 支持渠道级模型名映射
6. **多模态**: 支持文本、图片等内容

### 转换优先级

1. **入口**: 客户端指定的格式（OpenAI/Anthropic/Gemini）
2. **识别**: 从API key获取目标渠道（提供商）
3. **转换**: source_format → target_format（provider）
4. **映射**: 应用渠道配置的模型映射
5. **返回**: 转换回客户端期望的格式

### 必需配置清单

- `ANTHROPIC_MAX_TOKENS`: Anthropic的最大输出token数
- `OPENAI_*_TO_ANTHROPIC_TOKENS`: 思考预算映射
- 渠道的`models_mapping`: 模型名映射（可选）
