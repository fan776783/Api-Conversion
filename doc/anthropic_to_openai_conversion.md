# Anthropic格式转OpenAI格式完整逻辑分析文档

## 一、整体架构概览

本项目采用**多格式代理转换架构**，支持客户端使用Anthropic API格式发起请求，但实际后端渠道使用OpenAI服务。

### 转换流程总览

```
客户端请求(Anthropic格式)
        ↓
POST /v1/messages
x-api-key: <custom_key> 或 Authorization: Bearer <custom_key>
        ↓
extract_anthropic_api_key → 提取API key
        ↓
handle_unified_request(source_format="anthropic")
        ↓
channel_manager.get_channel_by_custom_key() → 获取目标渠道(provider="openai")
        ↓
forward_request_to_channel()
        ├→ convert_request("anthropic" → "openai")
        ├→ apply_model_mapping (可选)
        ├→ POST /v1/chat/completions (发送到OpenAI)
        └→ handle_streaming_response() / handle_non_streaming_response()
        ↓
convert_response(openai_response → anthropic_format)
        ↓
返回Anthropic格式响应给客户端
```

### 核心文件

| 文件 | 职责 |
|-----|-----|
| `src/api/unified_api.py` | API入口，请求路由与转发 |
| `src/formats/anthropic_converter.py` | Anthropic格式转换器（核心） |
| `src/formats/converter_factory.py` | 转换器工厂 |
| `src/formats/base_converter.py` | 转换器基类 |
| `src/channels/channel_manager.py` | 渠道管理与模型映射 |

---

## 二、请求入口与认证

### 2.1 入口点

**文件**: `src/api/unified_api.py`

```python
@router.post("/v1/messages")
async def unified_anthropic_format_endpoint(
    request: Request,
    api_key: str = Depends(extract_anthropic_api_key)
):
    """Anthropic格式统一端点（使用标准Anthropic认证）"""
    return await handle_unified_request(request, api_key, source_format="anthropic")
```

### 2.2 认证方式

Anthropic格式支持两种认证方式（优先级从高到低）：

1. **x-api-key Header**: `x-api-key: <custom_key>`
2. **Authorization Header**: `Authorization: Bearer <custom_key>`

```python
def extract_anthropic_api_key(
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
    authorization: Optional[str] = Header(None, alias="authorization")
) -> str:
    # 优先从x-api-key获取
    if x_api_key:
        return x_api_key

    # 否则从Authorization Bearer获取
    if authorization and authorization.startswith("Bearer "):
        return authorization[7:]

    raise HTTPException(status_code=401, detail="Missing authentication")
```

### 2.3 必须字段验证

```python
# handle_unified_request中的验证
if not request_data.get("model"):
    raise HTTPException(status_code=400, detail="Model name is required")

# Anthropic格式特有要求
if source_format == "anthropic" and not request_data.get("max_tokens"):
    raise HTTPException(status_code=400, detail="max_tokens is required for Anthropic format")
```

---

## 三、请求转换：Anthropic → OpenAI

### 3.1 调用路径

```
unified_api.forward_request_to_channel(source_format="anthropic", channel.provider="openai")
    ↓
converter_factory.convert_request("anthropic", "openai", data, headers)
    ↓
AnthropicConverter.convert_request(data, target_format="openai", headers)
    ↓
AnthropicConverter._convert_to_openai_request(data)
```

### 3.2 消息格式转换

#### 3.2.1 System消息处理

Anthropic使用顶层`system`字段，OpenAI使用messages数组中的system角色：

**Anthropic格式**:
```json
{
  "model": "claude-3-opus",
  "system": "你是一个有帮助的助手",
  "messages": [
    {"role": "user", "content": "你好"}
  ]
}
```

**转换为OpenAI格式**:
```json
{
  "model": "claude-3-opus",
  "messages": [
    {"role": "system", "content": "你是一个有帮助的助手"},
    {"role": "user", "content": "你好"}
  ]
}
```

#### 3.2.2 消息角色转换矩阵

| Anthropic角色 | Anthropic内容类型 | OpenAI角色 | 转换逻辑 |
|-------------|----------------|---------|--------|
| `user` | 普通文本 | `user` | 直接映射 |
| `user` | `tool_result` | `tool` | 拆分为tool消息 |
| `assistant` | 普通文本 | `assistant` | 直接映射 |
| `assistant` | `tool_use` | `assistant` + `tool_calls` | 转换工具调用格式 |

#### 3.2.3 工具调用消息转换

**Anthropic格式（assistant带tool_use）**:
```json
{
  "role": "assistant",
  "content": [{
    "type": "tool_use",
    "id": "toolu_123",
    "name": "get_weather",
    "input": {"location": "北京"}
  }]
}
```

**转换为OpenAI格式**:
```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [{
    "id": "toolu_123",
    "type": "function",
    "function": {
      "name": "get_weather",
      "arguments": "{\"location\": \"北京\"}"
    }
  }]
}
```

#### 3.2.4 工具结果消息转换

**Anthropic格式（user带tool_result）**:
```json
{
  "role": "user",
  "content": [{
    "type": "tool_result",
    "tool_use_id": "toolu_123",
    "content": "北京今天晴，25度"
  }]
}
```

**转换为OpenAI格式**:
```json
{
  "role": "tool",
  "tool_call_id": "toolu_123",
  "content": "北京今天晴，25度"
}
```

### 3.3 参数映射表

#### 3.3.1 基础参数映射

| Anthropic参数 | OpenAI参数 | 转换逻辑 |
|-------------|-----------|--------|
| `model` | `model` | 直接透传（可配置映射） |
| `system` | `messages[0]` | 作为system角色消息 |
| `messages` | `messages[1:]` | 消息格式转换 |
| `temperature` | `temperature` | 直接映射 |
| `top_p` | `top_p` | 直接映射 |
| `top_k` | - | **不支持，被忽略** |
| `stop_sequences` | `stop` | 直接映射 |
| `max_tokens` | `max_tokens` | 直接映射 |
| `stream` | `stream` | 直接映射 |
| `tools` | `tools` | 函数定义转换 |

#### 3.3.2 工具定义转换

**Anthropic工具格式**:
```json
{
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
  }]
}
```

**转换为OpenAI格式**:
```json
{
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
  "tool_choice": "auto"
}
```

### 3.4 Thinking/Reasoning模式转换

当Anthropic请求包含`thinking`参数时，需要转换为OpenAI的reasoning模式：

**Anthropic格式**:
```json
{
  "model": "claude-sonnet-4",
  "thinking": {
    "type": "enabled",
    "budget_tokens": 5000
  },
  "max_tokens": 4096,
  "messages": [...]
}
```

**转换为OpenAI格式**:
```json
{
  "model": "claude-sonnet-4",
  "reasoning_effort": "medium",
  "max_completion_tokens": 4096,
  "messages": [...]
}
```

#### reasoning_effort映射规则

通过环境变量配置阈值：

| 环境变量 | 用途 |
|---------|-----|
| `ANTHROPIC_TO_OPENAI_LOW_REASONING_THRESHOLD` | budget_tokens低阈值 |
| `ANTHROPIC_TO_OPENAI_HIGH_REASONING_THRESHOLD` | budget_tokens高阈值 |

映射逻辑：
```python
if budget_tokens <= low_threshold:
    reasoning_effort = "low"
elif budget_tokens <= high_threshold:
    reasoning_effort = "medium"
else:
    reasoning_effort = "high"
```

#### max_completion_tokens优先级

1. **最高优先级**: 请求中的`max_tokens`
2. **次优先级**: 环境变量`OPENAI_REASONING_MAX_TOKENS`
3. **都没有**: 抛出错误

### 3.5 多模态内容转换

#### 图片格式转换

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
  }, {
    "type": "text",
    "text": "这张图片里有什么？"
  }]
}
```

**转换为OpenAI格式**:
```json
{
  "role": "user",
  "content": [{
    "type": "image_url",
    "image_url": {
      "url": "data:image/png;base64,iVBORw0KG..."
    }
  }, {
    "type": "text",
    "text": "这张图片里有什么？"
  }]
}
```

### 3.6 消息验证与修正

转换完成后，会进行`validated_messages`安全修正：

```python
# 检查每个assistant消息的tool_calls是否有对应的tool响应
for idx, m in enumerate(messages):
    if m.get("role") == "assistant" and m.get("tool_calls"):
        call_ids = [tc.get("id") for tc in m["tool_calls"]]

        # 检查后续是否有匹配的tool响应
        unmatched = set(call_ids)
        for later in messages[idx + 1:]:
            if later.get("role") == "tool" and later.get("tool_call_id") in unmatched:
                unmatched.discard(later["tool_call_id"])

        if unmatched:
            # 移除无匹配的tool_calls
            m["tool_calls"] = [tc for tc in m["tool_calls"] if tc.get("id") not in unmatched]
            if not m["tool_calls"]:
                m.pop("tool_calls", None)
                if m.get("content") is None:
                    m["content"] = ""  # 避免空assistant消息
```

---

## 四、URL与Headers构建

当`channel.provider == "openai"`时：

```python
# 构建目标URL
url = f"{channel.base_url.rstrip('/')}/chat/completions"

# 构建请求Headers
target_headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {channel.api_key}"
}
```

---

## 五、响应转换：OpenAI → Anthropic

### 5.1 非流式响应转换

#### 5.1.1 调用路径

```
unified_api.handle_non_streaming_response(response, channel, request_data, source_format)
    ↓
converter = ConverterFactory().get_converter("anthropic")
    ↓
converter.set_original_model(request_data["model"])  # 保留原始模型名
    ↓
converter.convert_response(response_data, source_format="openai", target_format="anthropic")
    ↓
AnthropicConverter._convert_from_openai_response(data)
```

#### 5.1.2 响应格式转换

**OpenAI响应格式**:
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
      "content": "这是回复内容",
      "tool_calls": [{
        "id": "call_123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\":\"北京\"}"
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

**转换为Anthropic响应格式**:
```json
{
  "id": "chatcmpl-abc123",
  "type": "message",
  "role": "assistant",
  "model": "claude-3-opus",
  "content": [
    {"type": "text", "text": "这是回复内容"},
    {
      "type": "tool_use",
      "id": "call_123",
      "name": "get_weather",
      "input": {"location": "北京"}
    }
  ],
  "stop_reason": "tool_use",
  "usage": {
    "input_tokens": 100,
    "output_tokens": 50
  }
}
```

### 5.2 字段映射详解

#### 5.2.1 顶层字段映射

| OpenAI字段 | Anthropic字段 | 转换逻辑 |
|----------|-------------|--------|
| `id` | `id` | 直接透传 |
| - | `type` | 固定为`"message"` |
| `choices[0].message.role` | `role` | 固定为`"assistant"` |
| - | `model` | 使用`original_model`（客户端原始请求的模型名） |
| `choices[0].message.content` | `content[*].text` | 转换为text块 |
| `choices[0].message.tool_calls` | `content[*].tool_use` | 转换为tool_use块 |
| `choices[0].finish_reason` | `stop_reason` | 映射结束原因 |
| `usage` | `usage` | 字段名转换 |

#### 5.2.2 结束原因映射

| OpenAI finish_reason | Anthropic stop_reason |
|---------------------|----------------------|
| `stop` | `end_turn` |
| `length` | `max_tokens` |
| `content_filter` | `stop_sequence` |
| `tool_calls` | `tool_use` |
| 其他 | `end_turn` (默认) |

#### 5.2.3 Usage字段映射

| OpenAI字段 | Anthropic字段 |
|----------|-------------|
| `prompt_tokens` | `input_tokens` |
| `completion_tokens` | `output_tokens` |

### 5.3 Thinking内容提取

当OpenAI响应包含`<thinking>`标签时，会被提取为Anthropic的thinking块：

**OpenAI响应内容**:
```
<thinking>
让我分析一下这个问题...
首先需要考虑...
</thinking>

根据我的分析，答案是...
```

**转换为Anthropic格式**:
```json
{
  "content": [
    {
      "type": "thinking",
      "thinking": "让我分析一下这个问题...\n首先需要考虑..."
    },
    {
      "type": "text",
      "text": "根据我的分析，答案是..."
    }
  ]
}
```

---

## 六、流式响应转换：OpenAI SSE → Anthropic SSE

### 6.1 调用路径

```
unified_api.handle_streaming_response(response, channel, request_data, source_format)
    ↓
逐行解析OpenAI SSE流（data: {...}）
    ↓
convert_streaming_chunk("openai", "anthropic", chunk_data, original_model)
    ↓
AnthropicConverter._convert_from_openai_streaming_chunk(data)
    ↓
返回Anthropic SSE格式字符串
```

### 6.2 状态管理

流式转换使用`_streaming_state`维护跨chunk状态：

```python
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

### 6.3 事件转换流程

#### 6.3.1 OpenAI流式chunk示例

```
data: {"id":"chatcmpl-123","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","choices":[{"index":0,"delta":{"content":"你"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","choices":[{"index":0,"delta":{"content":"好"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{...}}

data: [DONE]
```

#### 6.3.2 转换为Anthropic SSE事件

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-opus","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"你"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"好"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":10,"output_tokens":5}}

event: message_stop
data: {"type":"message_stop"}
```

### 6.4 事件类型映射

| 场景 | 生成的Anthropic事件 |
|-----|-------------------|
| 首个有意义chunk | `message_start` |
| 首次文本内容 | `content_block_start` (type="text") |
| 文本增量 | `content_block_delta` (type="text_delta") |
| 首次工具调用 | `content_block_start` (type="tool_use") |
| 工具参数增量 | `content_block_delta` (type="input_json_delta") |
| 流结束 | `content_block_stop` + `message_delta` + `message_stop` |

### 6.5 流式工具调用处理

**OpenAI工具调用流式chunk**:
```json
{
  "choices": [{
    "delta": {
      "tool_calls": [{
        "index": 0,
        "id": "call_123",
        "function": {
          "name": "get_weather",
          "arguments": ""
        }
      }]
    }
  }]
}
```

**后续参数增量chunk**:
```json
{
  "choices": [{
    "delta": {
      "tool_calls": [{
        "index": 0,
        "function": {
          "arguments": "{\"loc"
        }
      }]
    }
  }]
}
```

**转换为Anthropic事件**:
```
event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"call_123","name":"get_weather","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"loc"}}
```

### 6.6 JSON片段清理

为避免不完整的Unicode字符或转义序列导致解析错误：

```python
def _clean_json_fragment(self, fragment: str) -> str:
    """清理JSON片段"""
    cleaned = fragment

    # 处理悬挂的反斜杠
    if cleaned.endswith('\\') and not cleaned.endswith('\\\\'):
        cleaned = cleaned[:-1]

    # 处理不完整的Unicode转义
    elif cleaned.endswith('\\u') or cleaned.endswith('\\u0'):
        idx = cleaned.rfind('\\u')
        cleaned = cleaned[:idx]

    return cleaned
```

---

## 七、模型映射机制

### 7.1 映射配置

在渠道配置中设置`models_mapping`：

```python
channel.models_mapping = {
    "claude-3-opus-20240229": "gpt-4-turbo",
    "claude-3-5-sonnet-20241022": "gpt-4o",
    "claude-3-haiku-20240307": "gpt-4o-mini"
}
```

### 7.2 映射应用时机

```python
# 在forward_request_to_channel中应用
if channel.models_mapping and isinstance(request_data, dict):
    original_model = request_data.get("model")  # 保留原始模型名
    if original_model:
        mapped_model = channel.models_mapping.get(original_model)
        if mapped_model:
            # 仅修改发送到下游的请求
            conversion_result.data["model"] = mapped_model
```

### 7.3 关键特性

| 特性 | 说明 |
|-----|-----|
| 下游请求 | 使用映射后的模型名 |
| 响应model字段 | **始终使用原始模型名** |
| 客户端视角 | 无感知，始终看到请求的模型名 |

---

## 八、完整请求示例

### 8.1 简单文本对话

**客户端请求（Anthropic格式）**:
```http
POST /v1/messages
x-api-key: sk-custom-key
Content-Type: application/json

{
  "model": "claude-3-opus-20240229",
  "system": "你是一个有帮助的助手",
  "messages": [
    {"role": "user", "content": "什么是Python?"}
  ],
  "max_tokens": 1000,
  "temperature": 0.7
}
```

**内部转换后（OpenAI格式）**:
```json
{
  "model": "gpt-4-turbo",
  "messages": [
    {"role": "system", "content": "你是一个有帮助的助手"},
    {"role": "user", "content": "什么是Python?"}
  ],
  "max_tokens": 1000,
  "temperature": 0.7
}
```

**返回客户端（Anthropic格式）**:
```json
{
  "id": "chatcmpl-xxx",
  "type": "message",
  "role": "assistant",
  "model": "claude-3-opus-20240229",
  "content": [
    {"type": "text", "text": "Python是一种高级编程语言..."}
  ],
  "stop_reason": "end_turn",
  "usage": {
    "input_tokens": 25,
    "output_tokens": 150
  }
}
```

### 8.2 工具调用示例

**客户端请求（Anthropic格式）**:
```json
{
  "model": "claude-3-5-sonnet-20241022",
  "messages": [
    {"role": "user", "content": "北京今天天气怎么样?"}
  ],
  "tools": [{
    "name": "get_weather",
    "description": "获取指定城市的天气",
    "input_schema": {
      "type": "object",
      "properties": {
        "city": {"type": "string", "description": "城市名称"}
      },
      "required": ["city"]
    }
  }],
  "max_tokens": 1000
}
```

**返回客户端（Anthropic格式）**:
```json
{
  "id": "msg_xxx",
  "type": "message",
  "role": "assistant",
  "model": "claude-3-5-sonnet-20241022",
  "content": [
    {
      "type": "tool_use",
      "id": "toolu_xxx",
      "name": "get_weather",
      "input": {"city": "北京"}
    }
  ],
  "stop_reason": "tool_use",
  "usage": {
    "input_tokens": 50,
    "output_tokens": 30
  }
}
```

### 8.3 Thinking模式示例

**客户端请求（Anthropic格式）**:
```json
{
  "model": "claude-sonnet-4-20250514",
  "messages": [
    {"role": "user", "content": "解方程: 2x + 5 = 13"}
  ],
  "thinking": {
    "type": "enabled",
    "budget_tokens": 8000
  },
  "max_tokens": 4096
}
```

**内部转换后（OpenAI格式）**:
```json
{
  "model": "o3-mini",
  "messages": [
    {"role": "user", "content": "解方程: 2x + 5 = 13"}
  ],
  "reasoning_effort": "high",
  "max_completion_tokens": 4096
}
```

---

## 九、环境变量配置

### 必需配置

```bash
# Thinking/Reasoning参数映射阈值
ANTHROPIC_TO_OPENAI_LOW_REASONING_THRESHOLD=2000   # budget_tokens <= 此值 -> low
ANTHROPIC_TO_OPENAI_HIGH_REASONING_THRESHOLD=8000  # budget_tokens <= 此值 -> medium, > 此值 -> high

# OpenAI Reasoning模式max_completion_tokens默认值
OPENAI_REASONING_MAX_TOKENS=8192
```

---

## 十、错误处理

### 10.1 请求验证错误

| 错误场景 | HTTP状态码 | 错误信息 |
|---------|----------|---------|
| 缺少认证 | 401 | Missing authentication |
| 无效API key | 401 | Invalid API key |
| 缺少model | 400 | Model name is required |
| 缺少max_tokens | 400 | max_tokens is required for Anthropic format |

### 10.2 转换错误

| 错误场景 | 处理方式 |
|---------|---------|
| 请求转换失败 | 抛出ConversionError |
| 响应转换失败 | 透传原始响应 |
| JSON解析错误 | 记录日志，尝试继续 |
| 缺少环境变量（Thinking模式） | 抛出ConversionError |

### 10.3 流式处理容错

```python
# 对于流式JSON解析错误，尝试继续处理
try:
    chunk_data = json.loads(data_content)
except json.JSONDecodeError:
    if "[DONE]" in data_content:
        yield end_marker
        break
    # 尝试透传
    yield f"data: {data_content}\n\n"
    continue
```

---

## 十一、完整流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    客户端发送Anthropic格式请求                        │
│  POST /v1/messages                                                  │
│  x-api-key: <custom_key>                                            │
│  {"model":"claude-3-opus","system":"...","messages":[...],...}      │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│          unified_anthropic_format_endpoint                          │
│  - extract_anthropic_api_key() 获取custom_key                       │
│  - call handle_unified_request(source_format="anthropic")           │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│          handle_unified_request                                      │
│  - channel_manager.get_channel_by_custom_key(api_key)               │
│  - 获取目标渠道（OpenAI Provider）                                   │
│  - 验证必须字段（model, max_tokens）                                 │
│  - 检查stream标志，选择流式/非流式处理                               │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
    (stream)                     (非stream)
        │                             │
        ▼                             ▼
┌─────────────────────────────┐  ┌──────────────────────────┐
│  convert_request()          │  │  convert_request()       │
│  Anthropic -> OpenAI        │  │  Anthropic -> OpenAI     │
│  - system字段转消息         │  │  - 转换消息格式          │
│  - 消息格式转换             │  │  - 转换参数              │
│  - tool_use/tool_result     │  │  - 处理thinking          │
│  - thinking参数转换         │  │  - 工具定义转换          │
│  - 工具定义转换             │  │  - 应用模型映射          │
│  - 应用模型映射             │  │                          │
└────────────┬────────────────┘  └────────┬─────────────────┘
             │                            │
             ▼                            ▼
┌─────────────────────────────┐  ┌──────────────────────────┐
│  POST /chat/completions     │  │  POST /chat/completions  │
│  (流式)                     │  │  httpx.post()            │
│  httpx.stream("POST",...)   │  │  - 获取完整响应          │
└────────────┬────────────────┘  └────────┬─────────────────┘
             │                            │
             ▼                            ▼
┌─────────────────────────────┐  ┌──────────────────────────┐
│  handle_streaming_response()│  │  handle_non_streaming()  │
│  - 逐行解析OpenAI SSE       │  │  - convert_response()    │
│  - convert_streaming_chunk()│  │    OpenAI -> Anthropic   │
│  - 状态跟踪与事件转换       │  │  - 保留原始model名       │
└────────────┬────────────────┘  └────────┬─────────────────┘
             │                            │
             ▼                            ▼
┌─────────────────────────────────────────────────────────────┐
│      StreamingResponse / JSONResponse                        │
│      返回转换后的Anthropic格式响应给客户端                    │
│      model字段始终为客户端请求的原始模型名                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 十二、总结

### 核心特性

| 特性 | 支持情况 |
|-----|---------|
| 非流式请求/响应 | ✅ 完整支持 |
| 流式请求/响应 | ✅ 完整支持 |
| 工具/函数调用 | ✅ 完整支持 |
| 工具结果返回 | ✅ 完整支持 |
| Thinking/Reasoning模式 | ✅ 完整支持 |
| 多模态（图片） | ✅ 完整支持 |
| 模型映射 | ✅ 完整支持 |
| 原始模型名保留 | ✅ 完整支持 |

### 关键转换点

1. **System消息**: 顶层`system`字段 → messages数组system角色
2. **工具定义**: `input_schema` → `parameters`，扁平 → 嵌套
3. **工具调用**: `tool_use` → `tool_calls`，JSON对象 → JSON字符串
4. **工具结果**: `tool_result`(user角色) → `tool`角色
5. **Thinking**: `thinking.budget_tokens` → `reasoning_effort` + `max_completion_tokens`
6. **结束原因**: `stop_reason` ↔ `finish_reason`映射
7. **Usage**: `input_tokens/output_tokens` ↔ `prompt_tokens/completion_tokens`
