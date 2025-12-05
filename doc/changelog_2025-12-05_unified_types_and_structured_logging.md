# 变更日志：统一类型层与结构化日志

**日期**: 2025-12-05
**版本**: 未标记
**状态**: 待审核

---

## 概述

本次变更引入两大核心改进：

1. **统一类型中间层 (Unified Types Layer)** - 新增 `src/formats/unified/` 模块，为 Anthropic、OpenAI、Gemini 三种格式提供统一的中间表示
2. **结构化错误日志 (Structured Error Logging)** - 在 `src/utils/logger.py` 中新增标准化错误日志函数，支持完整的请求/响应上下文记录

---

## 新增文件

### `src/formats/unified/` 目录

| 文件 | 用途 |
|------|------|
| `__init__.py` | 模块导出，统一对外接口 |
| `types.py` | 核心类型定义：`UnifiedContent`, `UnifiedMessage`, `UnifiedChatRequest`, `UnifiedChatResponse` |
| `stream_state.py` | 流式转换状态管理：`StreamState`, `StreamPhase`, `ToolCallState` |
| `exceptions.py` | 统一异常类型 |
| `adapters/` | 适配器基类（预留扩展） |

### 核心类型说明

```
UnifiedContentType (枚举)
├── TEXT          # 文本内容
├── THINKING      # 思考内容 (Anthropic thinking / OpenAI reasoning_content)
├── SIGNATURE     # 签名 (Anthropic)
├── TOOL_USE      # 工具调用请求
├── TOOL_RESULT   # 工具调用结果
├── IMAGE         # 图片内容
├── ANNOTATION    # 注解
└── WEB_SEARCH    # 网页搜索结果

UnifiedChatRequest
├── from_anthropic() -> 解析 Anthropic 请求
├── from_openai()    -> 解析 OpenAI 请求
├── to_anthropic()   -> 输出 Anthropic 格式
└── to_openai()      -> 输出 OpenAI 格式

StreamState
├── 统一管理流式响应状态
├── 替代分散的实例属性 (_streaming_state, _gemini_sent_start 等)
└── 提供 start_thinking_block(), start_text_block(), start_tool_call() 等方法
```

---

## 修改文件

### 1. `src/utils/logger.py`

**新增内容**:
- 错误类型常量: `ERROR_TYPE_NETWORK`, `ERROR_TYPE_AUTH`, `ERROR_TYPE_RATE_LIMIT`, `ERROR_TYPE_CONVERSION`, `ERROR_TYPE_UPSTREAM_API`
- `log_structured_error()` - 统一的结构化错误日志入口
- `log_request_entry()` - 请求入口日志（INFO 级别）
- 辅助函数: `_extract_request_id()`, `_summarize_headers()`, `_preview_body()`

**日志格式**:
```json
{
  "timestamp": "2025-12-05T10:30:00.000Z",
  "logger": "unified_api",
  "level": "ERROR",
  "error_type": "upstream_api",
  "request_id": "abc123",
  "request": { "method": "POST", "url": "...", "headers": {...}, "body": "..." },
  "response": { "status": 500, "headers": {...}, "body": "..." },
  "exception": { "type": "APIError", "message": "...", "traceback": "..." },
  "extra": { "channel_name": "...", "stage": "..." }
}
```

### 2. `src/api/unified_api.py`

**主要变更**:
- 导入新的结构化日志函数
- 所有错误处理点替换为 `log_structured_error()` 调用
- 新增 `/v1/messages/count_tokens` 端点（Anthropic Token 计数）
- 新增请求入口日志 `log_request_entry()`
- 新增辅助函数:
  - `handle_anthropic_count_tokens()` - 处理 Anthropic count_tokens 请求
  - `_forward_anthropic_count_tokens()` - 转发到 Anthropic API
  - `_estimate_tokens_with_tiktoken()` - 使用 tiktoken 估算
  - `_count_tokens_via_gemini()` - 通过 Gemini API 计算
  - `_convert_anthropic_messages_to_gemini_contents()` - 消息格式转换
  - `_extract_text_from_anthropic_request()` - 提取文本内容

### 3. `src/api/conversion_api.py`

**主要变更**:
- 导入结构化日志函数
- `forward_request()` 中的错误处理替换为结构化日志
- 根据 HTTP 状态码分类错误类型 (401/403 → AUTH, 429 → RATE_LIMIT)
- `test_proxy_connection()` 中 `proxies=` 改为 `proxy=`（httpx 兼容性修复）

### 4. `src/formats/anthropic_converter.py`

**重大重构**:
- `_convert_to_openai_request()` 改为基于统一类型层实现
- `_convert_from_openai_response()` 改为基于统一类型层实现
- `_convert_from_openai_streaming_chunk()` 改为基于 `StreamState` 实现
- 新增 `_extract_annotations_from_openai()` 方法
- 错误日志替换为 `log_structured_error()`

### 5. `src/utils/http_client.py`

**httpx 兼容性修复**:
- `httpx.AsyncClient(proxies=proxy_config)` → `httpx.AsyncClient(proxy=proxy)`
- 从字典配置中提取单个代理 URL

### 6. `src/core/capability_detector.py`

**httpx 兼容性修复**:
- 同上，`proxies=` 改为 `proxy=`

### 7. `requirements.txt`

**依赖升级**:
| 包名 | 旧版本 | 新版本 |
|------|--------|--------|
| fastapi | ==0.104.1 | >=0.115.0 |
| uvicorn | ==0.24.0 | >=0.32.0 |
| pydantic | ==2.5.0 | >=2.10.0 |
| httpx | ==0.25.2 | >=0.27.0 |
| pytest | ==7.4.3 | >=8.3.0 |
| cryptography | ==41.0.7 | >=44.0.0 |

**新增依赖**:
- `tiktoken>=0.7.0` - Token 计算

### 8. `README.md`

- `pip install` 改为 `pip3 install`

### 9. `.env.example`

- **已删除** - 环境变量示例文件

---

## 已知问题 (待修复)

| 优先级 | 问题 | 文件 | 建议 |
|--------|------|------|------|
| P0 | `url` 变量可能未定义时被记录 | unified_api.py:416-555 | 确保所有路径都有 url 赋值 |
| P1 | `proxy` 参数只取第一个代理配置 | http_client.py:136 | 使用 `proxy_config.get("https://")` |
| P1 | `request_data` 检测使用 `dir()` 不可靠 | unified_api.py:1285 | 改用 `locals().get()` |
| P2 | 依赖版本约束过于宽松 | requirements.txt | 考虑使用 `~=` 或固定版本 |

---

## 迁移指南

### 使用统一类型层

```python
from src.formats.unified import (
    UnifiedChatRequest,
    UnifiedChatResponse,
    UnifiedContent,
    UnifiedContentType,
    StreamState,
)

# 解析 Anthropic 请求
unified_req = UnifiedChatRequest.from_anthropic(anthropic_data)

# 转换为 OpenAI 格式
openai_data = unified_req.to_openai()

# 流式响应状态管理
state = StreamState(model="gpt-4", original_model="claude-3-opus")
text_idx = state.start_text_block()
```

### 使用结构化日志

```python
from src.utils.logger import (
    setup_logger,
    log_structured_error,
    log_request_entry,
    ERROR_TYPE_NETWORK,
    ERROR_TYPE_UPSTREAM_API,
)

logger = setup_logger("my_module")

# 记录请求入口
request_id = log_request_entry(
    logger,
    request_method="POST",
    request_url=url,
    source_format="anthropic",
    model="claude-3-opus",
    channel_name="default",
)

# 记录结构化错误
try:
    response = await client.post(url, json=data)
except Exception as e:
    log_structured_error(
        logger,
        error_type=ERROR_TYPE_NETWORK,
        exc=e,
        request_method="POST",
        request_url=url,
        request_body=data,
        extra={"channel": "default"},
    )
```

---

## 测试建议

1. **单元测试**: 验证 `UnifiedChatRequest` / `UnifiedChatResponse` 在各格式间的往返转换
2. **集成测试**: 验证新 `/v1/messages/count_tokens` 端点
3. **回归测试**: 确保现有转换逻辑行为不变
4. **代理测试**: 验证 httpx `proxy=` 参数在各代理配置下的行为

---

## 相关文档

- `doc/anthropic_to_openai_conversion.md` - Anthropic 到 OpenAI 转换规范
- `doc/gemini_to_openai_conversion.md` - Gemini 到 OpenAI 转换规范
