"""
日志配置
统一日志管理，支持单文件和时间轮转
"""
import json
import logging
import logging.handlers
import os
import sys
import traceback
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# 全局日志器缓存，避免重复创建
_loggers = {}

# 北京时间（UTC+8）
_BEIJING_TIMEZONE = timezone(timedelta(hours=8), name="CST")

# 全局调试模式开关
_debug_mode = False

# 双模式日志常量
LOG_MODE_OBSERVE = "OBSERVE"
LOG_MODE_DIAGNOSE = "DIAGNOSE"
_VALID_LOG_MODES = (LOG_MODE_OBSERVE, LOG_MODE_DIAGNOSE)

# 运行时日志模式覆盖（None 表示使用环境/默认模式）
_runtime_log_mode: Optional[str] = None


class BeijingFormatter(logging.Formatter):
    """统一使用北京时间输出日志时间。"""

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=_BEIJING_TIMEZONE)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]


def _beijing_timestamp_iso() -> str:
    """生成 ISO 风格北京时间字符串。"""
    return datetime.now(_BEIJING_TIMEZONE).isoformat(timespec="milliseconds")


def _normalize_log_mode(mode: Optional[str]) -> Optional[str]:
    if not mode:
        return None
    mode_upper = str(mode).strip().upper()
    if mode_upper in _VALID_LOG_MODES:
        return mode_upper
    return None


def _resolve_log_mode() -> str:
    if _runtime_log_mode in _VALID_LOG_MODES:
        return _runtime_log_mode

    env_mode = _normalize_log_mode(os.environ.get("LOG_MODE"))
    if env_mode:
        return env_mode

    return LOG_MODE_DIAGNOSE if _debug_mode else LOG_MODE_OBSERVE


def get_runtime_log_mode() -> str:
    """获取当前运行时日志模式。"""
    return _resolve_log_mode()


def is_debug_enabled() -> bool:
    """检查是否启用调试模式"""
    return _debug_mode


def is_observe_mode() -> bool:
    return _resolve_log_mode() == LOG_MODE_OBSERVE


def is_diagnose_mode() -> bool:
    return _resolve_log_mode() == LOG_MODE_DIAGNOSE


def _resolve_handler_levels(base_level: str) -> tuple[str, str]:
    """根据当前日志模式解析 console/file handler 级别。"""
    mode = _resolve_log_mode()
    if mode == LOG_MODE_DIAGNOSE:
        return ("INFO", "DEBUG")
    if mode == LOG_MODE_OBSERVE:
        return ("INFO", "INFO")
    return (base_level, base_level)


def enable_debug(enabled: bool = True):
    """
    启用或禁用调试模式。
    调试模式下，默认切换为 DIAGNOSE 模式（若未显式指定运行时模式）。

    Args:
        enabled: True 启用调试模式，False 禁用
    """
    global _debug_mode
    _debug_mode = enabled

    # 更新所有已创建的 logger 的级别。
    for logger in _loggers.values():
        try:
            _reconfigure_logger_handlers(logger)
        except Exception:
            logger.setLevel(logging.DEBUG if enabled else logging.INFO)


def setup_logger(name: str, level: str = None) -> logging.Logger:
    """设置日志器"""
    if name in _loggers:
        return _loggers[name]

    try:
        from src.utils.env_config import env_config
        base_level = (level or env_config.log_level).upper()
        log_file = env_config.log_file
        log_max_days = env_config.log_max_days
    except ImportError:
        base_level = (level or ("DEBUG" if _debug_mode else "WARNING")).upper()
        log_file = "logs/app.log"
        log_max_days = 1

    console_level, file_level = _resolve_handler_levels(base_level)

    def _lvl(s: str) -> int:
        return int(getattr(logging, s, logging.INFO))

    logger = logging.getLogger(name)
    logger.setLevel(min(_lvl(console_level), _lvl(file_level)))
    logger.handlers.clear()

    formatter = BeijingFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(_lvl(console_level))
    console_handler.setFormatter(formatter)
    console_handler._handler_role = "console"  # type: ignore[attr-defined]
    logger.addHandler(console_handler)

    try:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=log_file,
            when='midnight',
            interval=1,
            backupCount=log_max_days,
            encoding='utf-8',
            utc=False
        )
        file_handler.suffix = "%Y-%m-%d"
        file_handler.setLevel(_lvl(file_level))
        file_handler.setFormatter(formatter)
        file_handler._handler_role = "file"  # type: ignore[attr-defined]
        logger.addHandler(file_handler)

    except Exception as e:
        console_handler.emit(logging.LogRecord(
            name=name,
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg=f"Failed to initialize file log handler: {e}",
            args=(),
            exc_info=None
        ))

    logger.propagate = False
    _loggers[name] = logger
    return logger


def _reconfigure_logger_handlers(logger: logging.Logger) -> None:
    """Refresh handler levels for an already-created logger based on current mode."""
    try:
        from src.utils.env_config import env_config
        base_level = env_config.log_level.upper()
    except Exception:
        base_level = "WARNING"

    console_level, file_level = _resolve_handler_levels(base_level)

    def _lvl(s: str) -> int:
        return int(getattr(logging, s, logging.INFO))

    logger.setLevel(min(_lvl(console_level), _lvl(file_level)))

    for h in list(getattr(logger, "handlers", [])):
        role = getattr(h, "_handler_role", None)
        if role == "console":
            h.setLevel(_lvl(console_level))
        elif role == "file":
            h.setLevel(_lvl(file_level))


def set_runtime_log_mode(mode: str) -> None:
    """运行时切换日志模式（无需重启）。"""
    global _runtime_log_mode
    mode_upper = _normalize_log_mode(mode)
    if not mode_upper:
        raise ValueError(f"无效的日志模式: {mode}，支持: {', '.join(_VALID_LOG_MODES)}")

    _runtime_log_mode = mode_upper
    for logger in _loggers.values():
        _reconfigure_logger_handlers(logger)

    print(f"[logger] 运行时日志模式已切换为: {mode_upper}")


def set_runtime_log_level(level: str) -> None:
    """
    兼容旧接口：将传统日志级别映射到双模式。
    DEBUG -> DIAGNOSE，其余级别 -> OBSERVE
    """
    level_upper = level.upper() if level else "INFO"
    mapped_mode = LOG_MODE_DIAGNOSE if level_upper == "DEBUG" else LOG_MODE_OBSERVE
    set_runtime_log_mode(mapped_mode)


def get_runtime_log_level() -> str:
    """
    兼容旧接口：将双模式映射回传统级别。
    OBSERVE -> INFO，DIAGNOSE -> DEBUG
    """
    return "DEBUG" if is_diagnose_mode() else "INFO"


def get_logger(name: str) -> logging.Logger:
    """获取日志器"""
    return _loggers.get(name) or setup_logger(name)


def cleanup_old_logs():
    """清理旧的日志文件"""
    try:
        from src.utils.env_config import env_config
        log_file = env_config.log_file
        log_max_days = env_config.log_max_days

        log_path = Path(log_file)
        log_dir = log_path.parent

        if not log_dir.exists():
            return

        import time
        current_time = time.time()
        max_age = log_max_days * 24 * 60 * 60

        for file_path in log_dir.glob(f"{log_path.stem}.*"):
            if file_path.stat().st_mtime < (current_time - max_age):
                try:
                    file_path.unlink()
                    print(f"Cleaned up old log file: {file_path}")
                except Exception as e:
                    print(f"Failed to clean up log file {file_path}: {e}")

    except Exception as e:
        print(f"Failed to cleanup old logs: {e}")


# 在模块加载时清理旧日志
cleanup_old_logs()


# ===================== 结构化错误日志 =====================

ERROR_TYPE_NETWORK = "network"
ERROR_TYPE_AUTH = "auth"
ERROR_TYPE_RATE_LIMIT = "rate_limit"
ERROR_TYPE_CONVERSION = "conversion"
ERROR_TYPE_UPSTREAM_API = "upstream_api"


def _extract_request_id(
    headers: Optional[Mapping[str, Any]] = None,
    explicit_request_id: Optional[str] = None,
) -> str:
    """从请求头或显式参数中提取/生成 request_id"""
    if explicit_request_id:
        return str(explicit_request_id)

    if headers:
        try:
            lower_map = {str(k).lower(): v for k, v in headers.items()}
            for key in ("x-request-id", "x-correlation-id", "x-trace-id", "x-amzn-trace-id"):
                if key in lower_map and lower_map[key]:
                    return str(lower_map[key])
        except Exception:
            pass

    return uuid.uuid4().hex[:16]


def _summarize_headers(
    headers: Optional[Mapping[str, Any]],
    max_headers: int = 20,
    max_value_length: int = 200,
) -> Optional[Dict[str, Any]]:
    """对请求/响应头做掩码和长度控制"""
    if not headers:
        return None

    try:
        from src.utils.security import mask_sensitive_data
        masked = mask_sensitive_data(dict(headers))
        summary: Dict[str, Any] = {}
        items = list(masked.items())

        for idx, (key, value) in enumerate(items):
            if idx >= max_headers:
                summary["_truncated"] = f"{len(items) - max_headers} more headers"
                break
            if isinstance(value, str) and len(value) > max_value_length:
                summary[key] = value[:max_value_length] + "...[truncated]"
            else:
                summary[key] = value
        return summary
    except Exception as e:
        return {"_error": f"failed to summarize: {e.__class__.__name__}"}


def _preview_body(body: Any, max_length: int = 2000) -> Optional[str]:
    """生成请求/响应体的安全预览"""
    if body is None:
        return None

    try:
        from src.utils.security import safe_log_data
        return safe_log_data(body, max_length=max_length)
    except Exception as e:
        return f"***preview failed: {e.__class__.__name__}***"


def sanitize_url_for_log(url: Optional[str]) -> Optional[str]:
    """对 URL 中的敏感 query 参数做脱敏。"""
    if not url:
        return url

    try:
        parsed = urlsplit(url)
        query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
        sanitized_pairs = []
        sensitive_keys = {
            "key", "api_key", "x-api-key", "authorization", "token", "access_token",
            "refresh_token", "id_token", "sig", "signature",
        }
        for key, value in query_pairs:
            if key.lower() in sensitive_keys:
                sanitized_pairs.append((key, "***"))
            else:
                sanitized_pairs.append((key, value))
        sanitized_query = urlencode(sanitized_pairs, doseq=True)
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, sanitized_query, parsed.fragment))
    except Exception:
        return url


def _format_observe_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, (int, str)):
        text = str(value).strip()
        return text or None
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(value)


def _build_observe_message(event: str, fields: Dict[str, Any]) -> str:
    parts = [f"[observe] {event}"]
    for key, value in fields.items():
        formatted = _format_observe_value(value)
        if formatted is None:
            continue
        parts.append(f"{key}={formatted}")
    return " ".join(parts)


def log_diagnose_event(
    logger: logging.Logger,
    message: str,
    *,
    extra: Optional[Dict[str, Any]] = None,
    level: int = logging.DEBUG,
) -> None:
    """问题定位模式下的上下文日志。"""
    if not is_diagnose_mode():
        return

    if extra:
        try:
            message = f"{message} | {json.dumps(extra, ensure_ascii=False, default=str)}"
        except Exception:
            message = f"{message} | {extra}"
    logger.log(level, message)


def log_observe_request_start(
    logger: logging.Logger,
    *,
    request_id: Optional[str] = None,
    request_method: Optional[str] = None,
    request_url: Optional[str] = None,
    request_headers: Optional[Mapping[str, Any]] = None,
    source_format: Optional[str] = None,
    model: Optional[str] = None,
    is_streaming: bool = False,
    channel_name: Optional[str] = None,
    channel_provider: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """记录请求开始的观测摘要。"""
    effective_request_id = _extract_request_id(request_headers, request_id)
    fields: Dict[str, Any] = {
        "request_id": effective_request_id,
        "method": request_method,
        "path": urlsplit(request_url).path if request_url else None,
        "source": source_format,
        "client_model": model,
        "streaming": is_streaming,
        "channel": channel_name,
        "provider": channel_provider,
    }
    if extra:
        fields.update(extra)
    logger.info(_build_observe_message("request_start", fields))
    return effective_request_id


def log_observe_request_done(
    logger: logging.Logger,
    *,
    request_id: str,
    source_format: Optional[str] = None,
    request_method: Optional[str] = None,
    request_path: Optional[str] = None,
    channel_name: Optional[str] = None,
    channel_provider: Optional[str] = None,
    upstream_url: Optional[str] = None,
    client_model: Optional[str] = None,
    upstream_model: Optional[str] = None,
    is_streaming: Optional[bool] = None,
    status_code: Optional[int] = None,
    duration_ms: Optional[float] = None,
    ttfb_ms: Optional[float] = None,
    input_chars: Optional[int] = None,
    output_chars: Optional[int] = None,
    usage: Optional[Dict[str, Any]] = None,
    reasoning_mode: Optional[str] = None,
    thinking_budget: Optional[int] = None,
    chunk_count: Optional[int] = None,
    note: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """记录请求完成的观测摘要。"""
    fields: Dict[str, Any] = {
        "request_id": request_id,
        "source": source_format,
        "method": request_method,
        "path": request_path,
        "provider": channel_provider,
        "channel": channel_name,
        "upstream_url": sanitize_url_for_log(upstream_url),
        "client_model": client_model,
        "upstream_model": upstream_model,
        "streaming": is_streaming,
        "status": status_code,
        "duration_ms": duration_ms,
        "ttfb_ms": ttfb_ms,
        "input_chars": input_chars,
        "output_chars": output_chars,
        "chunk_count": chunk_count,
        "reasoning_mode": reasoning_mode,
        "thinking_budget": thinking_budget,
        "note": note,
    }
    if usage:
        fields.update({
            "usage_in": usage.get("input_tokens"),
            "usage_out": usage.get("output_tokens"),
            "usage_total": usage.get("total_tokens"),
            "reasoning_tokens": usage.get("reasoning_tokens"),
            "cache_create_tokens": usage.get("cache_creation_input_tokens"),
            "cache_read_tokens": usage.get("cache_read_input_tokens"),
        })
    if extra:
        fields.update(extra)
    logger.info(_build_observe_message("request_done", fields))


def log_observe_request_failed(
    logger: logging.Logger,
    *,
    request_id: Optional[str] = None,
    request_headers: Optional[Mapping[str, Any]] = None,
    source_format: Optional[str] = None,
    request_method: Optional[str] = None,
    request_path: Optional[str] = None,
    channel_name: Optional[str] = None,
    channel_provider: Optional[str] = None,
    upstream_url: Optional[str] = None,
    client_model: Optional[str] = None,
    upstream_model: Optional[str] = None,
    is_streaming: Optional[bool] = None,
    status_code: Optional[int] = None,
    duration_ms: Optional[float] = None,
    error_type: Optional[str] = None,
    stage: Optional[str] = None,
    message: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """记录请求失败的观测摘要。"""
    effective_request_id = _extract_request_id(request_headers, request_id)
    fields: Dict[str, Any] = {
        "request_id": effective_request_id,
        "source": source_format,
        "method": request_method,
        "path": request_path,
        "provider": channel_provider,
        "channel": channel_name,
        "upstream_url": sanitize_url_for_log(upstream_url),
        "client_model": client_model,
        "upstream_model": upstream_model,
        "streaming": is_streaming,
        "status": status_code,
        "duration_ms": duration_ms,
        "error_type": error_type,
        "stage": stage,
        "message": message,
    }
    if extra:
        fields.update(extra)
    logger.info(_build_observe_message("request_failed", fields))
    return effective_request_id


def log_structured_error(
    logger: logging.Logger,
    *,
    error_type: str,
    exc: Optional[BaseException] = None,
    request_id: Optional[str] = None,
    request_method: Optional[str] = None,
    request_url: Optional[str] = None,
    request_headers: Optional[Mapping[str, Any]] = None,
    request_body: Any = None,
    response_status: Optional[int] = None,
    response_headers: Optional[Mapping[str, Any]] = None,
    response_body: Any = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """
    统一的结构化错误日志入口。

    Args:
        logger: 日志器实例
        error_type: 错误类型（network/auth/rate_limit/conversion/upstream_api）
        exc: 异常对象
        request_id: 请求ID（自动从headers提取或生成）
        request_method: 请求方法
        request_url: 请求URL
        request_headers: 请求头
        request_body: 请求体
        response_status: 响应状态码
        response_headers: 响应头
        response_body: 响应体
        extra: 额外上下文信息

    Returns:
        生成或提取的 request_id
    """
    effective_request_id = _extract_request_id(request_headers, request_id)
    timestamp = _beijing_timestamp_iso()

    exception_block: Dict[str, Any] = {}
    if exc is not None:
        exception_block["type"] = exc.__class__.__name__
        exception_block["message"] = str(exc)
        exception_block["traceback"] = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
    else:
        exception_block["traceback"] = "".join(traceback.format_stack()[:-1])

    payload: Dict[str, Any] = {
        "timestamp": timestamp,
        "logger": logger.name,
        "level": "ERROR",
        "mode": get_runtime_log_mode(),
        "error_type": error_type,
        "request_id": effective_request_id,
        "request": {
            "method": request_method,
            "url": sanitize_url_for_log(request_url),
            "headers": _summarize_headers(request_headers),
            "body": _preview_body(request_body),
        },
        "response": {
            "status": response_status,
            "headers": _summarize_headers(response_headers),
            "body": _preview_body(response_body),
        },
        "exception": exception_block,
    }

    if extra:
        payload["extra"] = extra

    try:
        serialized = json.dumps(payload, ensure_ascii=False)
    except Exception as serialize_exc:
        fallback = {
            "timestamp": timestamp,
            "logger": logger.name,
            "level": "ERROR",
            "error_type": error_type,
            "request_id": effective_request_id,
            "serialization_error": str(serialize_exc),
        }
        serialized = json.dumps(fallback, ensure_ascii=False)

    logger.error(f"[structured_error] {serialized}")
    return effective_request_id


def log_request_entry(
    logger: logging.Logger,
    *,
    request_id: Optional[str] = None,
    request_method: Optional[str] = None,
    request_url: Optional[str] = None,
    request_headers: Optional[Mapping[str, Any]] = None,
    source_format: Optional[str] = None,
    model: Optional[str] = None,
    is_streaming: bool = False,
    channel_name: Optional[str] = None,
    channel_provider: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """兼容旧入口，转发到新的 OBSERVE 请求开始日志。"""
    return log_observe_request_start(
        logger,
        request_id=request_id,
        request_method=request_method,
        request_url=request_url,
        request_headers=request_headers,
        source_format=source_format,
        model=model,
        is_streaming=is_streaming,
        channel_name=channel_name,
        channel_provider=channel_provider,
        extra=extra,
    )
