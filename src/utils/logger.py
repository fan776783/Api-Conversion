"""
日志配置
统一日志管理，支持单文件和时间轮转
"""
import logging
import logging.handlers
import sys
import os
import json
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

# 全局日志器缓存，避免重复创建
_loggers = {}

# 全局调试模式开关
_debug_mode = False


def is_debug_enabled() -> bool:
    """检查是否启用调试模式"""
    return _debug_mode


def enable_debug(enabled: bool = True):
    """
    启用或禁用调试模式。
    调试模式下，所有 logger 的级别会被设置为 DEBUG。

    Args:
        enabled: True 启用调试模式，False 禁用
    """
    global _debug_mode
    _debug_mode = enabled

    # 更新所有已创建的 logger 的级别。
    # 注意：调试模式下我们希望“文件里更详细、控制台更简洁”，所以不要把所有 handler 都强行设成 DEBUG。
    for logger in _loggers.values():
        try:
            # 复用 setup_logger 的环境变量策略重新设置 handler 等级
            _reconfigure_logger_handlers(logger)
        except Exception:
            # 保底：至少把 logger 本身放到 DEBUG，避免完全丢失 debug 记录
            logger.setLevel(logging.DEBUG if enabled else logging.INFO)


def setup_logger(name: str, level: str = None) -> logging.Logger:
    """设置日志器"""
    # 如果已经创建过，直接返回
    if name in _loggers:
        return _loggers[name]

    # 导入环境配置
    try:
        from src.utils.env_config import env_config
        base_level = (level or env_config.log_level).upper()
        log_file = env_config.log_file
        log_max_days = env_config.log_max_days
    except ImportError:
        # 如果无法导入环境配置，使用默认值
        base_level = (level or ("DEBUG" if _debug_mode else "WARNING")).upper()
        log_file = "logs/app.log"
        log_max_days = 1

    # 支持把控制台和文件的日志级别拆开：
    # - 默认不设置 CONSOLE_LOG_LEVEL/FILE_LOG_LEVEL 时，保持兼容（两者都跟 base_level 一致）
    # - 调试模式下优先提升 FILE_LOG_LEVEL 到 DEBUG，便于落盘分析，同时控制台可保持更简洁
    console_level = os.environ.get("CONSOLE_LOG_LEVEL", base_level).upper()
    file_level = os.environ.get("FILE_LOG_LEVEL", base_level).upper()
    if _debug_mode:
        file_level = "DEBUG"
        # Debug runs are typically for investigation; keep console readable by default.
        if "CONSOLE_LOG_LEVEL" not in os.environ:
            console_level = "INFO"

    def _lvl(s: str) -> int:
        return int(getattr(logging, s, logging.INFO))

    logger = logging.getLogger(name)
    # logger level must be <= each handler level to allow records through.
    logger.setLevel(min(_lvl(console_level), _lvl(file_level)))

    # 清除现有处理器
    logger.handlers.clear()

    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(_lvl(console_level))
    console_handler.setFormatter(formatter)
    # mark for enable_debug() reconfigure
    console_handler._handler_role = "console"  # type: ignore[attr-defined]
    logger.addHandler(console_handler)

    # 创建文件处理器（时间轮转）
    try:
        # 确保日志目录存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # 使用TimedRotatingFileHandler进行时间轮转
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=log_file,
            when='midnight',  # 每天午夜轮转
            interval=1,       # 间隔1天
            backupCount=log_max_days,  # 保留的备份文件数量
            encoding='utf-8',
            utc=False
        )

        # 设置轮转文件的命名格式
        file_handler.suffix = "%Y-%m-%d"

        file_handler.setLevel(_lvl(file_level))
        file_handler.setFormatter(formatter)
        file_handler._handler_role = "file"  # type: ignore[attr-defined]
        logger.addHandler(file_handler)

    except Exception as e:
        # 文件处理器失败时仅打印一次警告，不中断程序
        console_handler.emit(logging.LogRecord(
            name=name,
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg=f"Failed to initialize file log handler: {e}",
            args=(),
            exc_info=None
        ))

    # 防止日志传播到根日志器
    logger.propagate = False

    # 缓存日志器
    _loggers[name] = logger

    return logger


def _reconfigure_logger_handlers(logger: logging.Logger) -> None:
    """Refresh handler levels for an already-created logger based on current env/debug flags."""
    try:
        from src.utils.env_config import env_config
        base_level = env_config.log_level.upper()
    except Exception:
        base_level = "WARNING"

    console_level = os.environ.get("CONSOLE_LOG_LEVEL", base_level).upper()
    file_level = os.environ.get("FILE_LOG_LEVEL", base_level).upper()
    if _debug_mode:
        file_level = "DEBUG"
        if "CONSOLE_LOG_LEVEL" not in os.environ:
            console_level = "INFO"

    def _lvl(s: str) -> int:
        return int(getattr(logging, s, logging.INFO))

    logger.setLevel(min(_lvl(console_level), _lvl(file_level)))

    for h in list(getattr(logger, "handlers", [])):
        role = getattr(h, "_handler_role", None)
        if role == "console":
            h.setLevel(_lvl(console_level))
        elif role == "file":
            h.setLevel(_lvl(file_level))


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
        max_age = log_max_days * 24 * 60 * 60  # 转换为秒

        # 清理旧的日志文件
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

# 错误类型常量
ERROR_TYPE_NETWORK = "network"           # 网络错误：超时、连接失败、代理错误
ERROR_TYPE_AUTH = "auth"                 # 认证错误：401/403
ERROR_TYPE_RATE_LIMIT = "rate_limit"     # 限流错误：429
ERROR_TYPE_CONVERSION = "conversion"     # 格式转换错误
ERROR_TYPE_UPSTREAM_API = "upstream_api" # 上游API错误：非2xx响应、流式解析失败


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
    timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    # 异常信息
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
        "error_type": error_type,
        "request_id": effective_request_id,
        "request": {
            "method": request_method,
            "url": request_url,
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
    """
    记录请求入口日志（INFO级别）。

    Returns:
        生成或提取的 request_id
    """
    effective_request_id = _extract_request_id(request_headers, request_id)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    payload: Dict[str, Any] = {
        "timestamp": timestamp,
        "logger": logger.name,
        "level": "INFO",
        "type": "request_entry",
        "request_id": effective_request_id,
        "request": {
            "method": request_method,
            "url": request_url,
        },
        "source_format": source_format,
        "model": model,
        "is_streaming": is_streaming,
        "channel": {
            "name": channel_name,
            "provider": channel_provider,
        },
    }

    if extra:
        payload["extra"] = extra

    try:
        serialized = json.dumps(payload, ensure_ascii=False)
    except Exception:
        serialized = json.dumps({"request_id": effective_request_id, "error": "serialization_failed"})

    logger.info(f"[request_entry] {serialized}")

    return effective_request_id