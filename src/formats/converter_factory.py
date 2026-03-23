"""
转换器工厂
负责创建和管理不同格式的转换器
"""
import contextvars
import os
from typing import Dict, Optional

from .base_converter import BaseConverter, ConversionResult
from .openai_converter import OpenAIConverter
from .openai_responses_converter import OpenAIResponsesConverter
from .anthropic_converter import AnthropicConverter
from .gemini_converter import GeminiConverter
from src.utils.logger import setup_logger

logger = setup_logger("converter_factory")

OPENAI_CHAT_COMPLETIONS_FORMAT = "openai_chat_completions"
OPENAI_RESPONSES_FORMAT = "openai_responses"
_OPENAI_ALIASES = {
    "openai": OPENAI_CHAT_COMPLETIONS_FORMAT,
    OPENAI_CHAT_COMPLETIONS_FORMAT: OPENAI_CHAT_COMPLETIONS_FORMAT,
    OPENAI_RESPONSES_FORMAT: OPENAI_RESPONSES_FORMAT,
}

# Streaming debug can explode quickly. Default behavior: log only the first few
# chunks and then every N chunks, unless STREAM_TRACE_LOG=1 is enabled.
_STREAM_TRACE_LOG = os.environ.get("STREAM_TRACE_LOG", "").lower() in ("1", "true", "yes", "on")
try:
    _STREAM_CHUNK_LOG_EVERY_N = int(os.environ.get("STREAM_CHUNK_LOG_EVERY_N", "50"))
except ValueError:
    _STREAM_CHUNK_LOG_EVERY_N = 50


def canonical_format_name(format_name: str) -> str:
    """规范化格式名称，兼容 openai 别名与子协议。"""
    return _OPENAI_ALIASES.get(format_name, format_name)


class ConverterFactory:
    """转换器工厂"""

    _converters: Dict[str, BaseConverter] = {}

    @classmethod
    def get_converter(cls, format_name: str) -> Optional[BaseConverter]:
        """获取指定格式的转换器"""
        canonical_name = canonical_format_name(format_name)
        if canonical_name not in cls._converters:
            cls._converters[canonical_name] = cls._create_converter(canonical_name)

        return cls._converters[canonical_name]

    @classmethod
    def _create_converter(cls, format_name: str) -> Optional[BaseConverter]:
        """创建转换器实例"""
        converters = {
            OPENAI_CHAT_COMPLETIONS_FORMAT: OpenAIConverter,
            OPENAI_RESPONSES_FORMAT: OpenAIResponsesConverter,
            "anthropic": AnthropicConverter,
            "gemini": GeminiConverter,
        }

        converter_class = converters.get(format_name)
        if converter_class:
            logger.info(f"Created converter for format: {format_name}")
            return converter_class()

        logger.error(f"Unsupported format: {format_name}")
        return None

    @classmethod
    def get_supported_formats(cls) -> list[str]:
        """获取支持的格式列表"""
        return [
            "openai",
            OPENAI_CHAT_COMPLETIONS_FORMAT,
            OPENAI_RESPONSES_FORMAT,
            "anthropic",
            "gemini",
        ]

    @classmethod
    def is_format_supported(cls, format_name: str) -> bool:
        """检查格式是否支持"""
        return canonical_format_name(format_name) in {
            canonical_format_name(name) for name in cls.get_supported_formats()
        }


# Per-request streaming converters.
#
# ConverterFactory caches singletons. Streaming conversion relies on mutable
# per-stream state (e.g. "sent message_start" flags). Using cached singletons can
# cause state bleed between concurrent requests. ContextVar keeps an isolated
# cache per asyncio task/request.
_STREAM_CONVERTERS_CTX: contextvars.ContextVar[Optional[Dict[str, BaseConverter]]] = contextvars.ContextVar(
    "STREAM_CONVERTERS_CTX",
    default=None,
)


def _get_stream_converters() -> Dict[str, BaseConverter]:
    converters = _STREAM_CONVERTERS_CTX.get()
    if converters is None:
        converters = {}
        _STREAM_CONVERTERS_CTX.set(converters)
    return converters


def clear_stream_converters() -> None:
    """Clear per-request streaming converter cache."""
    _STREAM_CONVERTERS_CTX.set(None)



def _get_stream_converter(format_name: str) -> Optional[BaseConverter]:
    canonical_name = canonical_format_name(format_name)
    converters = _get_stream_converters()
    converter = converters.get(canonical_name)
    if converter is not None:
        return converter

    converter_class = {
        OPENAI_CHAT_COMPLETIONS_FORMAT: OpenAIConverter,
        OPENAI_RESPONSES_FORMAT: OpenAIResponsesConverter,
        "anthropic": AnthropicConverter,
        "gemini": GeminiConverter,
    }.get(canonical_name)
    if converter_class is None:
        logger.error(f"Unsupported format: {canonical_name}")
        return None

    converter = converter_class()
    converters[canonical_name] = converter
    return converter


# 便捷函数
def convert_request(source_format: str, target_format: str, data: dict, headers: dict = None):
    """转换请求格式"""
    canonical_source = canonical_format_name(source_format)
    canonical_target = canonical_format_name(target_format)

    converter = ConverterFactory.get_converter(canonical_source)
    if not converter:
        raise ValueError(f"Unsupported source format: {source_format}")

    if hasattr(converter, "set_original_model") and isinstance(data, dict) and "model" in data:
        converter.set_original_model(data["model"])

    return converter.convert_request(data, canonical_target, headers)



def convert_response(source_format: str, target_format: str, data: dict, original_model: str = None):
    """转换响应格式"""
    canonical_source = canonical_format_name(source_format)
    canonical_target = canonical_format_name(target_format)

    converter = ConverterFactory.get_converter(canonical_target)
    if not converter:
        raise ValueError(f"Unsupported target format: {target_format}")

    if hasattr(converter, "set_original_model") and original_model:
        converter.set_original_model(original_model)

    return converter.convert_response(data, canonical_source, canonical_target)



def convert_streaming_chunk(source_format: str, target_format: str, data: dict, original_model: str = None):
    """转换流式响应 chunk 格式"""
    import logging

    logger = logging.getLogger("unified_api")
    canonical_source = canonical_format_name(source_format)
    canonical_target = canonical_format_name(target_format)

    if not data:
        logger.warning("Empty data passed to convert_streaming_chunk")
        return ConversionResult(success=True, data={})

    if canonical_source == canonical_target:
        logger.debug(f"Same format ({canonical_source}), returning data without conversion")
        return ConversionResult(success=True, data=data)

    converter = _get_stream_converter(canonical_target)
    if not converter:
        raise ValueError(f"Unsupported target format: {target_format}")

    cnt = getattr(converter, "_cf_stream_chunk_count", 0) + 1
    setattr(converter, "_cf_stream_chunk_count", cnt)

    is_finish = False
    if isinstance(data, dict):
        if canonical_source in {"openai", OPENAI_CHAT_COMPLETIONS_FORMAT} and "choices" in data:
            choices = data.get("choices", [])
            if choices and choices[0].get("finish_reason"):
                is_finish = True
        elif canonical_source == "gemini" and "candidates" in data:
            cands = data.get("candidates", [])
            if cands and cands[0].get("finishReason"):
                is_finish = True
        elif canonical_source == "anthropic" and data.get("type") in ("message_stop",):
            is_finish = True

    should_log = (
        _STREAM_TRACE_LOG
        or cnt <= 3
        or (_STREAM_CHUNK_LOG_EVERY_N > 0 and cnt % _STREAM_CHUNK_LOG_EVERY_N == 0)
        or is_finish
    )
    if should_log:
        keys = list(data.keys()) if isinstance(data, dict) else "not dict"
        logger.debug(
            f"CONVERTER_FACTORY: convert_streaming_chunk called: {canonical_source} -> {canonical_target} "
            f"(n={cnt}, finish={is_finish}), keys={keys}"
        )

    if hasattr(converter, "set_original_model") and original_model:
        converter.set_original_model(original_model)

    should_reset_state = False
    is_stream_start = False
    if isinstance(data, dict):
        if canonical_source == "gemini" and "responseId" in data and "candidates" in data:
            candidates = data.get("candidates", [])
            if candidates and not candidates[0].get("finishReason"):
                is_stream_start = True
        elif canonical_source in {"openai", OPENAI_CHAT_COMPLETIONS_FORMAT} and "choices" in data:
            stream_id = data.get("id")
            if isinstance(stream_id, str) and stream_id:
                last_stream_id = getattr(converter, "_cf_last_openai_stream_id", None)
                if last_stream_id != stream_id:
                    is_stream_start = True
                    setattr(converter, "_cf_last_openai_stream_id", stream_id)
            else:
                choices = data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {}) or {}
                    if isinstance(delta, dict) and delta.get("role") == "assistant":
                        delta_keys = set(delta.keys())
                        if delta_keys == {"role"}:
                            is_stream_start = True
        elif canonical_source == "anthropic" and data.get("type") == "message_start":
            is_stream_start = True

    if hasattr(converter, "reset_streaming_state") and hasattr(converter, "original_model"):
        current_model = getattr(converter, "original_model", None)
        if current_model != original_model:
            should_reset_state = True
            logger.debug(f"Model changed from {current_model} to {original_model}, resetting state")
        elif is_stream_start:
            should_reset_state = True
            logger.debug("Stream start detected, resetting state")

    if should_reset_state:
        logger.debug(f"Calling reset_streaming_state() on {converter.__class__.__name__}")
        converter.reset_streaming_state()

    if canonical_target == OPENAI_CHAT_COMPLETIONS_FORMAT:
        if canonical_source == "gemini" and hasattr(converter, "_convert_from_gemini_streaming_chunk"):
            return converter._convert_from_gemini_streaming_chunk(data)
        if canonical_source == "anthropic" and hasattr(converter, "_convert_from_anthropic_streaming_chunk"):
            return converter._convert_from_anthropic_streaming_chunk(data)
    elif canonical_target == OPENAI_RESPONSES_FORMAT:
        if canonical_source == "anthropic" and hasattr(converter, "_convert_from_anthropic_streaming_chunk"):
            return converter._convert_from_anthropic_streaming_chunk(data)
    elif canonical_target == "anthropic":
        if canonical_source in {"openai", OPENAI_CHAT_COMPLETIONS_FORMAT} and hasattr(converter, "_convert_from_openai_streaming_chunk"):
            return converter._convert_from_openai_streaming_chunk(data)
        if canonical_source == "gemini" and hasattr(converter, "_convert_from_gemini_streaming_chunk"):
            return converter._convert_from_gemini_streaming_chunk(data)
    elif canonical_target == "gemini":
        if canonical_source in {"openai", OPENAI_CHAT_COMPLETIONS_FORMAT} and hasattr(converter, "_convert_from_openai_streaming_chunk"):
            return converter._convert_from_openai_streaming_chunk(data)
        if canonical_source == "anthropic" and hasattr(converter, "_convert_from_anthropic_streaming_chunk"):
            return converter._convert_from_anthropic_streaming_chunk(data)
        if canonical_source == "gemini" and hasattr(converter, "_convert_from_gemini_streaming_chunk"):
            return converter._convert_from_gemini_streaming_chunk(data)

    if not data or (isinstance(data, dict) and not data) or (isinstance(data, str) and data.strip() == "[DONE]"):
        return ConversionResult(success=True, data=data)

    return converter.convert_response(data, canonical_source, canonical_target)
