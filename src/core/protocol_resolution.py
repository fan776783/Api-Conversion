"""
统一协议解析
负责在 provider 与 protocol/format 之间建立稳定的一致语义
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from src.formats.converter_factory import (
    OPENAI_CHAT_COMPLETIONS_FORMAT,
    OPENAI_RESPONSES_FORMAT,
    canonical_format_name,
)
from src.formats.openai_responses_request_adapter import OpenAIResponsesRequestAdapter

OPENAI_PROVIDER = "openai"
OPENAI_PROTOCOLS = (
    OPENAI_CHAT_COMPLETIONS_FORMAT,
    OPENAI_RESPONSES_FORMAT,
)
OPENAI_PROTOCOL_LABELS = {
    OPENAI_CHAT_COMPLETIONS_FORMAT: "OpenAI Chat Completions",
    OPENAI_RESPONSES_FORMAT: "OpenAI Responses",
}


def get_openai_protocol_options() -> list[dict[str, str]]:
    """返回统一的 OpenAI 子协议选项定义。"""
    return [
        {"value": protocol, "label": OPENAI_PROTOCOL_LABELS.get(protocol, protocol)}
        for protocol in OPENAI_PROTOCOLS
    ]


@dataclass
class ResolvedProtocolContext:
    """统一协议解析结果"""

    client_format: str
    target_provider: str
    target_format: str
    normalized_request_data: Dict[str, Any]
    request_path: str = ""
    explicit_client_format: Optional[str] = None


def is_openai_provider(provider: Optional[str]) -> bool:
    if not provider:
        return False
    return provider == OPENAI_PROVIDER


def is_openai_format(format_name: Optional[str]) -> bool:
    if not format_name:
        return False
    return canonical_format_name(format_name) in OPENAI_PROTOCOLS


def normalize_supported_openai_formats(
    default_target_format: Optional[str] = None,
    supported_formats: Optional[Sequence[str]] = None,
) -> list[str]:
    """规范化 OpenAI 可用子协议集合。"""
    normalized: list[str] = []

    if supported_formats:
        for format_name in supported_formats:
            if not is_openai_format(format_name):
                continue
            canonical = canonical_format_name(format_name)
            if canonical not in normalized:
                normalized.append(canonical)

    canonical_default = (
        canonical_format_name(default_target_format)
        if is_openai_format(default_target_format)
        else None
    )
    if canonical_default and canonical_default not in normalized:
        normalized.insert(0, canonical_default)

    if not normalized:
        normalized = [OPENAI_CHAT_COMPLETIONS_FORMAT, OPENAI_RESPONSES_FORMAT]

    return normalized


def resolve_openai_target_format(
    client_format: Optional[str],
    default_target_format: Optional[str] = None,
    supported_formats: Optional[Sequence[str]] = None,
) -> str:
    """为 OpenAI provider 解析最终 target_format。"""
    normalized_supported = normalize_supported_openai_formats(
        default_target_format=default_target_format,
        supported_formats=supported_formats,
    )
    canonical_client = canonical_format_name(client_format) if client_format else None

    if canonical_client in normalized_supported:
        return canonical_client

    return normalized_supported[0]


def provider_supports_format(
    provider: str,
    format_name: str,
    *,
    default_target_format: Optional[str] = None,
    supported_formats: Optional[Sequence[str]] = None,
) -> bool:
    canonical = canonical_format_name(format_name)
    if is_openai_provider(provider):
        return canonical in normalize_supported_openai_formats(
            default_target_format=default_target_format,
            supported_formats=supported_formats,
        )
    return provider == canonical


def same_protocol(source_format: str, target_format: str) -> bool:
    return canonical_format_name(source_format) == canonical_format_name(target_format)


def detect_client_format(
    request_data: Dict[str, Any],
    request_path: Optional[str] = None,
    explicit_client_format: Optional[str] = None,
) -> str:
    """检测客户端契约格式。显式路径优先于 body 形状。"""
    path = request_path or ""

    if explicit_client_format == OPENAI_RESPONSES_FORMAT:
        return OPENAI_RESPONSES_FORMAT
    if explicit_client_format == OPENAI_CHAT_COMPLETIONS_FORMAT:
        return OPENAI_CHAT_COMPLETIONS_FORMAT

    if path.endswith("/v1/responses") or "/responses" in path:
        return OPENAI_RESPONSES_FORMAT
    if path.endswith("/v1/chat/completions") or "/chat/completions" in path:
        return OPENAI_CHAT_COMPLETIONS_FORMAT
    if "/anthropic/" in path or path.endswith("/messages") or "/v1/messages" in path:
        return "anthropic"
    if "/gemini/" in path or "generateContent" in path:
        return "gemini"

    if isinstance(request_data, dict):
        if OpenAIResponsesRequestAdapter.looks_like_responses_request(request_data):
            return OPENAI_RESPONSES_FORMAT
        if "contents" in request_data:
            return "gemini"
        if "messages" in request_data and "system" in request_data:
            return "anthropic"
        if "messages" in request_data and "model" in request_data:
            return OPENAI_CHAT_COMPLETIONS_FORMAT

    if explicit_client_format:
        return canonical_format_name(explicit_client_format)

    return OPENAI_CHAT_COMPLETIONS_FORMAT


def normalize_request_for_client_format(client_format: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """按客户端契约规范化请求体。"""
    if not isinstance(request_data, dict):
        return request_data

    normalized = copy.deepcopy(request_data)
    canonical_client = canonical_format_name(client_format)

    if (
        canonical_client == OPENAI_CHAT_COMPLETIONS_FORMAT
        and OpenAIResponsesRequestAdapter.looks_like_responses_request(normalized)
    ):
        return OpenAIResponsesRequestAdapter.adapt(normalized)

    return normalized


def resolve_target_format(
    client_format: str,
    target_provider: str,
    *,
    default_target_format: Optional[str] = None,
    supported_formats: Optional[Sequence[str]] = None,
) -> str:
    """根据 provider 能力解析最终 target_format。"""
    if is_openai_provider(target_provider):
        return resolve_openai_target_format(
            client_format=client_format,
            default_target_format=default_target_format,
            supported_formats=supported_formats,
        )
    return canonical_format_name(target_provider)


def resolve_protocol_context(
    *,
    request_data: Dict[str, Any],
    request_path: Optional[str],
    target_provider: str,
    explicit_client_format: Optional[str] = None,
    default_target_format: Optional[str] = None,
    supported_formats: Optional[Sequence[str]] = None,
) -> ResolvedProtocolContext:
    """统一解析 client_format / target_format / normalized_request_data。"""
    client_format = detect_client_format(
        request_data=request_data,
        request_path=request_path,
        explicit_client_format=explicit_client_format,
    )
    normalized_request_data = normalize_request_for_client_format(client_format, request_data)
    target_format = resolve_target_format(
        client_format=client_format,
        target_provider=target_provider,
        default_target_format=default_target_format,
        supported_formats=supported_formats,
    )

    return ResolvedProtocolContext(
        client_format=client_format,
        target_provider=target_provider,
        target_format=target_format,
        normalized_request_data=normalized_request_data,
        request_path=request_path or "",
        explicit_client_format=explicit_client_format,
    )
