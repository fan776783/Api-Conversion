"""
Payload 配置处理器

支持渠道级别的请求参数默认值设置和强制覆盖。

配置格式：
{
    "default": {                  # 仅在参数不存在时设置
        "temperature": 0.7,
        "max_tokens": 32000
    },
    "override": {                 # 总是覆盖
        "top_p": 0.9
    }
}

参考 CLIProxyAPI 的 payload.default / payload.override 机制。
"""

import logging
import copy
from typing import Any, Dict, Optional

logger = logging.getLogger("payload_config")


def apply_payload_config(
    request_data: dict,
    payload_config: Optional[Dict[str, Any]],
) -> dict:
    """
    将渠道的 payload 配置应用到请求数据中。

    处理顺序：
    1. default 规则：仅在请求中不存在该参数时设置
    2. override 规则：总是覆盖请求中的参数

    支持嵌套路径（用 "." 分隔），例如：
      "generationConfig.temperature" → request_data["generationConfig"]["temperature"]

    Args:
        request_data: 请求数据（会被原地修改）
        payload_config: payload 配置字典

    Returns:
        修改后的 request_data
    """
    if not payload_config or not isinstance(payload_config, dict):
        return request_data

    if not isinstance(request_data, dict):
        return request_data

    # 1. 应用 default 规则
    defaults = payload_config.get("default", {})
    if isinstance(defaults, dict):
        for path, value in defaults.items():
            if not _has_nested_key(request_data, path):
                _set_nested_value(request_data, path, copy.deepcopy(value))
                logger.debug(f"payload default: 设置 {path} = {value}")

    # 2. 应用 override 规则
    overrides = payload_config.get("override", {})
    if isinstance(overrides, dict):
        for path, value in overrides.items():
            old_value = _get_nested_value(request_data, path)
            _set_nested_value(request_data, path, copy.deepcopy(value))
            if old_value is not None:
                logger.debug(f"payload override: 覆盖 {path}: {old_value} → {value}")
            else:
                logger.debug(f"payload override: 设置 {path} = {value}")

    return request_data


def _has_nested_key(data: dict, path: str) -> bool:
    """
    检查嵌套路径是否存在。

    Args:
        data: 字典数据
        path: 点分隔路径（如 "generationConfig.temperature"）

    Returns:
        路径是否存在
    """
    keys = path.split(".")
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return True


def _get_nested_value(data: dict, path: str) -> Optional[Any]:
    """
    获取嵌套路径的值。

    Args:
        data: 字典数据
        path: 点分隔路径

    Returns:
        路径对应的值，不存在返回 None
    """
    keys = path.split(".")
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _set_nested_value(data: dict, path: str, value: Any) -> None:
    """
    设置嵌套路径的值。自动创建中间层级的字典。

    Args:
        data: 字典数据
        path: 点分隔路径（如 "generationConfig.thinkingConfig.thinkingBudget"）
        value: 要设置的值
    """
    keys = path.split(".")
    current = data
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value
