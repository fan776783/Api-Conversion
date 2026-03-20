"""
模型名后缀解析器

参考 CLIProxyAPI 的 thinking.ParseSuffix 机制，从模型名中提取 thinking 配置和 token 限制。

支持的模型名格式：
  - gpt-5.4                          → 无后缀，原样透传
  - gpt-5.4(xhigh)                   → thinking_level = xhigh
  - gpt-5.4(xhigh)[1m]               → thinking_level = xhigh, max_tokens = 1000000
  - claude-sonnet-4(8192)             → thinking_budget = 8192
  - claude-sonnet-4(none)             → 禁用 thinking
  - claude-sonnet-4(auto)             → 自动 thinking
  - gpt-5.4[500k]                     → max_tokens = 500000（无 thinking 后缀）
  - gpt-5.4[8192]                     → max_tokens = 8192
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("model_suffix")


# ── 常量定义 ──────────────────────────────────────────────────

# thinking 模式
class ThinkingMode:
    """thinking 模式枚举"""
    NONE = "none"        # 禁用 thinking
    AUTO = "auto"        # 自动 thinking
    BUDGET = "budget"    # 数值型 budget
    LEVEL = "level"      # 等级型（minimal/low/medium/high/xhigh/max）


# 支持的 thinking 等级（大小写不敏感）
VALID_LEVELS = {"minimal", "low", "medium", "high", "xhigh", "max"}

# 支持的特殊值
SPECIAL_VALUES = {
    "none": ThinkingMode.NONE,
    "auto": ThinkingMode.AUTO,
    "-1": ThinkingMode.AUTO,
}

# token 数量后缀乘数（大小写不敏感）
TOKEN_SUFFIXES = {
    "k": 1_000,
    "m": 1_000_000,
}


# ── 数据类型 ──────────────────────────────────────────────────

@dataclass
class ThinkingConfig:
    """thinking 配置"""
    mode: Optional[str] = None   # ThinkingMode 的值
    budget: Optional[int] = None   # 数值型 budget（如 8192）
    level: Optional[str] = None    # 等级型（如 "xhigh"）

    @property
    def has_config(self) -> bool:
        """是否包含有效的 thinking 配置"""
        return self.mode is not None


@dataclass
class ModelSuffixResult:
    """模型名后缀解析结果"""
    # 基础模型名（去除所有后缀后的名称）
    base_model: str
    # 原始模型名
    original_model: str
    # 是否有圆括号后缀
    has_thinking_suffix: bool = False
    # 原始 thinking 后缀文本
    raw_thinking_suffix: str = ""
    # 解析后的 thinking 配置
    thinking: ThinkingConfig = field(default_factory=ThinkingConfig)
    # 是否有方括号后缀
    has_token_suffix: bool = False
    # 原始 token 后缀文本
    raw_token_suffix: str = ""
    # 解析后的 max_tokens 值
    max_tokens: Optional[int] = None


# ── 解析函数 ──────────────────────────────────────────────────

def parse_model_suffix(model: str) -> ModelSuffixResult:
    """
    解析模型名中的后缀。

    格式: base_model[(thinking_suffix)][[token_suffix]]

    支持的组合：
      - "gpt-5.4"                    → 无后缀
      - "gpt-5.4(xhigh)"             → thinking level
      - "gpt-5.4(xhigh)[1m]"         → thinking level + token limit
      - "gpt-5.4[500k]"              → 仅 token limit
      - "claude-sonnet-4(8192)"       → thinking budget
      - "claude-sonnet-4(none)"       → 禁用 thinking
      - "claude-sonnet-4(auto)"       → 自动 thinking

    Args:
        model: 完整的模型名（可能包含后缀）

    Returns:
        ModelSuffixResult: 解析结果
    """
    if not model or not isinstance(model, str):
        return ModelSuffixResult(base_model=model or "", original_model=model or "")

    result = ModelSuffixResult(base_model=model, original_model=model)
    remaining = model

    # ── 第 1 步：解析方括号后缀 [xxx] ──
    # 必须在末尾，如 "gpt-5.4(xhigh)[1m]" 或 "gpt-5.4[500k]"
    remaining, token_result = _parse_bracket_suffix(remaining)
    if token_result is not None:
        result.has_token_suffix = True
        result.raw_token_suffix = token_result["raw"]
        result.max_tokens = token_result["value"]

    # ── 第 2 步：解析圆括号后缀 (xxx) ──
    # 在方括号之前（如果有的话），如 "gpt-5.4(xhigh)"
    remaining, thinking_result = _parse_paren_suffix(remaining)
    if thinking_result is not None:
        result.has_thinking_suffix = True
        result.raw_thinking_suffix = thinking_result["raw"]
        result.thinking = thinking_result["config"]

    result.base_model = remaining
    return result


def _parse_bracket_suffix(model: str):
    """
    解析末尾的方括号后缀 [xxx]。

    支持：
      - [1m] → 1,000,000
      - [500k] → 500,000
      - [8192] → 8192

    Returns:
        (remaining_model, token_result_dict_or_None)
    """
    # 匹配末尾的 [...] 格式
    match = re.search(r'\[([^\]]+)\]$', model)
    if not match:
        return model, None

    raw = match.group(1)
    value = _parse_token_value(raw)
    if value is None:
        # 不是有效的 token 值，不做截断（可能是模型名的一部分）
        return model, None

    remaining = model[:match.start()]
    return remaining, {"raw": raw, "value": value}


def _parse_paren_suffix(model: str):
    """
    解析末尾的圆括号后缀 (xxx)。

    参考 CLIProxyAPI 的 ParseSuffix 逻辑：
    - 查找最后一个 "(" 的位置
    - 检查字符串是否以 ")" 结尾
    - 提取括号内容作为 raw suffix

    Returns:
        (remaining_model, thinking_result_dict_or_None)
    """
    if not model.endswith(")"):
        return model, None

    last_open = model.rfind("(")
    if last_open == -1 or last_open == 0:
        return model, None

    raw = model[last_open + 1:-1]
    if not raw:
        return model, None

    config = _parse_thinking_value(raw)
    remaining = model[:last_open]
    return remaining, {"raw": raw, "config": config}


def _parse_token_value(raw: str) -> Optional[int]:
    """
    解析 token 数量值。

    支持格式：
      - "1m" / "1M" → 1,000,000
      - "500k" / "500K" → 500,000
      - "8192" → 8192

    Returns:
        解析后的整数值，无效时返回 None
    """
    raw = raw.strip()
    if not raw:
        return None

    # 尝试匹配 数字+后缀（如 1m, 500k）
    suffix_match = re.match(r'^(\d+(?:\.\d+)?)\s*([a-zA-Z])$', raw)
    if suffix_match:
        num_str, suffix = suffix_match.group(1), suffix_match.group(2).lower()
        multiplier = TOKEN_SUFFIXES.get(suffix)
        if multiplier is None:
            return None
        try:
            return int(float(num_str) * multiplier)
        except (ValueError, OverflowError):
            return None

    # 尝试纯数字
    try:
        value = int(raw)
        if value < 0:
            return None
        return value
    except ValueError:
        return None


def _parse_thinking_value(raw: str) -> ThinkingConfig:
    """
    解析 thinking 后缀值。

    解析优先级（与 CLIProxyAPI 一致）：
    1. 特殊值：none → 禁用, auto/-1 → 自动
    2. 等级名：minimal/low/medium/high/xhigh/max
    3. 数值：正整数 → budget, 0 → 禁用

    Returns:
        ThinkingConfig 实例
    """
    normalized = raw.strip().lower()

    # 1. 尝试特殊值
    if normalized in SPECIAL_VALUES:
        mode = SPECIAL_VALUES[normalized]
        if mode == ThinkingMode.NONE:
            return ThinkingConfig(mode=ThinkingMode.NONE, budget=0)
        if mode == ThinkingMode.AUTO:
            return ThinkingConfig(mode=ThinkingMode.AUTO, budget=-1)

    # 2. 尝试等级名
    if normalized in VALID_LEVELS:
        return ThinkingConfig(mode=ThinkingMode.LEVEL, level=normalized)

    # 3. 尝试数值
    try:
        value = int(raw.strip())
        if value == 0:
            return ThinkingConfig(mode=ThinkingMode.NONE, budget=0)
        if value > 0:
            return ThinkingConfig(mode=ThinkingMode.BUDGET, budget=value)
    except ValueError:
        pass

    # 无法解析，返回空配置
    logger.debug(f"模型后缀 '{raw}' 无法识别为有效的 thinking 配置，忽略")
    return ThinkingConfig()


def apply_suffix_to_request(
    request_data: dict,
    suffix_result: ModelSuffixResult,
    source_format: str,
    target_format: str,
) -> dict:
    """
    将解析出的后缀配置应用到请求数据中。

    处理逻辑：
    1. 将 model 替换为 base_model（去掉后缀部分）
    2. 如果有 thinking 配置，注入到请求参数中（根据目标格式）
    3. 如果有 max_tokens 配置，注入到请求参数中（根据目标格式）

    后缀配置优先级高于请求 body 中的同名参数（与 CLIProxyAPI 行为一致）。

    Args:
        request_data: 请求数据（会被原地修改）
        suffix_result: 模型后缀解析结果
        source_format: 来源格式（openai/anthropic/gemini）
        target_format: 目标格式
    Returns:
        修改后的 request_data
    """
    if not isinstance(request_data, dict):
        return request_data

    # 替换 model 为 base_model
    if suffix_result.base_model != suffix_result.original_model:
        request_data["model"] = suffix_result.base_model
        logger.info(
            f"模型名后缀解析: {suffix_result.original_model} → base_model={suffix_result.base_model}"
        )

    # 应用 thinking 配置（后缀优先级高于 body）
    if suffix_result.thinking.has_config:
        _apply_thinking_config(request_data, suffix_result.thinking, source_format)

    # 应用 max_tokens 配置
    if suffix_result.max_tokens is not None:
        _apply_max_tokens(request_data, suffix_result.max_tokens, source_format)

    return request_data


def _apply_thinking_config(
    request_data: dict,
    thinking: ThinkingConfig,
    source_format: str,
):
    """
    将 thinking 配置注入到请求数据中。

    根据 source_format 使用对应的字段名：
    - openai: reasoning_effort
    - anthropic: thinking.type + thinking.budget_tokens
    - gemini: generationConfig.thinkingConfig
    """
    if thinking.mode == ThinkingMode.NONE:
        # 禁用 thinking：移除相关参数
        if source_format == "openai":
            request_data.pop("reasoning_effort", None)
        elif source_format == "anthropic":
            request_data.pop("thinking", None)
        elif source_format == "gemini":
            gen_config = request_data.get("generationConfig", {})
            gen_config.pop("thinkingConfig", None)
        logger.info("后缀配置: 禁用 thinking")

    elif thinking.mode == ThinkingMode.AUTO:
        if source_format == "openai":
            request_data["reasoning_effort"] = "high"
        elif source_format == "anthropic":
            request_data["thinking"] = {"type": "enabled", "budget_tokens": 32000}
        elif source_format == "gemini":
            gen_config = request_data.setdefault("generationConfig", {})
            gen_config["thinkingConfig"] = {"thinkingBudget": -1, "includeThoughts": True}
        logger.info("后缀配置: 自动 thinking")

    elif thinking.mode == ThinkingMode.LEVEL:
        # 等级模式：根据目标格式映射
        level = thinking.level
        if source_format == "openai":
            # OpenAI 只支持 low/medium/high，需要做映射
            effort_map = {
                "minimal": "low",
                "low": "low",
                "medium": "medium",
                "high": "high",
                "xhigh": "xhigh",
                "max": "xhigh",
            }
            request_data["reasoning_effort"] = effort_map.get(level, "high")
        elif source_format == "anthropic":
            # 等级映射到 budget_tokens
            budget_map = {
                "minimal": 1024,
                "low": 2048,
                "medium": 8192,
                "high": 16384,
                "xhigh": 24576,
                "max": 32000,
            }
            budget = budget_map.get(level, 16384)
            request_data["thinking"] = {"type": "enabled", "budget_tokens": budget}
        elif source_format == "gemini":
            gen_config = request_data.setdefault("generationConfig", {})
            gen_config["thinkingConfig"] = {
                "thinkingLevel": level.upper(),
                "includeThoughts": True,
            }
        logger.info(f"后缀配置: thinking level = {level}")

    elif thinking.mode == ThinkingMode.BUDGET:
        budget = thinking.budget
        if source_format == "openai":
            # 数值 budget 映射到 reasoning_effort
            if budget <= 2048:
                request_data["reasoning_effort"] = "low"
            elif budget <= 16384:
                request_data["reasoning_effort"] = "medium"
            else:
                request_data["reasoning_effort"] = "high"
        elif source_format == "anthropic":
            request_data["thinking"] = {"type": "enabled", "budget_tokens": budget}
        elif source_format == "gemini":
            gen_config = request_data.setdefault("generationConfig", {})
            gen_config["thinkingConfig"] = {
                "thinkingBudget": budget,
                "includeThoughts": True,
            }
        logger.info(f"后缀配置: thinking budget = {budget}")


def _apply_max_tokens(request_data: dict, max_tokens: int, source_format: str):
    """
    将 max_tokens 配置注入到请求数据中。

    根据 source_format 使用对应的字段名：
    - openai: max_tokens 或 max_completion_tokens
    - anthropic: max_tokens
    - gemini: generationConfig.maxOutputTokens
    """
    if source_format == "openai":
        # 如果请求中有 reasoning_effort（推理模型），使用 max_completion_tokens
        if "reasoning_effort" in request_data:
            request_data["max_completion_tokens"] = max_tokens
        else:
            request_data["max_tokens"] = max_tokens
    elif source_format == "anthropic":
        request_data["max_tokens"] = max_tokens
    elif source_format == "gemini":
        gen_config = request_data.setdefault("generationConfig", {})
        gen_config["maxOutputTokens"] = max_tokens

    logger.info(f"后缀配置: max_tokens = {max_tokens}")
