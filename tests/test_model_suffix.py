"""
模型名后缀解析器测试

测试 parse_model_suffix 和 apply_suffix_to_request 的各种场景。
"""

import sys
import os
# 确保项目根路径在 sys.path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.model_suffix import (
    parse_model_suffix,
    apply_suffix_to_request,
    ThinkingMode,
)
from src.utils.payload_config import apply_payload_config


def test_no_suffix():
    """无后缀的普通模型名"""
    result = parse_model_suffix("gpt-5.4")
    assert result.base_model == "gpt-5.4"
    assert result.original_model == "gpt-5.4"
    assert not result.has_thinking_suffix
    assert not result.has_token_suffix
    assert result.max_tokens is None
    assert not result.thinking.has_config
    print("✅ test_no_suffix passed")


def test_thinking_level_suffix():
    """thinking 等级后缀 (xhigh)"""
    result = parse_model_suffix("gpt-5.4(xhigh)")
    assert result.base_model == "gpt-5.4"
    assert result.has_thinking_suffix
    assert result.raw_thinking_suffix == "xhigh"
    assert result.thinking.mode == ThinkingMode.LEVEL
    assert result.thinking.level == "xhigh"
    assert not result.has_token_suffix
    print("✅ test_thinking_level_suffix passed")


def test_thinking_budget_suffix():
    """数值型 thinking budget (8192)"""
    result = parse_model_suffix("claude-sonnet-4(8192)")
    assert result.base_model == "claude-sonnet-4"
    assert result.has_thinking_suffix
    assert result.thinking.mode == ThinkingMode.BUDGET
    assert result.thinking.budget == 8192
    print("✅ test_thinking_budget_suffix passed")


def test_thinking_none_suffix():
    """禁用 thinking (none)"""
    result = parse_model_suffix("gpt-5.4(none)")
    assert result.base_model == "gpt-5.4"
    assert result.thinking.mode == ThinkingMode.NONE
    assert result.thinking.budget == 0
    print("✅ test_thinking_none_suffix passed")


def test_thinking_auto_suffix():
    """自动 thinking (auto)"""
    result = parse_model_suffix("gpt-5.4(auto)")
    assert result.base_model == "gpt-5.4"
    assert result.thinking.mode == ThinkingMode.AUTO
    assert result.thinking.budget == -1
    print("✅ test_thinking_auto_suffix passed")


def test_token_suffix_only():
    """仅 token 限制后缀 [1m]"""
    result = parse_model_suffix("gpt-5.4[1m]")
    assert result.base_model == "gpt-5.4"
    assert not result.has_thinking_suffix
    assert result.has_token_suffix
    assert result.max_tokens == 1_000_000
    print("✅ test_token_suffix_only passed")


def test_token_suffix_k():
    """token 限制后缀 [500k]"""
    result = parse_model_suffix("gpt-5.4[500k]")
    assert result.base_model == "gpt-5.4"
    assert result.max_tokens == 500_000
    print("✅ test_token_suffix_k passed")


def test_token_suffix_numeric():
    """纯数字 token 限制后缀 [8192]"""
    result = parse_model_suffix("gpt-5.4[8192]")
    assert result.base_model == "gpt-5.4"
    assert result.max_tokens == 8192
    print("✅ test_token_suffix_numeric passed")


def test_combined_suffix():
    """组合后缀 gpt-5.4(xhigh)[1m]"""
    result = parse_model_suffix("gpt-5.4(xhigh)[1m]")
    assert result.base_model == "gpt-5.4"
    assert result.has_thinking_suffix
    assert result.thinking.mode == ThinkingMode.LEVEL
    assert result.thinking.level == "xhigh"
    assert result.has_token_suffix
    assert result.max_tokens == 1_000_000
    print("✅ test_combined_suffix passed")


def test_combined_suffix_budget():
    """组合后缀 claude-sonnet-4(16384)[500k]"""
    result = parse_model_suffix("claude-sonnet-4(16384)[500k]")
    assert result.base_model == "claude-sonnet-4"
    assert result.thinking.mode == ThinkingMode.BUDGET
    assert result.thinking.budget == 16384
    assert result.max_tokens == 500_000
    print("✅ test_combined_suffix_budget passed")


def test_apply_to_openai():
    """将后缀应用到 OpenAI 格式请求"""
    result = parse_model_suffix("gpt-5.4(xhigh)[1m]")
    data = {"model": "gpt-5.4(xhigh)[1m]", "messages": []}
    data = apply_suffix_to_request(data, result, "openai", "openai")
    assert data["model"] == "gpt-5.4"
    assert data["reasoning_effort"] == "xhigh"  # xhigh 原样透传
    # OpenAI 推理模型使用 max_completion_tokens（因为有 reasoning_effort）
    assert data["max_completion_tokens"] == 1_000_000
    print("✅ test_apply_to_openai passed")


def test_apply_to_anthropic():
    """将后缀应用到 Anthropic 格式请求"""
    result = parse_model_suffix("claude-sonnet-4(high)[500k]")
    data = {"model": "claude-sonnet-4(high)[500k]", "messages": [], "max_tokens": 4096}
    data = apply_suffix_to_request(data, result, "anthropic", "anthropic")
    assert data["model"] == "claude-sonnet-4"
    assert data["thinking"]["type"] == "enabled"
    assert data["thinking"]["budget_tokens"] == 16384  # high 对应 16384
    assert data["max_tokens"] == 500_000  # 后缀覆盖原始值
    print("✅ test_apply_to_anthropic passed")


def test_apply_none_thinking():
    """禁用 thinking 时移除相关参数"""
    result = parse_model_suffix("gpt-5.4(none)")
    data = {"model": "gpt-5.4(none)", "reasoning_effort": "high", "messages": []}
    data = apply_suffix_to_request(data, result, "openai", "openai")
    assert data["model"] == "gpt-5.4"
    assert "reasoning_effort" not in data
    print("✅ test_apply_none_thinking passed")


def test_case_insensitive():
    """大小写不敏感"""
    result = parse_model_suffix("gpt-5.4(XHIGH)")
    assert result.thinking.level == "xhigh"

    result2 = parse_model_suffix("gpt-5.4(None)")
    assert result2.thinking.mode == ThinkingMode.NONE

    result3 = parse_model_suffix("gpt-5.4[1M]")
    assert result3.max_tokens == 1_000_000
    print("✅ test_case_insensitive passed")


def test_empty_and_none():
    """空值和 None"""
    result = parse_model_suffix("")
    assert result.base_model == ""
    assert not result.has_thinking_suffix

    result2 = parse_model_suffix(None)
    assert result2.base_model == ""
    print("✅ test_empty_and_none passed")


def test_no_valid_bracket():
    """非有效方括号内容不会被解析"""
    result = parse_model_suffix("gpt-5.4[invalid]")
    assert result.base_model == "gpt-5.4[invalid]"
    assert not result.has_token_suffix
    print("✅ test_no_valid_bracket passed")


def test_payload_config_default():
    """payload 默认值：仅在不存在时设置"""
    data = {"model": "gpt-5.4", "temperature": 0.5}
    config = {"default": {"temperature": 0.7, "max_tokens": 32000}}
    result = apply_payload_config(data, config)
    assert result["temperature"] == 0.5  # 不覆盖已存在的值
    assert result["max_tokens"] == 32000  # 设置不存在的值
    print("✅ test_payload_config_default passed")


def test_payload_config_override():
    """payload 强制覆盖"""
    data = {"model": "gpt-5.4", "temperature": 0.5}
    config = {"override": {"temperature": 0.9}}
    result = apply_payload_config(data, config)
    assert result["temperature"] == 0.9  # 强制覆盖
    print("✅ test_payload_config_override passed")


def test_payload_config_nested():
    """payload 嵌套路径"""
    data = {"model": "gpt-5.4"}
    config = {"default": {"generationConfig.temperature": 0.7}}
    result = apply_payload_config(data, config)
    assert result["generationConfig"]["temperature"] == 0.7
    print("✅ test_payload_config_nested passed")


if __name__ == "__main__":
    test_no_suffix()
    test_thinking_level_suffix()
    test_thinking_budget_suffix()
    test_thinking_none_suffix()
    test_thinking_auto_suffix()
    test_token_suffix_only()
    test_token_suffix_k()
    test_token_suffix_numeric()
    test_combined_suffix()
    test_combined_suffix_budget()
    test_apply_to_openai()
    test_apply_to_anthropic()
    test_apply_none_thinking()
    test_case_insensitive()
    test_empty_and_none()
    test_no_valid_bracket()
    test_payload_config_default()
    test_payload_config_override()
    test_payload_config_nested()
    print("\n🎉 全部测试通过！")
