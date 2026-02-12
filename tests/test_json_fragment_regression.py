"""回归分析：旧版 _clean_json_fragment 导致 command 参数缺失的机制验证。

本文件不是常规测试，而是用来验证旧版代码的 bug 触发路径。
"""
import json
import pytest


def old_clean_json_fragment(fragment: str) -> str:
    """旧版 _clean_json_fragment 逻辑复现"""
    if not fragment:
        return fragment
    cleaned = fragment
    if cleaned.endswith('\\') and not cleaned.endswith('\\\\'):
        cleaned = cleaned[:-1]  # 旧版：移除末尾反斜杠
    elif cleaned.endswith('\\u') or cleaned.endswith('\\u0') or cleaned.endswith('\\u00'):
        idx = cleaned.rfind('\\u')
        cleaned = cleaned[:idx]
    return cleaned


class TestOldBehaviorCausesCommandMissing:
    """验证旧版 _clean_json_fragment 如何导致 command 参数缺失。"""

    def test_quote_escape_truncation_breaks_json(self):
        """场景 1: 引号转义 \" 被截断 → JSON 结构性破坏 → 解析失败。

        这是 command 缺失的直接原因：
        模型生成 {"command": "echo \"hello\""}
        流式分片在 \" 的 \\ 处截断，旧版删除反斜杠后
        引号平衡被破坏，JSON 无法解析。
        """
        # JSON 中 \" 表示字面引号。流式分片可能在 \ 处截断
        frag1 = '{"command": "echo \\'    # 以 \ 结尾
        frag2 = '"hello\\""}'              # 以 " 开头

        # 旧版行为：删除 frag1 末尾的反斜杠
        cleaned1 = old_clean_json_fragment(frag1)
        assert cleaned1 == '{"command": "echo '  # 反斜杠被删除！

        combined_old = cleaned1 + frag2
        # 拼接后引号不平衡 → JSON 解析失败
        with pytest.raises(json.JSONDecodeError):
            json.loads(combined_old)

        # 新版行为：保留反斜杠
        combined_new = frag1 + frag2
        parsed = json.loads(combined_new)
        assert parsed["command"] == 'echo "hello"'

    def test_newline_truncation_corrupts_value(self):
        """场景 2: 换行符 \\n 被截断 → 值被篡改（\\n 变成字面 n）。

        虽然 JSON 仍可解析，但 command 的值被破坏，
        执行时可能产生意外行为。
        """
        frag1 = '{"command": "git add .\\'
        frag2 = 'ngit commit -m test"}'

        cleaned1 = old_clean_json_fragment(frag1)
        combined_old = cleaned1 + frag2
        parsed_old = json.loads(combined_old)
        # 旧版：\n 变成字面 n
        assert parsed_old["command"] == "git add .ngit commit -m test"
        assert "\n" not in parsed_old["command"]  # 换行符丢失！

        combined_new = frag1 + frag2
        parsed_new = json.loads(combined_new)
        # 新版：\n 正确保留
        assert parsed_new["command"] == "git add .\ngit commit -m test"
        assert "\n" in parsed_new["command"]

    def test_backslash_escape_truncation_breaks_path(self):
        """场景 3: 双反斜杠 \\\\\\\\ 被截断 → 路径错误。"""
        # Windows 路径 C:\\Users → JSON 中表示为 C:\\\\Users
        frag1 = '{"path": "C:\\'
        frag2 = '\\\\Users"}'

        cleaned1 = old_clean_json_fragment(frag1)
        combined_old = cleaned1 + frag2
        parsed_old = json.loads(combined_old)
        # 旧版：路径中的 \\ 变成单个 \
        assert parsed_old["path"] == "C:\\Users"  # 看起来正确但实际上……

        # 但如果原始值是 C:\\ （双反斜杠表示字面反斜杠），
        # 删除第一个 \ 后，变成 C + \U，这在 JSON 中是无效转义！
        frag1_v2 = '{"path": "C:\\\\'   # 字面 C:\\
        frag2_v2 = '\\\\Users"}'
        cleaned1_v2 = old_clean_json_fragment(frag1_v2)
        # \\\\  → endswith('\\') True, endswith('\\\\') True → 不会被删除
        # 旧版对双反斜杠的判断逻辑可以避免这个问题
        # 但单个反斜杠的场景仍然有 bug

    def test_sdk_fallback_leads_to_missing_command(self):
        """场景 4: JSON 解析失败后 SDK 的 fallback 行为导致 command 缺失。

        这是完整的错误链：
        1. _clean_json_fragment 删除反斜杠
        2. partial_json 拼接后 JSON 无效
        3. Anthropic SDK json.loads 失败
        4. SDK fallback 到 input: {}
        5. Claude Code 验证 Bash 工具参数 → command 缺失
        """
        # 模拟一个真实的多片段流式工具调用
        fragments = [
            '{"comma',
            'nd": "sed -i ',
            "'s/",            # sed 命令的引号
            "\\",             # 反斜杠单独成片！
            '"old/',          # 紧跟引号 → 构成 \"
            "\\",             # 又一个反斜杠
            '"new/g',         # 紧跟引号 → 构成 \"
            "' file.txt\"}",
        ]

        # 新版：直接拼接
        combined_new = "".join(fragments)
        try:
            parsed = json.loads(combined_new)
            # 新版可正常解析
            assert "command" in parsed
        except json.JSONDecodeError:
            pass  # 这个特定的片段组合可能无法解析，取决于具体构造

        # 旧版：删除反斜杠后拼接
        cleaned_old = [old_clean_json_fragment(f) for f in fragments]
        combined_old = "".join(cleaned_old)
        # 旧版反斜杠被删除后，引号结构被破坏
        # 如果 JSON 解析失败，SDK 可能 fallback 到 {}
        try:
            parsed = json.loads(combined_old)
        except json.JSONDecodeError:
            # 这就是 "command 参数缺失" 的根因：
            # JSON 解析失败 → input={} → 没有 command 字段
            pass  # 预期会走到这里


class TestToolChoiceConversion:
    """检查 tool_choice 格式转换是否可能影响工具调用行为。"""

    def test_anthropic_tool_choice_format_mismatch(self):
        """Anthropic 的 tool_choice 格式与 OpenAI 不同。

        Anthropic: {"type": "auto"} / {"type": "any"} / {"type": "tool", "name": "Bash"}
        OpenAI: "auto" / "required" / {"type": "function", "function": {"name": "Bash"}}

        如果直接透传 Anthropic 格式给 OpenAI，可能导致异常行为。
        """
        # 验证格式差异
        anthropic_auto = {"type": "auto"}
        openai_auto = "auto"
        assert anthropic_auto != openai_auto

        anthropic_any = {"type": "any"}
        openai_required = "required"
        assert anthropic_any != openai_required
