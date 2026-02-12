"""Tests for AnthropicConverter._clean_json_fragment.

回归测试：确保流式工具调用参数中的 JSON 转义序列不会被破坏。

背景：之前的实现会删除片段末尾的反斜杠（认为是"悬挂"的反斜杠），
但当流式分片恰好在转义序列（如 \\n）的 \\ 处截断时，这会导致转义序列
被破坏——例如 \\n 变成字面 n。
"""

import json
import pytest

from src.formats.anthropic_converter import AnthropicConverter


def _make_converter(model: str = "claude-3-5") -> AnthropicConverter:
    converter = AnthropicConverter()
    converter.set_original_model(model)
    return converter


class TestCleanJsonFragmentBasic:
    """基本行为测试。"""

    def test_empty_string_returns_empty(self):
        """空字符串应原样返回。"""
        converter = _make_converter()
        assert converter._clean_json_fragment("") == ""

    def test_none_returns_none(self):
        """None 应原样返回。"""
        converter = _make_converter()
        assert converter._clean_json_fragment(None) is None

    def test_normal_fragment_preserved(self):
        """普通 JSON 片段应原样返回。"""
        converter = _make_converter()
        fragment = '{"command": "echo hello"}'
        assert converter._clean_json_fragment(fragment) == fragment

    def test_partial_json_preserved(self):
        """不完整的 JSON 片段应原样返回。"""
        converter = _make_converter()
        fragment = '{"comma'
        assert converter._clean_json_fragment(fragment) == fragment


class TestCleanJsonFragmentEscapeSequences:
    """转义序列在截断边界的保留测试——之前 bug 的回归覆盖。"""

    def test_trailing_backslash_preserved(self):
        """末尾反斜杠不应被删除（可能是 \\n 等转义序列的前半部分）。"""
        converter = _make_converter()
        # 模拟 {"command": "echo hello\nworld"} 在 \ 处被截断
        fragment = '{"command": "echo hello\\'
        result = converter._clean_json_fragment(fragment)
        assert result == fragment
        assert result.endswith("\\")

    def test_newline_escape_split_produces_valid_json(self):
        """\\n 转义序列被分成两个片段后，拼接应产生合法 JSON。

        这是之前 bug 的核心回归测试：
        片段1: {"command": "echo hello\\     （末尾反斜杠）
        片段2: necho world"}                 （以 n 开头）
        拼接后应为合法 JSON: {"command": "echo hello\\necho world"}
        """
        converter = _make_converter()

        # 模拟流式分片在 \n 的 \ 处截断
        fragment1 = '{"command": "echo hello\\'
        fragment2 = 'necho world"}'

        cleaned1 = converter._clean_json_fragment(fragment1)
        cleaned2 = converter._clean_json_fragment(fragment2)

        combined = cleaned1 + cleaned2
        # 拼接后应该能正确解析为 JSON
        parsed = json.loads(combined)
        assert parsed["command"] == "echo hello\necho world"

    def test_tab_escape_split_produces_valid_json(self):
        """\\t 转义序列被截断后拼接应正确。"""
        converter = _make_converter()

        fragment1 = '{"text": "col1\\'
        fragment2 = 'tcol2"}'

        combined = converter._clean_json_fragment(fragment1) + converter._clean_json_fragment(fragment2)
        parsed = json.loads(combined)
        assert parsed["text"] == "col1\tcol2"

    def test_carriage_return_escape_split_produces_valid_json(self):
        """\\r 转义序列被截断后拼接应正确。"""
        converter = _make_converter()

        fragment1 = '{"text": "line1\\'
        fragment2 = 'rline2"}'

        combined = converter._clean_json_fragment(fragment1) + converter._clean_json_fragment(fragment2)
        parsed = json.loads(combined)
        assert parsed["text"] == "line1\rline2"

    def test_double_backslash_escape_split_produces_valid_json(self):
        """\\\\\\\\ 转义序列（字面反斜杠）被截断后拼接应正确。"""
        converter = _make_converter()

        # JSON 中 \\\\ 表示字面反斜杠 \
        # 如果在第一个 \ 处截断
        fragment1 = '{"path": "C:\\'
        fragment2 = '\\Users"}'

        combined = converter._clean_json_fragment(fragment1) + converter._clean_json_fragment(fragment2)
        parsed = json.loads(combined)
        assert parsed["path"] == "C:\\Users"

    def test_quote_escape_split_produces_valid_json(self):
        """\\\" 转义序列被截断后拼接应正确。"""
        converter = _make_converter()

        fragment1 = '{"text": "say \\'
        fragment2 = '"hello\\""}'

        combined = converter._clean_json_fragment(fragment1) + converter._clean_json_fragment(fragment2)
        parsed = json.loads(combined)
        assert parsed["text"] == 'say "hello"'

    def test_unicode_escape_split_produces_valid_json(self):
        """\\uXXXX 转义序列被截断后拼接应正确。"""
        converter = _make_converter()

        # \u4f60 = 你
        fragment1 = '{"text": "\\'
        fragment2 = 'u4f60好"}'

        combined = converter._clean_json_fragment(fragment1) + converter._clean_json_fragment(fragment2)
        parsed = json.loads(combined)
        assert parsed["text"] == "你好"

    def test_unicode_escape_partial_split(self):
        """\\uXXXX 在 \\u00 处截断后拼接应正确。"""
        converter = _make_converter()

        # \u0041 = A
        fragment1 = '{"text": "\\u00'
        fragment2 = '41"}'

        combined = converter._clean_json_fragment(fragment1) + converter._clean_json_fragment(fragment2)
        parsed = json.loads(combined)
        assert parsed["text"] == "A"


class TestCleanJsonFragmentMultiChunk:
    """多片段拼接的端到端测试，模拟真实流式场景。"""

    def test_realistic_bash_command_with_newlines(self):
        """模拟真实的 Bash 工具调用参数，包含多个换行符。

        这是会话中 "Claude-Code 工作流" 报错的直接复现场景。
        """
        converter = _make_converter()

        # 模拟模型输出: {"command": "git add .\ngit commit -m 'fix'\ngit push"}
        # 在 JSON 中表示: {"command": "git add .\\ngit commit -m 'fix'\\ngit push"}
        fragments = [
            '{"command": "git add .',
            "\\",      # 反斜杠单独成片
            'ngit commit -m ',
            "'fix'",
            "\\",      # 又一个反斜杠单独成片
            'ngit push"}',
        ]

        cleaned_fragments = [converter._clean_json_fragment(f) for f in fragments]
        combined = "".join(cleaned_fragments)
        parsed = json.loads(combined)

        assert parsed["command"] == "git add .\ngit commit -m 'fix'\ngit push"

    def test_file_content_with_mixed_escapes(self):
        """模拟文件编辑工具的参数，包含多种转义字符。"""
        converter = _make_converter()

        # 文件内容包含换行、制表符和引号
        # JSON: {"content": "line1\\n\\tindented\\n\\"quoted\\""}
        fragments = [
            '{"content": "line1',
            "\\",
            "n",
            "\\",
            "tindented",
            "\\",
            "n",
            "\\",
            '"quoted',
            "\\",
            '""}'
        ]

        cleaned_fragments = [converter._clean_json_fragment(f) for f in fragments]
        combined = "".join(cleaned_fragments)
        parsed = json.loads(combined)

        assert parsed["content"] == 'line1\n\tindented\n"quoted"'

    def test_gitignore_update_scenario(self):
        """复现会话中 .gitignore 更新场景：.env.local + __pycache__/

        之前的 bug 会导致 .env.local\\n__pycache__/ 变成 .env.localn__pycache__/
        """
        converter = _make_converter()

        # 模拟工具参数包含文件内容 ".env.local\n__pycache__/"
        # JSON 编码: {"content": ".env.local\\n__pycache__/"}
        fragments = [
            '{"content": ".env.local',
            "\\",        # 反斜杠在片段边界
            'n__pycache__/"}',
        ]

        cleaned_fragments = [converter._clean_json_fragment(f) for f in fragments]
        combined = "".join(cleaned_fragments)
        parsed = json.loads(combined)

        assert parsed["content"] == ".env.local\n__pycache__/"
        # 确保不是字面的 "n"
        assert "localn__pycache__" not in parsed["content"]
