import logging

from src.formats.anthropic_converter import AnthropicConverter
from src.formats.unified.types import UnifiedChatRequest, UnifiedMessage, UnifiedContent, UnifiedContentType
from src.utils.logger import BeijingFormatter, _beijing_timestamp_iso


def _make_converter(model: str = "claude-3-5") -> AnthropicConverter:
    converter = AnthropicConverter()
    converter.set_original_model(model)
    return converter


class TestCompatibilityImprovements:
    def test_anthropic_tool_choice_maps_to_openai_shape(self):
        converter = _make_converter()
        anthropic_request = {
            "model": "claude-3-5",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "tools": [{"name": "Bash", "description": "shell", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "tool", "name": "Bash"},
        }

        result = converter.convert_request(anthropic_request, target_format="openai")
        assert result.success is True
        assert result.data["tool_choice"] == {
            "type": "function",
            "function": {"name": "Bash"},
        }

    def test_unified_to_openai_supports_parameters_json_schema(self):
        unified = UnifiedChatRequest(
            model="gpt-4.1",
            messages=[
                UnifiedMessage(
                    role="user",
                    content=[UnifiedContent(type=UnifiedContentType.TEXT, text="hi")],
                )
            ],
            tools=[
                {
                    "name": "calc",
                    "description": "calculator",
                    "input_schema": {"type": "object", "properties": {"x": {"type": "number"}}},
                }
            ],
        )

        openai_req = unified.to_openai()
        function_def = openai_req["tools"][0]["function"]
        assert function_def["parameters"] == function_def["parametersJsonSchema"]

    def test_beijing_timestamp_helper_uses_utc_plus_8(self):
        timestamp = _beijing_timestamp_iso()
        assert timestamp.endswith("+08:00")

    def test_beijing_formatter_outputs_local_time(self):
        formatter = BeijingFormatter("%(asctime)s")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )
        record.created = 0
        formatted = formatter.format(record)
        assert formatted.startswith("1970-01-01 08:00:00")
