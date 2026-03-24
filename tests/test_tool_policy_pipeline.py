import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from src.core.tool_policy import (
    apply_openai_request_tool_policy,
    run_openai_tool_policy_pipeline,
)


def test_run_openai_tool_policy_pipeline_reports_stage_sequence_and_repairs_tool_response():
    payload = {
        "model": "gpt-4.1",
        "metadata": {"discovered_tools": ["Edit"]},
        "x-tool-schemas": {
            "Edit": {
                "type": "function",
                "function": {
                    "name": "Edit",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "old_string": {"type": "string"},
                            "new_string": {"type": "string"},
                        },
                        "required": ["old_string", "new_string"],
                        "additionalProperties": False,
                    },
                },
            }
        },
        "messages": [
            {"role": "user", "content": "继续编辑"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_edit_pipeline_1",
                        "type": "function",
                        "function": {
                            "name": "Edit",
                            "arguments": '{"old_string":"before","new_string":"after"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "name": "Edit",
                "content": "ok",
            },
        ],
    }

    pipeline_result = run_openai_tool_policy_pipeline(payload)

    assert pipeline_result.stages == [
        "collect_context",
        "repair_tool_messages",
        "annotate_unmatched_tool_calls",
        "refresh_discovered_tools",
        "rehydrate_tool_definitions",
        "finalize_tool_policy_state",
    ]

    result = pipeline_result.request
    assert result["messages"][2]["tool_call_id"] == "call_edit_pipeline_1"
    assert result["tools"][0]["function"]["name"] == "Edit"
    assert result["metadata"]["discovered_tools"] == ["Edit"]
    assert result["x-tool-policy"]["rehydrated_tools"] == ["Edit"]


def test_apply_openai_request_tool_policy_is_idempotent_and_preserves_unmatched_diagnostics():
    payload = {
        "model": "gpt-4.1",
        "metadata": {"discovered_tools": ["Edit"]},
        "x-tool-schemas": {
            "Edit": {
                "type": "function",
                "function": {
                    "name": "Edit",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "old_string": {"type": "string"},
                            "new_string": {"type": "string"},
                        },
                        "required": ["old_string", "new_string"],
                    },
                },
            }
        },
        "messages": [
            {"role": "user", "content": "继续编辑"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_edit_pipeline_2",
                        "type": "function",
                        "function": {
                            "name": "Edit",
                            "arguments": '{"old_string":"before","new_string":"after"}',
                        },
                    }
                ],
            },
        ],
    }

    first = apply_openai_request_tool_policy(payload)
    second = apply_openai_request_tool_policy(first)

    assistant_message = second["messages"][1]
    unmatched = assistant_message["metadata"]["tool_history"]["unmatched_tool_calls"]

    assert unmatched["unmatched_tool_call_ids"] == ["call_edit_pipeline_2"]
    assert second["x-tool-policy"]["rehydrated_tools"] == ["Edit"]
    assert len(second["tools"]) == 1
    assert second["tools"][0]["function"]["name"] == "Edit"
    assert second["metadata"]["discovered_tools"] == ["Edit"]
