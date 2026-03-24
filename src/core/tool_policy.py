"""
Tool policy helpers.

参考兼容代理的分层思路：
- converter 负责协议结构转换
- tool policy 负责 tool history 修补、诊断、schema 保留与降级/补齐决策

进一步抽象为 capability/policy pipeline：
1. collect context
2. repair tool messages
3. annotate unmatched tool calls
4. rehydrate tool schemas
5. finalize policy state
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


_DISCOVERED_TOOLS_METADATA_KEYS = (
    "discovered_tools",
    "discovered_tool_set",
    "discovered-tool-set",
)

_TOOL_SCHEMA_SNAPSHOT_KEYS = (
    "x-tool-schemas",
    "x-discovered-tool-schemas",
)

_TOOL_POLICY_STATE_KEY = "x-tool-policy"


@dataclass
class ToolPolicyContext:
    request: Dict[str, Any]
    logger: Any = None
    messages: List[Dict[str, Any]] = field(default_factory=list)
    discovered_tool_names: List[str] = field(default_factory=list)
    tool_schema_registry: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    existing_tool_names: List[str] = field(default_factory=list)
    rehydrated_tools: List[str] = field(default_factory=list)


@dataclass
class ToolPolicyPipelineResult:
    request: Dict[str, Any]
    stages: List[str] = field(default_factory=list)


def _ensure_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _ensure_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _dedupe_preserve_order(names: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for name in names:
        if not isinstance(name, str):
            continue
        normalized = name.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _extract_discovered_tools_from_metadata(metadata: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for key in _DISCOVERED_TOOLS_METADATA_KEYS:
        value = metadata.get(key)
        if isinstance(value, list):
            names.extend(str(item).strip() for item in value if isinstance(item, str))
        elif isinstance(value, dict):
            names.extend(str(item).strip() for item in value.keys() if isinstance(item, str))
    return _dedupe_preserve_order(names)


def _normalize_tool_definition(tool: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(tool, dict):
        return None

    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        function = tool["function"]
        name = str(function.get("name") or "").strip()
        if not name:
            return None
        parameters = copy.deepcopy(
            function.get("parameters")
            or function.get("parametersJsonSchema")
            or tool.get("input_schema")
            or {}
        )
        normalized = {
            "type": "function",
            "function": {
                "name": name,
                "description": function.get("description", ""),
                "parameters": parameters,
                "parametersJsonSchema": copy.deepcopy(parameters),
            },
        }
        if isinstance(tool.get("metadata"), dict):
            normalized["metadata"] = copy.deepcopy(tool["metadata"])
        return normalized

    name = str(tool.get("name") or "").strip()
    if not name:
        return None

    parameters = copy.deepcopy(
        tool.get("input_schema")
        or tool.get("parameters")
        or tool.get("parametersJsonSchema")
        or {}
    )
    normalized = {
        "type": "function",
        "function": {
            "name": name,
            "description": tool.get("description", ""),
            "parameters": parameters,
            "parametersJsonSchema": copy.deepcopy(parameters),
        },
    }
    if isinstance(tool.get("metadata"), dict):
        normalized["metadata"] = copy.deepcopy(tool["metadata"])
    return normalized


def _tool_name_from_definition(tool: Dict[str, Any]) -> Optional[str]:
    function = _ensure_dict(tool.get("function"))
    name = function.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def _load_tool_schema_registry(request_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    registry: Dict[str, Dict[str, Any]] = {}

    for key in _TOOL_SCHEMA_SNAPSHOT_KEYS:
        snapshot = request_data.get(key)
        if isinstance(snapshot, dict):
            for name, raw_tool in snapshot.items():
                normalized = _normalize_tool_definition(raw_tool)
                if normalized:
                    registry[name] = normalized
        elif isinstance(snapshot, list):
            for raw_tool in snapshot:
                normalized = _normalize_tool_definition(raw_tool)
                if normalized:
                    name = _tool_name_from_definition(normalized)
                    if name:
                        registry[name] = normalized

    for raw_tool in _ensure_list(request_data.get("tools")):
        normalized = _normalize_tool_definition(raw_tool)
        if normalized:
            name = _tool_name_from_definition(normalized)
            if name:
                registry[name] = normalized

    return registry


def _collect_tool_names(request_data: Dict[str, Any], messages: List[Dict[str, Any]]) -> List[str]:
    metadata = _ensure_dict(request_data.get("metadata"))
    names: List[str] = []

    names.extend(_extract_discovered_tools_from_metadata(metadata))

    for raw_tool in _ensure_list(request_data.get("tools")):
        normalized = _normalize_tool_definition(raw_tool)
        if normalized:
            name = _tool_name_from_definition(normalized)
            if name:
                names.append(name)

    for message in messages:
        if not isinstance(message, dict):
            continue

        message_metadata = _ensure_dict(message.get("metadata"))
        names.extend(_extract_discovered_tools_from_metadata(message_metadata))

        for tool_call in _ensure_list(message.get("tool_calls")):
            function = _ensure_dict(tool_call.get("function"))
            name = function.get("name")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())

        tool_name = message.get("name")
        if isinstance(tool_name, str) and tool_name.strip():
            names.append(tool_name.strip())

    return _dedupe_preserve_order(names)


def _collect_existing_tool_names(request_data: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for raw_tool in _ensure_list(request_data.get("tools")):
        normalized = _normalize_tool_definition(raw_tool)
        if normalized:
            name = _tool_name_from_definition(normalized)
            if name:
                names.append(name)
    return _dedupe_preserve_order(names)


def _build_unmatched_tool_call_diagnostics(
    assistant_index: int,
    call_ids: List[str],
    unmatched: List[str],
    later_messages_checked: int,
    unresolved_tool_messages: int,
) -> Dict[str, Any]:
    return {
        "assistant_message_index": assistant_index,
        "original_tool_call_count": len(call_ids),
        "unmatched_tool_call_ids": unmatched,
        "later_messages_checked": later_messages_checked,
        "unresolved_tool_messages": unresolved_tool_messages,
    }


def _attach_unmatched_tool_call_metadata(
    message: Dict[str, Any],
    diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    msg_copy = copy.deepcopy(message)
    metadata = _ensure_dict(msg_copy.get("metadata"))
    tool_history = _ensure_dict(metadata.get("tool_history"))
    tool_history["unmatched_tool_calls"] = diagnostics
    metadata["tool_history"] = tool_history
    msg_copy["metadata"] = metadata
    return msg_copy


def collect_openai_tool_policy_context(request_data: Dict[str, Any], logger: Any = None) -> ToolPolicyContext:
    request_copy = copy.deepcopy(request_data) if isinstance(request_data, dict) else {}
    messages = [copy.deepcopy(message) for message in _ensure_list(request_copy.get("messages"))]
    context = ToolPolicyContext(request=request_copy, logger=logger, messages=messages)
    context.tool_schema_registry = _load_tool_schema_registry(context.request)
    context.existing_tool_names = _collect_existing_tool_names(context.request)
    context.discovered_tool_names = _collect_tool_names(context.request, context.messages)
    return context


def stage_repair_tool_messages(context: ToolPolicyContext) -> ToolPolicyContext:
    pending_names_by_id: Dict[str, str] = {}
    for message in context.messages:
        if message.get("role") == "assistant":
            for tool_call in _ensure_list(message.get("tool_calls")):
                tool_call_id = tool_call.get("id")
                function_name = (_ensure_dict(tool_call.get("function")).get("name") or "").strip()
                if tool_call_id:
                    pending_names_by_id[tool_call_id] = function_name

    repaired_messages: List[Dict[str, Any]] = []
    pending_ids = list(pending_names_by_id.keys())
    patched_tool_messages = 0
    ambiguous_tool_messages = 0

    for message in context.messages:
        msg_copy = copy.deepcopy(message)
        if msg_copy.get("role") == "tool":
            tool_call_id = (msg_copy.get("tool_call_id") or "").strip()
            fallback_name = (msg_copy.get("name") or "").strip()

            if not tool_call_id and fallback_name:
                matching_ids = [
                    call_id for call_id, function_name in pending_names_by_id.items()
                    if function_name and function_name == fallback_name and call_id in pending_ids
                ]
                if len(matching_ids) == 1:
                    tool_call_id = matching_ids[0]
                    msg_copy["tool_call_id"] = tool_call_id
                    patched_tool_messages += 1
                elif len(matching_ids) > 1:
                    ambiguous_tool_messages += 1

            if not tool_call_id and len(pending_ids) == 1:
                tool_call_id = pending_ids[0]
                msg_copy["tool_call_id"] = tool_call_id
                patched_tool_messages += 1
            elif not tool_call_id and len(pending_ids) > 1:
                ambiguous_tool_messages += 1

            if tool_call_id in pending_ids:
                pending_ids.remove(tool_call_id)

        repaired_messages.append(msg_copy)

    if (patched_tool_messages or ambiguous_tool_messages) and context.logger:
        context.logger.info(
            "Tool response compatibility repair: patched_tool_messages=%s, ambiguous_tool_messages=%s",
            patched_tool_messages,
            ambiguous_tool_messages,
        )

    context.messages = repaired_messages
    return context


def stage_annotate_unmatched_tool_calls(context: ToolPolicyContext) -> ToolPolicyContext:
    validated_messages: List[Dict[str, Any]] = []
    for index, message in enumerate(context.messages):
        msg_copy = copy.deepcopy(message)
        if msg_copy.get("role") == "assistant" and msg_copy.get("tool_calls"):
            call_ids = [tool_call.get("id") for tool_call in _ensure_list(msg_copy.get("tool_calls")) if tool_call.get("id")]
            unmatched = set(call_ids)

            for later in context.messages[index + 1:]:
                if later.get("role") == "tool" and later.get("tool_call_id") in unmatched:
                    unmatched.discard(later["tool_call_id"])
                if not unmatched:
                    break

            if unmatched:
                unresolved_tool_messages = [
                    later for later in context.messages[index + 1:]
                    if later.get("role") == "tool" and not later.get("tool_call_id")
                ]
                diagnostics = _build_unmatched_tool_call_diagnostics(
                    assistant_index=index,
                    call_ids=call_ids,
                    unmatched=sorted(unmatched),
                    later_messages_checked=len(context.messages) - index - 1,
                    unresolved_tool_messages=len(unresolved_tool_messages),
                )
                if context.logger:
                    context.logger.warning(
                        "Unmatched tool_call IDs preserved for policy layer: %s; assistant_message_index=%s, original_tool_call_count=%s, later_messages_checked=%s, unresolved_tool_messages=%s",
                        diagnostics["unmatched_tool_call_ids"],
                        diagnostics["assistant_message_index"],
                        diagnostics["original_tool_call_count"],
                        diagnostics["later_messages_checked"],
                        diagnostics["unresolved_tool_messages"],
                    )
                msg_copy = _attach_unmatched_tool_call_metadata(msg_copy, diagnostics)

        validated_messages.append(msg_copy)

    context.messages = validated_messages
    return context


def stage_refresh_discovered_tools(context: ToolPolicyContext) -> ToolPolicyContext:
    context.discovered_tool_names = _collect_tool_names(context.request, context.messages)
    context.existing_tool_names = _collect_existing_tool_names(context.request)
    return context


def stage_rehydrate_tool_definitions(context: ToolPolicyContext) -> ToolPolicyContext:
    missing_tool_names = [
        name for name in context.discovered_tool_names
        if name in context.tool_schema_registry and name not in context.existing_tool_names
    ]

    existing_tools = _ensure_list(context.request.get("tools"))
    if missing_tool_names:
        context.request["tools"] = existing_tools + [copy.deepcopy(context.tool_schema_registry[name]) for name in missing_tool_names]
        if context.logger:
            context.logger.info(
                "Tool policy rehydrated missing tool definitions: %s",
                missing_tool_names,
            )
    elif existing_tools:
        context.request["tools"] = existing_tools

    context.rehydrated_tools = _dedupe_preserve_order(context.rehydrated_tools + missing_tool_names)
    context.existing_tool_names = _collect_existing_tool_names(context.request)
    return context


def stage_finalize_tool_policy_state(context: ToolPolicyContext) -> ToolPolicyContext:
    context.request["messages"] = context.messages

    metadata = _ensure_dict(context.request.get("metadata"))
    if context.discovered_tool_names:
        metadata["discovered_tools"] = context.discovered_tool_names
        context.request["metadata"] = metadata

    if context.tool_schema_registry:
        context.request[_TOOL_SCHEMA_SNAPSHOT_KEYS[0]] = copy.deepcopy(context.tool_schema_registry)

    tool_policy_state = _ensure_dict(context.request.get(_TOOL_POLICY_STATE_KEY))
    prior_rehydrated_tools = _dedupe_preserve_order(_ensure_list(tool_policy_state.get("rehydrated_tools")))
    merged_rehydrated_tools = _dedupe_preserve_order(prior_rehydrated_tools + context.rehydrated_tools)

    if context.discovered_tool_names or context.tool_schema_registry or context.rehydrated_tools or prior_rehydrated_tools:
        tool_policy_state.update({
            "discovered_tool_count": len(context.discovered_tool_names),
            "schema_registry_size": len(context.tool_schema_registry),
            "rehydrated_tools": merged_rehydrated_tools,
        })
        context.request[_TOOL_POLICY_STATE_KEY] = tool_policy_state

    return context


def run_openai_tool_policy_pipeline(request_data: Dict[str, Any], logger: Any = None) -> ToolPolicyPipelineResult:
    context = collect_openai_tool_policy_context(request_data, logger=logger)
    stages: List[str] = ["collect_context"]

    for stage_name, stage_fn in (
        ("repair_tool_messages", stage_repair_tool_messages),
        ("annotate_unmatched_tool_calls", stage_annotate_unmatched_tool_calls),
        ("refresh_discovered_tools", stage_refresh_discovered_tools),
        ("rehydrate_tool_definitions", stage_rehydrate_tool_definitions),
        ("finalize_tool_policy_state", stage_finalize_tool_policy_state),
    ):
        context = stage_fn(context)
        stages.append(stage_name)

    return ToolPolicyPipelineResult(request=context.request, stages=stages)


def repair_openai_tool_messages(messages: List[Dict[str, Any]], logger: Any = None) -> List[Dict[str, Any]]:
    context = ToolPolicyContext(request={}, logger=logger, messages=[copy.deepcopy(message) for message in messages])
    return stage_repair_tool_messages(context).messages


def annotate_unmatched_openai_tool_calls(messages: List[Dict[str, Any]], logger: Any = None) -> List[Dict[str, Any]]:
    context = ToolPolicyContext(request={}, logger=logger, messages=[copy.deepcopy(message) for message in messages])
    return stage_annotate_unmatched_tool_calls(context).messages


def apply_openai_request_tool_policy(request_data: Dict[str, Any], logger: Any = None) -> Dict[str, Any]:
    """在 OpenAI 请求侧统一处理 tool history、schema 快照与缺失工具定义补齐。"""
    return run_openai_tool_policy_pipeline(request_data, logger=logger).request
