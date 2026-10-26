# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from trace_ingest.loaders.helper.tool_definitions import tool_definitions
from trace_ingest.models import UNSET, Span, SpanKind, Trace, TraceAggregate, TraceSnapshot


@pytest.mark.parametrize("kind", [SpanKind.AGENT, SpanKind.LLM])
def test_recorded_entries_survive_normalization_and_snapshot(kind):
    schema = {"type": "object", "properties": {"query": {"type": "string"}}}
    recorded = [
        {"name": "search", "input_schema": schema, "cache_control": {"type": "ephemeral"}},
        {"name": "search", "inputSchema": schema},  # Do not deduplicate names.
        json.dumps({"type": "function", "function": {"name": "read", "parameters": schema}}),
        {"function": {"name": "nested", "inputSchema": schema}},
        {"name": "deny", "parameters_json_schema": "false"},
        {"name": "allow", "parameters": True},
        {"type": "web_search"},
        {"function_declarations": [{"name": "one"}, {"name": "two"}]},
        "unreadable",
        None,
    ]
    definitions = tool_definitions(recorded)
    trace = Trace(
        id="t",
        aggregate=TraceAggregate(),
        root_spans=[Span(id="owner", kind=kind, tool_definitions=definitions)],
    )
    restored = TraceSnapshot([trace]).get_trace_by_id("t").root_spans[0]
    assert restored.tool_definitions is not UNSET
    assert [entry.raw for entry in restored.tool_definitions] == recorded
    assert [entry.name for entry in restored.tool_definitions] == [
        "search",
        "search",
        "read",
        "nested",
        "deny",
        "allow",
        None,
        None,
        None,
        None,
    ]
    assert [entry.input_schema for entry in restored.tool_definitions] == [
        schema,
        schema,
        schema,
        schema,
        False,
        True,
        None,
        None,
        None,
        None,
    ]


def test_missing_and_explicitly_empty_lists_remain_distinct():
    assert tool_definitions(None) is None
    assert tool_definitions({"tools": []}) is None
    assert tool_definitions([]) == []
    for recorded in (UNSET, []):
        span = Span(id="a", kind=SpanKind.AGENT, tool_definitions=recorded)
        restored = Span.model_validate_json(span.model_dump_json())
        assert restored.tool_definitions == recorded
