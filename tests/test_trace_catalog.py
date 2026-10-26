# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import JsonValue

from trace_ingest.catalog import tool_catalog
from trace_ingest.models import Span, SpanKind, ToolDefinition, Trace, TraceAggregate


def definition(name: str, schema: JsonValue = None) -> ToolDefinition:
    return ToolDefinition(name=name, input_schema=schema, raw={"name": name})


def tool(span_id: str) -> Span:
    return Span(id=span_id, kind=SpanKind.TOOL, tool_name="search")


def trace(*roots: Span, **attributes: JsonValue) -> Trace:
    return Trace(id="t", root_spans=list(roots), aggregate=TraceAggregate(), attributes=attributes)


def test_declared_empty_list_is_an_empty_catalog_not_a_missing_one():
    declared = trace(Span(id="a", kind=SpanKind.AGENT, tool_definitions=[], children=[tool("t")]))
    undeclared = trace(Span(id="a", kind=SpanKind.AGENT, children=[tool("t")]))

    assert tool_catalog(declared) == {}
    assert tool_catalog(undeclared) is None


def test_legacy_attribute_catalog_applies_only_without_span_declarations():
    legacy = {"search": {"type": "object"}}

    assert tool_catalog(trace(tool("t"), tool_catalog=legacy)) == legacy
    declared = Span(
        id="a",
        kind=SpanKind.AGENT,
        tool_definitions=[definition("open")],
        children=[tool("t")],
    )
    assert tool_catalog(trace(declared, tool_catalog=legacy)) == {"open": None}


def test_unknown_entries_disable_catalog_instead_of_becoming_unknown_tools():
    from compass.compass_signals.tool_issues.signal import detect_trace, to_tool_issue_trace

    source = trace(
        Span(
            id="a",
            kind=SpanKind.AGENT,
            tool_definitions=[definition("search"), ToolDefinition(raw={"type": "unrecognized"})],
            children=[Span(id="call", kind=SpanKind.TOOL, tool_name="other", input={})],
        )
    )
    assert tool_catalog(source) is None
    assert not any(
        item["issue_type"] == "unknown_tool" for item in detect_trace(to_tool_issue_trace(source))
    )


def test_boolean_schemas_reach_tool_validation():
    from compass.compass_signals.tool_issues.signal import schema_errors, to_tool_issue_trace

    source = trace(
        Span(
            id="a",
            kind=SpanKind.AGENT,
            tool_definitions=[definition("allow", True), definition("deny", False)],
        )
    )
    catalog = to_tool_issue_trace(source).tool_catalog
    assert catalog == {"allow": True, "deny": False}
    assert schema_errors(catalog, "allow", {}) == []
    assert [error["type"] for error in schema_errors(catalog, "deny", {})] == [
        "json_schema_violation"
    ]
