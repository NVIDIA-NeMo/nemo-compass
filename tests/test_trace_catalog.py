# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime

from pydantic import JsonValue

from trace_ingest.catalog import record_tool_catalogs, tool_catalog
from trace_ingest.loaders.helper.merge import merge_records, prepare_record
from trace_ingest.models import UNSET, Span, SpanKind, ToolDefinition, Trace, TraceAggregate


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
            children=[tool("t")],
        )
    )
    (call,) = to_tool_issue_trace(source).calls
    catalog = call.tool_catalog
    assert catalog == {"allow": True, "deny": False}
    assert schema_errors(catalog, "allow", {}) == []
    assert [error["type"] for error in schema_errors(catalog, "deny", {})] == [
        "json_schema_violation"
    ]


def test_merged_session_checks_each_call_against_its_own_records_tools():
    from compass.compass_signals.tool_issues.signal import detect_trace, to_tool_issue_trace

    schema: JsonValue = {"type": "object", "required": ["q"]}

    def record(record_id: str, minute: int, offered: list[str] | None, called: str) -> Trace:
        start = datetime(2026, 1, 1, 0, minute, tzinfo=UTC)
        llm = Span(
            id="llm",
            kind=SpanKind.LLM,
            start_time=start,
            end_time=start,
            tool_definitions=[definition(name, schema) for name in offered]
            if offered is not None
            else UNSET,
        )
        call = Span(id="call", kind=SpanKind.TOOL, tool_name=called, input={})
        root = Span(
            id="root",
            kind=SpanKind.AGENT,
            tool_definitions=[definition("note", schema)] if offered is None else UNSET,
            children=[llm, call],
        )
        return Trace(id=record_id, root_spans=[root], aggregate=TraceAggregate())

    # Turn 1 offers only search; turn 2 adds lookup; turn 3 declares on its agent instead.
    prepared = [
        prepare_record(record("r1", 0, ["search"], "lookup")),
        prepare_record(record("r2", 1, ["search", "lookup"], "lookup")),
        prepare_record(record("r3", 2, None, "note")),
    ]
    merged = merge_records(
        [record for record, _ in prepared],
        session_id="s",
        session_basis="test.session",
        definitions=[item for _, definitions in prepared for item in definitions],
    )

    assert tool_catalog(merged) is None
    assert [catalog for _, catalog in record_tool_catalogs(merged)] == [
        {"search": schema},
        {"search": schema, "lookup": schema},
        {"note": schema},
    ]
    findings = detect_trace(to_tool_issue_trace(merged))
    assert {
        (item["call_id"], item["issue_type"])
        for item in findings
        if item["issue_type"] != "missing_tool_result"
    } == {
        ("r1/call", "unknown_tool"),
        ("r2/call", "missing_required_argument"),
        ("r3/call", "missing_required_argument"),
    }
