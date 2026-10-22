# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import JsonValue

from trace_ingest.catalog import definitions_in_scope, tool_catalog
from trace_ingest.models import Span, SpanKind, ToolDefinition, Trace, TraceAggregate


def definition(name: str, schema: JsonValue = None) -> ToolDefinition:
    return ToolDefinition(name=name, input_schema=schema, raw={"name": name})


def tool(span_id: str) -> Span:
    return Span(id=span_id, kind=SpanKind.TOOL, tool_name="search")


def trace(*roots: Span, **attributes: JsonValue) -> Trace:
    return Trace(id="t", root_spans=list(roots), aggregate=TraceAggregate(), attributes=attributes)


def names(scope: dict[str, list[ToolDefinition]]) -> dict[str, list[str]]:
    return {span_id: [d.name for d in found] for span_id, found in scope.items()}


def test_declared_empty_list_is_an_empty_catalog_not_a_missing_one():
    declared = trace(Span(id="a", kind=SpanKind.AGENT, tool_definitions=[], children=[tool("t")]))
    undeclared = trace(Span(id="a", kind=SpanKind.AGENT, children=[tool("t")]))

    assert tool_catalog(declared) == {}
    assert definitions_in_scope(declared) == {"t": []}
    assert tool_catalog(undeclared) is None
    assert definitions_in_scope(undeclared) == {}


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


def test_llm_declarations_cover_sibling_tools_and_nearest_declaration_wins():
    outer = Span(
        id="chain",
        kind=SpanKind.CHAIN,
        children=[
            Span(id="g1", kind=SpanKind.LLM, tool_definitions=[definition("search")]),
            Span(id="g2", kind=SpanKind.LLM, tool_definitions=[definition("open")]),
            tool("sibling"),
            Span(
                id="agent",
                kind=SpanKind.AGENT,
                tool_definitions=[definition("inner")],
                children=[tool("nested")],
            ),
        ],
    )
    root_llm = Span(id="g0", kind=SpanKind.LLM, tool_definitions=[definition("root")])

    assert names(definitions_in_scope(trace(outer, root_llm, tool("root-tool")))) == {
        "sibling": ["search", "open"],
        "nested": ["inner"],
        "root-tool": ["root"],
    }
