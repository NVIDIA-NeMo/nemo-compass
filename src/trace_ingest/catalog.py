# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve declared tool definitions from canonical traces."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from pydantic import JsonValue

from trace_ingest.models import UNSET, Span, SpanKind, ToolDefinition, Trace


def definitions_in_scope(trace: Trace) -> dict[str, list[ToolDefinition]]:
    """Map each TOOL span id to the definitions of its nearest declaration.

    A declaring span is an AGENT or LLM span with ``tool_definitions``. An AGENT
    declaration covers its descendants. An LLM declaration covers its own
    descendants and its siblings' subtrees, because the TOOL spans a model call
    triggers are usually recorded beside it; when several sibling LLM spans
    declare, their distinct definitions are combined. TOOL spans with no
    declaration in scope are omitted, so an empty list always means the source
    declared that no tools were available.
    """

    scoped: dict[str, list[ToolDefinition]] = {}
    pending: list[tuple[Span, list[ToolDefinition] | None]] = []

    def push_children(children: Sequence[Span], inherited: list[ToolDefinition] | None) -> None:
        offered = _offered_by_llm_siblings(children)
        scope = inherited if offered is None else offered
        pending.extend((child, scope) for child in reversed(children))

    push_children(trace.root_spans, None)
    while pending:
        span, inherited = pending.pop()
        if span.tool_definitions is not UNSET:
            inherited = list(span.tool_definitions)
        if span.kind is SpanKind.TOOL and inherited is not None:
            scoped[span.id] = inherited
        push_children(span.children, inherited)
    return scoped


def tool_catalog(trace: Trace) -> dict[str, JsonValue] | None:
    """Return ``{name: input_schema}`` over all declaring spans.

    Returns None when no span declares definitions or when two declarations give
    one name different schemas. A declared empty list yields ``{}``. Traces
    written before schema version 2 carried the catalog in
    ``attributes["tool_catalog"]``; it is used when no span declares definitions.
    """

    catalog: dict[str, JsonValue] = {}
    declared = False
    pending = list(trace.root_spans)
    while pending:
        span = pending.pop()
        pending.extend(span.children)
        if span.tool_definitions is UNSET:
            continue
        declared = True
        for definition in span.tool_definitions:
            if definition.name in catalog and _canonical(catalog[definition.name]) != _canonical(
                definition.input_schema
            ):
                return None
            catalog[definition.name] = definition.input_schema
    if declared:
        return catalog
    legacy = trace.attributes.get("tool_catalog")
    return dict(legacy) if isinstance(legacy, Mapping) else None


def _offered_by_llm_siblings(spans: Sequence[Span]) -> list[ToolDefinition] | None:
    offered: list[ToolDefinition] | None = None
    for span in spans:
        if span.kind is not SpanKind.LLM or span.tool_definitions is UNSET:
            continue
        offered = [] if offered is None else offered
        offered.extend(
            definition for definition in span.tool_definitions if definition not in offered
        )
    return offered


def _canonical(value: JsonValue) -> str:
    return json.dumps(value, sort_keys=True)


__all__ = ["definitions_in_scope", "tool_catalog"]
