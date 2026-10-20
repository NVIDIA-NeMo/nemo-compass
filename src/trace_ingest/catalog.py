# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve declared tool definitions from canonical traces."""

from __future__ import annotations

import json

from pydantic import JsonValue

from trace_ingest.models import UNSET, Span, SpanKind, ToolDefinition, Trace


def definitions_in_scope(trace: Trace) -> dict[str, list[ToolDefinition]]:
    """Map each TOOL span id to the definitions of its nearest declaring ancestor.

    A declaring span is an AGENT or LLM span with ``tool_definitions``.
    """

    scoped: dict[str, list[ToolDefinition]] = {}
    pending: list[tuple[Span, list[ToolDefinition]]] = [
        (span, []) for span in reversed(trace.root_spans)
    ]
    while pending:
        span, inherited = pending.pop()
        if span.tool_definitions is not UNSET:
            inherited = list(span.tool_definitions)
        if span.kind is SpanKind.TOOL:
            scoped[span.id] = inherited
        pending.extend((child, inherited) for child in reversed(span.children))
    return scoped


def tool_catalog(trace: Trace) -> dict[str, JsonValue] | None:
    """Return ``{name: input_schema}`` over all declaring spans, or None when absent or conflicting."""

    catalog: dict[str, JsonValue] = {}
    pending = list(trace.root_spans)
    while pending:
        span = pending.pop()
        pending.extend(span.children)
        if span.tool_definitions is UNSET:
            continue
        for definition in span.tool_definitions:
            if definition.name in catalog and _canonical(catalog[definition.name]) != _canonical(
                definition.input_schema
            ):
                return None
            catalog[definition.name] = definition.input_schema
    return catalog or None


def _canonical(value: JsonValue) -> str:
    return json.dumps(value, sort_keys=True)


__all__ = ["definitions_in_scope", "tool_catalog"]
