# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve declared tool definitions from canonical traces."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from pydantic import JsonValue

from trace_ingest.models import UNSET, Span, SpanKind, Trace


def tool_catalog(trace: Trace) -> dict[str, JsonValue] | None:
    """Project recorded declarations into one catalog for trace-wide validation.

    Prefer call-level lists when any LLM records one. Every LLM must then record
    the same names and schemas; otherwise a single catalog cannot describe the
    calls. Without call-level lists, combine agent declarations as before.
    Unknown entries or conflicting schemas make the catalog unavailable.

    An explicit empty declaration yields {}. Legacy attributes are read only
    when no span declares definitions. This projection does not prove that the
    source captured every tool or establish availability for an individual call.
    """
    return _catalog(trace.root_spans, (), trace.attributes.get("tool_catalog"))


def record_tool_catalogs(trace: Trace) -> list[tuple[list[Span], dict[str, JsonValue] | None]]:
    """Pair each provider record's root spans with that record's ``tool_catalog``.

    A trace that merges a session's records (``attributes.records``, under one
    session root span) gets one catalog per record, so turns may offer different
    tools. Definitions on the session root, moved there from the records' demoted
    AGENT roots, count as every record's agent declarations. Any other trace is
    one record.
    """
    legacy = trace.attributes.get("tool_catalog")
    records = trace.attributes.get("records")
    if len(trace.root_spans) != 1 or not isinstance(records, list):
        return [(trace.root_spans, tool_catalog(trace))]
    (session,) = trace.root_spans
    children = {span.id: span for span in session.children}
    scopes: list[list[Span]] = []
    for record in records:
        ids = record.get("root_span_ids") if isinstance(record, dict) else None
        if not isinstance(ids, list) or not all(
            isinstance(span_id, str) and span_id in children for span_id in ids
        ):
            return [(trace.root_spans, tool_catalog(trace))]
        scopes.append([children[span_id] for span_id in ids])
    inherited = (session,) if session.tool_definitions is not UNSET else ()
    return [(roots, _catalog(roots, inherited, legacy)) for roots in scopes]


def _catalog(
    roots: Sequence[Span], inherited: Sequence[Span], legacy: JsonValue
) -> dict[str, JsonValue] | None:
    agents: list[Span] = list(inherited)
    calls: list[Span] = []
    pending = list(roots)
    while pending:
        span = pending.pop()
        pending.extend(span.children)
        if span.kind is SpanKind.LLM:
            calls.append(span)
        elif span.tool_definitions is not UNSET:
            agents.append(span)

    use_calls = any(span.tool_definitions is not UNSET for span in calls)
    selected = calls if use_calls else agents
    if not selected:
        return dict(legacy) if isinstance(legacy, Mapping) else None

    catalog: dict[str, JsonValue] = {}
    call_names: set[str] | None = None
    for span in selected:
        if span.tool_definitions is UNSET:
            return None
        names: set[str] = set()
        for definition in span.tool_definitions:
            name = definition.name
            if name is None:
                return None
            names.add(name)
            if name in catalog and _canonical(catalog[name]) != _canonical(definition.input_schema):
                return None
            catalog[name] = definition.input_schema
        if use_calls:
            if call_names is not None and names != call_names:
                return None
            call_names = names
    return catalog


def _canonical(value: JsonValue) -> str:
    return json.dumps(value, sort_keys=True)


__all__ = ["record_tool_catalogs", "tool_catalog"]
