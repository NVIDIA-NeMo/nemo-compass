# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve declared tool definitions from canonical traces."""

from __future__ import annotations

import json
from collections.abc import Mapping

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

    agents: list[Span] = []
    calls: list[Span] = []
    pending = list(trace.root_spans)
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
        legacy = trace.attributes.get("tool_catalog")
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


__all__ = ["tool_catalog"]
