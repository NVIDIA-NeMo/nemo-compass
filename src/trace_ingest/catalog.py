# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve declared tool definitions from canonical traces."""

from __future__ import annotations

import json
from collections.abc import Mapping

from pydantic import JsonValue

from trace_ingest.models import UNSET, Trace


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


def _canonical(value: JsonValue) -> str:
    return json.dumps(value, sort_keys=True)


__all__ = ["tool_catalog"]
