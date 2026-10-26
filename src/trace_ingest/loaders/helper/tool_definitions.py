# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read recorded tool lists without filtering or deduplicating their entries."""

from __future__ import annotations

import json
from collections.abc import Mapping

from pydantic import JsonValue

from trace_ingest.models import ToolDefinition


def tool_definitions(value: object) -> list[ToolDefinition] | None:
    """None means no list was recorded; an empty list is an explicit declaration.

    Understand the function wrappers and schema spellings used by our supported
    sources. Built-ins, grouped declarations and unreadable entries remain raw.
    """

    return [_definition(item) for item in value] if isinstance(value, list) else None


def _decode(value: object) -> object:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def _definition(raw: JsonValue) -> ToolDefinition:
    value = _decode(raw)
    if not isinstance(value, Mapping):
        return ToolDefinition(raw=raw)
    nested = value.get("function")
    function = nested if isinstance(nested, Mapping) else {}
    name = value.get("name") or function.get("name")
    description = value.get("description") or function.get("description")
    schema = None
    for fields in (value, function):
        for key in ("parameters", "parameters_json_schema", "inputSchema", "input_schema"):
            if fields.get(key) is not None:
                schema = fields[key]
                break
        if schema is not None:
            break
    schema = _decode(schema)
    extra = value.get("extra")
    scope = extra.get("tool_scope") if isinstance(extra, Mapping) else None
    server = scope.get("server") if isinstance(scope, Mapping) else None
    return ToolDefinition(
        raw=raw,
        name=name if isinstance(name, str) and name else None,
        description=description if isinstance(description, str) else None,
        input_schema=schema if isinstance(schema, (dict, bool)) else None,
        server=server if isinstance(server, str) else None,
    )
