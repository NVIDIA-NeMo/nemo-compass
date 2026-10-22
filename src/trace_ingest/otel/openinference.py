# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read OpenInference span attributes (``openinference.span.kind`` and ``llm.*``).

See https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md.
Any OpenInference instrumentation is read the same way, not only NeMo Relay.
Each function takes a Langfuse observation's metadata, which keeps the span
attributes under ``attributes``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pydantic import JsonValue

from trace_ingest.models import TokenCounts
from trace_ingest.token_usage import token_counts as _token_counts

_SPAN_KIND = "openinference.span.kind"


def is_openinference(metadata: object) -> bool:
    """The spec requires ``openinference.span.kind`` on every span."""

    return _SPAN_KIND in span_attributes(metadata)


def params(metadata: object) -> dict[str, JsonValue] | None:
    attributes = span_attributes(metadata)
    return request_params(json_object(attributes.get("llm.invocation_parameters")))


def token_counts(metadata: object) -> TokenCounts | None:
    attributes = span_attributes(metadata)
    # The spec counts cache reads and writes inside llm.token_count.prompt.
    return _token_counts(
        input_tokens=otel_int(attributes.get("llm.token_count.prompt")),
        output_tokens=otel_int(attributes.get("llm.token_count.completion")),
        cache_read=otel_int(attributes.get("llm.token_count.prompt_details.cache_read")),
        cache_write=otel_int(attributes.get("llm.token_count.prompt_details.cache_write")),
        input_includes_cache=True,
    )


def tools(metadata: object) -> list[JsonValue] | None:
    indexed = []
    for key, value in span_attributes(metadata).items():
        parts = key.split(".")
        if (
            len(parts) == 5
            and parts[:2] == ["llm", "tools"]
            and parts[3:] == ["tool", "json_schema"]
            and parts[2].isdigit()
        ):
            indexed.append((int(parts[2]), to_json(maybe_json(value))))
    return [tool for _, tool in sorted(indexed, key=lambda item: item[0])] or None


# Value decoding, shared with nemo_relay.

_NON_PARAM_KEYS = frozenset({"model", "messages", "input", "system", "instructions", "tools"})


def request_params(request: Mapping[str, Any] | None) -> dict[str, JsonValue] | None:
    """Top-level request parameters, without the model, tools and prompt content."""

    if request is None:
        return None
    return {key: to_json(value) for key, value in request.items() if key not in _NON_PARAM_KEYS}


def span_attributes(metadata: object) -> Mapping[str, Any]:
    """The span attributes in a Langfuse observation's metadata (``metadata.attributes``).

    Langfuse keeps an OTel span's attributes there. Readers find their own data
    by key, since users can also set metadata.attributes.
    """

    attributes = metadata.get("attributes") if isinstance(metadata, Mapping) else None
    return attributes if isinstance(attributes, Mapping) else {}


def otel_int(value: object) -> int | None:
    """An integer attribute, also when Langfuse kept it as a string or an OTLP AnyValue."""

    value = maybe_json(value)
    if isinstance(value, Mapping) and "intValue" in value:
        value = value["intValue"]
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, str):
        try:
            value = int(value)
        except ValueError:
            return None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def maybe_json(value: object) -> object:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def json_object(value: object) -> dict[str, Any] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    return dict(value) if isinstance(value, Mapping) else None


def to_json(value: object) -> JsonValue:
    return json.loads(json.dumps(value, ensure_ascii=False))


__all__ = ["is_openinference", "params", "token_counts", "tools"]
