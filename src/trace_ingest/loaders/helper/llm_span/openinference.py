# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read OpenInference span attributes (``openinference.span.kind`` and ``llm.*``).

See https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md.
Any OpenInference instrumentation is read the same way, not only NeMo Relay.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import JsonValue

from trace_ingest.loaders.helper.llm_span.token_usage import token_counts as _token_counts
from trace_ingest.models import TokenCounts

if TYPE_CHECKING:
    from trace_ingest.loaders.helper.llm_span import SpanData


def model(span: SpanData) -> str | None:
    value = span.attributes.get("llm.model_name")
    return value if isinstance(value, str) and value else None


def params(span: SpanData) -> dict[str, JsonValue] | None:
    return request_params(json_object(span.attributes.get("llm.invocation_parameters")))


def token_counts(span: SpanData) -> TokenCounts | None:
    attributes = span.attributes
    # The spec counts cache reads and writes inside llm.token_count.prompt.
    return _token_counts(
        input_total=otel_int(attributes.get("llm.token_count.prompt")),
        cached_input_tokens=otel_int(attributes.get("llm.token_count.prompt_details.cache_read")),
        output_tokens=otel_int(attributes.get("llm.token_count.completion")),
        cache_write_tokens=otel_int(attributes.get("llm.token_count.prompt_details.cache_write")),
        source="openinference",
    )


def tools(span: SpanData) -> list[JsonValue] | None:
    indexed = []
    for key, value in span.attributes.items():
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


__all__ = ["model", "params", "token_counts", "tools"]
