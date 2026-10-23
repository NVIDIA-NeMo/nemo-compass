# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the spans NeMo Relay exports: its own ``nemo_relay.*`` attributes and the
OpenInference attributes it writes beside them.

Relay has written three layouts: flattened ``start.data.*``/``end.data.*``
attributes, whole ``start.data_json``/``end.data_json`` documents (older
versions), and, with ``openinference`` export, OpenInference attributes beside
the flattened ones. Only Relay records the request body, the provider's own
usage object and agent identity; the OpenInference attributes fill what a layout leaves out, such as tools and
parameters when the request body is not recorded.

Each generation route passes its provider's usage object through unchanged, so
the route decides how to read it: Anthropic Messages counts cache reads and
writes outside ``input_tokens``, OpenAI Responses counts them inside.

Each function takes a Langfuse observation's metadata, which keeps the span
attributes under ``attributes``. Where Relay's own attributes say nothing, the
OpenInference reader fills in, so these functions also read spans of other
OpenInference instrumentations.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import JsonValue

from trace_ingest.models import TokenCounts
from trace_ingest.otel import openinference
from trace_ingest.otel.openinference import (
    json_object,
    maybe_json,
    request_params,
    span_attributes,
    to_json,
)
from trace_ingest.token_usage import anthropic, openai

_SCOPE_TYPE = "nemo_relay.scope_type"
# The usage reader of each generation route.
_ROUTES = {"anthropic.messages": anthropic.token_counts, "openai.responses": openai.token_counts}


def is_relay(metadata: object) -> bool:
    return _SCOPE_TYPE in span_attributes(metadata)


def agent_id(metadata: object) -> str | None:
    """Agent ID Relay records on a span; None when it records none.

    An LLM span carries the subagent Relay attributes the call to; subagent,
    turn and tool spans carry their agent's own ID.
    """

    attributes = span_attributes(metadata)

    def relay_metadata(key: str) -> Any:  # noqa: ANN401
        # Flat layout, else the OpenInference layout.
        value = attributes.get(f"nemo_relay.start.metadata.{key}")
        return value if value is not None else attributes.get(f"openinference.metadata.{key}")

    for key in ("llm_correlation_subagent_id", "agent_id"):
        value = relay_metadata(key)
        if isinstance(value, str) and value:
            return value
    return None


def token_counts(metadata: object, route: str | None) -> TokenCounts | None:
    """Counts of one LLM span.

    ``route`` is the span name, which Relay sets to the API route. A provider
    usage object is self-describing, while llm.token_count.prompt depends on the
    exporter following the spec (July 2026 Relay builds excluded cache from it),
    so Relay's own usage wins over the OpenInference counts. When Relay recorded
    a usage object but it gives no count, the OpenInference counts say why in
    ``fallback_reason``.
    """

    usage = _response(span_attributes(metadata)).get("usage")
    if not isinstance(usage, Mapping) or not usage:
        return openinference.token_counts(metadata)
    if (own := _own_token_counts(usage, route)) is not None:
        return own.model_copy(update={"source": "nemo_relay"})
    fallback = openinference.token_counts(metadata)
    if fallback is None:
        return None
    reason = "unreadable_provider_usage" if route in _ROUTES else "unsupported_route"
    return fallback.model_copy(update={"fallback_reason": reason})


# Tools and parameters: OpenInference attributes first, the request body fills
# the rest. Exports with the request body record every request parameter,
# including the ones that affect caching (thinking, tool_choice, effort), while
# OpenInference-only exports have no body and their invocation parameters hold
# few of them.


def tools(metadata: object) -> list[JsonValue] | None:
    if (oi_tools := openinference.tools(metadata)) is not None:
        return oi_tools
    body_tools = _request(span_attributes(metadata)).get("tools")
    return [to_json(tool) for tool in body_tools] if isinstance(body_tools, list) else None


def params(metadata: object) -> dict[str, JsonValue] | None:
    if (oi_params := openinference.params(metadata)) is not None:
        return oi_params
    body = _request(span_attributes(metadata))
    return request_params(body) if body else None


def _own_token_counts(usage: Mapping[str, Any], route: str | None) -> TokenCounts | None:
    """Counts from the provider usage of a generation route."""

    reader = _ROUTES.get(route) if route else None
    return reader(usage) if reader else None


def _request(attributes: Mapping[str, Any]) -> dict[str, Any]:
    return (
        json_object(attributes.get("nemo_relay.start.data.content"))
        or json_object(attributes.get("nemo_relay.start.data_json"))
        or {}
    )


def _response(attributes: Mapping[str, Any]) -> dict[str, Any]:
    whole = json_object(attributes.get("nemo_relay.end.data_json"))
    if whole is not None:
        return whole
    prefix = "nemo_relay.end.data."
    return {
        key[len(prefix) :]: maybe_json(value)
        for key, value in attributes.items()
        if key.startswith(prefix)
    }


__all__ = ["agent_id", "is_relay", "params", "token_counts", "tools"]
