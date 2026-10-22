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
from typing import Any, TypeVar

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
_ANTHROPIC_MESSAGES = "anthropic.messages"
_OPENAI_RESPONSES = "openai.responses"

_T = TypeVar("_T")


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
    so Relay's own usage wins over the OpenInference counts.
    """

    own = _own_token_counts(span_attributes(metadata), route)
    return _first(own, openinference.token_counts(metadata))


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


def _own_token_counts(attributes: Mapping[str, Any], route: str | None) -> TokenCounts | None:
    """Counts from the provider usage of a generation route."""

    response = _response(attributes)
    usage = response.get("usage")
    if not isinstance(usage, Mapping) or not usage:
        return None
    if route == _ANTHROPIC_MESSAGES:
        return anthropic.token_counts(usage, request=_request(attributes))
    if route == _OPENAI_RESPONSES:
        return openai.token_counts(usage, request=_request(attributes), response=response)
    return None


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


def _first(*values: _T | None) -> _T | None:
    return next((value for value in values if value is not None), None)


__all__ = ["agent_id", "is_relay", "params", "token_counts", "tools"]
