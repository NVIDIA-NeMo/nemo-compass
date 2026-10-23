# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the LLM spans NeMo Relay exports: its own ``nemo_relay.*`` attributes and,
with ``openinference`` export, the OpenInference attributes it writes beside them.

Relay flattens the request and response into ``nemo_relay.start.data.*`` and
``nemo_relay.end.data.*`` attributes. With ``full`` export the request body is in
``nemo_relay.start.data.content``, and a platform may not type the span as an
LLM call; ``nemo_relay.scope_type`` says it is one. With ``openinference`` export
there is no request body, and OpenInference attributes stand beside Relay's own.

Each generation route passes its provider's usage object through unchanged, so
the route decides how to read it: Anthropic Messages counts cache reads and
writes outside ``input_tokens``, OpenAI Responses counts them inside. The usage
object is the only source of counts read here: ``llm.token_count.prompt``
excluded cache in July 2026 Relay builds, against the OpenInference spec.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import JsonValue

from trace_ingest.loaders.helper.llm_span import openinference
from trace_ingest.loaders.helper.llm_span.openinference import json_object, request_params, to_json
from trace_ingest.loaders.helper.llm_span.token_usage import anthropic, openai
from trace_ingest.models import TokenCounts

if TYPE_CHECKING:
    from trace_ingest.loaders.helper.llm_span import SpanData

_SCOPE_TYPE = "nemo_relay.scope_type"
# The usage reader of each generation route.
_ROUTES = {"anthropic.messages": anthropic.token_counts, "openai.responses": openai.token_counts}


def agent_id(span: SpanData) -> str | None:
    """Agent ID Relay records on a span; None when it records none or did not export the span.

    An LLM span carries the subagent Relay attributes the call to; subagent,
    turn and tool spans carry their agent's own ID.
    """

    attributes = span.attributes
    if _SCOPE_TYPE not in attributes:
        return None

    def relay_metadata(key: str) -> Any:  # noqa: ANN401
        # full: nemo_relay.start.metadata.*; openinference: openinference.metadata.*.
        value = attributes.get(f"nemo_relay.start.metadata.{key}")
        return value if value is not None else attributes.get(f"openinference.metadata.{key}")

    for key in ("llm_correlation_subagent_id", "agent_id"):
        value = relay_metadata(key)
        if isinstance(value, str) and value:
            return value
    return None


def token_counts(span: SpanData) -> TokenCounts | None:
    """Counts of the provider usage object, read by the route in the span name.

    None when Relay recorded no usage, such as for a failed call, or the route
    has no usage reader, such as ``anthropic.count_tokens``.
    """

    # Both exports: Relay writes the usage object in full and openinference export.
    usage = json_object(span.attributes.get("nemo_relay.end.data.usage"))
    reader = _ROUTES.get(span.name) if span.name else None
    if usage is None or reader is None:
        return None
    counts = reader(usage)
    return counts.model_copy(update={"source": "nemo_relay"}) if counts else None


def model(span: SpanData) -> str | None:
    """The model the response names."""

    # Both exports. With openinference export a platform can also take the model
    # from llm.model_name; with full export this is the only model on the span.
    value = span.attributes.get("nemo_relay.end.data.model")
    return value if isinstance(value, str) and value else None


# Tools and parameters: the request body when Relay recorded it (``full``
# export), which holds every request parameter, including the ones that affect
# caching (thinking, tool_choice, effort); else the OpenInference attributes of
# ``openinference`` export.


def tools(span: SpanData) -> list[JsonValue] | None:
    # full: the request body.
    body = json_object(span.attributes.get("nemo_relay.start.data.content"))
    if body is None:
        # openinference: llm.tools.*.
        return openinference.tools(span)
    body_tools = body.get("tools")
    return [to_json(tool) for tool in body_tools] if isinstance(body_tools, list) else None


def params(span: SpanData) -> dict[str, JsonValue] | None:
    # full: the request body; openinference: llm.invocation_parameters.
    body = json_object(span.attributes.get("nemo_relay.start.data.content"))
    return request_params(body) if body is not None else openinference.params(span)


__all__ = ["agent_id", "model", "params", "token_counts", "tools"]
