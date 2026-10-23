# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the LLM fields of a span, whichever platform stored it.

A loader finds what the readers need wherever its platform keeps it
(``SpanData``); ``select_reader`` picks the one reader that knows the span's
format, from scope names, namespaced attributes and API routes; the reader
applies that format's counting rules. Each reader has the same functions:
``token_counts``, ``model``, ``tools`` and ``params``, each taking a ``SpanData``.

- ``nemo_relay``: NeMo Relay's ``nemo_relay.*`` attributes, with the
  OpenInference attributes it writes beside them.
- ``openinference``: any OpenInference instrumentation.
- ``langfuse_sdk``: Langfuse SDK integrations, whose usage only the platform
  records (``usage_details``).

To support another format, such as OTel GenAI (``gen_ai.*``), add a module with
those functions and select it here. Readers use key names, numbers and request
structure only, never message content.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import ModuleType
from typing import Any, NamedTuple

from pydantic import JsonValue

from trace_ingest.loaders.helper.llm_span import langfuse_sdk, nemo_relay, openinference
from trace_ingest.models import TokenCounts


class SpanData(NamedTuple):
    """What the readers need from one span; a platform without a field leaves it empty."""

    # The OTel span name; Relay names its LLM calls after the API route.
    name: str | None
    # The platform's type of the span, such as Langfuse's GENERATION or SPAN.
    type: str | None
    # The instrumentation scope's name.
    scope: str | None
    # The OTel span attributes with flat dotted keys, values as the platform stores them.
    attributes: Mapping[str, Any]
    # Usage the platform records itself, such as Langfuse's usageDetails.
    usage_details: Mapping[str, Any]


class LlmFields(NamedTuple):
    token_counts: TokenCounts | None
    # The model the span records; a platform may record its own.
    model: str | None
    tools: list[JsonValue] | None
    params: dict[str, JsonValue] | None


def select_reader(span: SpanData) -> ModuleType | None:
    """The reader of an LLM span's own format; None for any other span.

    Relay marks its LLM calls with ``nemo_relay.scope_type``, whatever type the
    platform gives them, and also writes OpenInference attributes beside its own,
    so it is checked first. Turn and tool spans that carry a copy of a call's
    usage are not LLM calls and get no reader. Langfuse SDK generations carry no
    OTel attributes; users can set attributes too, so the scope decides.
    """

    attributes = span.attributes
    if attributes.get("nemo_relay.scope_type") == "llm":
        return nemo_relay
    if attributes.get("openinference.span.kind") == "LLM":
        return openinference
    if span.scope == "langfuse-sdk" and span.type == "GENERATION":
        return langfuse_sdk
    return None


def normalize(span: SpanData) -> LlmFields | None:
    """The LLM fields of a span; None when it is not an LLM call any reader knows."""

    reader = select_reader(span)
    if reader is None:
        return None
    return LlmFields(
        reader.token_counts(span), reader.model(span), reader.tools(span), reader.params(span)
    )


__all__ = ["LlmFields", "SpanData", "normalize", "select_reader"]
