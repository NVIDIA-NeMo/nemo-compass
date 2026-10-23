# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

from trace_ingest.loaders.helper.llm_span import (
    SpanData,
    langfuse_sdk,
    nemo_relay,
    openinference,
    select_reader,
)


def span(
    name: str | None = None,
    attributes: dict | None = None,
    usage_details: dict | None = None,
    *,
    scope: str | None = "opentelemetry",
    span_type: str = "GENERATION",
) -> SpanData:
    return SpanData(name, span_type, scope, attributes or {}, usage_details or {})


# Usage objects as Relay recorded them in testbed runs.
ANTHROPIC = {
    "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 19491},
    "cache_creation_input_tokens": 19491,
    "cache_read_input_tokens": 0,
    "cost": 0.038175975,
    "input_tokens": 3020,
    "output_tokens": 56,
    "output_tokens_details": {"thinking_tokens": 0},
    "service_tier": "standard",
}
OPENAI = {
    "cost": 0.010736144999999999,
    "input_tokens": 11409,
    "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 3712},
    "output_tokens": 87,
    "output_tokens_details": {"reasoning_tokens": 25},
    "total_tokens": 11496,
}


OI_COUNTS = {"llm.token_count.prompt": 900, "llm.token_count.completion": 5}


@pytest.mark.parametrize(
    ("route", "usage"),
    [
        # A failed call: Relay records no usage.
        ("anthropic.messages", None),
        ("anthropic.messages", {"input_tokens": "n/a"}),
        ("anthropic.count_tokens", ANTHROPIC),
        (None, OPENAI),
    ],
)
def test_relay_counts_only_its_provider_usage(route, usage):
    # llm.token_count.prompt excluded cache in some Relay builds, so it is not read.
    attributes: dict = {"nemo_relay.scope_type": "llm", **OI_COUNTS}
    if usage is not None:
        attributes["nemo_relay.end.data.usage"] = json.dumps(usage)
    assert nemo_relay.token_counts(span(route, attributes)) is None


def oi_counts(**token_count: object) -> dict | None:
    attributes = {f"llm.token_count.{key.replace('__', '.')}": v for key, v in token_count.items()}
    counts = openinference.token_counts(span(attributes=attributes))
    return None if counts is None else counts.model_dump(exclude_none=True, exclude={"source"})


def test_openinference_prompt_includes_cache():
    assert oi_counts(
        prompt="23290",
        prompt_details__cache_read="19673",
        prompt_details__cache_write={"intValue": 3591},
        completion=213,
        total=23503,
    ) == {
        "input_total": 23290,
        "cached_input_tokens": 19673,
        "output_tokens": 213,
        "cache_write_tokens": 3591,
    }


def test_openinference_without_cache_keys_is_cache_unknown():
    assert oi_counts(prompt=900) == {"input_total": 900}
    assert oi_counts(total=900) is None


def test_openinference_params_are_only_what_it_records():
    def params(invocation: dict) -> dict | None:
        attributes = {"llm.invocation_parameters": json.dumps(invocation)}
        return openinference.params(span(attributes=attributes))

    assert params({"model": "m", "max_tokens": 10}) == {"max_tokens": 10}
    # Only the model recorded: the parameters are unknown, not all defaults.
    assert params({"model": "m"}) is None


def sdk_counts(usage: dict | None) -> dict | None:
    counts = langfuse_sdk.token_counts(span(usage_details=usage, scope="langfuse-sdk"))
    return None if counts is None else counts.model_dump(exclude_none=True, exclude={"source"})


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        # First call: the integration leaves out the zero cache read, so it stays unknown.
        (
            {"input": 2, "output": 81, "input_cache_creation_5m": 14000, "total": 14083},
            {"input_tokens": 2, "output_tokens": 81, "cache_writes_by_ttl": {300: 14000}},
        ),
        ({"output": 52, "output_reasoning_tokens": -1}, None),
        # An unsupported input bucket leaves input unknown and keeps the other counts.
        ({"input": 8, "input_audio_tokens": 2, "output": 5}, {"output_tokens": 5}),
        ({"output": 4}, {"output_tokens": 4}),
        (None, None),
    ],
)
def test_langfuse_sdk_usage_edge_cases(usage, expected):
    assert sdk_counts(usage) == expected


RELAY_USAGE = {"nemo_relay.end.data.usage": json.dumps(ANTHROPIC)}


# One span per layout found in Langfuse, by its attribute keys.
@pytest.mark.parametrize(
    ("data", "reader"),
    [
        # Relay full export: Langfuse types the call a plain span.
        (
            span(
                "anthropic.messages",
                {"nemo_relay.scope_type": "llm", **RELAY_USAGE},
                span_type="SPAN",
            ),
            nemo_relay,
        ),
        # Relay openinference export, with OpenInference attributes beside its own.
        (
            span(
                "anthropic.messages",
                {"nemo_relay.scope_type": "llm", "openinference.span.kind": "LLM", **OI_COUNTS},
            ),
            nemo_relay,
        ),
        # Turn and tool spans carry a copy of a call's usage; they are not calls.
        (
            span(
                "claude-code-turn",
                {"nemo_relay.scope_type": "custom", **RELAY_USAGE},
                span_type="SPAN",
            ),
            None,
        ),
        (span("Agent", {"nemo_relay.scope_type": "tool", **RELAY_USAGE}, span_type="SPAN"), None),
        (span("ChatOpenAI", {"openinference.span.kind": "LLM", **OI_COUNTS}), openinference),
        (span("chain", {"openinference.span.kind": "CHAIN"}, span_type="CHAIN"), None),
        (span("LLM Call", {"team": "a"}, {"input": 2}, scope="langfuse-sdk"), langfuse_sdk),
        (span("Tool: Read", scope="langfuse-sdk", span_type="TOOL"), None),
        # OTel GenAI (Claude Code's own telemetry) has no reader yet.
        (span("chat", {"gen_ai.operation.name": "chat"}, {"input": 2}), None),
        # No scope, such as Langfuse SDK v2 ingestion: its usage keys are not verified.
        (span("LLM", usage_details={"input": 2}, scope=None), None),
    ],
)
def test_each_layout_gets_one_reader(data, reader):
    assert select_reader(data) is reader
