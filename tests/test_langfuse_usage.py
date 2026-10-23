# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

from trace_ingest import nemo_relay
from trace_ingest.loaders.langfuse import langfuse_sdk
from trace_ingest.otel import openinference

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


def relay_counts(route: str | None, usage: dict) -> dict | None:
    attributes = {"nemo_relay.scope_type": "llm", "nemo_relay.end.data.usage": json.dumps(usage)}
    counts = nemo_relay.token_counts({"attributes": attributes}, route)
    return None if counts is None else counts.model_dump(exclude_none=True, exclude={"source"})


def test_anthropic_input_excludes_cache():
    assert relay_counts("anthropic.messages", ANTHROPIC) == {
        "input_tokens": 3020,
        "cached_input_tokens": 0,
        "output_tokens": 56,
        "cache_write_tokens": 19491,
        "cache_writes_by_ttl": {300: 19491, 3600: 0},
    }


def test_openai_input_includes_cache():
    assert relay_counts("openai.responses", OPENAI) == {
        "input_total": 11409,
        "cached_input_tokens": 3712,
        "output_tokens": 87,
        "cache_write_tokens": 0,
    }


@pytest.mark.parametrize(
    ("route", "usage", "expected"),
    [
        # A missing count is unknown, not zero.
        (
            "anthropic.messages",
            {"input_tokens": 900, "output_tokens": 5},
            {"input_tokens": 900, "output_tokens": 5},
        ),
        ("openai.responses", {"input_tokens": 900}, {"input_total": 900}),
        # Counts that disagree with the route's convention are kept as recorded.
        (
            "openai.responses",
            {"input_tokens": 4, "input_tokens_details": {"cached_tokens": 2170}},
            {"input_total": 4, "cached_input_tokens": 2170},
        ),
        ("anthropic.count_tokens", ANTHROPIC, None),
        (None, OPENAI, None),
    ],
)
def test_relay_usage_edge_cases(route, usage, expected):
    assert relay_counts(route, usage) == expected


OI_COUNTS = {"llm.token_count.prompt": 900, "llm.token_count.completion": 5}


@pytest.mark.parametrize(
    ("route", "usage", "source", "reason"),
    [
        ("anthropic.messages", ANTHROPIC, "nemo_relay", None),
        # No provider usage: OpenInference is the only source, not a fallback.
        ("anthropic.messages", None, "openinference", None),
        (
            "anthropic.messages",
            {"input_tokens": "n/a"},
            "openinference",
            "unreadable_provider_usage",
        ),
        ("anthropic.count_tokens", ANTHROPIC, "openinference", "unsupported_route"),
    ],
)
def test_relay_counts_name_their_source(route, usage, source, reason):
    attributes: dict = {"nemo_relay.scope_type": "llm", **OI_COUNTS}
    if usage is not None:
        attributes["nemo_relay.end.data.usage"] = json.dumps(usage)
    counts = nemo_relay.token_counts({"attributes": attributes}, route)
    assert counts is not None
    assert (counts.source, counts.fallback_reason) == (source, reason)


def oi_counts(**token_count: object) -> dict | None:
    attributes = {f"llm.token_count.{key.replace('__', '.')}": v for key, v in token_count.items()}
    counts = openinference.token_counts({"attributes": attributes})
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


SDK_METADATA = {"scope": {"name": "langfuse-sdk"}}


def sdk_counts(usage: dict | None) -> dict | None:
    counts = langfuse_sdk.token_counts(SDK_METADATA, usage)
    return None if counts is None else counts.model_dump(exclude_none=True, exclude={"source"})


# usageDetails as Langfuse's Claude Code and Codex integrations recorded them in
# testbed/langfuse-hook runs.
@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (
            {
                "input": 2,
                "output": 52,
                "cache_read_input_tokens": 22509,
                "input_cache_creation_5m": 298,
                "total": 22861,
            },
            {
                "input_tokens": 2,
                "cached_input_tokens": 22509,
                "output_tokens": 52,
                "cache_writes_by_ttl": {300: 298},
            },
        ),
        # First call: the integration leaves out the zero cache read, so it stays unknown.
        (
            {"input": 2, "output": 81, "input_cache_creation_5m": 14000, "total": 14083},
            {"input_tokens": 2, "output_tokens": 81, "cache_writes_by_ttl": {300: 14000}},
        ),
        # Codex: with reasoning split out, ``output`` alone is not the output count.
        (
            {
                "input": 724,
                "output": 52,
                "total": 12436,
                "input_cached_tokens": 11648,
                "output_reasoning_tokens": 12,
            },
            {"input_tokens": 724, "cached_input_tokens": 11648},
        ),
        ({"output": 4}, {"output_tokens": 4}),
        (None, None),
    ],
)
def test_langfuse_sdk_usage_details(usage, expected):
    assert sdk_counts(usage) == expected


@pytest.mark.parametrize(
    "metadata", [{"scope": {"name": "opentelemetry"}}, {}, None, "not a mapping"]
)
def test_langfuse_sdk_reads_only_its_own_observations(metadata):
    assert langfuse_sdk.token_counts(metadata, {"input": 10, "output": 4}) is None
