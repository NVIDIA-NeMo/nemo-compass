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


def relay_counts(route: str, usage: dict) -> tuple[int, int | None, int, int | None] | None:
    attributes = {"nemo_relay.scope_type": "llm", "nemo_relay.end.data.usage": json.dumps(usage)}
    counts = nemo_relay.token_counts({"attributes": attributes}, route)
    if counts is None:
        return None
    return (
        counts.input_tokens,
        counts.cached_input_tokens,
        counts.output_tokens,
        counts.cache_write_tokens,
    )


def test_anthropic_input_excludes_cache():
    assert relay_counts("anthropic.messages", ANTHROPIC) == (3020, 0, 56, 19491)


def test_openai_input_includes_cache():
    assert relay_counts("openai.responses", OPENAI) == (7697, 3712, 87, 0)


@pytest.mark.parametrize(
    ("route", "usage", "expected"),
    [
        # A missing cache key is unknown, not zero.
        ("anthropic.messages", {"input_tokens": 900, "output_tokens": 5}, (900, None, 5, None)),
        (
            "openai.responses",
            {
                "input_tokens": 1594,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 78,
            },
            (1594, 0, 78, None),
        ),
        ("openai.responses", {"input_tokens": 900, "output_tokens": 5}, (900, None, 5, None)),
        # Cache larger than input: the usage does not follow the route's convention.
        (
            "openai.responses",
            {
                "input_tokens": 4,
                "input_tokens_details": {"cached_tokens": 2170},
                "output_tokens": 1,
            },
            None,
        ),
        ("anthropic.messages", {"input_tokens": 1}, None),
        ("anthropic.count_tokens", ANTHROPIC, None),
        (None, OPENAI, None),
    ],
)
def test_relay_usage_edge_cases(route, usage, expected):
    assert relay_counts(route, usage) == expected


def relay_ttl(usage: dict) -> int | None:
    attributes = {"nemo_relay.scope_type": "llm", "nemo_relay.end.data.usage": json.dumps(usage)}
    counts = nemo_relay.token_counts({"attributes": attributes}, "anthropic.messages")
    assert counts is not None
    return counts.cache_write_ttl_seconds


def iterations_usage(*creations: dict) -> dict:
    """Usage as the nim-infra gateway records it: the TTL breakdown only per iteration."""

    return {
        "cache_creation_input_tokens": 4548,
        "cache_read_input_tokens": 68378,
        "input_tokens": 2,
        "iterations": [
            {
                "cache_creation": creation,
                "cache_creation_input_tokens": 4548,
                "cache_read_input_tokens": 68378,
                "input_tokens": 2,
                "output_tokens": 645,
                "type": "message",
            }
            for creation in creations
        ],
        "output_tokens": 645,
        "output_tokens_details": {"thinking_tokens": 317},
    }


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (ANTHROPIC, 300),
        (
            iterations_usage({"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 4548}),
            300,
        ),
        (
            iterations_usage({"ephemeral_1h_input_tokens": 4548, "ephemeral_5m_input_tokens": 0}),
            3600,
        ),
        # The shortest TTL written in any iteration wins.
        (
            iterations_usage(
                {"ephemeral_1h_input_tokens": 4000, "ephemeral_5m_input_tokens": 0},
                {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 548},
            ),
            300,
        ),
        # No write, no TTL.
        (iterations_usage({"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0}), None),
        ({**iterations_usage(), "iterations": "not a list"}, None),
        ({"input_tokens": 900, "output_tokens": 5, "cache_creation_input_tokens": 10}, None),
    ],
)
def test_relay_usage_ttl(usage, expected):
    assert relay_ttl(usage) == expected


@pytest.mark.parametrize(
    ("retention", "where", "expected"),
    [
        # The response echoes the request's retention; codex exports record only the response.
        ("24h", "response", 86400),
        ("24h", "request", 86400),
        # in_memory has no fixed TTL.
        ("in_memory", "response", None),
        (None, "response", None),
    ],
)
def test_relay_openai_retention_ttl(retention, where, expected):
    attributes = {"nemo_relay.scope_type": "llm", "nemo_relay.end.data.usage": json.dumps(OPENAI)}
    if retention is not None and where == "response":
        attributes["nemo_relay.end.data.prompt_cache_retention"] = retention
    if retention is not None and where == "request":
        attributes["nemo_relay.start.data.content"] = json.dumps(
            {"prompt_cache_retention": retention}
        )
    counts = nemo_relay.token_counts({"attributes": attributes}, "openai.responses")
    assert counts is not None
    assert counts.cache_write_ttl_seconds == expected


def oi_counts(**token_count: object) -> tuple[int, int | None, int, int | None] | None:
    attributes = {f"llm.token_count.{key.replace('__', '.')}": v for key, v in token_count.items()}
    counts = openinference.token_counts({"attributes": attributes})
    if counts is None:
        return None
    return (
        counts.input_tokens,
        counts.cached_input_tokens,
        counts.output_tokens,
        counts.cache_write_tokens,
    )


def test_openinference_prompt_includes_cache():
    assert oi_counts(
        prompt="23290",
        prompt_details__cache_read="19673",
        prompt_details__cache_write={"intValue": 3591},
        completion=213,
        total=23503,
    ) == (26, 19673, 213, 3591)


def test_openinference_without_cache_keys_is_cache_unknown():
    assert oi_counts(prompt=900, completion=5) == (900, None, 5, None)


def test_openinference_rejects_counts_against_the_spec():
    # July 2026 Relay builds excluded cache from llm.token_count.prompt.
    assert oi_counts(prompt=2, prompt_details__cache_read=20502, completion=40) is None
    assert oi_counts(prompt=900) is None


SDK_METADATA = {"scope": {"name": "langfuse-sdk"}}


def sdk_counts(usage: dict) -> dict | None:
    counts = langfuse_sdk.token_counts(SDK_METADATA, usage)
    return None if counts is None else counts.model_dump()


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
                "cache_write_tokens": 298,
                "cache_write_ttl_seconds": 300,
            },
        ),
        # First call: the integration leaves out the zero cache read, so it stays unknown.
        (
            {"input": 2, "output": 81, "input_cache_creation_5m": 14000, "total": 14083},
            {
                "input_tokens": 2,
                "cached_input_tokens": None,
                "output_tokens": 81,
                "cache_write_tokens": 14000,
                "cache_write_ttl_seconds": 300,
            },
        ),
        # Codex: Langfuse splits OpenAI usage; output_tokens includes reasoning.
        (
            {
                "input": 724,
                "output": 52,
                "total": 12436,
                "input_cached_tokens": 11648,
                "output_reasoning_tokens": 12,
            },
            {"input_tokens": 724, "cached_input_tokens": 11648, "output_tokens": 64},
        ),
        (
            {"input": 10, "output": 4},
            {"input_tokens": 10, "cached_input_tokens": None, "output_tokens": 4},
        ),
        ({"output": 4}, None),
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
