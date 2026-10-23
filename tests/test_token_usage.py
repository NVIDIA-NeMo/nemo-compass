# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from trace_ingest.loaders.helper.llm_span import SpanData, langfuse_sdk
from trace_ingest.loaders.helper.llm_span.openinference import otel_int
from trace_ingest.loaders.helper.llm_span.token_usage import anthropic, openai, token_counts


def counts(value) -> dict | None:
    return None if value is None else value.model_dump(exclude_none=True, exclude={"source"})


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        # Counts are copied, never derived from one another.
        (
            {"input_total": 100, "cached_input_tokens": 60, "output_tokens": 5},
            {"input_total": 100, "cached_input_tokens": 60, "output_tokens": 5},
        ),
        # A cache count larger than the total is kept as the source records it.
        (
            {"input_total": 10, "cached_input_tokens": 60},
            {"input_total": 10, "cached_input_tokens": 60},
        ),
        # An invalid count is unknown; the others stay.
        ({"input_tokens": -1, "output_tokens": 5}, {"output_tokens": 5}),
        ({"input_tokens": True, "output_tokens": None}, None),
        ({"cache_writes_by_ttl": {300: 0, 3600: -1}}, {"cache_writes_by_ttl": {300: 0}}),
    ],
)
def test_core_copies_valid_counts(kwargs, expected):
    assert counts(token_counts(**kwargs)) == expected


def test_openai_chat_completions_usage():
    usage = {
        "prompt_tokens": 1200,
        "prompt_tokens_details": {"cached_tokens": 1024},
        "completion_tokens": 30,
    }
    assert counts(openai.token_counts(usage)) == {
        "input_total": 1200,
        "cached_input_tokens": 1024,
        "output_tokens": 30,
    }


def test_openai_unknown_usage_shape():
    assert openai.token_counts({"input": 10, "output": 4}) is None


USAGE = {"input_tokens": 3, "output_tokens": 50, "cache_creation_input_tokens": 1000}


def creation(five_minutes: int, one_hour: int) -> dict:
    return {"ephemeral_5m_input_tokens": five_minutes, "ephemeral_1h_input_tokens": one_hour}


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        # Writes to both TTLs stay apart.
        ({**USAGE, "cache_creation": creation(100, 900)}, {300: 100, 3600: 900}),
        # Gateways that keep the breakdown only per iteration.
        ({**USAGE, "iterations": [{"cache_creation": creation(0, 1000)}]}, {300: 0, 3600: 1000}),
        # Several iterations would need summing, so the split stays unknown.
        (
            {**USAGE, "iterations": [{"cache_creation": creation(0, 400)}] * 2},
            None,
        ),
        (USAGE, None),
    ],
)
def test_anthropic_cache_writes_by_ttl(usage, expected):
    result = anthropic.token_counts(usage)
    assert result is not None
    assert result.cache_write_tokens == 1000
    assert result.cache_writes_by_ttl == expected


def sdk(usage: dict) -> SpanData:
    return SpanData("LLM", "GENERATION", "langfuse-sdk", {}, usage)


def test_langfuse_sdk_reads_anthropic_cache_creation_key():
    # The Claude Code hook can send Anthropic's own write key instead of Langfuse's split.
    usage = {
        "input": 3,
        "output": 50,
        "cache_read_input_tokens": 40000,
        "cache_creation_input_tokens": 21621,
    }
    assert counts(langfuse_sdk.token_counts(sdk(usage))) == {
        "input_tokens": 3,
        "cached_input_tokens": 40000,
        "output_tokens": 50,
        "cache_write_tokens": 21621,
    }


def test_langfuse_sdk_leaves_invalid_counts_unknown():
    usage = {"input": 2, "output": 5, "cache_read_input_tokens": -7, "input_cache_creation_5m": 100}
    assert counts(langfuse_sdk.token_counts(sdk(usage))) == {
        "input_tokens": 2,
        "output_tokens": 5,
        "cache_writes_by_ttl": {300: 100},
    }


def test_langfuse_sdk_unknown_output_bucket_leaves_output_unknown():
    usage = {"input": 2, "output": 10, "output_audio_tokens": 3}
    assert counts(langfuse_sdk.token_counts(sdk(usage))) == {"input_tokens": 2}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (12, 12),
        ("12", 12),
        ({"intValue": 12}, 12),
        (12.0, 12),
        (12.7, None),
        ("12.7", None),
        (True, None),
    ],
)
def test_otel_int_does_not_round(value, expected):
    assert otel_int(value) == expected
