# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from trace_ingest.loaders.langfuse import langfuse_sdk
from trace_ingest.otel.openinference import otel_int
from trace_ingest.token_usage import anthropic, openai, token_counts


def counts(value) -> tuple[int, int | None, int, int | None] | None:
    if value is None:
        return None
    return (
        value.input_tokens,
        value.cached_input_tokens,
        value.output_tokens,
        value.cache_write_tokens,
    )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"input_tokens": 100, "output_tokens": 5, "cache_read": 60}, (40, 60, 5, None)),
        ({"input_tokens": 10, "output_tokens": 5, "cache_read": 60}, None),
        ({"input_tokens": -1, "output_tokens": 5}, None),
        ({"input_tokens": True, "output_tokens": 5}, None),
        ({"input_tokens": 10, "output_tokens": None}, None),
    ],
)
def test_core_subtracts_cache_from_inclusive_input(kwargs, expected):
    assert counts(token_counts(**kwargs, input_includes_cache=True)) == expected


def test_openai_chat_completions_usage():
    usage = {
        "prompt_tokens": 1200,
        "prompt_tokens_details": {"cached_tokens": 1024},
        "completion_tokens": 30,
    }
    assert counts(openai.token_counts(usage)) == (176, 1024, 30, None)


def test_openai_unknown_usage_shape():
    assert openai.token_counts({"input": 10, "output": 4}) is None


USAGE = {"input_tokens": 3, "output_tokens": 50, "cache_read_input_tokens": 40000}


@pytest.mark.parametrize(
    ("request_body", "expected"),
    [
        # A breakpoint without an explicit ttl leaves the TTL unknown.
        ({"system": [{"type": "text", "cache_control": {"type": "ephemeral"}}]}, None),
        (
            {
                "tools": [{"name": "a", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
                "system": [{"type": "text", "cache_control": {"type": "ephemeral"}}],
            },
            None,
        ),
        ({"system": [{"type": "text", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]}, 3600),
        ({"system": [{"type": "text", "cache_control": {"ttl": "2d"}}]}, None),
        ({"system": "no breakpoints"}, None),
        (None, None),
    ],
)
def test_anthropic_request_ttl(request_body, expected):
    result = anthropic.token_counts(USAGE, request=request_body)
    assert result is not None
    assert result.cache_write_ttl_seconds == expected


def test_langfuse_sdk_reads_anthropic_cache_creation_key():
    # The Claude Code hook can send Anthropic's own write key instead of Langfuse's split.
    usage = {
        "input": 3,
        "output": 50,
        "cache_read_input_tokens": 40000,
        "cache_creation_input_tokens": 21621,
    }
    result = langfuse_sdk.token_counts({"scope": {"name": "langfuse-sdk"}}, usage)
    assert result is not None
    assert counts(result) == (3, 40000, 50, 21621)
    assert result.cache_write_ttl_seconds is None


def test_langfuse_sdk_leaves_invalid_counts_unknown():
    usage = {"input": 2, "output": 5, "cache_read_input_tokens": -7, "input_cache_creation_5m": 100}
    result = langfuse_sdk.token_counts({"scope": {"name": "langfuse-sdk"}}, usage)
    assert counts(result) == (2, None, 5, 100)


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
