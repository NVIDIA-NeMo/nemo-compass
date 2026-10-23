# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the observations Langfuse SDK integrations create.

Langfuse SDKs from v3 on export through OTel under the ``langfuse-sdk``
instrumentation scope, and Langfuse keeps their usage only in its own
``usageDetails``. The keys read here are those of Langfuse's Claude Code and
Codex integrations, checked against the harnesses' own usage in
``testbed/langfuse-hook``: ``input`` excludes cache reads and writes, the Claude
Code integration sends Anthropic's cache keys, and Langfuse splits OpenAI usage
into ``input_cached_tokens`` and ``output_reasoning_tokens``. The buckets do not
overlap: they add up to ``total``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from pydantic import JsonValue

from trace_ingest.loaders.helper.llm_span.token_usage import token_counts as _token_counts
from trace_ingest.loaders.helper.llm_span.token_usage._core import count
from trace_ingest.models import TokenCounts

if TYPE_CHECKING:
    from trace_ingest.loaders.helper.llm_span import SpanData

# Keys by integration, as testbed/langfuse-hook runs recorded them:
#   both:                 input, output
#   Claude (Anthropic):   cache_read_input_tokens, input_cache_creation_5m/1h,
#                         cache_creation_input_tokens
#   Codex (OpenAI):       input_cached_tokens, output_reasoning_tokens

# Anthropic: Langfuse's split of Anthropic's cache writes by TTL.
_WRITE_BY_TTL = {"input_cache_creation_5m": 300, "input_cache_creation_1h": 3600}
# Both: output; OpenAI: output_reasoning_tokens, which Langfuse splits out of output.
_OUTPUT_KEYS = frozenset({"output", "output_reasoning_tokens"})


def token_counts(span: SpanData) -> TokenCounts | None:
    usage = span.usage_details
    # Anthropic sends its own cache read key; OpenAI's is split out as input_cached_tokens.
    cached = usage.get("cache_read_input_tokens")
    return _token_counts(
        # Both: input is uncached input, with cache split out.
        input_tokens=usage.get("input"),
        cached_input_tokens=cached if cached is not None else usage.get("input_cached_tokens"),
        # Both: output, plus OpenAI's reasoning bucket.
        output_tokens=_output_tokens(usage),
        # Anthropic only: the total write key, and the writes by TTL.
        cache_write_tokens=usage.get("cache_creation_input_tokens"),
        cache_writes_by_ttl={ttl: usage.get(key) for key, ttl in _WRITE_BY_TTL.items()},
        source="langfuse_sdk",
    )


def model(span: SpanData) -> str | None:
    """The integrations record the model only where the platform keeps it."""

    return None


def tools(span: SpanData) -> list[JsonValue] | None:
    """The integrations record no tool list."""

    return None


def params(span: SpanData) -> dict[str, JsonValue] | None:
    """The integrations record no request parameters."""

    return None


def _output_tokens(usage: Mapping[str, object]) -> int | None:
    """The sum of the output buckets; None when one is unknown or invalid.

    Anthropic sends ``output`` alone; OpenAI adds ``output_reasoning_tokens``.
    """

    keys = {key for key in usage if key.startswith("output")}
    if "output" not in keys or not keys <= _OUTPUT_KEYS:
        return None
    counts = [count(usage[key]) for key in keys]
    return None if None in counts else sum(c for c in counts if c is not None)


__all__ = ["model", "params", "token_counts", "tools"]
