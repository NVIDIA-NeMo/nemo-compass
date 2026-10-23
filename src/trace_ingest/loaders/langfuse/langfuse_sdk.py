# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the observations Langfuse SDK integrations create.

Langfuse SDKs from v3 on export through OTel under the ``langfuse-sdk``
instrumentation scope, and Langfuse keeps their usage only in its own
``usageDetails``. The keys read here are those of Langfuse's Claude Code and
Codex integrations, checked against the harnesses' own usage in
``testbed/langfuse-hook``: ``input`` excludes cache reads and writes, the Claude
Code integration sends Anthropic's cache keys, and Langfuse splits OpenAI usage
into ``input_cached_tokens`` and ``output_reasoning_tokens``.
"""

from __future__ import annotations

from collections.abc import Mapping

from trace_ingest.models import TokenCounts
from trace_ingest.token_usage import token_counts as _token_counts

# Langfuse's split of Anthropic's cache writes by TTL.
_WRITE_BY_TTL = {"input_cache_creation_5m": 300, "input_cache_creation_1h": 3600}
_INPUT_KEYS = frozenset({"input", "input_cached_tokens", *_WRITE_BY_TTL})


def is_langfuse_sdk(metadata: object) -> bool:
    """Whether an observation's metadata names the ``langfuse-sdk`` instrumentation scope."""

    scope = metadata.get("scope") if isinstance(metadata, Mapping) else None
    return isinstance(scope, Mapping) and scope.get("name") == "langfuse-sdk"


def token_counts(
    metadata: object, usage_details: Mapping[str, object] | None
) -> TokenCounts | None:
    """Counts of an observation a Langfuse SDK integration created; None for any other."""

    if not is_langfuse_sdk(metadata):
        return None
    usage = usage_details or {}
    # Langfuse keeps each usage bucket apart; with a bucket not listed here,
    # ``input`` or ``output`` alone is not the whole count, so it stays unknown.
    input_complete = {key for key in usage if key.startswith("input")} <= _INPUT_KEYS
    output_complete = {key for key in usage if key.startswith("output")} <= {"output"}
    cached = usage.get("cache_read_input_tokens")
    return _token_counts(
        input_tokens=usage.get("input") if input_complete else None,
        cached_input_tokens=cached if cached is not None else usage.get("input_cached_tokens"),
        output_tokens=usage.get("output") if output_complete else None,
        cache_write_tokens=usage.get("cache_creation_input_tokens"),
        cache_writes_by_ttl={ttl: usage.get(key) for key, ttl in _WRITE_BY_TTL.items()},
        source="langfuse_sdk",
    )


__all__ = ["is_langfuse_sdk", "token_counts"]
