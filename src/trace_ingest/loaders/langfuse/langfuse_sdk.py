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
from trace_ingest.token_usage import anthropic
from trace_ingest.token_usage import token_counts as _token_counts
from trace_ingest.token_usage._core import count

_ANTHROPIC_READ = "cache_read_input_tokens"
_ANTHROPIC_WRITE = "cache_creation_input_tokens"
# Langfuse's split of Anthropic's cache writes, keyed to Anthropic's cache_creation breakdown.
_WRITE_BY_TTL = {
    "input_cache_creation_5m": "ephemeral_5m_input_tokens",
    "input_cache_creation_1h": "ephemeral_1h_input_tokens",
}
_ANTHROPIC_KEYS = frozenset({_ANTHROPIC_READ, _ANTHROPIC_WRITE, *_WRITE_BY_TTL})


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
    usage = {
        key: tokens
        for key, value in (usage_details or {}).items()
        if (tokens := count(value)) is not None
    }
    if "input" not in usage or "output" not in usage:
        return None
    output = usage["output"] + usage.get("output_reasoning_tokens", 0)
    if _ANTHROPIC_KEYS.isdisjoint(usage):
        # Langfuse already took OpenAI's cached tokens out of input.
        return _token_counts(
            input_tokens=usage["input"],
            output_tokens=output,
            cache_read=usage.get("input_cached_tokens"),
            input_includes_cache=False,
        )
    # A key Langfuse leaves out stays unknown, also where the integration omits zeros.
    creation = {
        key: usage[langfuse_key]
        for langfuse_key, key in _WRITE_BY_TTL.items()
        if langfuse_key in usage
    }
    write = usage.get(_ANTHROPIC_WRITE, sum(creation.values()) if creation else None)
    return anthropic.token_counts(
        {
            "input_tokens": usage["input"],
            "output_tokens": output,
            _ANTHROPIC_READ: usage.get(_ANTHROPIC_READ),
            _ANTHROPIC_WRITE: write,
            "cache_creation": creation,
        }
    )


__all__ = ["is_langfuse_sdk", "token_counts"]
