# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI Responses and Chat Completions usage and prompt cache TTL.

Both APIs count cache reads (``cached_tokens``) and, where a gateway reports
them, cache writes (``cache_write_tokens``) inside the input count, and
reasoning tokens inside the output count.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trace_ingest.models import TokenCounts
from trace_ingest.token_usage._core import token_counts as _token_counts

# OpenAI's in_memory retention has no documented fixed TTL, so only 24h maps.
_RETENTION_TTL_SECONDS = {"24h": 86400}


def token_counts(
    usage: Mapping[str, Any],
    *,
    request: Mapping[str, Any] | None = None,
    response: Mapping[str, Any] | None = None,
) -> TokenCounts | None:
    """Counts of a Responses (``input_tokens``) or Chat Completions (``prompt_tokens``) usage."""

    if "input_tokens" in usage:
        input_tokens, output_tokens = usage.get("input_tokens"), usage.get("output_tokens")
        details = usage.get("input_tokens_details")
    elif "prompt_tokens" in usage:
        input_tokens, output_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
        details = usage.get("prompt_tokens_details")
    else:
        return None
    details = details if isinstance(details, Mapping) else {}
    return _token_counts(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read=details.get("cached_tokens"),
        cache_write=details.get("cache_write_tokens"),
        input_includes_cache=True,
        ttl=ttl(request, response),
    )


def ttl(request: Mapping[str, Any] | None, response: Mapping[str, Any] | None = None) -> int | None:
    """TTL of ``prompt_cache_retention``, from the request or its echo in the response."""

    retention = (request or {}).get("prompt_cache_retention") or (response or {}).get(
        "prompt_cache_retention"
    )
    return _RETENTION_TTL_SECONDS.get(str(retention)) if retention else None


__all__ = ["token_counts", "ttl"]
