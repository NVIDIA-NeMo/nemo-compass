# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI Responses and Chat Completions usage.

Both APIs count cache reads (``cached_tokens``) and, where a gateway reports
them, cache writes (``cache_write_tokens``) inside the input count, and
reasoning tokens inside the output count.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trace_ingest.models import TokenCounts
from trace_ingest.token_usage._core import token_counts as _token_counts


def token_counts(usage: Mapping[str, Any]) -> TokenCounts | None:
    """Counts of a Responses (``input_tokens``) or Chat Completions (``prompt_tokens``) usage."""

    if "input_tokens" in usage:
        input_total, output_tokens = usage.get("input_tokens"), usage.get("output_tokens")
        details = usage.get("input_tokens_details")
    elif "prompt_tokens" in usage:
        input_total, output_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
        details = usage.get("prompt_tokens_details")
    else:
        return None
    details = details if isinstance(details, Mapping) else {}
    return _token_counts(
        input_total=input_total,
        cached_input_tokens=details.get("cached_tokens"),
        output_tokens=output_tokens,
        cache_write_tokens=details.get("cache_write_tokens"),
    )


__all__ = ["token_counts"]
