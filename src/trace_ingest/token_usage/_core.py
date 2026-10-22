# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build TokenCounts from counts in either cache accounting."""

from __future__ import annotations

from trace_ingest.models import TokenCounts


def count(value: object) -> int | None:
    """A token count: a non-negative integer, else None."""

    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def token_counts(
    *,
    input_tokens: object,
    output_tokens: object,
    cache_read: object = None,
    cache_write: object = None,
    input_includes_cache: bool,
    ttl: int | None = None,
) -> TokenCounts | None:
    """Counts with uncached input; None when input or output is missing or the counts conflict.

    A missing cache count stays None (unknown), not zero.
    """

    input_count, output_count = count(input_tokens), count(output_tokens)
    if input_count is None or output_count is None:
        return None
    read, write = count(cache_read), count(cache_write)
    if input_includes_cache:
        input_count -= (read or 0) + (write or 0)
        if input_count < 0:
            return None
    return TokenCounts(
        input_tokens=input_count,
        cached_input_tokens=read,
        output_tokens=output_count,
        cache_write_tokens=write,
        cache_write_ttl_seconds=ttl,
    )


__all__ = ["count", "token_counts"]
