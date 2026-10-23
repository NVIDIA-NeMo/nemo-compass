# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build TokenCounts from source counts, copied as they are."""

from __future__ import annotations

from collections.abc import Mapping

from trace_ingest.models import TokenCounts


def count(value: object) -> int | None:
    """A token count: a non-negative integer, else None."""

    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def token_counts(
    *,
    input_total: object = None,
    input_tokens: object = None,
    cached_input_tokens: object = None,
    output_tokens: object = None,
    cache_write_tokens: object = None,
    cache_writes_by_ttl: Mapping[int, object] | None = None,
    source: str | None = None,
) -> TokenCounts | None:
    """Counts with each invalid or missing value left None; None when no count is known."""

    by_ttl = {
        ttl: tokens
        for ttl, value in (cache_writes_by_ttl or {}).items()
        if (tokens := count(value)) is not None
    }
    counts = TokenCounts(
        input_total=count(input_total),
        input_tokens=count(input_tokens),
        cached_input_tokens=count(cached_input_tokens),
        output_tokens=count(output_tokens),
        cache_write_tokens=count(cache_write_tokens),
        cache_writes_by_ttl=by_ttl or None,
    )
    if not counts.model_dump(exclude_none=True):
        return None
    return counts.model_copy(update={"source": source})


__all__ = ["count", "token_counts"]
