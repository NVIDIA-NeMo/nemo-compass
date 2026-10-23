# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CE-A3: a call reads none of the cache its reference call left.

Each call ``n`` of an agent sequence is compared with its reference ``r``, the
nearest earlier call that ended before ``n`` started (``CallSequence.reference``).
With ``cr`` the cache read and ``cw`` the cache write of a call, ``r`` left
``prefix(r) = cr(r) + cw(r)`` tokens in the cache. ``n`` misses when that cache is
still within its TTL and ``n`` reads none of it: ``cr(n) = 0``.

A call is skipped, not reported, when its data cannot rule out an innocent miss:

- ``n`` failed, or ``cr(n)`` is unknown;
- the TTL or either call's start time is unknown. Only the request that wrote the
  cache records its TTL; with several, the shortest counts;
- the cache is older than its TTL. An expired cache misses whatever changed, so no
  cause rule could explain it;
- either call's total input is unknown, or ``n`` sends less input than ``r``: a
  compaction or a new request;
- ``cr(r)`` or ``cw(r)`` is unknown, or ``prefix(r) = 0``: nothing known was cached;
- ``cr(n) > 0``: ``n`` read part of the cache.

A miss says nothing about its cause; cause rules such as ``tool_change`` explain it.
"""

from __future__ import annotations

from dataclasses import dataclass

from compass.compass_signals._trace import SpanVisit
from compass.compass_signals.cost_efficiency.prompt_caching_miss.sequences import CallSequence
from compass.traces import TokenCounts


@dataclass(frozen=True)
class CacheMiss:
    call: SpanVisit
    reference: SpanVisit
    reference_cached_tokens: int  # the reference's cache read + cache write, prefix(r)


def cache_misses(sequence: CallSequence) -> list[CacheMiss]:
    """Every call of ``sequence`` that missed the cache its reference left, in call order.

    Calls without a reference, such as the first call, are never misses.
    """

    misses = []
    for position in range(1, len(sequence.calls)):
        reference = sequence.reference(position)
        if reference is not None and (miss := _miss(sequence.calls[position], reference)):
            misses.append(miss)
    return misses


def _miss(call: SpanVisit, reference: SpanVisit) -> CacheMiss | None:
    """The miss of ``call`` against ``reference``, or None when the module rules skip it."""

    usage, reference_usage = call.span.token_counts, reference.span.token_counts
    if call.span.error is not None or usage is None or usage.cached_input_tokens is None:
        return None  # failed, or cache read unknown
    if reference_usage is None:
        return None  # reference() only returns calls with token counts
    ttl = cache_ttl(reference_usage)
    if ttl is None or call.span.start_time is None or reference.span.start_time is None:
        return None  # expiry cannot be ruled out
    if (call.span.start_time - reference.span.start_time).total_seconds() > ttl:
        return None  # expired
    total, reference_total = _total_input(usage), _total_input(reference_usage)
    if total is None or reference_total is None:
        return None  # input sizes cannot be compared
    if total < reference_total:
        return None  # compaction or a shorter request, not a miss
    if reference_usage.cached_input_tokens is None or reference_usage.cache_write_tokens is None:
        return None  # what the reference left in the cache is unknown
    if reference_usage.cached_input_tokens + reference_usage.cache_write_tokens == 0:
        return None  # nothing was cached
    if usage.cached_input_tokens != 0:
        return None  # the call read some of the cache
    return CacheMiss(
        call,
        reference,
        reference_usage.cached_input_tokens + reference_usage.cache_write_tokens,
    )


def cache_ttl(usage: TokenCounts) -> int | None:
    """Shortest TTL in seconds the call wrote cache tokens with; None when it recorded none."""

    ttls = [ttl for ttl, tokens in (usage.cache_writes_by_ttl or {}).items() if tokens > 0]
    return min(ttls, default=None)


def _total_input(usage: TokenCounts) -> int | None:
    """All input, cache included, as the source totals it, else summed from its parts."""

    if usage.input_total is not None:
        return usage.input_total
    if usage.input_tokens is None:
        return None
    return usage.input_tokens + (usage.cached_input_tokens or 0) + (usage.cache_write_tokens or 0)


__all__ = ["CacheMiss", "cache_misses", "cache_ttl"]
