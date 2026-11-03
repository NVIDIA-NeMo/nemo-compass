# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cache-miss detector.

Finds every cache miss once (``misses.cache_misses``) and runs each cause rule on it.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator

from compass.compass_signals.cost_efficiency.cards import Finding
from compass.compass_signals.cost_efficiency.prompt_caching_miss import tool_change
from compass.compass_signals.cost_efficiency.prompt_caching_miss.misses import (
    cache_misses,
    cache_ttl,
)
from compass.compass_signals.cost_efficiency.prompt_caching_miss.sequences import build_sequences
from compass.traces import TraceSnapshot

# Each cause rule compares one part of a missed call's span with its reference's and returns
# the change, or None. Every rule runs on every miss, so one miss can feed several.
CAUSES = ((tool_change.RULE, tool_change.check),)
SPECS = (tool_change.SPEC,)


def detect(snapshot: TraceSnapshot) -> Iterator[Finding]:
    """One Finding per cache miss and cause rule that explains it."""

    for sequence in build_sequences(snapshot):
        for miss, (rule, check) in itertools.product(cache_misses(sequence), CAUSES):
            call, reference = miss.call.span, miss.reference.span
            change = check(call, reference)
            if change is None:
                continue
            assert reference.token_counts is not None
            yield Finding(
                rule=rule,
                kind=change.kind,
                trace_id=miss.call.trace_id,
                session=sequence.session,
                session_id=sequence.session_id,
                span_ids=(call.id,),
                model=call.model,
                tokens=miss.reference_cached_tokens,
                change_summary=change.change_summary,
                reference_trace_id=miss.reference.trace_id,
                reference_span_id=reference.id,
                details={
                    "sequence": sequence.key,
                    "cache_age_s": _seconds(call.start_time, reference.start_time),
                    "idle_s": _seconds(call.start_time, reference.end_time),
                    "ttl_s": cache_ttl(reference.token_counts),
                },
            )


def _seconds(later, earlier) -> float | None:  # noqa: ANN001
    if later is None or earlier is None:
        return None
    return (later - earlier).total_seconds()


__all__ = ["CAUSES", "SPECS", "detect"]
