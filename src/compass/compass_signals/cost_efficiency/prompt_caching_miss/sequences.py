# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Order LLM calls into (session, agent) sequences and choose reference calls.

A sequence keyed by agent ID may span several traces of one session.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from compass.compass_signals._trace import SpanVisit, session_of, walk_spans
from compass.traces import SpanKind, Trace


@dataclass(frozen=True)
class CallSequence:
    session: str  # grouping key: the source session ID, else the trace ID
    session_id: str | None  # the trace's logical_case_id, if the source recorded one
    key: str
    calls: tuple[SpanVisit, ...]  # LLM spans, ordered by start

    def reference(self, position: int) -> SpanVisit | None:
        """Nearest earlier call that ended before this one started and records its cache read."""

        start = self.calls[position].span.start_time
        if start is None:
            return None
        for index in range(position - 1, -1, -1):
            earlier = self.calls[index]
            span = earlier.span
            if span.error is not None or span.end_time is None:
                continue
            if span.token_counts is None or span.token_counts.cached_input_tokens is None:
                continue
            if span.end_time <= start:
                return earlier
        return None


def sequence_keys(trace: Trace) -> Iterator[tuple[str, SpanVisit]]:
    """Each LLM span of a trace with its sequence key, in depth-first order.

    The key is the agent ID of the call, else of its nearest AGENT span (joins traces of
    the session); else the nearest AGENT span, one agent run within this trace; else the
    session's root sequence.
    """

    agents: dict[tuple[int, ...], str] = {}  # AGENT span path -> key; parents come first
    for visit in walk_spans(trace):
        span = visit.span
        if span.kind is SpanKind.AGENT:
            agents[visit.path] = (
                f"agent_id:{span.agent_id}" if span.agent_id else f"agent_span:{trace.id}/{span.id}"
            )
        elif span.kind is SpanKind.LLM:
            # The call's own agent ID wins over where the source placed it in the tree.
            if span.agent_id:
                yield f"agent_id:{span.agent_id}", visit
                continue
            ancestors = (visit.path[:n] for n in range(len(visit.path) - 1, 0, -1))
            yield next((agents[path] for path in ancestors if path in agents), "root"), visit


def build_sequences(traces: Iterable[Trace]) -> list[CallSequence]:
    """Group timed LLM calls by (session, agent); calls without a start time cannot be ordered."""

    groups: dict[tuple[str, str], list[SpanVisit]] = defaultdict(list)
    session_ids: dict[str, str | None] = {}
    for trace in traces:
        session_id = session_of(trace)
        session = session_id or trace.id
        session_ids[session] = session_id
        for key, visit in sequence_keys(trace):
            if visit.span.start_time is not None:
                groups[(session, key)].append(visit)
    sequences = []
    for (session, key), calls in sorted(groups.items()):
        ordered = sorted(calls, key=lambda c: (c.span.start_time, c.trace_id, c.index))
        sequences.append(CallSequence(session, session_ids[session], key, tuple(ordered)))
    return sequences


__all__ = ["CallSequence", "build_sequences", "sequence_keys"]
