# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cost analysis's evidence contract and private detector view."""

from __future__ import annotations

from collections import defaultdict

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from compass.compass_signals._trace import walk_spans
from compass.evidence import EvidenceBatch
from compass.traces import UNSET, SpanKind, TokenCounts, TraceSnapshot
from trace_ingest.models import ToolDefinition


class CacheEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: SpanKind | None = Field(
        default=None,
        description="Recorded operation kind, including LLM calls stored as generic spans. Unknown is null.",
    )
    session_id: str | None = Field(
        default=None,
        description="Recorded conversation/session ID, never a shared user or project ID.",
    )
    agent_id: str | None = Field(
        default=None, description="Recorded agent instance ID, not an agent type or display name."
    )
    start_time: AwareDatetime | None = None
    end_time: AwareDatetime | None = None
    error: str | None = Field(
        default=None, description="Recorded failure, null when no failure is recorded."
    )
    model: str | None = None
    token_counts: TokenCounts | None = None
    tool_definitions: list[ToolDefinition] | None = Field(
        default=None,
        description="Ordered tools actually sent on this LLM call. Preserve raw definitions and names. Null means unknown; [] means explicitly no tools.",
    )

    @model_validator(mode="after")
    def consistent_accounting(self) -> CacheEvidence:
        usage = self.token_counts
        if usage is None:
            return self
        parts = (usage.input_tokens, usage.cached_input_tokens, usage.cache_write_tokens)
        if usage.input_total is not None:
            known = sum(value for value in parts if value is not None)
            if known > usage.input_total or (
                all(v is not None for v in parts) and known != usage.input_total
            ):
                raise ValueError(
                    "Input total must equal uncached + cache-read + cache-write tokens; unknown is not zero"
                )
        buckets = usage.cache_writes_by_ttl or {}
        if any(ttl <= 0 or tokens < 0 for ttl, tokens in buckets.items()):
            raise ValueError("Cache write buckets require positive TTLs and nonnegative counts")
        if (
            usage.cache_write_tokens is not None
            and sum(buckets.values()) > usage.cache_write_tokens
        ):
            raise ValueError("TTL write buckets exceed total cache writes")
        return self


INSTRUCTIONS = """Extract the evidence needed to compare consecutive LLM requests.
Identify actual model calls from the recorded operation, including generic spans.
Do not count agent/turn summaries, tool invocations or token-counting requests as
model calls merely because they contain copied usage. Return kind for recognized
non-LLM operations too, so agent boundaries remain available.
Use canonical recorded fields when available, otherwise inspect the source-native
payloads. Token field names alone do not establish their accounting semantics.
input_total includes ALL input; input_tokens excludes cache reads AND cache writes.
Inspect nested usage details before classifying counts. Keep cache reads and writes
separate; missing buckets stay unknown. Do not subtract an unknown write count as
if it were zero. TTL write maps contain observed writes by lifetime in seconds;
never infer expiration from a model name or an assumed default. Partial TTL maps
cannot establish the shortest lifetime of all writes: leave the map unknown when
a positive write bucket has an unknown lifetime. Do not estimate usage from text.
Tool definitions must be the list sent on that exact call, not executed tool calls,
an ancestor's catalog or an application's current registry. Preserve tool order,
raw JSON key order, cache-control markers, definitions and an explicitly empty list.
Do not infer a tool-change cause, avoidability, savings or a cache miss.
Leave unsupported fields null. Return None only when nothing in the contract can
be established. Session and agent IDs must be recorded, not invented for grouping.
"""


def detector_view(snapshot: TraceSnapshot, evidence: EvidenceBatch[CacheEvidence]) -> TraceSnapshot:
    """Project evidence onto a private copy; compilation retains the original snapshot."""
    traces = {trace.id: trace.model_copy(deep=True) for trace in snapshot}
    # Never fall back to unextracted counts after a record abstained.
    for trace in traces.values():
        for visit in walk_spans(trace):
            visit.span.token_counts = None
            visit.span.tool_definitions = UNSET
    sessions: dict[str, set[str]] = defaultdict(set)
    for record in evidence.records:
        value = record.value
        children = traces[record.trace_id].root_spans
        for position in record.span_path:
            span = children[position]
            children = span.children
        if value.kind is not None:
            span.kind = value.kind
        span.agent_id = value.agent_id
        if value.session_id:
            sessions[record.trace_id].add(value.session_id)
        if span.kind is SpanKind.LLM:
            span.start_time, span.end_time = value.start_time, value.end_time
            span.error, span.model = value.error, value.model
            span.token_counts = value.token_counts
            span.tool_definitions = (
                value.tool_definitions if value.tool_definitions is not None else UNSET
            )
    for trace_id, identifiers in sessions.items():
        if len(identifiers) == 1:
            traces[trace_id].attributes["logical_case_id"] = next(iter(identifiers))
        else:
            # Conflicting session evidence must not join this trace to other traces.
            traces[trace_id].attributes.pop("logical_case_id", None)
    return TraceSnapshot(traces.values())
