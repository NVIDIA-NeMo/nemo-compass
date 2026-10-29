# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cost analysis's evidence contract and private detector view."""

from __future__ import annotations

from collections.abc import Iterator

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from compass.evidence import EvidenceBatch
from compass.traces import UNSET, SpanKind, TokenCounts, Trace, TraceSnapshot
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
Canonical recorded evidence is reused by the host. Extract only missing cost
fields from the source-native payloads; do not reconstruct or override known fields.
Return None when no missing evidence can be recovered. Never move tools from an
AGENT declaration to a child LLM call. Preserve per-call ownership.
Inspect the actual native payloads. Token field names alone do not establish their accounting semantics.
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
Leave unsupported fields null. Session and agent IDs must be recorded, not invented for grouping.
"""


def detector_view(snapshot: TraceSnapshot, evidence: EvidenceBatch[CacheEvidence]) -> TraceSnapshot:
    """Enrich fresh disk-backed traces; recorded facts always take precedence."""

    def enriched() -> Iterator[Trace]:
        records = iter(evidence.records)
        record = next(records, None)
        for trace in snapshot:
            sessions: set[str] = set()
            while record is not None and record.trace_id == trace.id:
                value = record.value
                children = trace.root_spans
                for position in record.span_path:
                    span = children[position]
                    children = span.children
                if span.kind is SpanKind.UNKNOWN and value.kind in (SpanKind.LLM, SpanKind.AGENT):
                    span.kind = value.kind
                if span.agent_id is None:
                    span.agent_id = value.agent_id
                if value.session_id:
                    sessions.add(value.session_id)
                if span.kind is SpanKind.LLM:
                    for name in ("start_time", "end_time", "error", "model"):
                        if getattr(span, name) is None:
                            setattr(span, name, getattr(value, name))
                    if value.token_counts is not None:
                        counts = span.token_counts.model_dump() if span.token_counts else {}
                        for name, count in value.token_counts.model_dump().items():
                            if counts.get(name) is None:
                                counts[name] = count
                        span.token_counts = TokenCounts.model_validate(counts)
                    if span.tool_definitions is UNSET and value.tool_definitions is not None:
                        span.tool_definitions = value.tool_definitions
                record = next(records, None)
            if trace.attributes.get("logical_case_id") is None and len(sessions) == 1:
                trace.attributes["logical_case_id"] = next(iter(sessions))
            yield trace
        if record is not None:
            raise ValueError("Evidence source order does not match the snapshot")

    return TraceSnapshot(enriched())
