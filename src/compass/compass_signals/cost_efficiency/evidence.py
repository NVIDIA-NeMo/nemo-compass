# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cost analysis's evidence contract and private detector view."""

from __future__ import annotations

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from compass.traces import UNSET, Span, SpanKind, TokenCounts, Trace
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


INSTRUCTIONS = """Identify model calls and extract their missing cost evidence:
usage, tools, timing, and identity. Existing canonical fields are reused.
Interpret native fields according to the schema; leave uncertain values unknown.
Keep evidence with the call that owns it, and preserve tool definitions verbatim
and in order. Only report cache lifetimes when all writes are accounted for.
Extract facts, not diagnoses.
"""


def enrich_trace(trace: Trace, evidence: list[tuple[Span, CacheEvidence]]) -> None:
    """Fill missing cost fields on a private trace; recorded facts take precedence."""
    sessions = {value.session_id for _, value in evidence if value.session_id}
    if trace.attributes.get("logical_case_id") is None and len(sessions) == 1:
        trace.attributes["logical_case_id"] = next(iter(sessions))
    for span, value in evidence:
        if span.kind is SpanKind.UNKNOWN and value.kind in (SpanKind.LLM, SpanKind.AGENT):
            span.kind = value.kind
        if span.agent_id is None:
            span.agent_id = value.agent_id
        if span.kind is not SpanKind.LLM:
            continue
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
