# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The typed Insight product contract and artifact loader."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import AliasChoices, BaseModel, Field, TypeAdapter

from compass.traces import TraceSnapshot
from trace_ingest.source_links import SourceURL

Rating = Literal["low", "med", "high"]
Resolution = Literal["resolved", "partial", "unresolved"]


class CommitEvidence(BaseModel):
    """One verified Git commit related to an insight."""

    commit: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    evidence_reason: str = Field(
        min_length=1,
        validation_alias=AliasChoices("evidence_reason", "why"),
        description="A concise explanation of how the commit relates to the insight.",
    )
    resolution: Resolution


class SpanEvidence(BaseModel):
    span_id: str = Field(min_length=1)
    url: SourceURL | None = Field(default=None, exclude_if=lambda value: value is None)


class TraceEvidence(BaseModel):
    trace_id: str = Field(min_length=1)
    url: SourceURL | None = Field(default=None, exclude_if=lambda value: value is None)
    spans: list[SpanEvidence] = Field(default_factory=list, exclude_if=lambda value: not value)


class Insight(BaseModel):
    """A validated, actionable problem found in the agent's traces."""

    id: str | None = Field(
        default=None,
        min_length=1,
        exclude_if=lambda value: value is None,
        description="Opaque storage ID of an existing insight. Preserve it exactly; new insights have no ID.",
    )
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    evidence: list[TraceEvidence] = Field(
        min_length=2,
        description="Supporting traces with optional relevant spans. URLs are populated by the application.",
    )
    updated_date: datetime | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="UTC timestamp of the last time a trace was added to this insight",
    )
    confidence: Rating | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="How well the codebase supports this insight's claim; set when code_base is configured.",
    )
    severity: Rating | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Observed consequence after recovery; set during final compilation without code access.",
    )
    severity_reason: str | None = Field(
        default=None,
        min_length=1,
        exclude_if=lambda value: value is None,
        description="Observed consequence that supports the severity rating, in one short sentence.",
    )
    repository_evidence: list[CommitEvidence] = Field(
        default_factory=list,
        max_length=3,
        exclude_if=lambda value: not value,
        description="Up to three verified commits related to this insight.",
    )


def resolve_trace_links(
    insights: Sequence[Insight], snapshot: TraceSnapshot, existing: Sequence[Insight] = ()
) -> list[Insight]:
    """Validate span references and attach only loader or previously saved URLs."""
    known_urls = {}
    known_spans: dict[str, dict[str, str | None]] = {}
    for insight in existing:
        for item in insight.evidence:
            if item.url is not None:
                known_urls[item.trace_id] = item.url
            known_spans.setdefault(item.trace_id, {}).update(
                (span.span_id, span.url) for span in item.spans
            )
    result = []
    for insight in insights:
        evidence = []
        for item in insight.evidence:
            try:
                trace = snapshot.get_trace_by_id(item.trace_id)
            except KeyError:
                url = known_urls.get(item.trace_id)
                available = known_spans.get(item.trace_id, {})
            else:
                url = trace.source_url
                available = {}
                pending = list(trace.root_spans)
                while pending:
                    span = pending.pop()
                    pending.extend(span.children)
                    available[span.id] = span.source_url
            evidence.append(
                TraceEvidence(
                    trace_id=item.trace_id,
                    url=url,
                    spans=[
                        SpanEvidence(span_id=sid, url=available[sid])
                        for sid in dict.fromkeys(span.span_id for span in item.spans)
                        if sid in available
                    ],
                )
            )
        result.append(insight.model_copy(update={"evidence": evidence}))
    return result


_INSIGHTS_ADAPTER = TypeAdapter(list[Insight])


def load_insights(path: Path) -> list[Insight]:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    return _INSIGHTS_ADAPTER.validate_python(payload)


__all__ = [
    "CommitEvidence",
    "Insight",
    "load_insights",
    "resolve_trace_links",
]
