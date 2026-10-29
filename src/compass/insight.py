# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The typed Insight product contract and artifact loader."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, TypeAdapter

from compass.traces import Trace, TraceSnapshot
from trace_ingest.source_links import SourceURL

Rating = Literal["low", "med", "high"]


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


def _trace_url(trace: Trace) -> str | None:
    """The trace's link; a merged session trace has none, so use its first linked record's."""
    if trace.source_url is not None:
        return trace.source_url
    records = trace.attributes.get("records")
    for record in records if isinstance(records, list) else ():
        url = record.get("source_url") if isinstance(record, dict) else None
        if isinstance(url, str):
            return url
    return None


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
                url = _trace_url(trace)
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


__all__ = ["Insight", "load_insights", "resolve_trace_links"]
