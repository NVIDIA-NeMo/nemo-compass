# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic inspection, evidence registration, and final compilation assembly."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from compass.compass_signals._trace import SpanVisit, walk_spans
from compass.compass_signals.compass_signals import CompassSignalResult, Problem
from compass.insight import Insight, TraceEvidence
from compass.insights_generation.evidence import (
    Decision,
    EvidenceCompletion,
    EvidenceDecision,
    apply_scan,
    merge_evidence,
    scan_completions,
)
from compass.traces import UNSET, Span, Trace, TraceSnapshot


def _text(value: object) -> str:
    if value is UNSET:
        return "<not recorded>"
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    return json.dumps(value, default=str, ensure_ascii=False)


def _excerpt(value: object, characters: int) -> dict[str, object]:
    text = _text(value)
    return {"text": text[:characters], "characters": len(text), "has_more": len(text) > characters}


def _page(rows: Sequence[Any], offset: int, limit: int) -> dict[str, Any]:
    if offset < 0 or not 1 <= limit <= 50:
        raise ValueError("Use offset >= 0 and 1 <= limit <= 50")
    end = offset + limit
    return {
        "items": list(rows[offset:end]),
        "total": len(rows),
        "next_offset": end if end < len(rows) else None,
    }


def tool_calls(trace: Trace, name: str | None = None) -> list[Span]:
    """Recorded TOOL spans in tree traversal order, not inferred chronological order."""
    return [
        visit.span
        for visit in walk_spans(trace)
        if visit.span.kind.value == "TOOL" and (name is None or visit.span.tool_name == name)
    ]


def recorded_messages(trace: Trace) -> list[dict[str, Any]]:
    """Source-recorded agent turns, preserving per-agent order and explicit authorship.

    Do not infer human authorship from role=user, deduplicate histories, or impose
    a global order across agents. Other source payloads remain available on spans.
    """
    return [
        {"agent_span_id": visit.span.id, "turn": turn.model_dump(mode="json")}
        for visit in walk_spans(trace)
        if visit.span.turns is not UNSET
        for turn in visit.span.turns
    ]


class CompilationTools:
    """Inspect bounded source facts and register checked claims for compilation.

    Helpers do not decide whether an issue is real, merge semantic claims, or
    interpret source text as instructions. Raw traces remain accessible in Python.
    """

    def __init__(
        self,
        signals: list[CompassSignalResult],
        snapshot: TraceSnapshot,
        existing: list[Insight],
        run_timestamp: datetime,
    ) -> None:
        self._signals = signals
        self._snapshot = snapshot
        self._existing = existing
        self._timestamp = run_timestamp
        self._registered: dict[str, EvidenceCompletion] = {}
        self._targets: dict[str, int | None] = {}
        identifiers = [insight.id for insight in existing if insight.id is not None]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Existing insights contain duplicate persistent IDs")

    def __repr__(self) -> str:
        return f"CompilationTools(traces={len(self._snapshot)}, candidates={sum(len(s.problems) for s in self._signals)}, existing={len(self._existing)})"

    def candidates(self, offset: int = 0, limit: int = 12) -> dict[str, Any]:
        """Page candidate descriptions, source signal, exact counts, and seed IDs.

        candidate(signal_index, problem_index) returns the full original problem.
        A supporting ID is a lead to inspect, not confirmation of the final claim.
        """
        rows = [
            {
                "signal_index": si,
                "problem_index": pi,
                "signal": signal.signal_name,
                "description": _excerpt(problem.description, 1000),
                "supporting_count": len(problem.supporting_trace_ids),
                "candidate_count": len(problem.candidate_trace_ids),
                "supporting_trace_ids": problem.supporting_trace_ids[:3],
            }
            for si, signal in enumerate(self._signals)
            for pi, problem in enumerate(signal.problems)
        ]
        return _page(rows, offset, limit)

    def signals(self, offset: int = 0, limit: int = 12) -> dict[str, Any]:
        """Page signal coverage, skip reasons, and limitations before judging candidates.

        signal(index) exposes the complete original record for targeted Python reads.
        Missing capabilities or evidence must not be mistaken for a clean result.
        """
        return _page(
            [
                {
                    "signal_index": index,
                    "name": signal.signal_name,
                    "finding_count": signal.finding_count,
                    "problem_count": len(signal.problems),
                    "skip_reason": _excerpt(signal.skip_reason, 1000),
                    "limitations": _excerpt(signal.limitations, 1000),
                }
                for index, signal in enumerate(self._signals)
            ],
            offset,
            limit,
        )

    def signal(self, signal_index: int) -> CompassSignalResult:
        """Full original signal, including capabilities, limitations, and artifacts."""
        return self._signals[signal_index]

    def candidate(self, signal_index: int, problem_index: int) -> Problem:
        """Full source problem object, usable in Python without printing it all."""
        return self._signals[signal_index].problems[problem_index]

    def artifacts(self, signal_index: int) -> object:
        """Original signal artifacts for targeted Python inspection; do not dump them."""
        return self._signals[signal_index].artifacts

    def existing(self, offset: int = 0, limit: int = 12) -> dict[str, Any]:
        """Page existing claims; all existing insights are preserved automatically."""
        return _page(
            [
                {
                    "existing_index": index,
                    "id": insight.id,
                    "name": insight.name,
                    "description": _excerpt(insight.description, 1000),
                    "severity": insight.severity,
                    "evidence_count": len(insight.evidence),
                }
                for index, insight in enumerate(self._existing)
            ],
            offset,
            limit,
        )

    def trace(self, trace_id: str) -> Trace:
        """Load the complete original trace for Python predicates or exact inspection."""
        return self._snapshot.get_trace_by_id(trace_id)

    def inspect_trace(
        self, trace_id: str, offset: int = 0, limit: int = 12, characters: int = 600
    ) -> dict[str, Any]:
        """Page source spans with bounded excerpts and exact IDs for read_span.

        Tree traversal is NOT chronological order. Recorded timestamps, tool-call
        indexes, and raw event metadata are exposed without inventing ordering.
        """
        if not 1 <= characters <= 4000:
            raise ValueError("Use 1 <= characters <= 4000")
        visits = list(walk_spans(self.trace(trace_id)))
        page = _page(visits, offset, limit)
        page["items"] = [self._span_view(visit, characters) for visit in page["items"]]
        page["ordering"] = "depth-first tree order; no global chronology inferred"
        return page

    def _span_view(self, visit: SpanVisit, characters: int) -> dict[str, Any]:
        span = visit.span
        return {
            "span_id": span.id,
            "path": visit.path,
            "kind": span.kind.value,
            "tool_name": span.tool_name,
            "start_time": span.start_time,
            "end_time": span.end_time,
            "tool_index": span.tool_call.index if span.tool_call else None,
            "source_pointer": _excerpt(visit.source_pointer, characters),
            "recorded_event": _excerpt(span.attributes.get("event", UNSET), characters),
            "input": _excerpt(span.input, characters),
            "output": _excerpt(span.output, characters),
            "error": _excerpt(span.error, characters),
        }

    def read_span(
        self,
        trace_id: str,
        span_id: str,
        field: str = "output",
        offset: int = 0,
        characters: int = 4000,
    ) -> dict[str, Any]:
        """Read an exact field in bounded chunks; next_offset exposes any remainder."""
        if offset < 0 or not 1 <= characters <= 12000:
            raise ValueError("Use offset >= 0 and 1 <= characters <= 12000")
        if field not in Span.model_fields:
            raise ValueError(f"Unknown Span field {field!r}")
        span = next(
            (v.span for v in walk_spans(self.trace(trace_id)) if v.span.id == span_id), None
        )
        if span is None:
            raise KeyError(span_id)
        text = _text(getattr(span, field))
        end = offset + characters
        return {
            "trace_id": trace_id,
            "span_id": span_id,
            "field": field,
            "text": text[offset:end],
            "characters": len(text),
            "next_offset": end if end < len(text) else None,
        }

    def register(
        self,
        insight: dict[str, Any],
        predicate: Callable[[Trace], EvidenceDecision | dict[str, object]] | None = None,
        checks: list[tuple[Trace, Decision]] | None = None,
        unresolved_reason: str | None = None,
        existing_index: int | None = None,
    ) -> str:
        """Register a final claim and return its handle after checking examples.

        insight supplies name, description, checked evidence (trace_id and optional
        spans) and optional severity/severity_reason. To update an existing claim,
        pass its existing_index from existing(), including when its id is None.
        Do not supply updated_date, confidence, or URLs.
        Pass the callable itself and match/no_match/unknown checks in Python.
        Return no predicate/checks only with a nonblank unresolved_reason.
        """
        target = self._existing_target(insight.get("id"), existing_index)
        prepared = self._prepare_insight(insight, target)
        completion = EvidenceCompletion(
            insight=prepared,
            predicate=predicate,
            checks=checks or [],
            unresolved_reason=unresolved_reason,
        )
        completion.validate_result()
        handle = uuid4().hex
        self._registered[handle] = completion
        self._targets[handle] = target
        return handle

    def _existing_target(self, identifier: object, existing_index: int | None) -> int | None:
        if existing_index is not None:
            if type(existing_index) is not int or not 0 <= existing_index < len(self._existing):
                raise ValueError("existing_index must name an insight from existing()")
            if identifier is not None and identifier != self._existing[existing_index].id:
                raise ValueError("Existing insight ID does not match existing_index")
            return existing_index
        if identifier is not None:
            for index, insight in enumerate(self._existing):
                if insight.id == identifier:
                    return index
            raise ValueError("Existing insight ID was not supplied to this compilation")
        return None

    def _prepare_insight(self, fields: dict[str, Any], target: int | None) -> Insight:
        supplied = dict(fields)
        unsupported = set(supplied) - {
            "id",
            "name",
            "description",
            "evidence",
            "severity",
            "severity_reason",
        }
        if unsupported:
            raise ValueError(f"Application-owned or unknown insight fields: {sorted(unsupported)}")
        existing = self._existing[target] if target is not None else None
        additions = self._checked_evidence(supplied.get("evidence", []), existing)
        if existing is None:
            return Insight.model_validate(
                {
                    **supplied,
                    "evidence": additions,
                    "updated_date": self._timestamp,
                    "confidence": None,
                }
            )
        # Existing claims and ratings are authoritative; only references and a
        # missing severity explanation may be added by the compiler.
        for field in ("name", "description", "severity"):
            if field in supplied and supplied[field] != getattr(existing, field):
                raise ValueError(f"Existing insight {field} must be preserved")
        merged, added = merge_evidence(existing.evidence, additions)
        return Insight.model_validate(
            {
                **existing.model_dump(),
                "evidence": merged,
                "updated_date": self._timestamp if added else existing.updated_date,
                "severity_reason": existing.severity_reason
                or (supplied.get("severity_reason") if existing.severity is not None else None),
            }
        )

    def _checked_evidence(
        self, items: list[dict[str, Any]], existing: Insight | None
    ) -> list[TraceEvidence]:
        historical = {item.trace_id: item for item in existing.evidence} if existing else {}
        checked = []
        for raw in items:
            item = TraceEvidence.model_validate(raw)
            if item.url is not None or any(span.url is not None for span in item.spans):
                raise ValueError("Omit evidence URLs; the application resolves source URLs")
            try:
                trace = self.trace(item.trace_id)
            except KeyError:
                if item.trace_id not in historical:
                    raise ValueError(f"Unknown evidence trace {item.trace_id!r}") from None
                if set(span.span_id for span in item.spans) - {
                    span.span_id for span in historical[item.trace_id].spans
                }:
                    raise ValueError("Cannot add unverified spans to a historical trace") from None
                checked.append(historical[item.trace_id])
                continue
            available = {visit.span.id for visit in walk_spans(trace)}
            if {span.span_id for span in item.spans} - available:
                raise ValueError(f"Evidence cites an unknown span in trace {item.trace_id!r}")
            checked.append(
                item.model_copy(
                    update={
                        "url": None,
                        "spans": [span.model_copy(update={"url": None}) for span in item.spans],
                    }
                )
            )
        return merge_evidence([], checked)[0]

    def _validate_handles(self, handles: list[str]) -> list[EvidenceCompletion]:
        """Reject unknown/duplicate handles and repeated updates to an existing ID."""
        if len(set(handles)) != len(handles) or any(h not in self._registered for h in handles):
            raise ValueError("Return unique handles obtained from tools.register in this session")
        completions = [self._registered[h] for h in handles]
        targets = [self._targets[h] for h in handles if self._targets[h] is not None]
        if len(set(targets)) != len(targets):
            raise ValueError("Register only one final update per existing insight")
        for completion in completions:
            completion.validate_result()
        return completions

    def scan(self, handles: list[str]) -> list[dict[str, Any]]:
        """Fresh batch scan; return compact counts/errors, retaining evidence in Python.

        No result is cached: application finalization scans again independently.
        """
        completions = self._validate_handles(handles)
        results = scan_completions(completions, self._snapshot)
        return [
            {
                "handle": handle,
                **(
                    {"status": "failed", "error": f"{type(result).__name__}: {str(result)[:500]}"}
                    if isinstance(result, Exception)
                    else {
                        "status": "incomplete" if result[1] else "complete",
                        "match_count": len(result[0]),
                        "unknown_count": result[1],
                        "example_trace_ids": [item.trace_id for item in result[0][:3]],
                    }
                ),
            }
            for handle, result in zip(handles, results, strict=True)
        ]

    def _finish(self, handles: list[str]) -> list[Insight]:
        """Independently verify selected claims and preserve all existing insights."""
        completions = self._validate_handles(handles)
        results = scan_completions(completions, self._snapshot)
        updates = [
            apply_scan(c, result, len(self._snapshot), self._timestamp)
            for c, result in zip(completions, results, strict=True)
        ]
        by_index = {
            self._targets[h]: insight
            for h, insight in zip(handles, updates, strict=True)
            if self._targets[h] is not None
        }
        return [by_index.get(index, insight) for index, insight in enumerate(self._existing)] + [
            insight
            for handle, insight in zip(handles, updates, strict=True)
            if self._targets[handle] is None
        ]
