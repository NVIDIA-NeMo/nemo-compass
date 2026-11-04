# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from collections.abc import Callable
from datetime import datetime
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema

from compass import telemetry_phases as telemetry
from compass.compass_signals._trace import walk_spans
from compass.insight import Insight, SpanEvidence, TraceEvidence
from compass.traces import Trace, TraceSnapshot

Decision = Literal["match", "no_match", "unknown"]


class EvidenceDecision(BaseModel):
    """Whether a trace matches the insight, with IDs of the spans that support it."""

    model_config = ConfigDict(extra="forbid", revalidate_instances="always")

    status: Decision
    witness_span_ids: tuple[str, ...] = ()


class EvidenceCompletion(BaseModel):
    """An insight with a function and examples for checking its evidence.

    The application saves only the insight. The function and checks stay in memory.
    """

    insight: Insight
    # JSON cannot carry a function. Return this object from the agent's Python session.
    predicate: SkipJsonSchema[Callable[[Trace], EvidenceDecision | dict[str, object]] | None] = (
        Field(default=None, exclude=True, repr=False)
    )
    checks: list[tuple[Trace, Decision]] = Field(default_factory=list, exclude=True, repr=False)
    unresolved_reason: str | None = None

    def _evaluate(self, trace: Trace, span_ids: set[str] | None = None) -> EvidenceDecision:
        assert self.predicate is not None
        if span_ids is None:
            span_ids = {visit.span.id for visit in walk_spans(trace)}
        decision = EvidenceDecision.model_validate(self.predicate(trace))
        if decision.status == "match" and not decision.witness_span_ids:
            raise ValueError(f"Match in trace {trace.id} must cite at least one span ID")
        if set(decision.witness_span_ids) - span_ids:
            raise ValueError(f"Cited span ID does not exist in trace {trace.id}")
        return decision

    def validate_result(self) -> None:
        """Check the function against examples, or require a reason for skipping the scan."""
        if self.predicate is None:
            if not self.unresolved_reason or not self.unresolved_reason.strip() or self.checks:
                raise ValueError(
                    "Without a predicate, leave checks empty. Explain why in unresolved_reason. "
                    "The reason must not be blank."
                )
            return
        if not callable(self.predicate) or self.unresolved_reason is not None:
            raise ValueError("Provide a Python function as predicate. Set unresolved_reason=None.")
        if {expected for _, expected in self.checks} != {"match", "no_match", "unknown"}:
            raise ValueError(
                "Checks must include expected results for match, no_match, and unknown"
            )
        for trace, expected in self.checks:
            actual = self._evaluate(trace.model_copy(deep=True)).status
            if actual != expected:
                raise ValueError(f"Check {trace.id}: expected {expected}, got {actual}")

    def scan(self, snapshot: TraceSnapshot) -> tuple[list[TraceEvidence], int]:
        """Scan all traces after the example checks pass.

        Return matching traces with their supporting spans and the count of unknown traces.
        """
        with telemetry.evidence_scan(self.insight, len(snapshot)) as record:
            try:
                matches, unknown = self._scan(snapshot)
            except Exception as error:
                skipped = (
                    self.predicate is None
                    and bool(self.unresolved_reason and self.unresolved_reason.strip())
                    and not self.checks
                )
                telemetry.record_scan_failure(record, error, skipped=skipped)
                failure = error
            else:
                telemetry.record_scan_result(record, len(matches), unknown)
                return matches, unknown
        # Close intentional skips normally so OTel does not mark them as errors.
        raise failure

    def _scan(self, snapshot: TraceSnapshot) -> tuple[list[TraceEvidence], int]:
        self.validate_result()
        if self.predicate is None:
            raise ValueError(self.unresolved_reason)
        matches = []
        unknown = 0
        for trace in snapshot:
            trace_id = trace.id
            decision = self._evaluate(trace)
            if decision.status == "match":
                matches.append(
                    TraceEvidence(
                        trace_id=trace_id,
                        spans=[
                            SpanEvidence(span_id=sid)
                            for sid in dict.fromkeys(decision.witness_span_ids)
                        ],
                    )
                )
            unknown += decision.status == "unknown"
        if not matches:
            raise ValueError(
                "No traces matched despite a passing positive check. Review the function and example."
            )
        return matches, unknown

    def apply(
        self,
        snapshot: TraceSnapshot,
        run_timestamp: datetime,
    ) -> Insight:
        """Add matching evidence, or keep the insight unchanged if the scan fails."""
        try:
            result = self.scan(snapshot)
        except Exception as error:
            result = error
        return apply_scan(self, result, len(snapshot), run_timestamp)


def apply_scan(
    completion: EvidenceCompletion,
    result: tuple[list[TraceEvidence], int] | Exception,
    trace_count: int,
    run_timestamp: datetime,
) -> Insight:
    """Apply a fresh batch result using the same preservation rules as a single scan."""
    if isinstance(result, Exception):
        warnings.warn(
            f"The evidence scan did not complete for {completion.insight.name!r}: "
            f"{type(result).__name__}: {str(result)[:300]}. "
            "The insight keeps its existing references. The scan added no new references.",
            stacklevel=2,
        )
        return completion.insight
    matches, unknown = result
    if unknown:
        warnings.warn(
            f"Evidence scan incomplete for {completion.insight.name!r}: "
            f"The function could not decide whether {unknown} of {trace_count} traces match. "
            "The scan added its matches, but other supporting traces may remain without references.",
            stacklevel=2,
        )
    evidence, added = merge_evidence(completion.insight.evidence, matches)
    return completion.insight.model_copy(
        update={
            "evidence": evidence,
            "updated_date": run_timestamp if added else completion.insight.updated_date,
        }
    )


def merge_evidence(
    original: list[TraceEvidence], additions: list[TraceEvidence]
) -> tuple[list[TraceEvidence], bool]:
    """Union references without replacing existing trace/span metadata."""
    original_ids = {item.trace_id for item in original}
    merged: dict[str, TraceEvidence] = {}
    for item in [*original, *additions]:
        previous = merged.get(item.trace_id, item)
        spans = {span.span_id: span for span in previous.spans}
        for span in item.spans:
            spans.setdefault(span.span_id, span)
        merged[item.trace_id] = previous.model_copy(update={"spans": list(spans.values())})
    return list(merged.values()), any(item.trace_id not in original_ids for item in additions)


def scan_completions(
    completions: list[EvidenceCompletion], snapshot: TraceSnapshot
) -> list[tuple[list[TraceEvidence], int] | Exception]:
    """Freshly verify predicates in one disk pass, isolating each predicate's input.

    Read each stored JSON payload once and share its authoritative span-ID set.
    Each predicate receives its own decoded trace so mutations cannot affect another.
    No predicate results, functions, or mutable closure state are cached.
    """
    if not completions:
        return []
    results: list[tuple[list[TraceEvidence], int] | Exception] = [([], 0) for _ in completions]
    evaluation_seconds = [0.0 for _ in completions]
    with telemetry.evidence_scan_batch([c.insight for c in completions], len(snapshot)) as record:
        for index, completion in enumerate(completions):
            try:
                completion.validate_result()
                if completion.predicate is None:
                    raise ValueError(completion.unresolved_reason)
            except Exception as error:
                results[index] = error
        if any(not isinstance(result, Exception) for result in results):
            for payload in snapshot.iter_json():
                _scan_trace(payload, completions, results, evaluation_seconds)
        summaries = []
        for index, completion in enumerate(completions):
            result = results[index]
            if not isinstance(result, Exception) and not result[0]:
                result = ValueError(
                    "No traces matched despite a passing positive check. Review the function and example."
                )
                results[index] = result
            summary = _scan_summary(completion, result, evaluation_seconds[index])
            if isinstance(result, Exception) and summary["status"] == "failed":
                record.record_exception(result)
            summaries.append(summary)
        telemetry.record_batch_scan_result(record, summaries)
    return results


def _scan_trace(
    payload: str,
    completions: list[EvidenceCompletion],
    results: list[tuple[list[TraceEvidence], int] | Exception],
    evaluation_seconds: list[float],
) -> None:
    """Evaluate one stored trace, sharing source IDs but isolating mutable objects."""
    trace = Trace.model_validate_json(payload)
    trace_id = trace.id
    span_ids = {visit.span.id for visit in walk_spans(trace)}
    first = True
    for index, completion in enumerate(completions):
        result = results[index]
        if isinstance(result, Exception):
            continue
        matches, unknown = result
        started = perf_counter()
        try:
            isolated = trace if first else Trace.model_validate_json(payload)
            first = False
            decision = completion._evaluate(isolated, span_ids)
            if decision.status == "match":
                matches.append(
                    TraceEvidence(
                        trace_id=trace_id,
                        spans=[
                            SpanEvidence(span_id=sid)
                            for sid in dict.fromkeys(decision.witness_span_ids)
                        ],
                    )
                )
            results[index] = matches, unknown + (decision.status == "unknown")
        except Exception as error:
            results[index] = error
        finally:
            evaluation_seconds[index] += perf_counter() - started


def _scan_summary(
    completion: EvidenceCompletion,
    result: tuple[list[TraceEvidence], int] | Exception,
    evaluation_seconds: float,
) -> dict:
    summary = {"insight_name": completion.insight.name, "evaluation_seconds": evaluation_seconds}
    if isinstance(result, Exception):
        skipped = (
            completion.predicate is None
            and bool(completion.unresolved_reason and completion.unresolved_reason.strip())
            and not completion.checks
        )
        summary.update(status="skipped" if skipped else "failed", reason=str(result)[:300])
    else:
        summary.update(
            status="incomplete" if result[1] else "complete",
            match_count=len(result[0]),
            unknown_count=result[1],
        )
    return summary
