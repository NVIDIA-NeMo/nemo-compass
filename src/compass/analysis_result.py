# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serializable results and common diagnostics for every Compass caller."""

from collections import defaultdict
from collections.abc import Sequence
from importlib.metadata import version
from typing import Literal

from pydantic import BaseModel, Field

from compass.analysis_options import AnalysisOptions
from compass.compass_signals.anomaly_and_patterns.signal import AnomalyAndPatternsArtifacts
from compass.compass_signals.compass_signals import CompassSignalResult
from compass.compass_signals.tool_issues.signal import ToolIssueEvidenceArtifacts
from compass.insight import Insight
from compass.insights_generation.validation import CandidateProblemRejection


class AnalysisProgress(BaseModel):
    stage: Literal["signals", "validation", "compilation", "confidence", "complete"]
    active_signals: tuple[str, ...] = ()


class StageResult(BaseModel):
    stage: str
    status: Literal["completed", "skipped"]
    reason: str | None = None


class SignalSummary(BaseModel):
    signal_name: str
    finding_count: int
    candidate_count: int
    skip_reason: str | None = None
    limitations: tuple[str, ...] = ()


class AnalysisResult(BaseModel):
    """Complete run artifact; JSON serialization needs no signal-specific knowledge."""

    trace_count: int
    evidence: list[CompassSignalResult]
    insights: list[Insight]
    existing_insights: list[Insight] = Field(default_factory=list)
    validation_rejections: list[CandidateProblemRejection] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    effective_options: AnalysisOptions = Field(default_factory=AnalysisOptions)
    compass_version: str = Field(default_factory=lambda: version("nemo-compass"))
    stages: list[StageResult] = Field(default_factory=list)
    signal_summaries: list[SignalSummary] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    below_threshold_count: int = 0


def signal_limitations(evidence: list[CompassSignalResult]) -> list[str]:
    limitations = []
    for result in evidence:
        if result.skip_reason:
            limitations.append(f"{result.signal_name}: {result.skip_reason}")
        limitations.extend(f"{result.signal_name}: {reason}" for reason in result.limitations)
    return limitations


def _one_line(text: str) -> str:
    return text.replace("\r", "\\r").replace("\n", "\\n")


def evidence_notes(evidence: list[CompassSignalResult]) -> tuple[list[str], int]:
    """Explain unpromoted evidence without exposing artifact types to integrations."""
    shortfalls: list[tuple[int, str]] = []
    skipped = []
    for item in evidence:
        artifacts = item.artifacts
        if isinstance(artifacts, ToolIssueEvidenceArtifacts):
            required = artifacts.config.minimum_independent_cases
            for card in artifacts.cards:
                if not card.eligible_for_insights:
                    tools = ", ".join(
                        dict.fromkeys(_one_line(e.tool_name) for e in card.representative_evidence)
                    )
                    shortfalls.append(
                        (
                            card.independent_case_count,
                            f"- below threshold: tool-issues {tools} {card.issue_type} "
                            f"({card.independent_case_count} of {required} independent cases)",
                        )
                    )
        elif isinstance(artifacts, AnomalyAndPatternsArtifacts):
            analysis = artifacts.result
            grouped = {group.signature for group in analysis.failure_groups}
            cross_tool = {group.message_signature for group in analysis.cross_tool_failure_groups}
            traces: dict[str, set[str]] = defaultdict(set)
            for event in analysis.failure_events:
                if (
                    event["signature"] not in grouped
                    and event.get("message_signature") not in cross_tool
                ):
                    traces[str(event["signature"])].add(str(event["trace_id"]))
            required = artifacts.config.minimum_independent_traces
            shortfalls.extend(
                (
                    len(ids),
                    f"- below threshold: anomaly-and-patterns {_one_line(sig)} ({len(ids)} of {required} traces)",
                )
                for sig, ids in traces.items()
            )
        if item.skip_reason is not None:
            skipped.append(f"- skipped: {item.signal_name} ({_one_line(item.skip_reason)})")
    shortfalls.sort(key=lambda entry: -entry[0])
    notes = [line for _, line in shortfalls[:5]]
    if len(shortfalls) > 5:
        notes.append(f"- ... and {len(shortfalls) - 5} more below threshold")
    return [*notes, *skipped], len(shortfalls)


def initial_result(
    trace_count: int,
    evidence: list[CompassSignalResult],
    existing_insights: Sequence[Insight],
    options: AnalysisOptions,
) -> AnalysisResult:
    """Capture effective inputs and signal diagnostics before subsequent review."""
    notes, below_threshold_count = evidence_notes(evidence)
    return AnalysisResult(
        trace_count=trace_count,
        evidence=evidence,
        insights=list(existing_insights),
        existing_insights=list(existing_insights),
        limitations=signal_limitations(evidence),
        effective_options=options.model_copy(deep=True),
        notes=notes,
        below_threshold_count=below_threshold_count,
        signal_summaries=[
            SignalSummary(
                signal_name=item.signal_name,
                finding_count=item.finding_count,
                candidate_count=len(item.problems),
                skip_reason=item.skip_reason,
                limitations=item.limitations,
            )
            for item in evidence
        ],
        stages=[StageResult(stage="signals", status="completed")],
    )


def append_run_notes(result: AnalysisResult) -> None:
    """Expose stage outcomes in the same report contract as signal diagnostics."""
    result.notes.extend(
        f"- limitation: {_one_line(reason)}"
        for reason in result.limitations
        if reason
        not in {
            f"{item.signal_name}: {item.skip_reason}"
            for item in result.evidence
            if item.skip_reason
        }
    )
    result.notes.extend(
        f"- skipped: {stage.stage} ({stage.reason})"
        for stage in result.stages
        if stage.status == "skipped"
    )
    if result.validation_rejections:
        result.notes.append(
            f"- validation: {len(result.validation_rejections)} candidate problem(s) rejected"
        )
