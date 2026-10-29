# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cost-efficiency Compass Signal.

Detection is deterministic: no LLM call, no network, and no prompt content in
any output. Each detector returns Findings; each rule's ``RuleSpec`` words the Problem
of each of its kinds.

The shared insight compiler validates the candidate Problems against their traces.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from nooa.unifiedllm import UnifiedLLM
from pydantic import BaseModel, Field

from compass.compass_signals._trace import session_of, walk_spans
from compass.compass_signals.compass_signals import CompassSignalResult
from compass.compass_signals.cost_efficiency import prompt_caching_miss
from compass.compass_signals.cost_efficiency.cards import (
    CostCard,
    Finding,
    RuleSpec,
    build_cards,
    problems_from_cards,
)
from compass.compass_signals.cost_efficiency.evidence import (
    INSTRUCTIONS,
    CacheEvidence,
    detector_view,
)
from compass.evidence import extract_evidence
from compass.traces import SpanKind, TraceSnapshot

# Each detector reads the snapshot and returns complete Findings, cause included.
# Add a rule by adding its detector and the RuleSpec of every rule it reports.
DETECTORS: tuple[Callable[[TraceSnapshot], Iterable[Finding]], ...] = (prompt_caching_miss.detect,)
SPECS: dict[str, RuleSpec] = {spec.rule: spec for spec in prompt_caching_miss.SPECS}


class CostEfficiencyConfig(BaseModel):
    """Typed configuration owned by cost-efficiency analysis."""

    minimum_independent_traces: int = Field(default=3, ge=1)


@dataclass(frozen=True)
class CostEfficiencyArtifacts:
    findings: tuple[Finding, ...]
    cards: tuple[CostCard, ...]
    config: CostEfficiencyConfig


@dataclass(frozen=True)
class CostEfficiencyCompassSignal:
    name = "cost-efficiency"

    config: CostEfficiencyConfig = field(default_factory=CostEfficiencyConfig)
    llm: UnifiedLLM | None = None

    def check_prerequisites(self, snapshot: TraceSnapshot) -> str | None:
        if not isinstance(self.config, CostEfficiencyConfig):
            raise TypeError("cost-efficiency requires CostEfficiencyConfig")
        if not any(trace.root_spans for trace in snapshot):
            return "No source spans"
        return None

    async def analyze(self, snapshot: TraceSnapshot) -> CompassSignalResult:
        if self.llm is None:
            raise ValueError("cost-efficiency evidence extraction requires an LLM client")
        evidence = await extract_evidence(snapshot, CacheEvidence, INSTRUCTIONS, self.llm)
        view = detector_view(snapshot, evidence)
        result = await asyncio.to_thread(self.analyze_snapshot, view)
        calls = [
            visit.span
            for trace in view
            for visit in walk_spans(trace)
            if visit.span.kind is SpanKind.LLM
        ]
        missing = sum(span.token_counts is None for span in calls)
        limitations = []
        if evidence.unavailable_count:
            limitations.append(
                f"No evidence extracted for {evidence.unavailable_count} of {evidence.source_count} source spans."
            )
        if missing:
            limitations.append(
                f"Token counts unavailable for {missing} of {len(calls)} recognized LLM calls."
            )
        if not calls:
            limitations.append("No LLM calls identified in the extracted evidence.")
        return result.model_copy(update={"limitations": tuple(limitations)})

    def analyze_snapshot(self, snapshot: TraceSnapshot) -> CompassSignalResult:
        """Run every detector and build candidate Problems."""
        findings = [finding for detect in DETECTORS for finding in detect(snapshot)]
        findings.sort(key=lambda f: (f.trace_id, f.span_ids))
        cards = build_cards(
            findings, SPECS, minimum_independent_traces=self.config.minimum_independent_traces
        )
        counted = [  # (trace, session) of each LLM call with token counts
            (trace.id, session_of(trace) or trace.id)
            for trace in snapshot
            for visit in walk_spans(trace)
            if visit.span.kind is SpanKind.LLM and visit.span.token_counts is not None
        ]
        totals = {
            "total_traces": len(snapshot),
            "traces_with_counts": len({trace_id for trace_id, _ in counted}),
            "sessions_with_counts": len({session for _, session in counted}),
            "calls_with_counts": len(counted),
        }
        problems = problems_from_cards(cards, findings, SPECS, totals)
        return CompassSignalResult(
            signal_name=self.name,
            problems=problems,
            finding_count=len(findings),
            artifacts=CostEfficiencyArtifacts(
                findings=tuple(findings),
                cards=tuple(cards),
                config=self.config,
            ),
        )


__all__ = [
    "CostEfficiencyArtifacts",
    "CostEfficiencyConfig",
    "CostEfficiencyCompassSignal",
]
