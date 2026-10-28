# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cost-efficiency Compass Signal.

Detection is deterministic: no LLM call, no network, and no prompt content in
any output. Each detector returns Findings; each rule's ``RuleSpec`` words the Problem
of each of its kinds.

When an LLM is set and ``llm_review`` is on, ``review.py`` then confirms each Problem
against its supporting traces; its confirmed trace IDs are kept in ``artifacts.review``.
"""

from __future__ import annotations

import asyncio
import dataclasses
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
from compass.compass_signals.cost_efficiency.review import review_problems
from compass.traces import SpanKind, TraceSnapshot

# Each detector reads the snapshot and returns complete Findings, cause included.
# Add a rule by adding its detector and the RuleSpec of every rule it reports.
DETECTORS: tuple[Callable[[TraceSnapshot], Iterable[Finding]], ...] = (prompt_caching_miss.detect,)
SPECS: dict[str, RuleSpec] = {spec.rule: spec for spec in prompt_caching_miss.SPECS}


class CostEfficiencyConfig(BaseModel):
    """Typed configuration owned by cost-efficiency analysis."""

    minimum_independent_traces: int = Field(default=3, ge=1)
    llm_review: bool = Field(
        default=True, description="Have an LLM confirm each Problem against its traces"
    )


@dataclass(frozen=True)
class CostEfficiencyArtifacts:
    findings: tuple[Finding, ...]
    cards: tuple[CostCard, ...]
    config: CostEfficiencyConfig
    review: dict[str, tuple[str, ...]] | None = None  # card ID -> LLM-confirmed trace IDs


@dataclass(frozen=True)
class CostEfficiencyCompassSignal:
    name = "cost-efficiency"

    config: CostEfficiencyConfig = field(default_factory=CostEfficiencyConfig)
    llm: UnifiedLLM | None = None  # always set by registration; None only in tools and tests

    def check_prerequisites(self, snapshot: TraceSnapshot) -> str | None:
        if not isinstance(self.config, CostEfficiencyConfig):
            raise TypeError("cost-efficiency requires CostEfficiencyConfig")
        if not any(
            visit.span.kind is SpanKind.LLM and visit.span.token_counts is not None
            for trace in snapshot
            for visit in walk_spans(trace)
        ):
            return "No LLM spans with token counts"
        return None

    async def analyze(self, snapshot: TraceSnapshot) -> CompassSignalResult:
        result = await asyncio.to_thread(self.analyze_snapshot, snapshot)
        if self.llm is None or not self.config.llm_review or not result.problems:
            return result
        artifacts = result.artifacts
        assert isinstance(artifacts, CostEfficiencyArtifacts)
        async with self.llm:
            problems, review = await review_problems(
                self.llm,
                snapshot,
                [card for card in artifacts.cards if card.eligible],
                result.problems,
            )
        return result.model_copy(
            update={
                "problems": problems,
                "artifacts": dataclasses.replace(artifacts, review=review),
            }
        )

    def analyze_snapshot(self, snapshot: TraceSnapshot) -> CompassSignalResult:
        """Run every detector; the result before any LLM review."""
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
