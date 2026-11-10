# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shared Compass analysis pipeline used by the CLI and platform integrations."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from nooa.context_blocks import EventBase
from nooa.runtime.event_manager import EventManager
from nooa.unifiedllm import UnifiedLLM

from compass import telemetry_phases as telemetry
from compass.analysis_options import AnalysisOptions, CompassSignalsConfig
from compass.analysis_result import (
    AnalysisProgress,
    AnalysisResult,
    SignalSummary,
    StageResult,
    append_run_notes,
    initial_result,
)
from compass.analysis_runtime import BorrowedModelClient, agent_scope, analysis_scope
from compass.compass_signals.builtins import registered_builtin_signals
from compass.compass_signals.compass_signals import CompassSignalResult, Problem
from compass.compass_signals.user_embedding.embedding import (
    LiteLLMEmbeddingConfig,
    UserEmbeddingGenerator,
)
from compass.insight import Insight, resolve_trace_links
from compass.insights_generation.confidence import InsightConfidence
from compass.insights_generation.insight_compilation import InsightCompilation
from compass.insights_generation.validation import (
    CandidateProblemDecision,
    CandidateProblemValidationResult,
    CandidateProblemValidator,
)
from compass.trace_loaders.intake import normalize_intake_trace
from compass.traces import Trace, TraceSnapshot


async def _run_compass_signals(
    config: CompassSignalsConfig,
    snapshot: TraceSnapshot,
    llm_factory: Callable[[], UnifiedLLM] | None = None,
    progress: Callable[[Sequence[str]], None] | None = None,
    *,
    ethos_path: Path | None = None,
    embedding_generator: UserEmbeddingGenerator | None = None,
) -> list[CompassSignalResult]:
    """Run the configured Compass Signals concurrently in registration order."""

    registry = registered_builtin_signals(
        anomaly_and_patterns=config.anomaly_and_patterns,
        tool_issues=config.tool_issues,
        cost_efficiency=config.cost_efficiency,
        ethos_divergence=config.ethos_divergence,
        ethos_path=ethos_path,
        user_sentiment=config.user_sentiment,
        grader_failure_patterns=config.grader_failure_patterns,
        llm_factory=llm_factory,
        embedding_generator=embedding_generator,
    )

    active: set[str] = set()

    def changed() -> None:
        if progress is not None:
            progress(tuple(name for name in registry.names if name in active))

    async def analyze(name: str) -> CompassSignalResult:
        configuration = registry.configuration(name)

        def started() -> None:
            active.add(name)
            changed()

        try:
            with telemetry.signal(name, configuration, len(snapshot)) as record:
                result = await registry.analyze(name, snapshot, on_start=started)
                telemetry.record_signal_result(record, result)
                return result
        finally:
            if name in active:
                active.remove(name)
                changed()

    # Cancel sibling work on failure; no analysis outlives this run.
    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(analyze(name)) for name in registry.names]
    return [task.result() for task in tasks]


async def _validate_candidate_problems(
    signal_results: list[CompassSignalResult],
    snapshot: TraceSnapshot,
    code_base_path: Path | None,
    llm: UnifiedLLM,
    ethos: str | None = None,
) -> CandidateProblemValidationResult:
    """Review the candidate problems in each signal result, preserving their order."""
    validator = CandidateProblemValidator(code_base_path, llm, ethos)

    async def validate_problem(problem: Problem) -> CandidateProblemDecision:
        traces = tuple(snapshot.get_trace_by_id(tid) for tid in problem.supporting_trace_ids)
        with telemetry.problem_validation(problem, code_base_path, ethos) as record:
            decision = await validator.validate(problem, traces)
            telemetry.record_problem_decision(record, decision)
            return decision

    async def validate_signal_problems(
        result: CompassSignalResult,
    ) -> list[CandidateProblemDecision]:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(validate_problem(problem)) for problem in result.problems]
        return [task.result() for task in tasks]

    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(validate_signal_problems(result)) for result in signal_results]
    return CandidateProblemValidationResult(signal_results, [task.result() for task in tasks])


async def _rate_confidence_with_code(
    insights: list[Insight],
    snapshot: TraceSnapshot,
    code_base_path: Path,
    llm: UnifiedLLM,
) -> list[Insight]:
    """Rate each insight's confidence against the supplied codebase."""

    code_base_path = code_base_path.expanduser().resolve()
    if not code_base_path.is_dir():
        raise ValueError(f"code_base must be an existing directory: {code_base_path}")
    rater = InsightConfidence(code_base_path, llm)

    def supporting_trace(ref: str) -> Trace | None:
        try:
            return snapshot.get_trace_by_id(ref)
        except KeyError:
            return None

    async def rate_insight(insight: Insight) -> Insight:
        supporting_traces = tuple(
            trace
            for trace in (supporting_trace(item.trace_id) for item in insight.evidence)
            if trace is not None
        )
        with telemetry.insight_confidence(insight, code_base_path) as record:
            rating = await rater.rate(insight, supporting_traces)
            telemetry.record_rating(record, insight, rating)
            return insight.model_copy(update={"confidence": rating})

    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(rate_insight(insight)) for insight in insights]
    return [task.result() for task in tasks]


async def analyze(
    snapshot: TraceSnapshot,
    *,
    llm_factory: Callable[[], UnifiedLLM],
    options: AnalysisOptions | None = None,
    existing_insights: Sequence[Insight] = (),
    ethos_path: Path | None = None,
    code_base: Path | None = None,
    fast_llm_factory: Callable[[], UnifiedLLM] | None = None,
    embedding_generator: UserEmbeddingGenerator | None = None,
    progress: Callable[[AnalysisProgress], None] | None = None,
    instrument_agent: Callable[[EventManager], AbstractAsyncContextManager[object]] | None = None,
    event_handler: Callable[[EventBase], None] | None = None,
) -> AnalysisResult:
    """Run the complete Compass pipeline against an already-loaded snapshot.

    Factories return fresh clients owned and closed by Compass. For a host-owned
    client, return BorrowedModelClient wrappers instead. Embedding generators are
    borrowed. Progress and instrumentation errors propagate and cancel sibling
    analysis tasks. Event observers follow Nooa's event delivery semantics.
    Configure telemetry exporters outside this function.
    """
    options = options or AnalysisOptions()
    if code_base is not None:
        code_base = code_base.expanduser().resolve()
        if not code_base.is_dir():
            raise ValueError(f"code_base must be an existing directory: {code_base}")
    ethos = ethos_path.read_text(encoding="utf-8") if ethos_path is not None else None
    if ethos is not None and not ethos.strip():
        raise ValueError("ETHOS document must not be empty")

    def notify(
        stage: Literal["signals", "validation", "compilation", "confidence", "complete"],
        active: Sequence[str] = (),
    ) -> None:
        if progress is not None:
            progress(AnalysisProgress(stage=stage, active_signals=tuple(active)))

    async with analysis_scope(instrument_agent, event_handler):
        notify("signals")
        evidence = await _run_compass_signals(
            options.compass_signals,
            snapshot,
            fast_llm_factory or llm_factory,
            lambda active: notify("signals", active),
            ethos_path=ethos_path,
            embedding_generator=embedding_generator,
        )
        result = initial_result(len(snapshot), evidence, existing_insights, options)
        await _compile_evidence(snapshot, result, llm_factory, code_base, ethos, notify)
        result.insights = resolve_trace_links(result.insights, snapshot, existing_insights)
        result.stages.append(StageResult(stage="evidence_links", status="completed"))
        if code_base is not None and result.insights:
            notify("confidence")
            with telemetry.confidence(result.insights, code_base) as record:
                async with llm_factory() as llm:
                    result.insights = await _rate_confidence_with_code(
                        result.insights, snapshot, code_base, llm
                    )
                telemetry.record_confidence(record, result.insights)
            result.stages.append(StageResult(stage="confidence", status="completed"))
        else:
            result.stages.append(
                StageResult(
                    stage="confidence",
                    status="skipped",
                    reason="No codebase supplied" if code_base is None else "No insights",
                )
            )
        append_run_notes(result)
        notify("complete")
        return result


async def _compile_evidence(
    snapshot: TraceSnapshot,
    result: AnalysisResult,
    llm_factory: Callable[[], UnifiedLLM],
    code_base: Path | None,
    ethos: str | None,
    notify: Callable[
        [Literal["signals", "validation", "compilation", "confidence", "complete"]], None
    ],
) -> None:
    signal_results = result.evidence
    if not any(item.problems for item in signal_results):
        result.stages.extend(
            StageResult(stage=stage, status="skipped", reason="No candidate problems")
            for stage in ("validation", "compilation")
        )
        return
    async with llm_factory() as llm:
        if code_base is not None or ethos is not None:
            notify("validation")
            with telemetry.candidate_problem_validation(signal_results, code_base, ethos) as record:
                validation = await _validate_candidate_problems(
                    signal_results, snapshot, code_base, llm, ethos
                )
                telemetry.record_candidate_problem_validation(record, validation)
            signal_results = validation.retained_signal_results()
            result.validation_rejections = validation.rejections
            if validation.unresolved_count:
                result.limitations.append(
                    f"Candidate problem validation was inconclusive for {validation.unresolved_count} problem(s); these were retained for compilation."
                )
            result.stages.append(StageResult(stage="validation", status="completed"))
        else:
            result.stages.append(
                StageResult(
                    stage="validation",
                    status="skipped",
                    reason="No codebase or ETHOS document supplied",
                )
            )
        if not any(item.problems for item in signal_results):
            result.stages.append(
                StageResult(
                    stage="compilation", status="skipped", reason="All candidate problems rejected"
                )
            )
            return
        notify("compilation")
        with telemetry.compilation(signal_results, result.existing_insights, ethos=ethos) as record:
            compiler = InsightCompilation(llm=llm)
            async with agent_scope(compiler):
                result.insights = await compiler.compile_insights(
                    signal_results,
                    snapshot,
                    result.existing_insights,
                    datetime.now(timezone.utc),
                    ethos=ethos,
                )
            telemetry.record_insights(record, result.insights)
        result.stages.append(StageResult(stage="compilation", status="completed"))


__all__ = [
    "analyze",
    "AnalysisOptions",
    "AnalysisResult",
    "AnalysisProgress",
    "StageResult",
    "SignalSummary",
    "TraceSnapshot",
    "Trace",
    "Insight",
    "BorrowedModelClient",
    "UserEmbeddingGenerator",
    "LiteLLMEmbeddingConfig",
    "normalize_intake_trace",
]
