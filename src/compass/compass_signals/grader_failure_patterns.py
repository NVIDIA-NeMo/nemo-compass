# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import dataclass, field
from typing import Any, Protocol

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.strategies import CodeActStrategy
from nooa.strategy_validation import InvariantError
from nooa.unifiedllm import UnifiedLLM
from pydantic import BaseModel, ConfigDict, Field

from compass.compass_signals._trace import walk_spans
from compass.compass_signals.compass_signals import (
    CompassSignalResult,
    Problem,
)
from compass.insights_generation.defaults import (
    DEFAULT_MAX_TOOL_ROUNDS,
)
from compass.traces import TraceSnapshot


def _trace_index(snapshot: TraceSnapshot) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for trace in snapshot:
        spans = [visit.span for visit in walk_spans(trace)]
        rows.append(
            {
                "trace_id": trace.id,
                "case_id": trace.case_id,
                "session_id": trace.session_id,
                "evaluator_results": trace.evaluator_results,
                "tools": sorted({span.tool_name for span in spans if span.tool_name}),
                "errors": sorted({span.error for span in spans if span.error}),
            }
        )
    return sorted(rows, key=lambda row: str(row["trace_id"]))


class _GraderFailureReport(BaseModel):
    problems: tuple[Problem, ...]


class _GraderFailureAgent(Protocol):
    async def find_problems(self, index: list[dict[str, object]]) -> _GraderFailureReport: ...


class GraderFailurePatternsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_tool_rounds: int = Field(default=DEFAULT_MAX_TOOL_ROUNDS, ge=1)


def _build_agent(
    snapshot: TraceSnapshot,
    llm: UnifiedLLM,
    config: GraderFailurePatternsConfig,
) -> _GraderFailureAgent:
    fetched: set[str] = set()
    fetch_calls = 0

    def validate_report(_agent: Agent, report: _GraderFailureReport, _call: object) -> None:
        for problem in report.problems:
            cited = set(problem.supporting_trace_ids)
            if missing := cited - fetched:
                raise InvariantError(f"fetch traces before citing them: {sorted(missing)}")

    class GraderFailureAgent(Agent):
        def fetch_traces(self, trace_ids: list[str]) -> list[dict[str, Any]]:
            """Fetch complete traces from the compact index by exact trace ID."""
            nonlocal fetch_calls
            if fetch_calls >= config.max_tool_rounds:
                raise ValueError("trace fetch limit reached")
            fetch_calls += 1
            found = []
            for trace_id in dict.fromkeys(trace_ids):
                trace = snapshot.get_trace_by_id(trace_id)
                found.append(trace.model_dump(mode="json", exclude_unset=True))
            fetched.update(trace["id"] for trace in found)
            return found

        @strategy(
            CodeActStrategy(
                config=CodeActConfig(
                    max_iterations=config.max_tool_rounds + 3,
                    postconditions=[validate_report],
                )
            )
        )
        async def find_problems(self, index: list[dict[str, object]]) -> _GraderFailureReport:  # ty: ignore[empty-body] -- Nooa implements the ellipsis method.
            """Find recurring Problems linked to recorded evaluation signals.

            Group the same failure across traces, not merely matching scores.
            Fetch every supporting trace. Describe the observed failure behavior and the
            relevant evaluator names and values in each Problem's description. Distinguish
            observed associations from proven causes. Call return_result when done.
            """
            ...

    return GraderFailureAgent(llm=llm)


@dataclass(frozen=True)
class GraderFailurePatternsCompassSignal:
    name = "grader-failure-patterns"

    llm: UnifiedLLM
    config: GraderFailurePatternsConfig = field(default_factory=GraderFailurePatternsConfig)

    def check_prerequisites(self, snapshot: TraceSnapshot) -> str | None:
        if not isinstance(self.config, GraderFailurePatternsConfig):
            raise TypeError("grader-failure-patterns requires GraderFailurePatternsConfig")
        if not any(
            value is not None for trace in snapshot for value in trace.evaluator_results.values()
        ):
            return "No evaluator results"
        return None

    async def analyze(self, snapshot: TraceSnapshot) -> CompassSignalResult:
        evaluated = await asyncio.to_thread(
            lambda: sum(
                any(value is not None for value in trace.evaluator_results.values())
                for trace in snapshot
            )
        )
        async with self.llm:
            agent = _build_agent(snapshot, self.llm, self.config)
            report = await agent.find_problems(await asyncio.to_thread(_trace_index, snapshot))
        return CompassSignalResult(
            signal_name=self.name,
            problems=report.problems,
            limitations=(
                f"{len(snapshot) - evaluated} of {len(snapshot)} traces lack evaluator results",
            )
            if evaluated < len(snapshot)
            else (),
        )
