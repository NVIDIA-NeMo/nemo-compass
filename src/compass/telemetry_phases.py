# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Presentation of Compass phases in telemetry.

Orchestration passes domain inputs and results here. Span naming, serialization,
and summaries stay at this boundary rather than influencing analysis decisions.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from openinference.instrumentation._spans import OpenInferenceSpan
from opentelemetry.trace import StatusCode
from pydantic import BaseModel

from compass.compass_signals.builtins import (
    ANOMALY_AND_PATTERNS,
    ETHOS_DIVERGENCE,
    GRADER_FAILURE_PATTERNS,
    TOOL_ISSUES,
    USER_SENTIMENT,
)
from compass.compass_signals.compass_signals import (
    CompassSignalResult,
    Problem,
)
from compass.insight import Insight, Rating
from compass.telemetry import record_span

if TYPE_CHECKING:
    from compass.cli.output import RunResult
    from compass.insights_generation.validation import EvidenceValidation

# These are display names, not keys used to look up configuration.
_SIGNAL_NAMES = {
    ANOMALY_AND_PATTERNS: "anomaly_and_patterns",
    ETHOS_DIVERGENCE: "ethos_divergence",
    GRADER_FAILURE_PATTERNS: "grader_failure_patterns",
    TOOL_ISSUES: "tool_issues",
    USER_SENTIMENT: "user_sentiment",
}


def _signal_name(name: str) -> str:
    return _SIGNAL_NAMES.get(name, name)


@contextmanager
def _phase(name: str, inputs: dict) -> Iterator[OpenInferenceSpan]:
    with record_span(name) as record:
        record.set_input(inputs)
        yield record


def signal(
    name: str, config: BaseModel, trace_count: int
) -> AbstractContextManager[OpenInferenceSpan]:
    return _phase(
        f"signal.{_signal_name(name)}",
        {
            "trace_count": trace_count,
            "config": config.model_dump(mode="json"),
        },
    )


def record_signal_result(record: OpenInferenceSpan, result: CompassSignalResult) -> None:
    record.set_output(
        {
            "status": "skipped" if result.skip_reason else "completed",
            **result.model_dump(mode="json", include={"problems", "skip_reason", "limitations"}),
        }
    )


def problem_validation(
    problem: Problem, code_base: Path
) -> AbstractContextManager[OpenInferenceSpan]:
    return _phase(
        "source_code_validation.problem",
        {
            "problem": problem.model_dump(mode="json"),
            "code_base": str(code_base),
        },
    )


def record_support(record: OpenInferenceSpan, decision: bool | None) -> None:
    record.set_output({"supported": decision})


def validation(
    evidence: list[CompassSignalResult], code_base: Path
) -> AbstractContextManager[OpenInferenceSpan]:
    return _phase(
        "source_code_validation",
        {
            "problems": [
                {"signal": _signal_name(item.signal_name), "description": problem.description}
                for item in evidence
                for problem in item.problems
            ],
            "code_base": str(code_base),
        },
    )


def record_validation(record: OpenInferenceSpan, result: EvidenceValidation) -> None:
    groups: dict[str, list[dict[str, str]]] = {"supported": [], "unsupported": [], "unresolved": []}
    categories = {True: "supported", False: "unsupported", None: "unresolved"}
    for signal_result, decisions in zip(result.evidence, result.decisions, strict=True):
        for problem, decision in zip(signal_result.problems, decisions, strict=True):
            groups[categories[decision]].append(
                {
                    "signal": _signal_name(signal_result.signal_name),
                    "description": problem.description,
                }
            )
    record.set_output(groups)


def compilation(
    evidence: list[CompassSignalResult], existing: list[Insight]
) -> AbstractContextManager[OpenInferenceSpan]:
    return _phase(
        "compilation",
        {
            "signals": [
                {
                    "signal": _signal_name(item.signal_name),
                    "problems": [problem.model_dump(mode="json") for problem in item.problems],
                    "limitations": item.limitations,
                }
                for item in evidence
            ],
            "existing_insights": [item.model_dump(mode="json") for item in existing],
        },
    )


def record_insights(record: OpenInferenceSpan, insights: list[Insight]) -> None:
    record.set_output({"insights": [item.model_dump(mode="json") for item in insights]})


def confidence(
    insights: list[Insight], code_base: Path
) -> AbstractContextManager[OpenInferenceSpan]:
    return _phase("confidence", {"insight_count": len(insights), "code_base": str(code_base)})


def insight_confidence(
    insight: Insight, code_base: Path
) -> AbstractContextManager[OpenInferenceSpan]:
    return _phase(
        "confidence.insight",
        {
            "insight_name": insight.name,
            "description": insight.description,
            "code_base": str(code_base),
        },
    )


def record_rating(record: OpenInferenceSpan, insight: Insight, rating: Rating | None) -> None:
    record.set_output({"insight_name": insight.name, "confidence": rating})


def record_confidence(record: OpenInferenceSpan, insights: list[Insight]) -> None:
    record.set_output(
        {
            "ratings": [
                {"insight_name": item.name, "confidence": item.confidence} for item in insights
            ]
        }
    )


def record_run(record: OpenInferenceSpan, result: RunResult) -> None:
    record.set_output(
        {
            "trace_count": result.trace_count,
            "insights": [item.model_dump(mode="json") for item in result.insights],
            "rejected_by_code": result.rejected_by_code,
            "ethos_rejections": [item.model_dump(mode="json") for item in result.ethos_rejections],
            "limitations": result.limitations,
        }
    )


def evidence_scan(insight: Insight, trace_count: int) -> AbstractContextManager[OpenInferenceSpan]:
    return _phase("evidence_scan", {"insight_name": insight.name, "trace_count": trace_count})


def record_scan_failure(record: OpenInferenceSpan, error: Exception, *, skipped: bool) -> None:
    if not skipped:
        record.record_exception(error)
        record.set_status(StatusCode.ERROR)
    record.set_output(
        {
            "status": "skipped" if skipped else "failed",
            "reason": str(error)[:300],
            "error_type": None if skipped else type(error).__name__,
        }
    )


def record_scan_result(record: OpenInferenceSpan, match_count: int, unknown_count: int) -> None:
    record.set_output(
        {
            "status": "incomplete" if unknown_count else "complete",
            "match_count": match_count,
            "unknown_count": unknown_count,
        }
    )
