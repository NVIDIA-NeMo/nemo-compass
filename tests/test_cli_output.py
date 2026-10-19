# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from compass.cli.output import RunOutput, RunResult
from compass.compass_signals.compass_signals import CompassSignalResult, Problem
from compass.insight import Insight


def make_insight(name="Agent calls tools absent from the active catalog"):
    return Insight.model_validate(
        {
            "name": name,
            "description": "A long explanation that belongs in the saved report.",
            "evidence": [
                {"trace_id": f"trace-{i}", "spans": [{"span_id": f"span-{i}"}]} for i in range(3)
            ],
        }
    )


@pytest.mark.parametrize("width", [40, 80])
def test_default_report_prioritizes_findings_and_keeps_coverage_visible(width):
    stream = StringIO()
    result = RunResult(
        200,
        [
            *[
                CompassSignalResult(
                    signal_name=name,
                    problems=tuple(
                        Problem(description=f"Candidate {i}", supporting_trace_ids=("trace-0",))
                        for i in range(candidates)
                    ),
                    limitations=("Missing schemas",) if name == "tool-issues" else (),
                )
                for name, candidates in (("anomaly-and-patterns", 5), ("tool-issues", 3))
            ],
            *[
                CompassSignalResult(signal_name=name, problems=(), skip_reason=reason)
                for name, reason in (
                    ("ethos-divergence", "No ethos document"),
                    ("grader-failure-patterns", "No evaluator results"),
                    ("user-sentiment", "No embedding backend configured"),
                )
            ],
        ],
        [
            make_insight(),
            make_insight("Retries repeat [invalid] requests"),
            make_insight("False success"),
        ],
    )
    RunOutput(Console(file=stream, width=width)).report(result, Path("insights.yml"))
    rendered = stream.getvalue()
    assert "3 insights from 200 traces" in rendered
    assert rendered.count("Evidence: 3 supporting traces") == 3
    assert "2. Retries repeat [invalid] requests" in " ".join(rendered.split())
    assert rendered.index("3. False success") < rendered.index("Ran:") < rendered.index("Saved:")
    normalized = " ".join(rendered.split())
    assert (
        "Ran: Anomalies and patterns (5 candidates), tool issues (3 candidates; Missing schemas)"
        in normalized
    )
    assert (
        "Skipped: Ethos — no document Grader failures — no results Sentiment — no embedding backend"
        in normalized
    )
    assert (
        "To run skipped analyses: https://github.com/NVIDIA-NeMo/labs-nemo-compass/blob/main/docs/compass-signals.md"
        in rendered
    )
    for hidden in ("trace-0", "span-0", "long explanation", "\x1b"):
        assert hidden not in rendered
    # Keep the setup URL intact for copying; other content wraps to the terminal width.
    assert all(len(line) <= width for line in rendered.splitlines() if "https://" not in line)


def test_report_lists_all_insights_in_order_and_omits_irrelevant_guidance():
    insights = [make_insight(f"Issue {i}") for i in range(6)]
    result = RunResult(
        200,
        [
            CompassSignalResult(signal_name="tool-issues", problems=(), finding_count=2),
            CompassSignalResult(signal_name="grader-failure-patterns", problems=()),
        ],
        insights,
    )
    stream = StringIO()
    RunOutput(Console(file=stream, width=120)).report(result, Path("-"))
    rendered = stream.getvalue()
    assert [line for line in rendered.splitlines() if ". Issue" in line] == [
        f"{i + 1}. Issue {i}" for i in range(6)
    ]
    assert "Tool issues (2 findings; no candidates)" in rendered
    assert "grader failures (no findings)" in rendered
    assert "Skipped" not in rendered and "To run skipped analyses" not in rendered
    assert "Saved:" not in rendered
    assert result.insights == insights


def test_report_shows_optional_severity_and_reason():
    insight = make_insight().model_copy(
        update={
            "severity": "high",
            "severity_reason": "The agent is blocked from completing the task.",
        }
    )
    stream = StringIO()
    RunOutput(Console(file=stream, width=120)).report(RunResult(3, [], [insight]), Path("-"))

    assert "Severity: HIGH — The agent is blocked from completing the task." in stream.getvalue()


def test_ethos_rejections_report_passage_and_reason():
    from compass.insights_generation.ethos_validation import EthosDecision, EthosRejection

    problem = Problem(description="Approval is unnecessary", supporting_trace_ids=("t",))
    rejection = EthosRejection(
        signal_name="test",
        problem=problem,
        decision=EthosDecision(
            contradicted=True,
            passage="Require [explicit] approval.",
            explanation="The reported friction is required.",
        ),
    )
    result = RunResult(
        1,
        [CompassSignalResult(signal_name="test", problems=(problem,))],
        [],
        ethos_rejections=[rejection],
    )
    stream = StringIO()
    RunOutput(Console(file=stream, width=120)).report(result, Path("insights.yml"))
    rendered = stream.getvalue()
    assert "ethos excluded 1 candidate issue." in rendered
    assert "Require [explicit] approval." in rendered
    assert "The reported friction is required." in rendered
    assert "none were retained after validation and review" in rendered
