# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import yaml
from nooa.unifiedllm import FakeLLMClient

from compass.compass_signals.compass_signals import CompassSignalResult, Problem
from compass.insight import Insight, TraceEvidence, load_insights


def trace(id):
    from compass.traces import Trace

    return Trace(id=id, root_spans=[], aggregate={})


def test_updated_date_round_trips_through_yaml(tmp_path):
    stamped_at = datetime(2026, 9, 29, 15, 0, tzinfo=timezone.utc)
    insight = Insight(
        name="Timeouts",
        description="Repeated timeouts",
        evidence=[TraceEvidence(trace_id="a"), TraceEvidence(trace_id="b")],
        updated_date=stamped_at,
    )
    path = tmp_path / "insights.yml"
    path.write_text(yaml.safe_dump([insight.model_dump()]))

    [loaded] = load_insights(path)

    assert loaded.updated_date == stamped_at


def test_insight_without_updated_date_does_not_serialize_the_field():
    insight = Insight(
        name="Timeouts",
        description="Repeated timeouts",
        evidence=[TraceEvidence(trace_id="a"), TraceEvidence(trace_id="b")],
    )
    assert "updated_date" not in insight.model_dump()


def test_cli_passes_through_updated_date_from_compilation(tmp_path, monkeypatch, select_signals):
    """End-to-end through cli.main(): updated_date is now set by the LLM inside
    compile_insights (given run_timestamp), not by any Python post-processing.
    This proves the CLI layer writes out exactly what compilation returned,
    unchanged, and that it actually passes compile_insights a run_timestamp."""
    import compass.cli.main as cli

    path = tmp_path / "traces.jsonl"
    path.write_text(
        trace("a").model_dump_json()
        + "\n"
        + trace("b").model_dump_json()
        + "\n"
        + trace("c").model_dump_json()
    )
    monkeypatch.setattr(cli, "_check_environment", lambda config: "test-key")
    monkeypatch.setattr(cli, "_build_llm", lambda *args: FakeLLMClient())
    monkeypatch.setattr(
        cli,
        "_run_compass_signals",
        AsyncMock(
            return_value=[
                CompassSignalResult(
                    signal_name="tool-issues",
                    problems=(Problem(description="Failure", supporting_trace_ids=("a", "b")),),
                )
            ]
        ),
    )

    def run_args(output):
        return [
            "--trace.filesystem.path",
            str(path),
            "--output-path",
            str(output),
            "--compass-signals",
            json.dumps(select_signals(tool_issues={})),
        ]

    # First run: the mock simulates the LLM creating a new insight and
    # stamping it with whatever run_timestamp it was given.
    first_stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    first_insight = Insight(
        id="platform-1",
        name="Repeated failure",
        description="Details",
        evidence=[TraceEvidence(trace_id="a"), TraceEvidence(trace_id="b")],
        updated_date=first_stamp,
    )
    first_compile = AsyncMock(return_value=[first_insight])
    monkeypatch.setattr(
        cli, "InsightCompilation", lambda **kwargs: SimpleNamespace(compile_insights=first_compile)
    )
    first_output = tmp_path / "first-run-insights.yml"
    assert cli.main(run_args(first_output)) == cli.EXIT_OK
    [after_first] = load_insights(first_output)
    assert after_first.updated_date == first_stamp  # untouched - exactly what the mock returned

    assert first_compile.await_args is not None
    run_timestamp_arg = first_compile.await_args.args[-1]
    assert isinstance(run_timestamp_arg, datetime)  # main.py actually threads one through

    # Second run: the mock simulates the LLM adding trace "c" and re-stamping
    # with a later run_timestamp - the CLI must not recompute or override it.
    second_stamp = datetime(2026, 9, 29, 15, 0, tzinfo=timezone.utc)
    grown_insight = first_insight.model_copy(
        update={
            "evidence": [TraceEvidence(trace_id=ref) for ref in ["a", "b", "c"]],
            "updated_date": second_stamp,
        }
    )
    second_compile = AsyncMock(return_value=[grown_insight])
    monkeypatch.setattr(
        cli, "InsightCompilation", lambda **kwargs: SimpleNamespace(compile_insights=second_compile)
    )
    second_output = tmp_path / "second-run-insights.yml"
    assert (
        cli.main(run_args(second_output) + ["--existing-insights", str(first_output)])
        == cli.EXIT_OK
    )
    [after_second] = load_insights(second_output)
    assert [item.trace_id for item in after_second.evidence] == ["a", "b", "c"]
    assert after_second.updated_date == second_stamp  # untouched - exactly what the mock returned

    # Third run: the mock simulates the LLM deciding nothing changed, so it
    # preserves the prior updated_date unchanged - the CLI must pass it
    # through exactly, not recompute it to "now".
    third_compile = AsyncMock(return_value=[grown_insight])
    monkeypatch.setattr(
        cli, "InsightCompilation", lambda **kwargs: SimpleNamespace(compile_insights=third_compile)
    )
    third_output = tmp_path / "third-run-insights.yml"
    assert (
        cli.main(run_args(third_output) + ["--existing-insights", str(second_output)])
        == cli.EXIT_OK
    )
    [after_third] = load_insights(third_output)
    assert after_third.updated_date == second_stamp
