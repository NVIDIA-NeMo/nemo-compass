# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json

import pytest
from nooa.unifiedllm import FakeLLMClient
from pydantic import ValidationError

from compass.cli import main as cli
from compass.cli.main import _run_compass_signals, get_config
from compass.compass_signals.builtins import registered_builtin_signals
from compass.compass_signals.compass_signals import Problem
from compass.compass_signals.ethos_divergence import ethos_divergence_detector as ethos
from compass.config import RunConfig
from compass.traces import Trace, TraceAggregate, TraceSnapshot


def test_ethos_cli_runs_registered_detector(tmp_path, monkeypatch, select_signals):
    path = tmp_path / "ethos.md"
    path.write_text("Never issue refunds.", encoding="utf-8")
    config = get_config(
        [
            "--trace.filesystem.path",
            "unused.jsonl",
            "--ethos-path",
            str(path),
            "--compass-signals",
            json.dumps(select_signals(ethos_divergence={})),
        ]
    )
    snapshot = TraceSnapshot([Trace(id="trace-1", root_spans=[], aggregate=TraceAggregate())])
    llm = FakeLLMClient()
    problem = Problem(description="Issued a refund", supporting_trace_ids=("trace-1",))

    class FakeDetector:
        def __init__(self, *, llm):
            self.llm = llm

        async def detect_issues(self, traces, description, **extra):
            assert self.llm is llm
            assert traces is snapshot
            assert description == ethos.ETHOS_DIVERGENCE
            assert extra == {"ethos": "Never issue refunds."}
            return [problem]

    monkeypatch.setattr(ethos, "IssueDetector", FakeDetector)
    results = asyncio.run(
        _run_compass_signals(
            config.compass_signals, snapshot, lambda: llm, ethos_path=config.ethos_path
        )
    )
    assert len(results) == 1
    assert results[0].signal_name == "ethos-divergence"
    assert results[0].problems == (problem,)


def test_ethos_rejects_missing_and_empty_documents(tmp_path):
    path = tmp_path / "ethos.md"
    with pytest.raises(ValidationError):
        RunConfig(trace={"filesystem": {"path": "unused.jsonl"}}, ethos_path=path)
    path.write_text(" \n", encoding="utf-8")
    signal = ethos.EthosDivergenceCompassSignal(
        config=ethos.EthosDivergenceConfig(), llm=FakeLLMClient(), ethos_path=path
    )
    with pytest.raises(ValueError, match="non-empty"):
        signal.check_prerequisites(TraceSnapshot([]))


@pytest.mark.parametrize("enabled", [True, False])
def test_shared_ethos_respects_signal_selection(tmp_path, select_signals, enabled):
    path = tmp_path / "ethos.md"
    path.write_text("Never issue refunds.")
    config = RunConfig(
        trace={"filesystem": {"path": "unused.jsonl"}},
        ethos_path=path,
        compass_signals=select_signals(tool_issues={}, ethos_divergence=enabled),
    )
    registry = registered_builtin_signals(
        ethos_divergence=config.compass_signals.ethos_divergence,
        ethos_path=config.ethos_path,
        llm_factory=FakeLLMClient,
    )
    assert ("ethos-divergence" in registry.names) is enabled


def test_generate_insights_passes_shared_ethos_to_detector(tmp_path, monkeypatch, select_signals):
    from rich.console import Console

    path = tmp_path / "ETHOS.md"
    path.write_text("Never issue refunds.")
    source = tmp_path / "traces.jsonl"
    source.write_text(
        Trace(id="trace-1", root_spans=[], aggregate=TraceAggregate()).model_dump_json() + "\n"
    )
    config = RunConfig(
        trace={"filesystem": {"path": source}},
        ethos_path=path,
        compass_signals=select_signals(ethos_divergence=True),
    )
    received = []

    async def detect(snapshot, llm, ethos):
        received.append(ethos)
        return []

    monkeypatch.setattr(ethos, "detect_ethos_divergence", detect)
    monkeypatch.setattr(cli, "_check_environment", lambda config: "test-key")
    monkeypatch.setattr(cli, "_build_llm", lambda *args: FakeLLMClient())
    result = asyncio.run(cli._generate_insights(config, cli.RunOutput(Console(quiet=True))))
    assert received == ["Never issue refunds."]
    assert result.evidence[0].signal_name == "ethos-divergence"


@pytest.mark.parametrize("is_directory", [False, True])
def test_invalid_ethos_path_has_actionable_cli_error(tmp_path, capsys, is_directory):
    path = tmp_path / "ETHOS.md"
    if is_directory:
        path.mkdir()
    assert cli.main(["--trace.filesystem.path", "unused.jsonl", "--ethos-path", str(path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert str(path) in captured.err
    assert "ETHOS file not found or not a regular file" in captured.err
    assert "working directory" in captured.err
    assert "--ethos-path" in captured.err
    assert "Traceback" not in captured.err


def test_no_ethos_path_skips_default_detector(monkeypatch):
    async def unexpected(*args, **kwargs):
        pytest.fail("Detector must not run without an ETHOS document")

    monkeypatch.setattr(ethos, "detect_ethos_divergence", unexpected)
    config = RunConfig(
        trace={"filesystem": {"path": "unused.jsonl"}},
        compass_signals={
            "anomaly_and_patterns": False,
            "tool_issues": False,
            "grader_failure_patterns": False,
            "user_sentiment": False,
        },
    )
    snapshot = TraceSnapshot([Trace(id="trace-1", root_spans=[], aggregate=TraceAggregate())])
    results = asyncio.run(_run_compass_signals(config.compass_signals, snapshot, FakeLLMClient))
    assert results[0].skip_reason == "No ethos document"
    assert not results[0].problems


@pytest.mark.parametrize("selection", [True, {}])
def test_explicit_ethos_without_path_fails_before_loading(tmp_path, monkeypatch, capsys, selection):
    from unittest.mock import Mock

    config = tmp_path / "config.yaml"
    config.write_text(
        "trace:\n  filesystem:\n    path: unused.jsonl\ncompass_signals:\n  ethos_divergence: "
        + json.dumps(selection)
        + "\n"
    )
    loader = Mock()
    monkeypatch.setattr(cli, "_configured_trace_loader", loader)
    assert cli.main(["--config", str(config)]) == 2
    loader.assert_not_called()
    captured = capsys.readouterr()
    assert "explicitly enabled" in captured.err
    assert "--ethos-path" in captured.err
    assert "Traceback" not in captured.err


def test_explicit_ethos_cli_without_path_fails():
    with pytest.raises(cli.SetupError, match="explicitly enabled"):
        cli.get_config(
            ["--trace.filesystem.path", "unused.jsonl", "--compass-signals.ethos-divergence", "{}"]
        )


@pytest.mark.parametrize("disabled", [False, True])
def test_default_ethos_warning_only_when_skipped(monkeypatch, capsys, disabled):
    from contextlib import nullcontext
    from unittest.mock import AsyncMock, Mock

    config = RunConfig(
        trace={"filesystem": {"path": "unused.jsonl"}},
        compass_signals={"ethos_divergence": False} if disabled else {},
    )
    monkeypatch.setattr(cli, "get_config", lambda argv: config)
    monkeypatch.setattr(cli, "_check_environment", lambda config: "test-key")
    monkeypatch.setattr(cli, "telemetry_session", lambda config: nullcontext(Mock()))
    run = AsyncMock(return_value=cli.RunResult(trace_count=0, evidence=[], insights=[]))
    monkeypatch.setattr(cli, "_generate_insights", run)
    assert cli.main(["--config", "unused.yaml"]) == 0
    run.assert_awaited_once()
    captured = capsys.readouterr()
    assert ("Warning: skipping ethos divergence" in captured.err) is (not disabled)
    if not disabled:
        assert "--ethos-path" in captured.err
