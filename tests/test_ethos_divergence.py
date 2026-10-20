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
