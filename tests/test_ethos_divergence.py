# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json

import pytest
from nooa.unifiedllm import FakeLLMClient
from openinference.instrumentation import OITracer, TraceConfig
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from pydantic import ValidationError

from compass import telemetry
from compass.cli.main import _run_compass_signals, get_config
from compass.compass_signals.compass_signals import Problem
from compass.compass_signals.ethos_divergence import ethos_divergence_detector as ethos
from compass.traces import Trace, TraceAggregate, TraceSnapshot


def test_ethos_cli_runs_registered_detector(tmp_path, monkeypatch, select_signals):
    path = tmp_path / "ethos.md"
    path.write_text("Never issue refunds.", encoding="utf-8")
    config = get_config(
        [
            "--trace.filesystem.path",
            "unused.jsonl",
            "--compass-signals",
            json.dumps(select_signals(ethos_divergence={"ethos_path": str(path)})),
        ]
    )
    snapshot = TraceSnapshot([Trace(id="trace-1", root_spans=[], aggregate=TraceAggregate())])
    llm = FakeLLMClient()
    monkeypatch.setattr(
        telemetry, "tracer", OITracer(TracerProvider().get_tracer(__name__), config=TraceConfig())
    )
    problem = Problem(description="Issued a refund", supporting_trace_ids=("trace-1",))

    class FakeDetector:
        def __init__(self, *, llm):
            self.llm = llm

        async def detect_issues(self, traces, description, **extra):
            assert self.llm is llm
            assert traces is snapshot
            assert description == ethos.ETHOS_DIVERGENCE
            assert extra == {"ethos": "Never issue refunds."}
            current = trace.get_current_span()
            assert isinstance(current, ReadableSpan) and current.attributes is not None
            inputs = json.loads(str(current.attributes["input.value"]))
            assert inputs == {
                "trace_count": 1,
                "config": {"ethos_path": str(path)},
                "ethos": extra["ethos"],
            }
            return [problem]

    monkeypatch.setattr(ethos, "IssueDetector", FakeDetector)
    results = asyncio.run(_run_compass_signals(config.compass_signals, snapshot, lambda: llm))
    assert len(results) == 1
    assert results[0].signal_name == "ethos-divergence"
    assert results[0].problems == (problem,)


def test_ethos_rejects_missing_and_empty_documents(tmp_path):
    path = tmp_path / "ethos.md"
    with pytest.raises(ValidationError):
        ethos.EthosDivergenceConfig(ethos_path=path)
    path.write_text(" \n", encoding="utf-8")
    signal = ethos.EthosDivergenceCompassSignal(
        config=ethos.EthosDivergenceConfig(ethos_path=path), llm=FakeLLMClient()
    )
    with pytest.raises(ValueError, match="non-empty"):
        signal.check_prerequisites(TraceSnapshot([]))
