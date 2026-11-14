# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from nooa.unifiedllm import FakeLLMClient

import compass.analysis as analysis
from compass.analysis import AnalysisOptions, AnalysisResult, BorrowedModelClient, analyze
from compass.analysis_runtime import agent_scope, analysis_scope
from compass.compass_signals.compass_signals import CompassSignalResult, Problem
from compass.insight import Insight
from compass.insights_generation.validation import (
    CandidateProblemDecision,
    CandidateProblemValidationResult,
)
from compass.traces import Trace, TraceAggregate, TraceSnapshot


def test_public_pipeline_preserves_validated_results_and_reports_stages(tmp_path, monkeypatch):
    snapshot = TraceSnapshot([Trace(id="a", root_spans=[], aggregate=TraceAggregate())])
    candidates = [
        CompassSignalResult(
            signal_name="test",
            problems=(
                Problem(description="Real failure", supporting_trace_ids=("a",)),
                Problem(description="Expected behavior", supporting_trace_ids=("a",)),
            ),
        )
    ]
    validation = CandidateProblemValidationResult(
        candidates,
        [
            [
                CandidateProblemDecision(supported=True, explanation="Confirmed"),
                CandidateProblemDecision(
                    supported=False, explanation="By design", rejection_basis="code"
                ),
            ]
        ],
    )
    insight = Insight(
        name="Real failure",
        description="Failure across requests",
        evidence=[{"trace_id": "a"}, {"trace_id": "b"}],
    )
    compiler = SimpleNamespace(compile_insights=AsyncMock(return_value=[insight]))
    monkeypatch.setattr(analysis, "_run_compass_signals", AsyncMock(return_value=candidates))
    monkeypatch.setattr(
        analysis, "_validate_candidate_problems", AsyncMock(return_value=validation)
    )
    monkeypatch.setattr(analysis, "InsightCompilation", lambda llm: compiler)
    monkeypatch.setattr(
        analysis,
        "_rate_confidence_with_code",
        AsyncMock(return_value=[insight.model_copy(update={"confidence": "high"})]),
    )
    progress = []
    result = asyncio.run(
        analyze(snapshot, llm_factory=FakeLLMClient, code_base=tmp_path, progress=progress.append)
    )
    compiled = compiler.compile_insights.await_args.args[0]
    assert len(compiled[0].problems) == 1
    assert result.insights[0].confidence == "high"
    assert len(result.validation_rejections) == 1
    assert [s.stage for s in result.stages] == [
        "signals",
        "validation",
        "compilation",
        "evidence_links",
        "confidence",
    ]
    assert all(s.status == "completed" for s in result.stages)
    assert [e.stage for e in progress] == [
        "signals",
        "validation",
        "compilation",
        "confidence",
        "complete",
    ]
    restored = AnalysisResult.model_validate_json(result.model_dump_json())
    assert restored == result


def test_builtin_results_serialize_and_missing_resources_are_explicit():
    result = asyncio.run(analyze(TraceSnapshot([]), llm_factory=FakeLLMClient))
    restored = AnalysisResult.model_validate_json(result.model_dump_json())
    assert restored.model_dump(mode="json") == result.model_dump(mode="json")
    assert len(result.signal_summaries) == 6
    assert any(s.signal_name == "user-sentiment" and s.skip_reason for s in result.signal_summaries)
    assert (
        next(s for s in result.stages if s.stage == "confidence").reason == "No codebase supplied"
    )
    assert result.effective_options == AnalysisOptions()
    assert result.compass_version


def test_run_hooks_are_isolated_and_cleaned_on_cancellation():
    from contextvars import ContextVar

    from nooa.runtime.event_manager import EventManager
    from nooa.unifiedllm import LLMResponse

    async def run():
        current = ContextVar("instrumented", default=None)
        observed = []
        seen = []

        @asynccontextmanager
        async def instrument(manager):
            token = current.set(manager)
            observed.append(manager)
            try:
                yield manager
            finally:
                current.reset(token)
                observed.remove(manager)

        started = asyncio.Event()

        async def child():
            agent = SimpleNamespace(event_manager=EventManager())
            async with agent_scope(agent):
                assert current.get() is agent.event_manager
                agent.event_manager.add(LLMResponse(parts=({"kind": "text", "text": "hello"},)))
                started.set()
                await asyncio.Event().wait()

        async with analysis_scope(instrument, seen.append):
            pending = asyncio.create_task(child())
            await started.wait()
            assert current.get() is None
            assert len(seen) == 1 and seen[0].content == "hello"
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        assert observed == []
        assert current.get() is None

    asyncio.run(run())


def test_borrowed_client_does_not_enter_or_close_host():
    class Client(FakeLLMClient):
        async def __aenter__(self):
            raise AssertionError("Host owns client entry")

        async def __aexit__(self, *args):
            raise AssertionError("Host owns client close")

    async def run():
        client = Client()
        async with BorrowedModelClient(client) as borrowed:
            assert borrowed.client is client

    asyncio.run(run())


@pytest.mark.parametrize("outcome", ["skipped", "failure", "cancelled", "construction_failure"])
def test_signal_clients_close_once_for_every_exit(outcome):
    from compass.analysis_runtime import owned_signal_clients

    clients = []

    class Client(FakeLLMClient):
        def __init__(self):
            super().__init__()
            self.closes = 0
            clients.append(self)

        async def aclose(self):
            self.closes += 1

    def factory():
        if outcome == "construction_failure" and len(clients) == 2:
            raise RuntimeError("Cannot construct third client")
        return Client()

    async def run():
        async with owned_signal_clients(factory) as create:
            assert create is not None
            create()  # A skipped signal never enters its client.
            active = create()
            if outcome == "construction_failure":
                create()
            if outcome == "skipped":
                return
            async with active:
                if outcome == "cancelled":
                    raise asyncio.CancelledError()
                raise RuntimeError("Signal failed")

    if outcome == "skipped":
        asyncio.run(run())
    else:
        with pytest.raises(asyncio.CancelledError if outcome == "cancelled" else RuntimeError):
            asyncio.run(run())
    assert len(clients) == 2
    assert [client.closes for client in clients] == [1, 1]


def test_borrowed_client_preserves_host_metadata():
    client = FakeLLMClient()
    client._context_window = 12345
    client.cache_breakpoint = "anthropic"
    client.reasoning_level = "high"
    client.config["max_tokens"] = 987
    borrowed = BorrowedModelClient(client)
    assert borrowed.context_window == client.context_window == 12345
    assert borrowed.cache_breakpoint == "anthropic"
    assert borrowed.reasoning_level == "high"
    assert borrowed.config["max_tokens"] == 987
