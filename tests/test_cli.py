# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import yaml
from nooa.unifiedllm import FakeLLMClient
from pydantic import BaseModel

import compass.cli.main as cli
from compass.compass_signals.compass_signals import CompassSignalResult, Problem
from compass.compass_signals.registry import CompassSignalRegistry
from compass.config import CompassSignalsConfig, RunConfig
from compass.insight import Insight, load_insights
from compass.insights_generation.config import load_dotenv
from compass.traces import Trace, TraceAggregate, TraceSnapshot


@pytest.fixture
def clean_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.os, "environ", {})
    monkeypatch.setattr(cli, "load_dotenv", lambda: load_dotenv(tmp_path / ".env"))


def test_missing_environment_exits_before_loading_traces(
    clean_environment, monkeypatch, tmp_path, capsys
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "trace:\n  langsmith:\n    project: production\n"
        "compass_signals:\n  user_sentiment:\n    litellm:\n"
        "      model: openai/embedding\n      api_key_env: EMBEDDING_API_KEY\n"
    )
    monkeypatch.setenv("LANGSMITH_API_KEY", "")
    monkeypatch.setenv("EMBEDDING_API_KEY", " \t")
    loader = Mock()
    monkeypatch.setattr(cli, "_configured_trace_loader", loader)
    output = tmp_path / "insights.yml"

    assert cli.main(["--config", str(config_path), "--output-path", str(output)]) == 2

    loader.assert_not_called()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "Missing required environment settings:\n"
        "  COMPASS_API_KEY — API key for the configured model\n"
        "  LANGSMITH_API_KEY — API key for LangSmith\n"
        "  EMBEDDING_API_KEY — API key named by compass_signals.user_sentiment.litellm.api_key_env\n\n"
        "Set these in .env in your working directory (NAME=value),\n"
        "or export them in your shell, then rerun the command.\n"
        "Setup: https://github.com/NVIDIA-NeMo/labs-nemo-compass/blob/main/docs/model-access.md\n"
    )
    assert not output.exists()


def test_langfuse_checks_only_missing_settings_after_cli_overrides(
    clean_environment, monkeypatch, tmp_path, capsys
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "trace:\n  langfuse:\n"
        "    from_timestamp: 2026-01-01T00:00:00Z\n"
        "    to_timestamp: 2026-01-02T00:00:00Z\n"
        "compass_signals:\n  tool_issues: {}\n"
    )
    monkeypatch.setenv("COMPASS_API_KEY", "private-model-key")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "private-project-key")
    args = ["--config", str(config_path)]
    assert cli.main(args) == 2
    error = capsys.readouterr().err
    assert "LANGFUSE_SECRET_KEY" in error
    assert "LANGFUSE_BASE_URL" in error
    assert "LANGFUSE_PUBLIC_KEY" not in error
    assert "private-" not in error

    args += ["--trace.langfuse.base-url", "https://langfuse.example.com"]
    assert cli.main(args) == 2
    assert "LANGFUSE_BASE_URL" not in capsys.readouterr().err
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "private-secret-key")
    assert cli._check_environment(cli.get_config(args)) == "private-model-key"


def test_environment_accepts_dotenv_and_existing_key_aliases(
    clean_environment, monkeypatch, tmp_path
):
    (tmp_path / ".env").write_text(
        "ANTHROPIC_API_KEY=dotenv-model-key\nLANGCHAIN_API_KEY=dotenv-trace-key\n"
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "exported-model-key")
    config = RunConfig(
        trace={"langsmith": {"project": "production"}}, compass_signals={"tool_issues": {}}
    )

    assert cli._check_environment(config) == "exported-model-key"


@pytest.mark.parametrize(
    "model",
    ["anthropic/claude-sonnet-4-6", "openrouter/anthropic/claude-sonnet-4.6", "openai/test"],
)
@pytest.mark.parametrize("fallback", ["OPENAI_API_BASE", "OPENAI_BASE_URL"])
def test_model_endpoint_does_not_inherit_another_provider_gateway(
    clean_environment, monkeypatch, model, fallback
):
    monkeypatch.setenv(fallback, "https://gateway.example/v1")
    client = Mock()
    monkeypatch.setattr(cli, "CompletionClient", client)
    config = RunConfig(trace={"filesystem": {"path": "traces.jsonl"}}, model=model)

    cli._build_llm(config, "test-key")
    expected = "https://gateway.example/v1" if model.startswith("openai/") else None
    assert client.call_args.kwargs["api_base"] == expected

    monkeypatch.setenv("COMPASS_API_BASE", "https://explicit.example/v1")
    cli._build_llm(config, "test-key")
    assert client.call_args.kwargs["api_base"] == "https://explicit.example/v1"

    config.api_base = "https://configured.example/v1"
    cli._build_llm(config, "test-key")
    assert client.call_args.kwargs["api_base"] == "https://configured.example/v1"


def test_anthropic_parameters_are_translated_to_native_fields(clean_environment, monkeypatch):
    from litellm.utils import get_optional_params

    client = Mock()
    monkeypatch.setattr(cli, "CompletionClient", client)
    config = RunConfig(
        trace={"filesystem": {"path": "traces.jsonl"}},
        model="anthropic/claude-sonnet-4-6",
        max_tokens=8192,
    )
    cli._build_llm(config, "test-key")
    params = dict(client.call_args.kwargs)
    params["model"] = "claude-sonnet-4-6"
    native = get_optional_params(
        **params,
        custom_llm_provider="anthropic",
        tool_choice="auto",
        tools=[
            {
                "type": "function",
                "function": {"name": "ping", "parameters": {"type": "object", "properties": {}}},
            }
        ],
    )
    assert "reasoning_effort" not in native
    assert native["output_config"]["effort"] == "high"
    assert native["tool_choice"] == {"type": "auto"}


@pytest.mark.parametrize("save_file", [True, False])
def test_cli_preserves_existing_insights_without_synthesizing_empty_evidence(
    clean_environment, tmp_path, monkeypatch, capsys, select_signals, save_file
):
    existing = [
        Insight.model_validate(
            {
                "name": "Search omits archived documents",
                "description": "Archived documents disappear from search results.",
                "evidence": [
                    {
                        "trace_id": f"historical-trace-{i}",
                        "url": f"https://provider.test/trace/{i}",
                        "spans": [
                            {"span_id": f"span-{j}", "url": f"https://provider.test/span/{i}/{j}"}
                            for j in range(3)
                        ],
                    }
                    for i in range(116)
                ],
            }
        )
    ]
    existing_path = tmp_path / "existing.json"
    existing_path.write_text(json.dumps([item.model_dump() for item in existing]), encoding="utf-8")
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_text('{"id":"unscored","root_spans":[],"aggregate":{}}', encoding="utf-8")
    output_path = tmp_path / "results" / "insights.yml"
    compilation = SimpleNamespace(compile_insights=AsyncMock(return_value=existing))
    monkeypatch.setenv("COMPASS_API_KEY", "test-key-not-real")
    monkeypatch.setattr(cli, "_build_llm", lambda config, api_key: FakeLLMClient())
    monkeypatch.setattr(cli, "InsightCompilation", lambda llm: compilation)

    result = cli.main(
        [
            "--trace.filesystem.path",
            str(trace_path),
            "--compass-signals",
            json.dumps(select_signals(tool_issues={"retry_threshold": 3})),
            "--existing-insights",
            str(existing_path),
            "--output-path",
            str(output_path) if save_file else "-",
        ]
    )

    compilation.compile_insights.assert_not_awaited()
    assert result == cli.EXIT_OK
    captured = capsys.readouterr()
    assert yaml.safe_load(captured.out) == [item.model_dump() for item in existing]
    assert "116 supporting traces" in captured.err
    assert "Trace " not in captured.err and "Span " not in captured.err
    assert "Ran:      None" in captured.err
    assert "Skipped:  Tool issues — No tool calls" in captured.err
    assert "No new insights from 1 trace" in captured.err
    assert "1 existing insight retained." in captured.err
    if save_file:
        assert f"Saved: {output_path}" in captured.err
        assert output_path.read_text(encoding="utf-8") == captured.out
        assert load_insights(output_path) == existing
    else:
        assert "Saved:" not in captured.err
        assert not output_path.exists()


def test_confidence_rating_preserves_compiled_severity(tmp_path, monkeypatch) -> None:
    trace = Trace(id="trace-1", root_spans=[], aggregate=TraceAggregate())
    snapshot = TraceSnapshot([trace])
    insight = Insight(
        name="Timeouts",
        description="Tools time out.",
        evidence=[{"trace_id": "trace-1"}, {"trace_id": "missing"}],
        severity="high",
    )
    received = []

    class FakeRater:
        def __init__(self, code_base_path, llm, ethos=None):
            assert code_base_path == tmp_path.resolve()

        async def rate(self, insight, supporting_traces):
            received.append((insight, supporting_traces))
            return "high"

    monkeypatch.setattr(cli, "InsightConfidence", FakeRater)

    result = asyncio.run(
        cli._rate_confidence_with_code([insight], snapshot, tmp_path, FakeLLMClient())
    )

    assert received == [(insight, (trace,))]
    assert result[0].confidence == "high"
    assert result[0].severity == "high"
    assert insight.confidence is None


def test_repository_history_refreshes_once_and_enriches_confidence(tmp_path, monkeypatch) -> None:
    trace = Trace(id="trace-1", root_spans=[], aggregate=TraceAggregate())
    snapshot = TraceSnapshot([trace])
    insight = Insight(
        name="Timeouts",
        description="Tools time out.",
        evidence=[{"trace_id": "trace-1"}, {"trace_id": "missing"}],
    )
    calls = []
    repository = SimpleNamespace(
        refresh_remote=Mock(side_effect=lambda: calls.append("refresh")),
    )

    class FakeRater:
        def __init__(self, code_base_path, llm, received_repository):
            assert code_base_path == tmp_path.resolve()
            assert received_repository is repository

        async def investigate(self, received_insight, supporting_traces):
            calls.append((received_insight, supporting_traces))
            return SimpleNamespace(confidence="med", repository_evidence=[])

    monkeypatch.setattr(cli, "GitHistory", lambda path: repository)
    monkeypatch.setattr(cli, "InsightConfidence", FakeRater)

    result = asyncio.run(
        cli._rate_confidence_with_code(
            [insight],
            snapshot,
            tmp_path,
            FakeLLMClient(),
            True,
        )
    )

    assert calls == ["refresh", (insight, (trace,))]
    assert result[0].confidence == "med"


@pytest.mark.parametrize(
    "with_code_base,repository_history", [(False, False), (True, False), (True, True)]
)
def test_final_compilation_severity_does_not_require_code_base(
    tmp_path, monkeypatch, with_code_base, repository_history
) -> None:
    from rich.console import Console

    trace = Trace(id="trace-1", root_spans=[], aggregate=TraceAggregate())
    snapshot = TraceSnapshot([trace])
    evidence = [
        CompassSignalResult(
            signal_name="test",
            problems=(Problem(description="Timeout", supporting_trace_ids=("trace-1",)),),
        )
    ]
    insight = Insight(
        name="Timeout",
        description="The task never completes.",
        evidence=[{"trace_id": "trace-1"}, {"trace_id": "trace-2"}],
        severity="high",
    )
    compilation = SimpleNamespace(compile_insights=AsyncMock(return_value=[insight]))
    review = AsyncMock(return_value=[insight.model_copy(update={"confidence": "high"})])
    monkeypatch.setattr(cli, "_check_environment", lambda config: "test-key")
    monkeypatch.setattr(
        cli, "_configured_trace_loader", lambda config: SimpleNamespace(load=lambda: snapshot)
    )
    monkeypatch.setattr(cli, "_run_compass_signals", AsyncMock(return_value=evidence))
    monkeypatch.setattr(
        cli,
        "_validate_candidate_problems",
        AsyncMock(
            return_value=cli.CandidateProblemValidationResult(
                evidence,
                [
                    [
                        cli.CandidateProblemDecision(supported=True, explanation="Supported")
                        for _ in item.problems
                    ]
                    for item in evidence
                ],
            )
        ),
    )
    monkeypatch.setattr(cli, "_build_llm", lambda *args: FakeLLMClient())
    monkeypatch.setattr(cli, "InsightCompilation", lambda llm: compilation)
    monkeypatch.setattr(cli, "_rate_confidence_with_code", review)
    config = RunConfig(
        trace={"filesystem": {"path": "traces.jsonl"}},
        code_base=tmp_path if with_code_base else None,
        repository_history=repository_history,
    )

    result = asyncio.run(cli._generate_insights(config, cli.RunOutput(Console(quiet=True))))

    compilation.compile_insights.assert_awaited_once()
    assert result.insights[0].severity == "high"
    assert result.insights[0].confidence == ("high" if with_code_base else None)
    assert review.await_count == int(with_code_base)
    if with_code_base:
        assert review.await_args.kwargs == {"repository_history": repository_history}


def test_compass_signals_share_cli_loop_and_run_concurrently(monkeypatch):
    async def run():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        registry = CompassSignalRegistry()

        class SignalConfig(BaseModel):
            pass

        class Signal:
            def __init__(self, name):
                self.name = name
                self.config = SignalConfig()

            def check_prerequisites(self, snapshot):
                pass

            async def analyze(self, snapshot):
                assert asyncio.get_running_loop() is loop
                if self.name == "anomaly-and-patterns":
                    await asyncio.wait_for(started.wait(), timeout=1)
                else:
                    started.set()
                return CompassSignalResult(signal_name=self.name, problems=())

        for signal in (Signal("anomaly-and-patterns"), Signal("tool-issues")):
            registry.register(signal)
        monkeypatch.setattr(cli, "registered_builtin_signals", lambda **kwargs: registry)
        progress = []
        results = await cli._run_compass_signals(
            CompassSignalsConfig(tool_issues={}),
            TraceSnapshot([Trace(id="trace-1", root_spans=[], aggregate=TraceAggregate())]),
            progress=progress.append,
        )
        assert [result.signal_name for result in results] == ["anomaly-and-patterns", "tool-issues"]
        assert set().union(*map(set, progress)) == {"anomaly-and-patterns", "tool-issues"}
        assert progress[-1] == ()

    asyncio.run(run())


def test_no_candidates_skips_synthesis_and_file_creation(
    clean_environment, tmp_path, monkeypatch, capsys, select_signals
):
    traces = tmp_path / "traces.jsonl"
    traces.write_text('{"id":"trace-1","root_spans":[],"aggregate":{}}')
    output = tmp_path / "insights.yml"
    compilation = Mock()
    monkeypatch.setenv("COMPASS_API_KEY", "test-key")
    monkeypatch.setattr(cli, "InsightCompilation", compilation)

    assert (
        cli.main(
            [
                "--trace.filesystem.path",
                str(traces),
                "--compass-signals",
                json.dumps(select_signals(tool_issues=True)),
                "--output-path",
                str(output),
            ]
        )
        == 0
    )
    compilation.assert_not_called()
    assert not output.exists()
    captured = capsys.readouterr()
    assert captured.out == "[]\n"
    assert "Saved:" not in captured.err


@pytest.mark.parametrize("code_enabled,ethos_enabled", [(True, True), (True, False), (False, True)])
def test_candidates_share_both_contexts_and_run_concurrently(
    tmp_path, monkeypatch, code_enabled, ethos_enabled
):
    traces = {
        name: Trace(id=name, root_spans=[], aggregate=TraceAggregate())
        for name in ("reject", "supported", "unresolved")
    }
    problems = tuple(
        Problem(
            description=name,
            supporting_trace_ids=(name,),
            candidate_trace_ids=("not-fetched-by-validator",),
        )
        for name in traces
    )
    evidence = [CompassSignalResult(signal_name="test", problems=problems, artifacts={"keep": 1})]
    started = {p.description: asyncio.Event() for p in problems}
    calls = []

    class Validator:
        def __init__(self, code_base, llm, ethos):
            assert code_base == (tmp_path if code_enabled else None)
            assert ethos == ("Explicit permission." if ethos_enabled else None)

        async def validate(self, problem, traces):
            started[problem.description].set()
            await asyncio.gather(*(event.wait() for event in started.values()))
            calls.append((problem, traces))
            rejected = problem.description == "reject"
            return cli.CandidateProblemDecision(
                supported={"reject": False, "supported": True, "unresolved": None}[
                    problem.description
                ],
                explanation="Considered available context.",
                rejection_basis=("ethos" if ethos_enabled else "code") if rejected else None,
                ethos_passage="Explicit permission." if rejected and ethos_enabled else None,
            )

    monkeypatch.setattr(cli, "CandidateProblemValidator", Validator)

    async def run():
        return await asyncio.wait_for(
            cli._validate_candidate_problems(
                evidence,
                TraceSnapshot(list(traces.values())),
                tmp_path if code_enabled else None,
                FakeLLMClient(),
                "Explicit permission." if ethos_enabled else None,
            ),
            timeout=2,
        )

    result = asyncio.run(run())
    assert result.retained_signal_results()[0].problems == problems[1:]
    assert result.retained_signal_results()[0].artifacts == {"keep": 1}
    assert result.rejected_count == 1
    assert result.unresolved_count == 1
    assert sorted(calls, key=lambda call: call[0].description) == [
        (problem, (traces[problem.description],)) for problem in problems
    ]
    assert evidence[0].problems == problems


def test_ethos_only_validation_filters_with_signal_disabled(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    from rich.console import Console

    from compass.insights_generation.validation import CandidateProblemDecision

    ethos_path = tmp_path / "ETHOS.md"
    ethos_path.write_text("Require approval.")
    config = RunConfig(
        trace={"filesystem": {"path": "unused.jsonl"}},
        ethos_path=ethos_path,
        compass_signals={"ethos_divergence": None},
    )
    problem = Problem(description="Approval is unnecessary", supporting_trace_ids=("t",))
    evidence = [CompassSignalResult(signal_name="test", problems=(problem,))]
    existing = Insight(
        name="Existing", description="Keep", evidence=[{"trace_id": "t"}, {"trace_id": "t2"}]
    )
    result = cli.RunResult(
        trace_count=1, evidence=evidence, insights=[existing], existing_insights=[existing]
    )
    compilation = AsyncMock()

    class Validator:
        def __init__(self, code_base, llm, ethos):
            assert ethos == "Require approval."

        async def validate(self, *args):
            return CandidateProblemDecision(
                supported=False,
                rejection_basis="ethos",
                ethos_passage="Require approval.",
                explanation="Required behavior.",
            )

    monkeypatch.setattr(cli, "CandidateProblemValidator", Validator)
    monkeypatch.setattr(cli, "_build_llm", lambda *args: FakeLLMClient())
    monkeypatch.setattr(cli, "InsightCompilation", compilation)
    insights = asyncio.run(
        cli._compile_evidence(
            config,
            "key",
            TraceSnapshot([Trace(id="t", root_spans=[], aggregate=TraceAggregate())]),
            result,
            cli.RunOutput(Console(quiet=True)),
            datetime.now(timezone.utc),
        )
    )
    assert insights == [existing]
    compilation.assert_not_called()
    assert len(result.validation_rejections) == 1


@pytest.mark.parametrize("missing_ethos", [False, True])
def test_configuration_errors_are_reported_together(tmp_path, monkeypatch, capsys, missing_ethos):
    monkeypatch.chdir(tmp_path)
    loader = Mock()
    monkeypatch.setattr(cli, "_configured_trace_loader", loader)
    args = ["--max-tokens", "0"]
    if missing_ethos:
        args += ["--ethos-path", str(tmp_path / "missing.md")]
    assert cli.main(args) == cli.EXIT_SETUP
    loader.assert_not_called()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Invalid configuration:" in captured.err
    assert "trace: Field required" in captured.err
    assert "max_tokens: Input should be greater than or equal to 1" in captured.err
    if missing_ethos:
        assert "ethos_path: ETHOS.md not found or not a regular file" in captured.err
        assert "--ethos-path" in captured.err
    assert "Traceback" not in captured.err
