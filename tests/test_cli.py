# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import yaml
from nooa.unifiedllm import FakeLLMClient

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


def test_code_validation_filters_problems_and_preserves_signal_result(
    tmp_path, monkeypatch
) -> None:
    trace = Trace(id="trace-1", root_spans=[], aggregate=TraceAggregate())
    snapshot = TraceSnapshot([trace])
    supported = Problem(
        description="Supported",
        supporting_trace_ids=("trace-1",),
        candidate_trace_ids=("trace-1", "not-fetched-by-code-validator"),
    )
    unsupported = Problem(description="Unsupported", supporting_trace_ids=("trace-1",))
    unknown = Problem(description="Unknown", supporting_trace_ids=("trace-1",))
    evidence = [
        CompassSignalResult(
            signal_name="test-signal",
            problems=(supported, unsupported, unknown),
            artifacts={"kept": True},
        )
    ]
    received = []

    class FakeValidator:
        def __init__(self, code_base_path, llm):
            assert code_base_path == tmp_path.resolve()

        async def is_supported(self, problem, supporting_traces):
            received.append((problem, supporting_traces))
            return {"Supported": True, "Unsupported": False, "Unknown": None}[problem.description]

    monkeypatch.setattr(cli, "ProblemValidation", FakeValidator)

    result, rejected, ethos_rejections = asyncio.run(
        cli._validate_evidence(evidence, snapshot, tmp_path, FakeLLMClient())
    )

    assert result[0].problems == (supported, unknown)
    assert result[0].artifacts == {"kept": True}
    assert received == [
        (supported, (trace,)),
        (unsupported, (trace,)),
        (unknown, (trace,)),
    ]


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
        def __init__(self, code_base_path, llm):
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


@pytest.mark.parametrize("with_code_base", [False, True])
def test_final_compilation_severity_does_not_require_code_base(
    tmp_path, monkeypatch, with_code_base
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
    monkeypatch.setattr(cli, "_validate_evidence", AsyncMock(return_value=(evidence, 0, [])))
    monkeypatch.setattr(cli, "_build_llm", lambda *args: FakeLLMClient())
    monkeypatch.setattr(cli, "InsightCompilation", lambda llm: compilation)
    monkeypatch.setattr(cli, "_rate_confidence_with_code", review)
    config = RunConfig(
        trace={"filesystem": {"path": "traces.jsonl"}},
        code_base=tmp_path if with_code_base else None,
    )

    result = asyncio.run(cli._generate_insights(config, cli.RunOutput(Console(quiet=True))))

    compilation.compile_insights.assert_awaited_once()
    assert result.insights[0].severity == "high"
    assert result.insights[0].confidence == ("high" if with_code_base else None)
    assert review.await_count == int(with_code_base)


def test_compass_signals_share_cli_loop_and_run_concurrently(monkeypatch):
    async def run():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        registry = CompassSignalRegistry()

        class Signal:
            def __init__(self, name):
                self.name = name

            def check_prerequisites(self, snapshot):
                pass

            async def analyze(self, snapshot):
                assert asyncio.get_running_loop() is loop
                if self.name == "first":
                    await asyncio.wait_for(started.wait(), timeout=1)
                else:
                    started.set()
                return CompassSignalResult(signal_name=self.name, problems=())

        for signal in (Signal("first"), Signal("second")):
            registry.register(signal)
        monkeypatch.setattr(cli, "registered_builtin_signals", lambda **kwargs: registry)
        progress = []
        results = await cli._run_compass_signals(
            CompassSignalsConfig(tool_issues={}),
            TraceSnapshot([Trace(id="trace-1", root_spans=[], aggregate=TraceAggregate())]),
            progress=progress.append,
        )
        assert [result.signal_name for result in results] == ["first", "second"]
        assert set().union(*map(set, progress)) == {"first", "second"}
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


@pytest.mark.parametrize(
    "code_enabled,ethos_enabled", [(True, True), (True, False), (False, True), (False, False)]
)
def test_candidate_validators_run_independently_and_concurrently(
    tmp_path, monkeypatch, code_enabled, ethos_enabled
):
    from compass.insights_generation.ethos_validation import EthosDecision

    trace = Trace(id="t", root_spans=[], aggregate=TraceAggregate())
    problems = tuple(
        Problem(description=name, supporting_trace_ids=("t",))
        for name in ("code", "ethos", "both", "unknown")
    )
    evidence = [CompassSignalResult(signal_name="test", problems=problems, artifacts={"keep": 1})]
    code_started = asyncio.Event()
    ethos_started = asyncio.Event()
    calls = []

    class CodeValidator:
        def __init__(self, *args):
            pass

        async def is_supported(self, problem, traces):
            code_started.set()
            if ethos_enabled:
                await ethos_started.wait()
            calls.append(("code", problem.description))
            return False if problem.description in ("code", "both") else None

    class EthosValidator:
        def __init__(self, ethos, llm):
            assert ethos == "Explicit permission."

        async def validate(self, problem, traces):
            ethos_started.set()
            if code_enabled:
                await code_started.wait()
            calls.append(("ethos", problem.description))
            return EthosDecision(
                contradicted=problem.description in ("ethos", "both"),
                explanation="Explicitly allowed within scope.",
                passage="Explicit permission.",
            )

    monkeypatch.setattr(cli, "ProblemValidation", CodeValidator)
    monkeypatch.setattr(cli, "EthosValidation", EthosValidator)

    async def run():
        return await asyncio.wait_for(
            cli._validate_evidence(
                evidence,
                TraceSnapshot([trace]),
                tmp_path if code_enabled else None,
                FakeLLMClient(),
                "Explicit permission." if ethos_enabled else None,
            ),
            timeout=2,
        )

    filtered, code_count, rejections = asyncio.run(run())
    expected = [
        p
        for p in problems
        if not (code_enabled and p.description in ("code", "both"))
        and not (ethos_enabled and p.description in ("ethos", "both"))
    ]
    assert list(filtered[0].problems) == expected
    assert filtered[0].artifacts == {"keep": 1}
    assert evidence[0].problems == problems
    assert code_count == (2 if code_enabled else 0)
    assert len(rejections) == (2 if ethos_enabled else 0)
    assert len(calls) == 4 * (int(code_enabled) + int(ethos_enabled))
    assert all(r.decision.passage == "Explicit permission." for r in rejections)


@pytest.mark.parametrize("signal_enabled", [True, False])
def test_ethos_validation_can_skip_compilation_and_preserve_existing(
    tmp_path, monkeypatch, signal_enabled
):
    from datetime import datetime, timezone

    from rich.console import Console

    from compass.insights_generation.ethos_validation import EthosDecision

    ethos_path = tmp_path / "ethos.md"
    ethos_path.write_text("Require approval.")
    config = RunConfig(
        trace={"filesystem": {"path": "unused.jsonl"}},
        ethos_path=ethos_path,
        compass_signals={"ethos_divergence": {} if signal_enabled else None},
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
        def __init__(self, ethos, llm):
            assert ethos == "Require approval."

        async def validate(self, *args):
            return EthosDecision(
                contradicted=True, passage="Require approval.", explanation="Required behavior."
            )

    monkeypatch.setattr(cli, "EthosValidation", Validator)
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
    assert len(result.ethos_rejections) == 1
    assert result.rejected_by_code == 0


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
