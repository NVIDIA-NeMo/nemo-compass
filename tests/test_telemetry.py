# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

import pytest
import yaml

from compass.traces import Trace, TraceAggregate


def test_execution_metadata_tracks_compass_checkout(monkeypatch, tmp_path):
    from compass import telemetry

    checkout = tmp_path / "compass"
    checkout.mkdir()
    monkeypatch.setattr(telemetry, "__file__", str(checkout / "src/compass/telemetry.py"))
    monkeypatch.chdir(tmp_path)

    def git(*args):
        return subprocess.check_output(["git", "-C", str(checkout), *args], text=True).strip()

    git("init", "-q")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "Initial",
        "--allow-empty",
    )
    metadata = telemetry._execution_metadata()
    assert metadata["compass.git_sha"] == git("rev-parse", "HEAD")
    assert metadata["compass.git_dirty"] is False

    file = checkout / "source.py"
    file.write_text("original")
    assert telemetry._execution_metadata()["compass.git_dirty"] is True  # Untracked.
    git("add", "source.py")
    assert telemetry._execution_metadata()["compass.git_dirty"] is True  # Staged only.
    git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "Source")
    assert telemetry._execution_metadata()["compass.git_dirty"] is False
    file.write_text("modified")
    assert telemetry._execution_metadata()["compass.git_dirty"] is True  # Unstaged.

    # A wheel inside another checkout must not claim that repository's identity.
    monkeypatch.setattr(
        telemetry, "__file__", str(checkout / ".venv/lib/site-packages/compass/telemetry.py")
    )
    metadata = telemetry._execution_metadata()
    assert set(metadata) == {"compass.version", "nooa.version", "python.version"}
    assert all(metadata.values())
    monkeypatch.setattr(telemetry, "__file__", str(checkout / "src/compass/telemetry.py"))
    monkeypatch.setenv("PATH", "")
    assert telemetry._execution_metadata() == metadata


@pytest.mark.parametrize("same_endpoint", [True, False])
def test_analyzing_telemetry_project_checks_source_endpoint(monkeypatch, capsys, same_endpoint):
    from unittest.mock import patch

    from compass.config import RunConfig
    from compass.telemetry import telemetry_session

    monkeypatch.setenv("COMPASS_TELEMETRY_LANGSMITH_API_KEY", "test-key")
    monkeypatch.setenv("COMPASS_TELEMETRY_LANGSMITH_PROJECT", "custom-project")
    monkeypatch.setenv("COMPASS_TELEMETRY_LANGSMITH_ENDPOINT", "https://telemetry.example.com")
    config = RunConfig(
        trace={
            "langsmith": {
                "project": "custom-project",
                "api_url": (
                    "https://telemetry.example.com/"
                    if same_endpoint
                    else "https://source.example.com"
                ),
            }
        }
    )
    with (
        patch("compass.telemetry.enable_tracing") as enable,
        patch("compass.telemetry.OTLPSpanExporter"),
        patch("compass.telemetry.shutdown_traces"),
    ):
        with telemetry_session(config):
            pass
    assert enable.called == (not same_endpoint)
    assert ("analyzing the telemetry project itself" in capsys.readouterr().err) == same_endpoint


@pytest.mark.parametrize(
    "api_key,project", [("", "custom-project"), ("test-key", ""), ("test-key", "custom-project")]
)
def test_cli_telemetry_in_fresh_process(tmp_path, api_key, project):
    """Exercise CLI opt-in, input provenance, library instrumentation, and YAML stdout."""
    source = tmp_path / "input.jsonl"
    source.write_text(
        Trace(
            id="input",
            root_spans=[],
            aggregate=TraceAggregate(),
        ).model_dump_json()
        + "\n"
    )
    result = subprocess.run(
        [sys.executable, __file__, str(source), str(bool(api_key and project))],
        env={
            **os.environ,
            "COMPASS_API_KEY": "dummy",
            "LANGSMITH_API_KEY": "source-key",
            "LANGCHAIN_API_KEY": "legacy-source-key",
            "LANGSMITH_ENDPOINT": "https://source.example.com",
            "LANGSMITH_WORKSPACE_ID": "source-workspace",
            "COMPASS_TELEMETRY_LANGSMITH_API_KEY": api_key,
            "COMPASS_TELEMETRY_LANGSMITH_PROJECT": project,
            "COMPASS_TELEMETRY_LANGSMITH_ENDPOINT": "https://telemetry.example.com/api/v1/",
            "COMPASS_TELEMETRY_LANGSMITH_WORKSPACE_ID": "telemetry-workspace",
            "LANGSMITH_TRACING": "true",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert yaml.safe_load(result.stdout) == []
    if api_key and not project:
        assert "set COMPASS_TELEMETRY_LANGSMITH_PROJECT" in result.stderr


# Nooa owns process-wide instrumentation. Each case gets the same lifecycle as the CLI.
def _run_cli(source, enabled):
    import json
    from unittest.mock import patch

    from nooa import Agent, strategy
    from nooa.strategies import CodeActStrategy
    from nooa.unifiedllm import CompletionClient, FakeLLMClient, LLMResponse, ToolCall
    from opentelemetry.exporter.otlp.proto.http import Compression
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from compass.cli.main import main
    from compass.compass_signals.compass_signals import CompassSignalResult
    from compass.telemetry import span

    class SmallAgent(Agent):
        @strategy(CodeActStrategy())
        async def answer(self) -> int:  # ty: ignore[empty-body] -- Nooa generates this method.
            """Return 7."""
            ...

    async def analyze(*args):
        fake = FakeLLMClient(
            [
                LLMResponse(
                    raw_response=None,
                    content="",
                    finish_reason="tool_calls",
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="execute_python",
                            arguments='{"code":"return_result(7)"}',
                        )
                    ],
                    assistant_message={"role": "assistant", "content": ""},
                )
            ]
        )
        with span("evidence.test") as record:
            assert await SmallAgent(llm=fake).answer() == 7
            async with CompletionClient(
                model="openai/gpt-4o-mini", api_key="dummy", mock_response="Hello"
            ) as llm:
                assert (
                    await llm.acall([{"role": "user", "content": "Say hello"}])
                ).content == "Hello"
            try:
                with span("failed_tool"):
                    raise ValueError("expected test failure")
            except ValueError:
                pass
            result = CompassSignalResult(signal_name="test", problems=())
            record.set_output(result)
            return [result]

    exporter = InMemorySpanExporter()
    args = ["--trace.filesystem.path", str(source)]
    with patch("compass.telemetry.OTLPSpanExporter", return_value=exporter) as factory:
        with patch("compass.cli.main._run_compass_signals", analyze):
            assert main(args) == 0
        assert factory.called == enabled
    records = exporter.get_finished_spans()
    if not enabled:
        assert not records
    else:
        factory.assert_called_once_with(
            endpoint="https://telemetry.example.com/api/v1/otel/v1/traces",
            headers={
                "x-api-key": "test-key",
                "Langsmith-Project": "custom-project",
                "X-Tenant-Id": "telemetry-workspace",
            },
            compression=Compression.Gzip,
            timeout=60,
        )
        root = next(r for r in records if r.name == "compass")
        metadata = json.loads(str((root.attributes or {})["metadata"]))
        assert metadata["compass.version"]
        assert metadata["nooa.version"]
        assert metadata["python.version"]
        assert all(r.context.trace_id == root.context.trace_id for r in records)
        ids = {r.context.span_id for r in records}
        assert all(r.parent.span_id in ids for r in records if r.parent)
        assert json.loads(str((root.attributes or {})["output.value"]))["insights"] == []

        inputs = json.loads(str((root.attributes or {})["input.value"]))
        assert inputs["input_source"] == {
            "source": f"fs:{source}",
            "trace_count": 1,
            "call_count": 0,
            "distinct_logical_cases": 1,
        }
        ingestion = next(r for r in records if r.name == "ingestion")
        assert json.loads(str((ingestion.attributes or {})["output.value"])) == {"trace_count": 1}
        assert any(r.name == "code_execution" for r in records)
        assert not next(r for r in records if r.name == "failed_tool").status.is_ok
        model = next(
            r for r in records if (r.attributes or {}).get("openinference.span.kind") == "LLM"
        )
        assert model.attributes is not None
        assert model.attributes["llm.input_messages.0.message.content"] == "Say hello"
        assert model.attributes["llm.output_messages.0.message.content"] == "Hello"


if __name__ == "__main__":
    _run_cli(sys.argv[1], sys.argv[2] == "True")
