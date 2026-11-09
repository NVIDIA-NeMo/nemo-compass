# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

import pytest
import yaml

from compass.traces import Trace, TraceAggregate


def test_export_batch_limit_preserves_other_spans(monkeypatch, caplog):
    from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from compass import telemetry_export

    provider = TracerProvider()
    captured = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(captured))
    tracer = provider.get_tracer("test")
    for name, size in [("first", 600), ("oversized", 2000), ("last", 600)]:
        with tracer.start_as_current_span(name, attributes={"input.value": "x" * size}):
            pass
    spans = captured.get_finished_spans()
    monkeypatch.setattr(telemetry_export, "_MAX_BATCH_BYTES", 1024)

    class Receiver(InMemorySpanExporter):
        def export(self, spans):
            assert encode_spans(spans).ByteSize() <= 1024
            return super().export(spans)

    receiver = Receiver()
    exporter = telemetry_export.CompassSpanExporter(receiver)
    assert exporter.export([spans[0], spans[2]]) is SpanExportResult.SUCCESS
    assert [span.name for span in receiver.get_finished_spans()] == ["first", "last"]
    receiver.clear()
    assert exporter.export(spans) is SpanExportResult.FAILURE
    assert [span.name for span in receiver.get_finished_spans()] == ["first", "last"]
    assert "Cannot export span oversized" in caplog.text
    provider.shutdown()


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


@pytest.mark.parametrize(
    "public_key,secret_key,base_url,tags",
    [
        ("", "", "https://telemetry.example.com/", None),
        ("test-public", "", "https://telemetry.example.com/", None),
        ("", "test-secret", "https://telemetry.example.com/", None),
        ("test-public", "test-secret", "", " , , "),
        (
            "test-public",
            "test-secret",
            "https://telemetry.example.com/",
            " development, experiment-a,development, ,",
        ),
    ],
)
def test_cli_telemetry_in_fresh_process(tmp_path, public_key, secret_key, base_url, tags):
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
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("COMPASS_TELEMETRY_")
    }
    if tags is not None:
        environment["COMPASS_TELEMETRY_TAGS"] = tags
    result = subprocess.run(
        [sys.executable, __file__, str(source), str(bool(public_key and secret_key))],
        env={
            **environment,
            "COMPASS_API_KEY": "dummy",
            "LANGSMITH_API_KEY": "source-key",
            "LANGCHAIN_API_KEY": "legacy-source-key",
            "LANGSMITH_ENDPOINT": "https://source.example.com",
            "LANGSMITH_WORKSPACE_ID": "source-workspace",
            "COMPASS_TELEMETRY_LANGFUSE_BASE_URL": base_url,
            "COMPASS_TELEMETRY_LANGFUSE_PUBLIC_KEY": public_key,
            "COMPASS_TELEMETRY_LANGFUSE_SECRET_KEY": secret_key,
            "LANGFUSE_PUBLIC_KEY": "source-public",
            "LANGFUSE_SECRET_KEY": "source-secret",
            "LANGFUSE_BASE_URL": "https://source.example.com",
            # Old sink credentials and trace-source credentials must not enable export.
            "COMPASS_TELEMETRY_LANGSMITH_API_KEY": "old-sink-key",
            "COMPASS_TELEMETRY_LANGSMITH_PROJECT": "old-project",
            "LANGSMITH_TRACING": "true",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert yaml.safe_load(result.stdout) == []
    assert ("set both COMPASS_TELEMETRY_LANGFUSE_PUBLIC_KEY" in result.stderr) == (
        bool(public_key) != bool(secret_key)
    )


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
    from compass.telemetry import record_span

    class SmallAgent(Agent):
        @strategy(CodeActStrategy())
        async def answer(self) -> int:  # ty: ignore[empty-body] -- Nooa generates this method.
            """Return 7."""
            ...

    async def analyze(*args, **kwargs):
        fake = FakeLLMClient(
            [
                LLMResponse(
                    raw_response=None,
                    finish_reason="tool_calls",
                    parts=(
                        ToolCall(
                            id="call-1",
                            name="execute_python",
                            arguments='{"code":"return_result(7)"}',
                        ),
                    ),
                )
            ]
        )
        with record_span("evidence.test") as record:
            assert await SmallAgent(llm=fake).answer() == 7
            async with CompletionClient(
                model="openai/gpt-4o-mini", api_key="dummy", mock_response="Hello"
            ) as llm:
                assert (
                    await llm.acall([{"role": "user", "content": "Say hello"}])
                ).content == "Hello"
            try:
                with record_span("failed_tool"):
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
        import base64

        base_url = os.getenv("COMPASS_TELEMETRY_LANGFUSE_BASE_URL") or "https://cloud.langfuse.com"
        auth = base64.b64encode(b"test-public:test-secret").decode()
        factory.assert_called_once_with(
            endpoint=f"{base_url.rstrip('/')}/api/public/otel/v1/traces",
            headers={"Authorization": f"Basic {auth}", "x-langfuse-ingestion-version": "4"},
            compression=Compression.Gzip,
            timeout=60,
        )
        root = next(r for r in records if r.name == "compass")
        expected_tags = (
            ("development", "experiment-a")
            if "development" in os.getenv("COMPASS_TELEMETRY_TAGS", "")
            else None
        )
        assert (root.attributes or {}).get("langfuse.trace.tags") == expected_tags
        assert all(
            (record.attributes or {}).get("langfuse.trace.tags") == expected_tags
            for record in records
        )
        metadata = json.loads(str((root.attributes or {})["metadata"]))
        assert metadata["compass.version"]
        assert metadata["nooa.version"]
        assert metadata["python.version"]
        assert all(r.resource.attributes["service.name"] == "compass" for r in records)
        assert len({(r.attributes or {})["session.id"] for r in records}) == 1
        assert all(r.context.trace_id == root.context.trace_id for r in records)
        ids = {r.context.span_id for r in records}
        assert all(r.parent.span_id in ids for r in records if r.parent)
        assert json.loads(str((root.attributes or {})["output.value"]))["insights"] == []

        inputs = json.loads(str((root.attributes or {})["input.value"]))
        assert inputs["input_source"] == {
            "source": f"fs:{source}",
            "trace_count": 1,
            "call_count": 0,
            "distinct_cases": 1,
            "distinct_sessions": 0,
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
