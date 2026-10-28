# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import socket
import subprocess
import sys

import pytest
from nooa.unifiedllm import FakeLLMClient

import compass.cli.main as cli
from compass.config import CompassSignalsConfig


class InferenceReached(BaseException):
    """Stop before inference without triggering the framework's error retries."""


def test_cli_reaches_compilation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COMPASS_API_KEY", "offline-smoke-check")
    monkeypatch.setenv("COMPASS_TELEMETRY_LANGSMITH_API_KEY", "")

    def deny_network(*args, **kwargs):
        raise OSError("Network disabled for the startup check")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket, "getaddrinfo", deny_network)
    llm = FakeLLMClient()

    async def stop_before_inference(*args, **kwargs):
        raise InferenceReached

    monkeypatch.setattr(llm, "acall", stop_before_inference)
    monkeypatch.setattr(cli, "_build_llm", lambda *args: llm)
    # Three independent tool failures produce a real candidate for compilation.
    span = {"id": "tool", "kind": "TOOL", "tool_name": "lookup", "error": "timeout"}
    (tmp_path / "traces.jsonl").write_text(
        "\n".join(
            json.dumps({"id": str(i), "aggregate": {}, "root_spans": [span]}) for i in range(3)
        ),
        encoding="utf-8",
    )
    signals = dict.fromkeys(CompassSignalsConfig.model_fields, False) | {"tool_issues": True}
    with pytest.raises(InferenceReached):
        cli.main(
            ["--trace.filesystem.path", "traces.jsonl", "--compass-signals", json.dumps(signals)]
        )


@pytest.mark.parametrize("model", ["ToolCall", "ToolResult", "Trace"])
def test_nooa_inspects_cold_trace_models(model):
    # Each entry starts cold: no earlier test, construction, or schema generation
    # can hide a first-use failure in NOOA's inspection of these nested models.
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            f"from nooa.agentdoc import doc; from trace_ingest import {model}; "
            f"doc({model}, inline_depth=4)",
        ],
        check=True,
        timeout=30,
    )
