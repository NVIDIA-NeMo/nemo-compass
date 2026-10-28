# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the installed CLI through compilation setup, stopping at its first LLM request."""

import json
import os
import sys
from contextlib import chdir
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from nooa.unifiedllm import FakeLLMClient

import compass.cli.main as cli
from compass.config import CompassSignalsConfig


class InferenceReached(BaseException):
    """Stop before inference without triggering the framework's error retries."""


def deny_network(event: str, args: tuple[object, ...]) -> None:
    if event in {"socket.connect", "socket.getaddrinfo"}:
        raise RuntimeError("The startup check must not access the network")


sys.addaudithook(deny_network)
os.environ["COMPASS_API_KEY"] = "offline-smoke-check"
os.environ["COMPASS_TELEMETRY_LANGSMITH_API_KEY"] = ""

signals = dict.fromkeys(CompassSignalsConfig.model_fields, False) | {"tool_issues": True}
llm = FakeLLMClient()
with (
    TemporaryDirectory() as directory,
    chdir(directory),
    patch.object(cli, "_build_llm", return_value=llm),
    patch.object(llm, "acall", side_effect=InferenceReached),
):
    # Three independent tool failures produce a real candidate for compilation.
    span = {"id": "tool", "kind": "TOOL", "tool_name": "lookup", "error": "timeout"}
    Path("traces.jsonl").write_text(
        "\n".join(
            json.dumps({"id": str(i), "aggregate": {}, "root_spans": [span]}) for i in range(3)
        ),
        encoding="utf-8",
    )
    try:
        cli.main(
            ["--trace.filesystem.path", "traces.jsonl", "--compass-signals", json.dumps(signals)]
        )
    except InferenceReached:
        print("Startup reached compilation's first LLM request.")
    else:
        raise AssertionError("Startup did not reach compilation's first LLM request")
