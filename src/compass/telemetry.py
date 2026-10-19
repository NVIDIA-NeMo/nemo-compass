# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-wide tracing for one Compass CLI invocation; enabled by dedicated credentials."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stdout
from functools import partial
from pathlib import Path

import nooa
from nooa.tracing import enable_tracing, exporters, shutdown_traces
from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation._spans import OpenInferenceSpan
from opentelemetry import trace

from compass.config import RunConfig
from compass.version import __version__

tracer = OITracer(trace.get_tracer("compass", __version__), config=TraceConfig())
span = partial(tracer.start_as_current_span, openinference_span_kind="chain")


def _execution_metadata() -> dict[str, str | bool]:
    metadata: dict[str, str | bool] = {
        "compass.version": __version__,
        "nooa.version": nooa.__version__,
        "python.version": platform.python_version(),
    }
    # Inspect Compass's checkout, not the caller's or analyzed agent's repository.
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / ".git").exists():
        git = partial(
            subprocess.check_output,
            cwd=checkout,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
        try:
            sha = git(["git", "rev-parse", "HEAD"]).strip()
            dirty = bool(git(["git", "status", "--porcelain", "--untracked-files=normal"]))
            metadata.update({"compass.git_sha": sha, "compass.git_dirty": dirty})
        except (OSError, subprocess.SubprocessError):
            pass  # Git is optional; installed wheels need only their package version.
    return metadata


@contextmanager
def telemetry_session(config: RunConfig) -> Iterator[OpenInferenceSpan]:
    api_key = os.getenv("COMPASS_TELEMETRY_LANGSMITH_API_KEY", "").strip()
    project = os.getenv("COMPASS_TELEMETRY_LANGSMITH_PROJECT", "").strip()
    if api_key and not project:
        print("Compass telemetry skipped: set COMPASS_TELEMETRY_LANGSMITH_PROJECT", file=sys.stderr)
        api_key = ""
    endpoint = (
        os.getenv("COMPASS_TELEMETRY_LANGSMITH_ENDPOINT") or "https://api.smith.langchain.com"
    ).rstrip("/")
    source = config.trace.langsmith
    if api_key and source and source.project == project:
        from langsmith.utils import get_api_url

        if get_api_url(source.api_url) == endpoint:
            print(
                "Compass telemetry skipped: analyzing the telemetry project itself", file=sys.stderr
            )
            api_key = ""
    if api_key:
        headers = {"x-api-key": api_key, "Langsmith-Project": project}
        if workspace := os.getenv("COMPASS_TELEMETRY_LANGSMITH_WORKSPACE_ID"):
            headers["X-Tenant-Id"] = workspace
        # Nooa prints initialization messages; stdout belongs to the CLI's YAML output.
        with redirect_stdout(sys.stderr):
            enable_tracing(
                exporters=[exporters.otlp(f"{endpoint}/otel/v1/traces", headers)],
            )
    try:
        with span("compass") as root:
            root.set_input(config.model_dump(mode="json", exclude={"api_base"}))
            if api_key:
                # Use OpenInference span metadata for LangSmith compatibility.
                root.set_attribute("metadata", json.dumps(_execution_metadata()))
                print(
                    f"Compass telemetry: full content to {project}; OTLP trace ID {root.get_span_context().trace_id:032x}",
                    file=sys.stderr,
                )
            yield root
    finally:
        if api_key:
            shutdown_traces()
