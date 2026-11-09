# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-wide tracing for one Compass CLI invocation; opt-in export to Langfuse."""

from __future__ import annotations

import asyncio
import base64
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
from nooa.tracing import enable_tracing, shutdown_traces
from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation._spans import OpenInferenceSpan
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from compass.config import RunConfig
from compass.telemetry_export import CompassSpanExporter
from compass.version import __version__
from trace_ingest.loaders.trace_loaders import TraceLoader

tracer = OITracer(trace.get_tracer("compass", __version__), config=TraceConfig())


@contextmanager
def record_span(name: str) -> Iterator[OpenInferenceSpan]:
    """Record one Compass operation, including cancellation as interrupted work."""
    with tracer.start_as_current_span(name, openinference_span_kind="chain") as record:
        try:
            yield record
        except asyncio.CancelledError as error:
            # OTel intentionally ignores BaseException subclasses. Cancellation is
            # not a completed Compass operation, so retain the actual exception.
            record.record_exception(error)
            record.set_status(trace.StatusCode.ERROR)
            raise


def record_input_metadata(config: RunConfig, loader: TraceLoader) -> None:
    """Attach provenance to the current Compass root after its ingestion span closes."""
    root = trace.get_current_span()
    if root.is_recording():
        root_input = {
            **config.model_dump(mode="json", exclude={"api_base"}),
            "input_source": loader.describe(),
        }
        root.set_attribute("input.value", json.dumps(root_input))
        root.set_attribute("input.mime_type", "application/json")


def _execution_metadata() -> dict[str, str | bool]:
    metadata = {
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
            metadata["compass.git_sha"] = sha
            metadata["compass.git_dirty"] = dirty
        except (OSError, subprocess.SubprocessError):
            pass  # Git is optional; installed wheels need only their package version.
    return metadata


@contextmanager
def telemetry_session(config: RunConfig) -> Iterator[OpenInferenceSpan]:
    public_key = os.getenv("COMPASS_TELEMETRY_LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.getenv("COMPASS_TELEMETRY_LANGFUSE_SECRET_KEY", "").strip()
    enabled = bool(public_key and secret_key)
    if bool(public_key) != bool(secret_key):
        print(
            "Compass telemetry skipped: set both COMPASS_TELEMETRY_LANGFUSE_PUBLIC_KEY "
            "and COMPASS_TELEMETRY_LANGFUSE_SECRET_KEY",
            file=sys.stderr,
        )
    base_url = (
        os.getenv("COMPASS_TELEMETRY_LANGFUSE_BASE_URL", "").strip() or "https://cloud.langfuse.com"
    ).rstrip("/")
    tags = list(
        dict.fromkeys(
            tag.strip() for tag in os.getenv("COMPASS_TELEMETRY_TAGS", "").split(",") if tag.strip()
        )
    )
    if enabled:
        auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        headers = {"Authorization": f"Basic {auth}", "x-langfuse-ingestion-version": "4"}
        # Nooa prints initialization messages; stdout belongs to the CLI's YAML output.
        with redirect_stdout(sys.stderr):
            enable_tracing(
                exporters=[
                    CompassSpanExporter(
                        OTLPSpanExporter(
                            endpoint=f"{base_url}/api/public/otel/v1/traces",
                            headers=headers,
                            compression=Compression.Gzip,
                            timeout=float(
                                os.getenv("OTEL_EXPORTER_OTLP_TRACES_TIMEOUT")
                                or os.getenv("OTEL_EXPORTER_OTLP_TIMEOUT")
                                or "60"
                            ),
                        ),
                        tags=tags,
                    )
                ],
                extra_resource_attrs={"service.name": "compass"},
            )
    try:
        with record_span("compass") as root:
            root.set_input(config.model_dump(mode="json", exclude={"api_base"}))
            if enabled:
                root.set_attribute("metadata", json.dumps(_execution_metadata()))
                print(
                    f"Compass telemetry: execution content and source provenance to {base_url}; "
                    f"OTLP trace ID {root.get_span_context().trace_id:032x}",
                    file=sys.stderr,
                )
            yield root
    finally:
        if enabled:
            shutdown_traces()
