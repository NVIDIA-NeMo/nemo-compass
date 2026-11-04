# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-wide tracing for one Compass CLI invocation; enabled by dedicated credentials."""

from __future__ import annotations

import asyncio
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
from opentelemetry.sdk.trace import TracerProvider

from compass.config import RunConfig
from compass.telemetry_buffer import BufferedSpanExporter
from compass.telemetry_export import CompassSpanExporter, SpanInventory, span_id_digest
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
        # Keep embedding spans and usage, without exporting their large numeric vectors.
        # NOOA's LiteLLM instrumentor reads this when tracing is initialized below.
        os.environ["OPENINFERENCE_HIDE_EMBEDDINGS_VECTORS"] = "true"
        headers = {"x-api-key": api_key, "Langsmith-Project": project}
        if workspace := os.getenv("COMPASS_TELEMETRY_LANGSMITH_WORKSPACE_ID"):
            headers["X-Tenant-Id"] = workspace
        transport = CompassSpanExporter(
            OTLPSpanExporter(
                endpoint=f"{endpoint}/otel/v1/traces",
                headers=headers,
                compression=Compression.Gzip,
                timeout=float(
                    os.getenv("OTEL_EXPORTER_OTLP_TRACES_TIMEOUT")
                    or os.getenv("OTEL_EXPORTER_OTLP_TIMEOUT")
                    or "60"
                ),
            )
        )
        archive_dir = os.getenv("COMPASS_TELEMETRY_ARCHIVE_DIR")
        exporter = BufferedSpanExporter(
            transport, archive_dir=Path(archive_dir) if archive_dir else None
        )
        # Nooa prints initialization messages; stdout belongs to the CLI's YAML output.
        with redirect_stdout(sys.stderr):
            enable_tracing(exporters=[exporter])
        inventory = SpanInventory()
        provider = trace.get_tracer_provider()
        if isinstance(provider, TracerProvider):
            provider.add_span_processor(inventory)
    try:
        with record_span("compass") as root:
            root.set_input(config.model_dump(mode="json", exclude={"api_base"}))
            if api_key:
                # Use OpenInference span metadata for LangSmith compatibility.
                root.set_attribute("metadata", json.dumps(_execution_metadata()))
                tags = [
                    tag.strip()
                    for tag in os.getenv("COMPASS_TELEMETRY_LANGSMITH_TAGS", "").split(",")
                    if tag.strip()
                ]
                if tags:
                    root.set_attribute("langsmith.span.tags", ",".join(dict.fromkeys(tags)))
                print(
                    f"Compass telemetry: execution content and source provenance to {project}; OTLP trace ID {root.get_span_context().trace_id:032x}",
                    file=sys.stderr,
                )
            try:
                yield root
            finally:
                if api_key:
                    # The root is still open and already in the inventory. Consumers
                    # must reconcile this manifest with stored IDs, not wait for a
                    # stable (but potentially incomplete) snapshot.
                    root.set_attribute(
                        "metadata",
                        json.dumps(
                            {
                                **_execution_metadata(),
                                "compass.telemetry": inventory.metadata(),
                            }
                        ),
                    )
    finally:
        if api_key:
            shutdown_traces()
            expected = inventory.started
            exported = exporter.exported_span_ids
            print(
                "Compass telemetry summary: "
                + json.dumps(
                    {
                        **inventory.metadata(),
                        "ended_span_count": len(inventory.ended),
                        "exported_span_count": len(exported),
                        "exported_span_id_sha256": span_id_digest(exported),
                        "failed_span_count": exporter.failed_span_count,
                        "status": "exported"
                        if expected == inventory.ended == exported
                        else "incomplete",
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
