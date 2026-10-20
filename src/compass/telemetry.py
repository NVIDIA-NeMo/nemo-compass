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
from opentelemetry.sdk.trace import ReadableSpan

from compass.config import RunConfig
from compass.config import TraceConfig as SourceConfig
from compass.telemetry_export import CompassSpanExporter
from compass.version import __version__
from trace_ingest.loaders.trace_loaders import TraceDescription, TraceLoader
from trace_ingest.models import TraceSnapshot

tracer = OITracer(trace.get_tracer("compass", __version__), config=TraceConfig())


@contextmanager
def span(name: str) -> Iterator[OpenInferenceSpan]:
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


def _input_metadata(
    config: SourceConfig, description: TraceDescription, snapshot: TraceSnapshot
) -> dict:
    """Identify the loaded corpus without per-trace lists or duplicate query settings."""
    source_type, source = next(
        (name, value) for name, value in config if name != "max_traces" and value is not None
    )
    configured = source.model_dump()
    resolved = dict(description)
    coordinates = {
        key: resolved.get(key)
        for key in (
            "project_id",
            "experiment_id",
            "workspace",
            "api_version",
            "max_traces",
            "truncated",
            "export_trace_count",
            "ignored_score_count",
        )
    }
    if source_type == "intake":
        query = resolved.get("query")
        if isinstance(query, dict):
            coordinates["max_traces"] = query.get("max_traces")
    coordinates["api_version"] = coordinates["api_version"] or configured.get("api_version")
    coordinates["endpoint"] = (
        resolved.get("base_url")
        or resolved.get("api_url")
        or resolved.get("tracking_uri")
        or configured.get("base_url")
    )
    for input_name, output_name in (("path", "source_path"), ("scores_path", "scores_path")):
        path = configured.get(input_name)
        if isinstance(path, Path):
            coordinates[output_name] = str(path.resolve())
    return {
        "source_type": source_type,
        "trace_count": len(snapshot),
        "snapshot_saved": False,
        **coordinates,
    }


def _input_metadata_fields(metadata: dict) -> dict:
    """Export searchable scalars; full coordinates remain in the root input."""
    return {
        f"compass.input.{key}": value
        for key, value in {**metadata, "details_location": "inputs.input_source"}.items()
        if value is not None and (not isinstance(value, str) or len(value) < 500)
    }


def record_input_metadata(config: RunConfig, loader: TraceLoader, snapshot: TraceSnapshot) -> None:
    """Attach provenance to the current Compass root after its ingestion span closes."""
    root = trace.get_current_span()
    if root.is_recording():
        input_source = _input_metadata(config.trace, loader.describe(), snapshot)
        root_input = {
            **config.model_dump(mode="json", exclude={"api_base"}),
            "input_source": input_source,
        }
        root.set_attribute("input.value", json.dumps(root_input))
        root.set_attribute("input.mime_type", "application/json")
        attributes = root.attributes if isinstance(root, ReadableSpan) else {}
        existing_metadata = (attributes or {}).get("metadata")
        metadata = json.loads(existing_metadata) if isinstance(existing_metadata, str) else {}
        metadata.update(_input_metadata_fields(input_source))
        root.set_attribute("metadata", json.dumps(metadata))


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
        headers = {"x-api-key": api_key, "Langsmith-Project": project}
        if workspace := os.getenv("COMPASS_TELEMETRY_LANGSMITH_WORKSPACE_ID"):
            headers["X-Tenant-Id"] = workspace
        # Nooa prints initialization messages; stdout belongs to the CLI's YAML output.
        with redirect_stdout(sys.stderr):
            enable_tracing(
                exporters=[
                    CompassSpanExporter(
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
                ],
            )
    try:
        with span("compass") as root:
            root.set_input(config.model_dump(mode="json", exclude={"api_base"}))
            if api_key:
                # Use OpenInference span metadata for LangSmith compatibility.
                root.set_attribute("metadata", json.dumps(_execution_metadata()))
                print(
                    f"Compass telemetry: execution content and source provenance to {project}; OTLP trace ID {root.get_span_context().trace_id:032x}",
                    file=sys.stderr,
                )
            yield root
    finally:
        if api_key:
            shutdown_traces()
