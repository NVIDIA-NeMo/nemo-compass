# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load native Langfuse v3 trace-detail exports."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from trace_ingest.loaders.langfuse.config import (
    LangfuseFileTraceConfig,
    LangfuseFileTraceDescription,
    LangfuseTraceLoadError,
    _describe_corpus,
)
from trace_ingest.loaders.langfuse.file_io import _parse_trace_export
from trace_ingest.loaders.langfuse.normalization import (
    _normalize_trace,
    _trace_id,
    _trace_timestamp,
)
from trace_ingest.models import Trace, TraceSnapshot
from trace_ingest.source_links import file_source_url


@dataclass
class LangfuseV3FileTraceLoader:
    """Read full v3 trace responses without contacting Langfuse.

    Accepts raw API bodies and the CLI's status/headers/body envelopes. All
    records are validated before selecting the newest complete traces.
    """

    config: LangfuseFileTraceConfig
    _snapshot: TraceSnapshot = field(
        default_factory=lambda: TraceSnapshot([]), init=False, repr=False
    )
    _export_trace_count: int = field(default=0, init=False, repr=False)

    def load(self) -> TraceSnapshot:
        """Validate native exports and reuse live trace normalization."""
        from langfuse.api.commons.types.trace_with_full_details import (
            TraceWithFullDetails,
        )
        from langfuse.api.core.pydantic_utilities import parse_obj_as

        export_path = self.config.path.resolve()
        files = (
            sorted(
                p for p in export_path.iterdir() if p.suffix in {".json", ".jsonl"} and p.is_file()
            )
            if export_path.is_dir()
            else [export_path]
        )
        if not files:
            raise LangfuseTraceLoadError(
                f"Langfuse export {export_path} contains no JSON/JSONL files"
            )
        traces: dict[tuple[datetime, str], Trace] = {}
        seen: set[str] = set()
        for path in files:
            contents = path.read_bytes()
            for index, record in enumerate(_parse_trace_export(contents, path), start=1):
                try:
                    provider = parse_obj_as(TraceWithFullDetails, record)
                    trace_id = _trace_id(provider)
                    if trace_id in seen:
                        raise LangfuseTraceLoadError(f"duplicate trace id {trace_id!r}")
                    seen.add(trace_id)
                    if any(obs.trace_id != trace_id for obs in provider.observations):
                        raise LangfuseTraceLoadError("observation traceId does not match its trace")
                    trace = _normalize_trace(
                        provider,
                        require_complete_tree=True,
                        source_pointer={
                            "provider": "langfuse",
                            "export_path": str(path),
                            "record": index,
                            "trace_id": trace_id,
                        },
                    )
                    trace.source_url = file_source_url(path)
                    traces[(_trace_timestamp(provider), trace_id)] = trace
                except (ValueError, LangfuseTraceLoadError) as error:
                    raise LangfuseTraceLoadError(
                        f"Invalid Langfuse export {path}, record {index}: {error}"
                    ) from error

        snapshot = TraceSnapshot(traces[key] for key in sorted(traces)[-self.config.max_traces :])
        self._export_trace_count = len(traces)
        self._snapshot = snapshot
        return snapshot

    def describe(self) -> LangfuseFileTraceDescription:
        """Describe the last load and its source export."""
        return {
            **_describe_corpus(f"langfuse-export:{self.config.path.resolve()}", self._snapshot),
            "export_path": str(self.config.path.resolve()),
            "export_trace_count": self._export_trace_count,
            "max_traces": self.config.max_traces,
            "truncated": self._export_trace_count > self._snapshot.trace_count,
        }
