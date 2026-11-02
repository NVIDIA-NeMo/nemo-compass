# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration and descriptions for Langfuse trace sources."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from trace_ingest.loaders.trace_loaders import TraceDescription
from trace_ingest.models import SpanKind, TraceSnapshot, case_key

LANGFUSE_DEFAULT_MAX_TRACES = 100


class LangfuseTraceLoadError(RuntimeError):
    """A Langfuse source could not be selected or normalized safely."""


class _LangfuseCorpusDescription(TraceDescription):
    """Corpus facts shared by live and file sources."""

    observation_count: int
    unresolved_parent_count: int


class LangfuseTraceDescription(_LangfuseCorpusDescription):
    """Run metadata for a live Langfuse trace query."""

    api_version: str | None
    base_url: str
    filter: str | None
    from_timestamp: str
    to_timestamp: str
    max_traces: int


@dataclass(frozen=True)
class LangfuseTraceConfig:
    """Configuration for loading traces through the Langfuse v3 or v4 API."""

    from_timestamp: datetime
    to_timestamp: datetime
    base_url: str | None = None
    filter_string: str | None = None
    max_traces: int = LANGFUSE_DEFAULT_MAX_TRACES

    def __post_init__(self) -> None:
        validate_langfuse_time_window(self.from_timestamp, self.to_timestamp)
        if self.max_traces < 1:
            raise ValueError("max_traces must be at least 1")


def validate_langfuse_time_window(from_timestamp: datetime, to_timestamp: datetime) -> None:
    """Require a timezone-aware, nonempty window for CLI and direct API loading."""
    for name, value in (("from_timestamp", from_timestamp), ("to_timestamp", to_timestamp)):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{name} must include a timezone")
    if to_timestamp <= from_timestamp:
        raise ValueError("to_timestamp must be after from_timestamp")


class LangfuseFileTraceDescription(_LangfuseCorpusDescription):
    """Run metadata for a local native trace-detail export."""

    export_path: str
    export_trace_count: int
    max_traces: int
    truncated: bool


@dataclass(frozen=True)
class LangfuseFileTraceConfig:
    """A native trace-detail JSON/JSONL file or directory of such files."""

    path: Path
    max_traces: int = LANGFUSE_DEFAULT_MAX_TRACES

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if self.max_traces < 1:
            raise ValueError("max_traces must be at least 1")


def _describe_corpus(source: str, snapshot: TraceSnapshot) -> _LangfuseCorpusDescription:
    observation_count = call_count = unresolved_parent_count = 0
    pending = [span for trace in snapshot for span in trace.root_spans]
    while pending:
        span = pending.pop()
        observation_count += 1
        call_count += span.kind is SpanKind.TOOL
        pointer = span.attributes.get("source_pointer")
        unresolved_parent_count += (
            isinstance(pointer, dict) and "unresolved_parent_observation_id" in pointer
        )
        pending.extend(span.children)
    return {
        "source": source,
        "trace_count": snapshot.trace_count,
        "call_count": call_count,
        "distinct_cases": len({case_key(trace) for trace in snapshot}),
        "distinct_sessions": len({trace.session_id for trace in snapshot if trace.session_id}),
        "observation_count": observation_count,
        "unresolved_parent_count": unresolved_parent_count,
    }


def _describe_live_source(
    config: LangfuseTraceConfig,
    base_url: str,
    api_version: str | None,
    snapshot: TraceSnapshot,
) -> LangfuseTraceDescription:
    return {
        **_describe_corpus(f"langfuse:{base_url}", snapshot),
        "base_url": base_url,
        "api_version": api_version,
        "filter": config.filter_string,
        "from_timestamp": config.from_timestamp.isoformat(),
        "to_timestamp": config.to_timestamp.isoformat(),
        "max_traces": config.max_traces,
    }
