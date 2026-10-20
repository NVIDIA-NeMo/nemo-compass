# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load native ``langsmith trace export --full`` JSONL trace export files."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from trace_ingest.loaders.langsmith._normalization import (
    LANGSMITH_DEFAULT_MAX_TRACES,
    LangSmithTraceLoadError,
    LangSmithTraceLoadReport,
    normalize_trace,
    trace_sort_key,
    walk_spans,
)
from trace_ingest.loaders.trace_loaders import TraceDescription
from trace_ingest.models import SpanKind, Trace, TraceSnapshot
from trace_ingest.source_links import file_source_url

__all__ = [
    "LangSmithTraceExportFileConfig",
    "LangSmithTraceExportFileDescription",
    "LangSmithTraceExportFileLoader",
]

_BASE_FIELDS = {
    "run_id",
    "trace_id",
    "parent_run_id",
    "name",
    "run_type",
    "start_time",
    "end_time",
}
_FULL_FIELDS = {
    "status",
    "duration_ms",
    "first_token_time",
    "token_usage",
    "costs",
    "tags",
    "custom_metadata",
    "inputs",
    "outputs",
    "error",
    "events",
    "feedback_stats",
}


class LangSmithTraceExportFileDescription(TraceDescription):
    """Corpus metadata for a native LangSmith CLI trace export."""

    run_count: int
    unresolved_parent_count: int
    export_path: str
    export_trace_count: int
    truncated: bool
    max_traces: int


@dataclass(frozen=True)
class LangSmithTraceExportFileConfig:
    """Configuration for LangSmith trace export files."""

    path: Path
    max_traces: int = LANGSMITH_DEFAULT_MAX_TRACES

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if self.max_traces < 1:
            raise ValueError("max_traces must be at least 1")


@dataclass(frozen=True)
class _ExportRun:
    """Run-shaped data parsed from one record in a LangSmith trace export file."""

    id: str
    trace_id: str
    parent_run_id: str | None
    name: str | None
    run_type: str | None
    start_time: datetime
    end_time: datetime | None
    inputs: JsonValue | None
    outputs: JsonValue | None
    error: str | None
    status: str | None
    extra: dict[str, JsonValue]
    tags: JsonValue | None
    total_cost: float | None
    feedback_stats: JsonValue
    dotted_order: None = None
    app_path: None = None


@dataclass(frozen=True)
class _ExportTrace:
    """One validated complete trace from a per-trace or stitched JSONL file."""

    path: Path
    runs: list[_ExportRun]
    root: _ExportRun
    line_numbers: dict[str, int]


@dataclass
class LangSmithTraceExportFileLoader:
    """Normalize complete traces emitted by ``langsmith trace export --full``.

    Current LangSmith CLI exports contain one JSONL file per trace and one Run
    object per line. A single JSONL file is also accepted, including the
    stitched format produced by concatenating a directory's trace files. The
    newest root Runs are selected before ``max_traces`` is applied, so the bound
    matches LangSmith Trace Loader behavior and can never split a trace.
    """

    config: LangSmithTraceExportFileConfig
    report: LangSmithTraceLoadReport = field(default_factory=LangSmithTraceLoadReport, init=False)
    _snapshot: TraceSnapshot | None = field(default=None, init=False, repr=False)
    _description: LangSmithTraceExportFileDescription | None = field(
        default=None, init=False, repr=False
    )

    def load(self) -> TraceSnapshot:
        """Parse, validate, and normalize the configured CLI export."""

        if self._snapshot is not None:
            return self._snapshot

        export_path = self.config.path.resolve()
        files = _trace_export_files(self.config.path)
        traces_from_export = [
            trace
            for path in files
            for trace in _parse_trace_export_file(
                path,
                allow_multiple_traces=self.config.path.is_file(),
            )
        ]
        seen_export_trace_ids: set[str] = set()
        for export_trace in traces_from_export:
            trace_id = export_trace.root.trace_id
            if trace_id in seen_export_trace_ids:
                raise LangSmithTraceLoadError(
                    f"LangSmith export {str(export_path)!r} contains duplicate trace id "
                    f"{trace_id!r}"
                )
            seen_export_trace_ids.add(trace_id)
        export_trace_count = len(traces_from_export)
        traces_from_export = sorted(
            traces_from_export,
            key=lambda trace: trace_sort_key(trace.root, trace.root.trace_id),
            reverse=True,
        )[: self.config.max_traces]
        run_count = 0
        call_count = 0
        logical_cases: set[str] = set()

        def traces() -> Iterator[Trace]:
            nonlocal run_count, call_count
            for export_trace in reversed(traces_from_export):
                path = export_trace.path
                provider_runs = export_trace.runs
                root = export_trace.root
                trace_id = root.trace_id
                trace, unresolved = normalize_trace(
                    trace_id,
                    root,
                    provider_runs,
                    source_pointer={
                        "provider": "langsmith",
                        "export_path": str(export_path),
                        "export_file": path.name,
                    },
                    langsmith_attributes={"export_file": path.name},
                    run_source_pointers={
                        run.id: {"line_number": export_trace.line_numbers[run.id]}
                        for run in provider_runs
                    },
                )
                trace.source_url = file_source_url(path, export_trace.line_numbers[root.id])
                if unresolved:
                    # _parse_trace_export_file rejects this first; keep the invariant
                    # explicit if the shared normalizer changes later.
                    raise LangSmithTraceLoadError(
                        f"LangSmith export file {str(path)!r} is incomplete"
                    )
                run_count += len(provider_runs)
                call_count += sum(
                    span.kind is SpanKind.TOOL for span in walk_spans(trace.root_spans)
                )
                logical_cases.add(str(trace.attributes.get("logical_case_id") or trace.id))
                yield trace

        snapshot = TraceSnapshot(traces())

        self._snapshot = snapshot
        self.report = LangSmithTraceLoadReport(
            trace_count=snapshot.trace_count,
            run_count=run_count,
            unresolved_parent_count=0,
        )
        self._description = {
            "source": f"langsmith-trace-export-file:{export_path}",
            "trace_count": snapshot.trace_count,
            "call_count": call_count,
            "distinct_logical_cases": len(logical_cases),
            "run_count": run_count,
            "unresolved_parent_count": 0,
            "export_path": str(export_path),
            "export_trace_count": export_trace_count,
            "truncated": export_trace_count > snapshot.trace_count,
            "max_traces": self.config.max_traces,
        }
        return snapshot

    def describe(self) -> LangSmithTraceExportFileDescription:
        """Describe the normalized corpus and its local export provenance."""

        self.load()
        assert self._description is not None
        return self._description


def _trace_export_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".jsonl":
            raise LangSmithTraceLoadError(
                f"LangSmith export file {str(path)!r} must have a .jsonl extension"
            )
        return [path]
    if not path.exists():
        raise LangSmithTraceLoadError(f"LangSmith export path {str(path)!r} does not exist")
    if not path.is_dir():
        raise LangSmithTraceLoadError(
            f"LangSmith export path {str(path)!r} is not a file or directory"
        )
    files = sorted(
        (
            candidate
            for candidate in path.iterdir()
            if candidate.is_file() and candidate.suffix.lower() == ".jsonl"
        ),
        key=lambda candidate: candidate.name,
    )
    if not files:
        raise LangSmithTraceLoadError(
            f"LangSmith export directory {str(path)!r} contains no .jsonl trace files"
        )
    return files


def _parse_trace_export_file(path: Path, *, allow_multiple_traces: bool) -> list[_ExportTrace]:
    records: list[_ExportRun] = []
    line_numbers: dict[str, int] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, raw in enumerate(source, start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as error:
                raise LangSmithTraceLoadError(
                    f"{path}:{line_number}: malformed JSON: {error.msg}"
                ) from error
            run = _parse_run(value, path=path, line_number=line_number)
            if run.id in line_numbers:
                raise LangSmithTraceLoadError(
                    f"LangSmith export file {str(path)!r} contains duplicate Run id {run.id!r}"
                )
            line_numbers[run.id] = line_number
            records.append(run)

    if not records:
        raise LangSmithTraceLoadError(f"LangSmith export file {str(path)!r} contains no Runs")

    trace_ids = {run.trace_id for run in records}
    if len(trace_ids) != 1 and not allow_multiple_traces:
        raise LangSmithTraceLoadError(
            f"LangSmith export file {str(path)!r} mixes trace ids: {sorted(trace_ids)!r}"
        )
    grouped = {
        trace_id: [run for run in records if run.trace_id == trace_id] for trace_id in trace_ids
    }
    return [
        _validate_complete_trace(
            path,
            trace_id,
            grouped[trace_id],
            {run.id: line_numbers[run.id] for run in grouped[trace_id]},
        )
        for trace_id in sorted(grouped)
    ]


def _validate_complete_trace(
    path: Path,
    trace_id: str,
    records: list[_ExportRun],
    line_numbers: dict[str, int],
) -> _ExportTrace:
    roots = [run for run in records if run.parent_run_id is None]
    if len(roots) != 1:
        raise LangSmithTraceLoadError(
            f"LangSmith export file {str(path)!r} must contain exactly one root Run; "
            f"found {len(roots)}"
        )
    root = roots[0]
    if root.id != trace_id:
        raise LangSmithTraceLoadError(
            f"LangSmith export file {str(path)!r} root Run {root.id!r} does not match "
            f"trace id {trace_id!r}"
        )

    run_ids = set(line_numbers)
    for run in records:
        if run.parent_run_id is not None and run.parent_run_id not in run_ids:
            raise LangSmithTraceLoadError(
                f"LangSmith export file {str(path)!r} is incomplete: Run {run.id!r} "
                f"references missing parent {run.parent_run_id!r}"
            )
    # The shared normalizer performs deterministic topological sorting and
    # rejects cycles after these stricter complete-file checks.
    return _ExportTrace(path=path, runs=records, root=root, line_numbers=line_numbers)


def _parse_run(value: object, *, path: Path, line_number: int) -> _ExportRun:
    if not isinstance(value, Mapping):
        raise LangSmithTraceLoadError(f"{path}:{line_number}: Run must be a JSON object")

    missing = sorted((_BASE_FIELDS | _FULL_FIELDS).difference(value))
    if missing:
        raise LangSmithTraceLoadError(
            f"{path}:{line_number}: Run is missing {', '.join(missing)}; "
            "export traces with 'langsmith trace export OUTPUT_DIR --full'"
        )

    run_id = _required_string(value["run_id"], path, line_number, "run_id")
    trace_id = _required_string(value["trace_id"], path, line_number, "trace_id")
    parent_run_id = _optional_string(value["parent_run_id"], path, line_number, "parent_run_id")
    name = _optional_string(value["name"], path, line_number, "name")
    run_type = _optional_string(value["run_type"], path, line_number, "run_type")
    start_time = _timestamp(value["start_time"], path, line_number, "start_time", required=True)
    end_time = _timestamp(value["end_time"], path, line_number, "end_time", required=False)
    assert start_time is not None
    if end_time is not None and end_time < start_time:
        raise LangSmithTraceLoadError(f"{path}:{line_number}: Run {run_id!r} ends before it starts")

    custom_metadata = _optional_mapping(
        value["custom_metadata"], path, line_number, "custom_metadata"
    )
    _optional_mapping(value["token_usage"], path, line_number, "token_usage")
    costs = _optional_mapping(value["costs"], path, line_number, "costs")
    tags = value["tags"]
    if tags is not None and not (
        isinstance(tags, list) and all(isinstance(tag, str) for tag in tags)
    ):
        raise LangSmithTraceLoadError(
            f"{path}:{line_number}: tags must be null or an array of strings"
        )
    error = _optional_string(value["error"], path, line_number, "error")
    status = _optional_string(value["status"], path, line_number, "status")

    export_metadata = {
        field: value[field]
        for field in (
            "duration_ms",
            "first_token_time",
            "token_usage",
            "costs",
            "events",
            "feedback_stats",
        )
    }
    unknown = {
        str(key): item for key, item in value.items() if key not in _BASE_FIELDS | _FULL_FIELDS
    }
    if unknown:
        export_metadata["additional_fields"] = unknown
    extra: dict[str, JsonValue] = {
        "metadata": _json_value(custom_metadata),
        "langsmith_cli_export": _json_value(export_metadata),
    }

    return _ExportRun(
        id=run_id,
        trace_id=trace_id,
        parent_run_id=parent_run_id,
        name=name,
        run_type=run_type,
        start_time=start_time,
        end_time=end_time,
        inputs=_json_value(value["inputs"]),
        outputs=_json_value(value["outputs"]),
        error=error,
        status=status,
        extra=extra,
        tags=_json_value(tags),
        total_cost=_total_cost(costs, path, line_number),
        feedback_stats=_json_value(value["feedback_stats"]),
    )


def _required_string(value: object, path: Path, line_number: int, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise LangSmithTraceLoadError(
            f"{path}:{line_number}: {field_name} must be a non-empty string"
        )
    return value


def _optional_string(value: object, path: Path, line_number: int, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise LangSmithTraceLoadError(
            f"{path}:{line_number}: {field_name} must be null or a non-empty string"
        )
    return value


def _timestamp(
    value: object,
    path: Path,
    line_number: int,
    field_name: str,
    *,
    required: bool,
) -> datetime | None:
    if value is None and not required:
        return None
    if not isinstance(value, str):
        raise LangSmithTraceLoadError(
            f"{path}:{line_number}: {field_name} must be an RFC 3339 timestamp"
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise LangSmithTraceLoadError(
            f"{path}:{line_number}: {field_name} must be an RFC 3339 timestamp"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise LangSmithTraceLoadError(f"{path}:{line_number}: {field_name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _optional_mapping(
    value: object, path: Path, line_number: int, field_name: str
) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise LangSmithTraceLoadError(
            f"{path}:{line_number}: {field_name} must be null or a JSON object"
        )
    return value


def _total_cost(costs: Mapping[str, Any] | None, path: Path, line_number: int) -> float | None:
    if costs is None or "total_cost" not in costs:
        return None
    value = costs["total_cost"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LangSmithTraceLoadError(
            f"{path}:{line_number}: costs.total_cost must be a non-negative number"
        )
    cost = float(value)
    if cost < 0:
        raise LangSmithTraceLoadError(
            f"{path}:{line_number}: costs.total_cost must be a non-negative number"
        )
    return cost


def _json_value(value: object) -> JsonValue:
    # Values arrived from json.loads; this also detaches mutable caller-owned
    # mappings and rejects non-finite numbers accepted by Python's JSON parser.
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise LangSmithTraceLoadError(
            f"LangSmith export contains invalid JSON data: {error}"
        ) from error
