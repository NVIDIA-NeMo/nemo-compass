# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load traces from the LangSmith API."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from trace_ingest.loaders.helper.merge import Record, assemble
from trace_ingest.loaders.langsmith._normalization import (
    LANGSMITH_DEFAULT_MAX_TRACES,
    LangSmithTraceLoadError,
    LangSmithTraceLoadReport,
    normalize_trace,
    required_id,
    session_key,
    trace_sort_key,
    walk_spans,
)
from trace_ingest.loaders.trace_loaders import TraceDescription
from trace_ingest.models import SpanKind, TraceSnapshot, case_key
from trace_ingest.source_links import http_source_url

if TYPE_CHECKING:
    from langsmith import Client
    from langsmith.schemas import Run

_HYDRATION_BATCH_SIZE = 50
_ROOT_SELECT_FIELDS = ["id", "name", "run_type", "trace_id", "start_time", "extra"]
_RUN_SELECT_FIELDS = [
    "id",
    "name",
    "run_type",
    "start_time",
    "end_time",
    "inputs",
    "outputs",
    "error",
    "status",
    "extra",
    "tags",
    "parent_run_id",
    "trace_id",
    "dotted_order",
    "total_cost",
    "app_path",
    "feedback_stats",
]

__all__ = [
    "LANGSMITH_DEFAULT_MAX_TRACES",
    "LangSmithTraceLoadReport",
    "LangSmithTraceConfig",
    "LangSmithTraceDescription",
    "LangSmithTraceLoadError",
    "LangSmithTraceLoader",
]


class LangSmithTraceDescription(TraceDescription):
    """Corpus metadata for a LangSmith API query."""

    run_count: int
    unresolved_parent_count: int
    project_name: str
    project_id: str
    api_url: str
    filter: str | None
    tree_filter: str | None
    start_time: str | None
    max_traces: int


@dataclass(frozen=True)
class LangSmithTraceConfig:
    """Configuration for loading traces from the LangSmith API."""

    project_name: str
    api_url: str | None = None
    filter: str | None = None
    tree_filter: str | None = None
    start_time: datetime | None = None
    max_traces: int = LANGSMITH_DEFAULT_MAX_TRACES

    def __post_init__(self) -> None:
        if not self.project_name.strip():
            raise ValueError("project_name must not be empty")
        if self.max_traces < 1:
            raise ValueError("max_traces must be at least 1")
        if self.start_time is not None and (
            self.start_time.tzinfo is None or self.start_time.utcoffset() is None
        ):
            raise ValueError("start_time must include a timezone")


@dataclass
class LangSmithTraceLoader:
    """Load a bounded selection of complete conversations from the LangSmith API.

    Root Runs that share a thread key (``thread_id``, ``session_id`` or
    ``conversation_id`` metadata) merge into one trace; ``max_traces`` bounds
    the number of merged conversations.

    ``client`` is an injection seam for tests and custom authentication. When it
    is omitted, LangSmith is imported only when :meth:`load` is called, so users
    of other trace sources do not need the optional dependency.
    """

    config: LangSmithTraceConfig
    client: Client | None = field(default=None, repr=False)
    report: LangSmithTraceLoadReport = field(default_factory=LangSmithTraceLoadReport, init=False)
    _project_id: str | None = field(default=None, init=False, repr=False)
    _resolved_api_url: str | None = field(default=None, init=False, repr=False)
    _call_count: int = field(default=0, init=False, repr=False)
    _case_count: int = field(default=0, init=False, repr=False)
    _session_count: int = field(default=0, init=False, repr=False)

    def load(self) -> TraceSnapshot:
        """Fetch and normalize the configured LangSmith corpus."""

        client = self.client or _new_langsmith_client(self.config.api_url)
        api_url = self.config.api_url or _client_api_url(client)
        project = client.read_project(project_name=self.config.project_name)
        project_id = required_id(project, "id", "LangSmith project")
        root_query = {
            "project_id": project_id,
            "is_root": True,
            "filter": self.config.filter,
            "tree_filter": self.config.tree_filter,
            "start_time": self.config.start_time,
            "select": list(_ROOT_SELECT_FIELDS),
        }
        roots, capped = self._select_roots(client.list_runs(**root_query))
        selected = _selected_roots(roots)
        unresolved_parent_count = 0
        run_count = 0
        records: list[Record] = []
        for trace_id, runs in self._hydrate_selected_runs(client, project_id, selected):
            trace, unresolved = normalize_trace(
                trace_id,
                selected[trace_id],
                runs,
                source_pointer={
                    "provider": "langsmith",
                    "api_url": api_url,
                    "project_name": self.config.project_name,
                    "project_id": project_id,
                },
                langsmith_attributes={
                    "project_name": self.config.project_name,
                    "project_id": project_id,
                },
            )
            trace.source_url = http_source_url(selected[trace_id].url)
            span_urls = {str(run.id): http_source_url(run.url) for run in runs}
            for span in walk_spans(trace.root_spans):
                span.source_url = span_urls.get(span.id)
            unresolved_parent_count += unresolved
            run_count += len(runs)
            root_run = next(run for run in runs if str(run.id) == str(selected[trace_id].id))
            records.append(
                Record(
                    trace,
                    conversation_input=root_run.inputs,
                    conversation_output=root_run.outputs,
                )
            )
        records.sort(key=lambda record: trace_sort_key(selected[record.trace.id], record.trace.id))
        try:
            traces = assemble(records, capped=capped)
        except ValueError as error:
            raise LangSmithTraceLoadError(str(error)) from error
        snapshot = TraceSnapshot(traces)

        self._project_id = project_id
        self._resolved_api_url = api_url
        self._call_count = sum(
            span.kind is SpanKind.TOOL for trace in traces for span in walk_spans(trace.root_spans)
        )
        self._case_count = len({case_key(trace) for trace in traces})
        self._session_count = len({trace.session_id for trace in traces if trace.session_id})
        self.report = LangSmithTraceLoadReport(
            trace_count=snapshot.trace_count,
            run_count=run_count,
            unresolved_parent_count=unresolved_parent_count,
        )
        return snapshot

    def _select_roots(self, roots: Iterable[Run]) -> tuple[list[Run], bool]:
        """Take roots until one would start conversation ``max_traces + 1``."""

        selected: list[Run] = []
        groups: set[tuple[str, str]] = set()
        for root in roots:
            session = session_key(root)
            group = ("session", session[0]) if session else ("trace", _root_trace_id(root))
            if group not in groups:
                if len(groups) == self.config.max_traces:
                    return selected, True
                groups.add(group)
            selected.append(root)
        return selected, False

    def _hydrate_selected_runs(
        self,
        client: Client,
        project_id: str,
        selected: Mapping[str, Run],
    ) -> Iterator[tuple[str, list[Run]]]:
        seen_run_ids: set[str] = set()
        trace_ids = list(selected)
        for batch in _batches(trace_ids, _HYDRATION_BATCH_SIZE):
            grouped: dict[str, list[Run]] = defaultdict(list)
            runs = client.list_runs(
                project_id=project_id,
                trace_filter=_root_id_filter(batch),
                select=list(_RUN_SELECT_FIELDS),
            )
            for run in runs:
                run_id = required_id(run, "id", "LangSmith Run")
                if run_id in seen_run_ids:
                    raise LangSmithTraceLoadError(f"LangSmith returned duplicate Run id {run_id!r}")
                seen_run_ids.add(run_id)
                trace_id = _run_trace_id(run, selected)
                if trace_id not in batch:
                    raise LangSmithTraceLoadError(
                        f"LangSmith returned Run {run_id!r} for unselected trace {trace_id!r}"
                    )
                grouped[trace_id].append(run)
            for trace_id in batch:
                if trace_id not in grouped:
                    raise LangSmithTraceLoadError(
                        f"LangSmith did not return Runs for selected trace {trace_id!r}"
                    )
                yield trace_id, grouped.pop(trace_id)

    def describe(self) -> LangSmithTraceDescription:
        """Describe the normalized corpus and its LangSmith coordinates."""

        api_url = self._resolved_api_url or self.config.api_url or "<configured>"
        project_id = self._project_id or "<unresolved>"
        return {
            "source": _source_label(api_url, project_id),
            "trace_count": self.report.trace_count,
            "call_count": self._call_count,
            "distinct_cases": self._case_count,
            "distinct_sessions": self._session_count,
            "run_count": self.report.run_count,
            "unresolved_parent_count": self.report.unresolved_parent_count,
            "project_name": self.config.project_name,
            "project_id": project_id,
            "api_url": api_url,
            "filter": self.config.filter,
            "tree_filter": self.config.tree_filter,
            "start_time": self.config.start_time.isoformat() if self.config.start_time else None,
            "max_traces": self.config.max_traces,
        }


def _new_langsmith_client(api_url: str | None) -> Client:
    try:
        from langsmith import Client
    except ImportError as error:
        raise LangSmithTraceLoadError(
            "LangSmith support is not installed; install the project with the 'langsmith' extra"
        ) from error
    # This adapter is read-only. Disabling the tracing queue avoids starting a
    # background upload worker or probing ingestion capabilities for a query client.
    return Client(api_url=api_url, auto_batch_tracing=False)


def _client_api_url(client: Client) -> str:
    value = client.api_url
    return str(value) if value else "<configured>"


def _source_label(api_url: str, project_id: str) -> str:
    return f"langsmith:{api_url}#projects/{project_id}"


def _root_trace_id(root: Run) -> str:
    root_id = required_id(root, "id", "LangSmith root Run")
    return str(getattr(root, "trace_id", None) or root_id)


def _selected_roots(roots: Sequence[Run]) -> dict[str, Run]:
    return {_root_trace_id(root): root for root in roots}


def _run_trace_id(run: Run, selected: Mapping[str, Run]) -> str:
    run_id = required_id(run, "id", "LangSmith Run")
    value = getattr(run, "trace_id", None)
    if value not in (None, ""):
        return str(value)
    if run_id in selected:
        return run_id
    raise LangSmithTraceLoadError(f"LangSmith Run {run_id!r} has no trace id")


def _batches(values: Sequence[str], size: int) -> Iterator[list[str]]:
    for offset in range(0, len(values), size):
        yield list(values[offset : offset + size])


def _root_id_filter(trace_ids: Sequence[str]) -> str:
    if len(trace_ids) == 1:
        return f"eq(id, {json.dumps(trace_ids[0])})"
    # LangSmith 0.15.x rejects an OR of root-id comparisons as crossing query
    # tables. The equivalent IN comparison stays on the root table and works on
    # both the current legacy server contract and newer deployments.
    return f"in(id, {json.dumps(list(trace_ids))})"
