# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load traces from the LangSmith API."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, cast

from trace_ingest.loaders.helper.merge import Record, assemble, select_sessions
from trace_ingest.loaders.langsmith._normalization import (
    LANGSMITH_DEFAULT_MAX_TRACES,
    SESSION_METADATA_KEYS,
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
    from langsmith import Client, Langsmith
    from langsmith.schemas import Run, TracerSession
    from langsmith.types.run import Run as V2Run
    from langsmith.types.run_select_field import RunSelectField

_HYDRATION_BATCH_SIZE = 50
# Run ids per feedback request, which sends them in the URL.
_FEEDBACK_BATCH_SIZE = 100
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
    "prompt_tokens",
    "completion_tokens",
    "prompt_token_details",
]
# v2 spells the same fields in upper case and returns only the id unless asked.
# v2 also returns run metadata apart from `extra`.
_V2_ROOT_SELECTS = cast(
    "list[RunSelectField]", [*(name.upper() for name in _ROOT_SELECT_FIELDS), "METADATA"]
)
_V2_RUN_SELECTS = cast(
    "list[RunSelectField]",
    [
        *(name.upper().replace("PARENT_RUN_ID", "PARENT_RUN_IDS") for name in _RUN_SELECT_FIELDS),
        "METADATA",
    ],
)
# The first window the v2 root query reads back from now; each next window is twice as long.
_V2_FIRST_WINDOW = timedelta(hours=1)
# v2 needs time bounds: these cover runs that start during the load or on a clock ahead of
# ours, and a project's first run, which LangSmith receives before creating the project.
_V2_CLOCK_MARGIN = timedelta(hours=1)
_V2_PROJECT_MARGIN = timedelta(days=1)

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
        queries = _runs_api(client, project_id, project, self.config)
        roots, partial = select_sessions(
            queries.roots(),
            key=_root_group_key,
            item_id=_root_trace_id,
            limit=self.config.max_traces,
            reread=queries.thread_roots,
        )
        selected = _selected_roots(roots)
        unresolved_parent_count = 0
        run_count = 0
        records: list[Record] = []
        for trace_id, runs in self._hydrate_selected_runs(queries, selected):
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
            traces = assemble(records, partial=partial)
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

    def _hydrate_selected_runs(
        self,
        queries: _V1Runs | _V2Runs,
        selected: Mapping[str, Run],
    ) -> Iterator[tuple[str, list[Run]]]:
        seen_run_ids: set[str] = set()
        trace_ids = list(selected)
        for batch in _batches(trace_ids, _HYDRATION_BATCH_SIZE):
            grouped: dict[str, list[Run]] = defaultdict(list)
            for run in queries.trace_runs([selected[trace_id] for trace_id in batch]):
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


def _runs_api(
    client: Client, project_id: str, project: TracerSession, config: LangSmithTraceConfig
) -> _V1Runs | _V2Runs:
    """The v2 query API where the server answers it (LangSmith Cloud, SmithDB), else v1.

    LangSmith removes the v1 run queries on Cloud on 2027-01-31 and on self-hosted in v0.19.
    """
    info = client.info
    if not info.version:
        # The SDK logs a failed /info and answers an empty one, which would pick v1 silently.
        raise LangSmithTraceLoadError(
            f"Could not read LangSmith server info from {client.api_url}/info; "
            "check LANGSMITH_ENDPOINT and the API key"
        )
    if (info.instance_flags or {}).get("sdb_query_enabled"):
        start_time = config.start_time or project.start_time - _V2_PROJECT_MARGIN
        return _V2Runs(client, project_id, config.filter, config.tree_filter, start_time)
    return _V1Runs(client, project_id, config.filter, config.tree_filter, config.start_time)


@dataclass(frozen=True)
class _V1Runs:
    client: Client
    project_id: str
    filter: str | None
    tree_filter: str | None
    start_time: datetime | None

    def roots(self) -> Iterator[Run]:
        return self._roots(self.filter)

    def thread_roots(self, session_id: str) -> Iterator[Run]:
        return self._roots(_session_filter(session_id, self.filter))

    def _roots(self, filter: str | None) -> Iterator[Run]:
        # LangSmith returns root runs newest first.
        return self._list_runs(
            project_id=self.project_id,
            is_root=True,
            filter=filter,
            tree_filter=self.tree_filter,
            start_time=self.start_time,
            select=list(_ROOT_SELECT_FIELDS),
        )

    def trace_runs(self, roots: Sequence[Run]) -> Iterator[Run]:
        return self._list_runs(
            project_id=self.project_id,
            trace_filter=_root_id_filter([_root_trace_id(root) for root in roots]),
            select=list(_RUN_SELECT_FIELDS),
        )

    def _list_runs(self, **query: object) -> Iterator[Run]:
        from langsmith._internal._beta_decorator import suppress_deprecation_warning

        # Only servers without the v2 API get here, so the SDK's advice to move does not apply.
        with suppress_deprecation_warning():
            return self.client.list_runs(**query)


@dataclass(frozen=True)
class _V2Runs:
    client: Client
    project_id: str
    filter: str | None
    tree_filter: str | None
    start_time: datetime
    end_time: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) + _V2_CLOCK_MARGIN
    )

    def roots(self) -> Iterator[Run]:
        # v2 has no result order, so read windows back from now, each sorted newest first,
        # and the selection stops reading once it has its conversations, as with v1.
        end, window = self.end_time, _V2_FIRST_WINDOW
        while end >= self.start_time:
            start = max(end - window, self.start_time)
            yield from self._roots(start, end, self.filter)
            end, window = start - timedelta(microseconds=1), window * 2

    def thread_roots(self, session_id: str) -> Iterator[Run]:
        filter = _session_filter(session_id, self.filter)
        return iter(self._roots(self.start_time, self.end_time, filter))

    def trace_runs(self, roots: Sequence[Run]) -> Iterator[Run]:
        # Subtree usage keeps v1's totals: a Run's cost includes its descendants'.
        runs = list(
            self._query(
                min_start_time=min(root.start_time for root in roots),
                max_start_time=self.end_time,
                trace_filter=_root_id_filter([_root_trace_id(root) for root in roots]),
                selects=_V2_RUN_SELECTS,
                include_subtree_usage=True,
            )
        )
        self._add_feedback_comments(runs)
        return iter(runs)

    def _add_feedback_comments(self, runs: Sequence[Run]) -> None:
        """Fill the comments v2 feedback statistics leave out, as v1 returns them."""
        rated = {str(run.id): run.feedback_stats for run in runs if run.feedback_stats}
        if not rated:
            return
        comments: dict[tuple[str, str], list[str]] = defaultdict(list)
        for batch in _batches(list(rated), _FEEDBACK_BATCH_SIZE):
            for feedback in self.client.list_feedback(run_ids=batch):
                comments[str(feedback.run_id), feedback.key].append(feedback.comment or "")
        for run_id, statistics in rated.items():
            for key, stats in statistics.items():
                if stats.get("comments") is None:
                    stats["comments"] = comments.get((run_id, key), [])

    def _roots(self, start: datetime, end: datetime, filter: str | None) -> list[Run]:
        query: dict[str, Any] = {"is_root": True, "selects": _V2_ROOT_SELECTS}
        if filter is not None:
            query["filter"] = filter
        if self.tree_filter is not None:
            query["tree_filter"] = self.tree_filter
        roots = list(self._query(min_start_time=start, max_start_time=end, **query))
        return sorted(roots, key=lambda run: (run.start_time, str(run.id)), reverse=True)

    def _query(self, **query: Any) -> Iterator[Run]:  # noqa: ANN401
        # The SDK's own v2 client, with the session, headers and timeouts of `client`.
        api: Langsmith = self.client._get_langsmith_api_sync()
        runs = api.runs.query(project_ids=[self.project_id], page_size=1000, **query)
        return (self._run(run) for run in runs)

    def _run(self, run: V2Run) -> Run:
        """The v1 Run shape the normalization reads, with its LangSmith URL."""
        from langsmith.schemas import Run

        parents = run.parent_run_ids or []
        # v2 returns metadata apart from `extra`; the normalization reads extra.metadata.
        extra = run.extra or {}
        metadata = extra.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        feedback = run.feedback_stats or {}
        fields = {
            "id": run.id,
            "name": run.name,
            "run_type": run.run_type.lower() if run.run_type else None,
            "start_time": run.start_time,
            "end_time": run.end_time,
            "trace_id": run.trace_id,
            "session_id": self.project_id,
            "parent_run_id": parents[-1] if parents else None,
            "dotted_order": run.dotted_order,
            "inputs": run.inputs or {},
            # v2 answers {} for a run that recorded no outputs, where v1 answers null.
            "outputs": run.outputs or None,
            "error": run.error,
            "status": run.status.lower() if run.status else None,
            "tags": run.tags,
            "extra": {**extra, "metadata": {**metadata, **run.metadata}} if run.metadata else extra,
            "app_path": run.app_path,
            "total_cost": run.total_cost,
            "feedback_stats": {
                name: stats.model_dump(exclude_none=True) for name, stats in feedback.items()
            }
            or None,
            "prompt_tokens": run.prompt_tokens,
            "completion_tokens": run.completion_tokens,
            "prompt_token_details": (
                run.prompt_token_details.raw if run.prompt_token_details else None
            ),
        }
        return Run(
            **{key: value for key, value in fields.items() if value is not None},
            _host_url=self.client._host_url,
        )


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


def _root_group_key(root: Run) -> tuple[str, str]:
    session = session_key(root)
    return ("session", session[0]) if session else ("trace", _root_trace_id(root))


def _session_filter(session_id: str, user_filter: str | None) -> str:
    """Roots whose thread metadata equals ``session_id``, within ``user_filter``.

    A root can match through a lower-precedence key than the one it is grouped by,
    so callers still check each returned root's thread key.
    """
    restriction = (
        f"and(in(metadata_key, {json.dumps(list(SESSION_METADATA_KEYS))}), "
        f"eq(metadata_value, {json.dumps(session_id)}))"
    )
    return restriction if user_filter is None else f"and({restriction}, {user_filter})"


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
