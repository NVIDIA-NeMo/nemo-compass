# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import json
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from uuid import UUID

import pytest
from langsmith import Client
from langsmith.schemas import Run
from test_langsmith_trace_export_file_loader import (
    trace_export_run_record,
    write_trace_export_file,
)

from compass.trace_loaders.langsmith import (
    LANGSMITH_DEFAULT_MAX_TRACES,
    LangSmithTraceConfig,
    LangSmithTraceExportFileConfig,
    LangSmithTraceExportFileLoader,
    LangSmithTraceLoader,
    LangSmithTraceLoadError,
)
from compass.traces import UNSET, SpanKind, Trace
from trace_ingest.loaders.langsmith._normalization import SESSION_METADATA_KEYS, walk_spans
from trace_ingest.models import TurnRole

PROJECT_ID = UUID("10000000-0000-0000-0000-000000000000")
TRACE_ID = UUID("20000000-0000-0000-0000-000000000000")
LLM_ID = UUID("30000000-0000-0000-0000-000000000000")
TOOL_ID = UUID("40000000-0000-0000-0000-000000000000")
START = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)


def _records(trace: Trace) -> list[dict]:
    """The merged session's per-record provenance entries."""
    records = trace.attributes["records"]
    assert isinstance(records, list)
    entries = [entry for entry in records if isinstance(entry, dict)]
    assert len(entries) == len(records)
    return entries


def langsmith_run(
    run_id: UUID,
    *,
    trace_id: UUID = TRACE_ID,
    parent_id: UUID | None = None,
    name: str = "agent",
    run_type: str = "chain",
    offset: int = 0,
    duration: int | None = 1,
    inputs=None,
    outputs=None,
    error: str | None = None,
    status: str | None = "success",
    extra=None,
    total_cost: float | None = None,
    app_path: str | None = None,
) -> Run:
    start = START + timedelta(seconds=offset)
    end = start + timedelta(seconds=duration) if duration is not None else None
    return Run(
        id=run_id,
        trace_id=trace_id,
        parent_run_id=parent_id,
        name=name,
        run_type=run_type,
        start_time=start,
        end_time=end,
        inputs=inputs,
        outputs=outputs,
        error=error,
        status=status,
        extra=extra,
        total_cost=total_cost,
        app_path=app_path,
    )


class FakeLangSmithClient(Client):
    api_url = "https://langsmith.test/api/v1"
    _info = SimpleNamespace(version="0.15.13", instance_flags={})  # a v1-only server

    def __init__(self, roots: list[Run], runs: list[Run], *, project_id: UUID = PROJECT_ID):
        self.roots = roots
        self._fake_runs = runs
        self.project_id = project_id
        self.project_calls = []
        self.list_calls = []

    def read_project(self, **kwargs):
        self.project_calls.append(kwargs)
        return SimpleNamespace(id=self.project_id)

    def list_runs(self, **kwargs):
        self.list_calls.append(kwargs)
        if kwargs.get("is_root"):
            roots = self.roots
            thread = re.search(r"eq\(metadata_value, (\".*?\")\)", kwargs.get("filter") or "")
            if thread:
                value = json.loads(thread.group(1))
                roots = [root for root in roots if value in thread_values(root.extra)]
            return iter(roots[: kwargs.get("limit", len(roots))])
        selected = set(re.findall(r'"([0-9a-f-]{36})"', kwargs["trace_filter"]))
        return iter(item for item in self._fake_runs if str(item.trace_id) in selected)


class JSONHandler(BaseHTTPRequestHandler):
    """A quiet local LangSmith stand-in; tests add do_GET and do_POST."""

    def log_message(self, format, *args):
        pass

    def send_json(self, payload):
        contents = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(contents)))
        self.end_headers()
        self.wfile.write(contents)


@contextmanager
def serving(handler: type[BaseHTTPRequestHandler]) -> Iterator[str]:
    """Serve ``handler`` on a free local port and yield its base URL."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()


def thread_values(extra) -> set[str]:
    """The values a LangSmith thread-metadata filter can match on a run's ``extra``."""
    metadata = (extra or {}).get("metadata") or {}
    return {str(metadata[key]) for key in SESSION_METADATA_KEYS if key in metadata}


def trace_runs() -> list[Run]:
    root = langsmith_run(
        TRACE_ID,
        inputs={"messages": [{"role": "user", "content": "Find the weather"}]},
        outputs={"answer": None},
        extra={
            "metadata": {
                "thread_id": "thread-7",
                "session_id": "session-lower-priority",
            }
        },
        total_cost=0.0125,
        app_path=f"/o/acme/projects/p/r/{TRACE_ID}",
    )
    llm = langsmith_run(
        LLM_ID,
        parent_id=TRACE_ID,
        name="acompletion",
        run_type="llm",
        offset=1,
        inputs={"model": "test-model"},
        outputs={"content": "I should use the weather tool."},
        total_cost=0.005,
    )
    tool = langsmith_run(
        TOOL_ID,
        parent_id=TRACE_ID,
        name="get_weather",
        run_type="tool",
        offset=2,
        inputs={"city": "Denver"},
        outputs={"temperature": 72},
        error="TimeoutError: upstream timed out",
        status="error",
    )
    return [tool, llm, root]


def test_loader_queries_bounded_roots_then_hydrates_and_normalizes_complete_traces():
    runs = trace_runs()
    root = runs[-1]
    client = FakeLangSmithClient([root], runs)
    start_time = START - timedelta(days=1)
    loader = LangSmithTraceLoader(
        LangSmithTraceConfig(
            project_name="glamr-ux",
            filter='eq(status, "success")',
            tree_filter='eq(name, "get_weather")',
            start_time=start_time,
            max_traces=25,
        ),
        client=client,
    )

    snapshot = loader.load()

    assert client.project_calls == [{"project_name": "glamr-ux"}]
    assert client.list_calls[0] == {
        "project_id": str(PROJECT_ID),
        "is_root": True,
        "filter": 'eq(status, "success")',
        "tree_filter": 'eq(name, "get_weather")',
        "start_time": start_time,
        "select": ["id", "name", "run_type", "trace_id", "start_time", "extra"],
    }
    assert client.list_calls[1]["project_id"] == str(PROJECT_ID)
    assert client.list_calls[1]["trace_filter"] == f'eq(id, "{TRACE_ID}")'
    assert "filter" not in client.list_calls[1]
    assert "limit" not in client.list_calls[1]

    assert snapshot.trace_count == 1
    trace = next(iter(snapshot))
    assert trace.id == "session:thread-7"
    # The root's total_cost already includes the LLM's 0.005.
    assert trace.aggregate.cost_usd == 0.0125
    assert trace.aggregate.latency_ms == 3000
    assert (trace.session_id, trace.session_basis) == ("thread-7", "langsmith.metadata.thread_id")
    assert trace.losses == []
    assert [record["trace_id"] for record in _records(trace)] == [str(TRACE_ID)]
    (session_root,) = trace.root_spans
    assert session_root.kind is SpanKind.AGENT
    assert [span.id for span in session_root.children] == [f"{TRACE_ID}/{TRACE_ID}"]
    root_span = session_root.children[0]
    assert root_span.input == {"messages": [{"role": "user", "content": "Find the weather"}]}
    assert root_span.output == {"answer": None}
    assert root_span.kind is SpanKind.CHAIN
    assert [span.id for span in root_span.children] == [
        f"{TRACE_ID}/{LLM_ID}",
        f"{TRACE_ID}/{TOOL_ID}",
    ]
    assert [span.kind for span in root_span.children] == [SpanKind.LLM, SpanKind.TOOL]
    assert [span.cost_usd for span in [root_span, *root_span.children]] == [None, 0.005, None]
    assert session_root.turns is not UNSET
    assert [
        (turn.role, turn.content, turn.llm_span_ids, turn.tool_call_span_ids)
        for turn in session_root.turns
    ] == [
        (TurnRole.USER, "Find the weather", [], []),
        (
            TurnRole.AGENT,
            "I should use the weather tool.",
            [f"{TRACE_ID}/{LLM_ID}"],
            [f"{TRACE_ID}/{TOOL_ID}"],
        ),
        (TurnRole.AGENT, '{"answer": null}', [], []),
    ]
    tool_span = root_span.children[-1]
    assert tool_span.tool_name == "get_weather"
    assert tool_span.attributes["status"] == "ERROR"
    assert tool_span.error == "TimeoutError: upstream timed out"
    assert tool_span.attributes["duration_ms"] == 1000
    tool_pointer = tool_span.attributes["source_pointer"]
    assert isinstance(tool_pointer, dict)
    assert tool_pointer["run_id"] == str(TOOL_ID)
    root_pointer = root_span.attributes["source_pointer"]
    assert isinstance(root_pointer, dict)
    app_path = root_pointer["app_path"]
    assert isinstance(app_path, str)
    assert app_path.endswith(str(TRACE_ID))
    assert tool_span.tool_call is not None
    assert tool_span.tool_call.index == 0
    assert tool_span.tool_call.result_count == 1

    assert loader.report.trace_count == 1
    assert loader.report.run_count == 3
    assert loader.report.unresolved_parent_count == 0
    assert loader.describe() == {
        "source": f"langsmith:https://langsmith.test/api/v1#projects/{PROJECT_ID}",
        "trace_count": 1,
        "call_count": 1,
        "distinct_cases": 1,
        "distinct_sessions": 1,
        "run_count": 3,
        "unresolved_parent_count": 0,
        "project_name": "glamr-ux",
        "project_id": str(PROJECT_ID),
        "api_url": "https://langsmith.test/api/v1",
        "filter": 'eq(status, "success")',
        "tree_filter": 'eq(name, "get_weather")',
        "start_time": start_time.isoformat(),
        "max_traces": 25,
    }


def test_loader_uses_langsmith_sdk_query_contract_without_importing_provider_objects(tmp_path):
    _, child, root = trace_runs()
    root.feedback_stats = {"quality": {"n": 2, "avg": 0.5}, "missing": None, "empty": {}}
    child.feedback_stats = {"quality": {"n": 1, "avg": 0}, "accepted": {"n": 1, "avg": False}}
    root_record = root.model_dump(mode="json", exclude_none=True)
    child_record = child.model_dump(mode="json", exclude_none=True)
    # The legacy Run wire model cannot distinguish this explicit top-level null
    # from a missing output; both arrive as Run.outputs=None.
    child_record["outputs"] = None
    requests = []

    class Handler(JSONHandler):
        def do_GET(self):
            if self.path.startswith("/info"):
                # A self-hosted 0.15 server: v1 queries only.
                return self.send_json({"version": "0.15.13", "instance_flags": {}})
            requests.append(("GET", self.path, None))
            self.send_json(
                [
                    {
                        "id": str(PROJECT_ID),
                        "tenant_id": "90000000-0000-0000-0000-000000000000",
                        "reference_dataset_id": None,
                        "name": "glamr-ux",
                        "start_time": START.isoformat(),
                    }
                ]
            )

        def do_POST(self):
            size = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(size))
            requests.append(("POST", self.path, body))

            def selected(record):
                return {key: value for key, value in record.items() if key in body["select"]}

            if body.get("is_root"):
                self.send_json({"runs": [selected(root_record)]})
            elif body.get("cursor") == "hydration-page-2":
                self.send_json({"runs": [selected(child_record)]})
            else:
                self.send_json(
                    {
                        "runs": [selected(root_record)],
                        "cursors": {"next": "hydration-page-2"},
                    }
                )

    with serving(Handler) as api_url:
        client = Client(api_url=api_url, auto_batch_tracing=False)
        loader = LangSmithTraceLoader(
            LangSmithTraceConfig(project_name="glamr-ux"),
            client=client,
        )
        snapshot = loader.load()

    assert snapshot.trace_count == 1
    trace = next(iter(snapshot))
    expected = {
        "langsmith.feedback_stats.quality": {
            str(TRACE_ID): {"n": 2, "avg": 0.5},
            str(LLM_ID): {"n": 1, "avg": 0},
        },
        "langsmith.feedback_stats.accepted": {str(LLM_ID): {"n": 1, "avg": False}},
    }
    expected_json = json.dumps(expected, sort_keys=True)
    # The root's thread_id metadata arrives with the root selection, so the live
    # trace is a merged session whose evaluator values name their record.
    assert trace.id == "session:thread-7"
    assert trace.evaluator_results == {
        name: [{"trace_id": str(TRACE_ID), "value": value}] for name, value in expected.items()
    }
    path = tmp_path / "trace.jsonl"
    write_trace_export_file(
        path,
        *(
            trace_export_run_record(
                str(run.id),
                trace_id=str(TRACE_ID),
                parent_run_id=str(run.parent_run_id) if run.parent_run_id else None,
                feedback_stats=run.feedback_stats,
            )
            for run in (child, root)
        ),
    )
    exported = next(
        iter(LangSmithTraceExportFileLoader(LangSmithTraceExportFileConfig(path)).load())
    )
    assert json.dumps(exported.evaluator_results, sort_keys=True) == expected_json
    (session_root,) = trace.root_spans
    assert [span.id for span in session_root.children] == [f"{TRACE_ID}/{TRACE_ID}"]
    record_root = session_root.children[0]
    assert [span.id for span in record_root.children] == [f"{TRACE_ID}/{LLM_ID}"]
    assert "output" not in record_root.children[0].model_dump()
    assert requests[0][0:2] == ("GET", "/sessions?limit=1&name=glamr-ux&include_stats=False")
    assert requests[1][2]["is_root"] is True
    assert "limit" not in requests[1][2]
    assert "extra" in requests[1][2]["select"]
    assert requests[2][2]["trace_filter"] == f'eq(id, "{TRACE_ID}")'
    assert [request[1] for request in requests[1:]] == ["/runs/query"] * 3
    assert requests[3][2]["cursor"] == "hydration-page-2"


def test_loader_uses_explicit_api_url_and_maps_unknown_pending_run():
    root = langsmith_run(
        TRACE_ID,
        run_type="prompt",
        outputs=None,
        duration=None,
        status="pending",
        extra={"metadata": {"conversation_id": "conversation-3"}},
    )
    loader = LangSmithTraceLoader(
        LangSmithTraceConfig(project_name="project", api_url="http://localhost/api/v1"),
        client=FakeLangSmithClient([root], [root]),
    )

    trace = next(iter(loader.load()))

    assert trace.evaluator_results == {}
    assert (trace.session_id, trace.session_basis) == (
        "conversation-3",
        "langsmith.metadata.conversation_id",
    )
    (record_root,) = trace.root_spans[0].children
    assert record_root.kind is SpanKind.UNKNOWN
    assert record_root.attributes["subtype"] == "PROMPT"
    assert record_root.attributes["status"] == "UNKNOWN"
    assert record_root.end_time is None
    assert "output" not in record_root.model_dump()
    assert loader.describe()["api_url"] == "http://localhost/api/v1"


def test_loader_detaches_and_reports_unresolved_parent():
    missing_parent = UUID("50000000-0000-0000-0000-000000000000")
    root = langsmith_run(TRACE_ID)
    child = langsmith_run(TOOL_ID, parent_id=missing_parent, run_type="tool", offset=1)
    loader = LangSmithTraceLoader(
        LangSmithTraceConfig(project_name="project"),
        client=FakeLangSmithClient([root], [child, root]),
    )

    trace = next(iter(loader.load()))

    assert [span.id for span in trace.root_spans] == [str(TRACE_ID), str(TOOL_ID)]
    pointer = trace.root_spans[1].attributes["source_pointer"]
    assert isinstance(pointer, dict)
    assert pointer["unresolved_parent_run_id"] == str(missing_parent)
    assert loader.report.unresolved_parent_count == 1


def test_loader_hydrates_root_ids_in_bounded_batches():
    roots = []
    runs = []
    for value in range(51):
        trace_id = UUID(int=value + 1)
        root = langsmith_run(trace_id, trace_id=trace_id, offset=value)
        child = langsmith_run(
            UUID(int=value + 1000),
            trace_id=trace_id,
            parent_id=trace_id,
            run_type="tool",
            offset=value + 1,
        )
        roots.append(root)
        runs.extend((root, child))
    client = FakeLangSmithClient(roots, list(reversed(runs)))
    loader = LangSmithTraceLoader(
        LangSmithTraceConfig(project_name="many", max_traces=51),
        client=client,
    )

    snapshot = loader.load()

    assert snapshot.trace_count == 51
    hydration_calls = client.list_calls[1:]
    assert len(hydration_calls) == 2
    assert hydration_calls[0]["trace_filter"].startswith("in(id, [")
    assert hydration_calls[0]["trace_filter"].count('"00000000-') == 50
    assert hydration_calls[1]["trace_filter"] == f'eq(id, "{UUID(int=51)}")'
    assert [trace.id for trace in snapshot] == [str(UUID(int=i)) for i in range(1, 52)]
    assert all(len(trace.root_spans) == 1 for trace in snapshot)
    assert all(len(trace.root_spans[0].children) == 1 for trace in snapshot)


def test_loader_paginates_root_selection_above_server_page_limit():
    roots = [
        langsmith_run(UUID(int=value + 1), trace_id=UUID(int=value + 1), offset=value)
        for value in range(102)
    ]
    client = FakeLangSmithClient(roots, roots)
    loader = LangSmithTraceLoader(
        LangSmithTraceConfig(project_name="many", max_traces=101),
        client=client,
    )

    snapshot = loader.load()

    assert snapshot.trace_count == 101
    assert "limit" not in client.list_calls[0]


def thread_roots() -> list[Run]:
    """Three single-run traces: two in thread t1, then one in thread t2."""
    return [
        langsmith_run(
            UUID(int=value + 1),
            trace_id=UUID(int=value + 1),
            offset=value,
            extra={"metadata": {"thread_id": thread}},
        )
        for value, thread in enumerate(["t1", "t1", "t2"])
    ]


def test_loader_merges_roots_that_share_a_thread_into_one_conversation():
    roots = thread_roots()
    loader = LangSmithTraceLoader(
        LangSmithTraceConfig(project_name="threads", max_traces=2),
        client=FakeLangSmithClient(list(reversed(roots)), roots),
    )

    snapshot = loader.load()

    assert [trace.id for trace in snapshot] == ["session:t1", "session:t2"]
    first = next(iter(snapshot))
    assert [span.id for span in first.root_spans[0].children] == [
        f"{UUID(int=1)}/{UUID(int=1)}",
        f"{UUID(int=2)}/{UUID(int=2)}",
    ]
    assert all(loss.code != "merge_incomplete" for trace in snapshot for loss in trace.losses)
    assert loader.report.trace_count == 2
    assert loader.report.run_count == 3
    description = loader.describe()
    assert (description["trace_count"], description["distinct_sessions"]) == (2, 2)


def single_run_roots(*metadata: dict) -> list[Run]:
    """Single-run traces ``UUID(int=1..)``, one second apart, with the given root metadata."""
    return [
        langsmith_run(
            UUID(int=value + 1),
            trace_id=UUID(int=value + 1),
            offset=value,
            extra={"metadata": entry},
        )
        for value, entry in enumerate(metadata)
    ]


def test_loader_rereads_selected_threads_past_the_cap():
    roots = single_run_roots(
        {"thread_id": "t1"}, {"thread_id": "t1"}, {"thread_id": "t2"}, {"thread_id": "t1"}
    )
    client = FakeLangSmithClient(roots, roots)
    loader = LangSmithTraceLoader(
        LangSmithTraceConfig(project_name="threads", max_traces=1),
        client=client,
    )

    (trace,) = loader.load()

    # Selection stops at t2's root, before the fourth root; the thread re-read finds it.
    assert trace.id == "session:t1"
    trace_ids = [str(UUID(int=value)) for value in (1, 2, 4)]
    assert [record["trace_id"] for record in _records(trace)] == trace_ids
    assert trace.losses == []
    assert client.list_calls[1]["filter"] == (
        'and(in(metadata_key, ["thread_id", "session_id", "conversation_id"]), '
        'eq(metadata_value, "t1"))'
    )
    assert client.list_calls[1]["is_root"] is True
    assert client.list_calls[2]["trace_filter"] == f"in(id, {json.dumps(trace_ids)})"


def test_reread_ands_the_user_filter_and_drops_other_thread_keys():
    roots = single_run_roots(
        {"thread_id": "t1"}, {"thread_id": "t2"}, {"thread_id": "t9", "session_id": "t1"}
    )
    client = FakeLangSmithClient(roots, roots)

    (trace,) = LangSmithTraceLoader(
        LangSmithTraceConfig(project_name="threads", max_traces=1, filter='eq(name, "agent")'),
        client=client,
    ).load()

    assert client.list_calls[1]["filter"] == (
        'and(and(in(metadata_key, ["thread_id", "session_id", "conversation_id"]), '
        'eq(metadata_value, "t1")), eq(name, "agent"))'
    )
    # The third root matches "t1" through session_id but belongs to thread t9.
    assert [record["trace_id"] for record in _records(trace)] == [str(UUID(int=1))]


def test_loader_rejects_missing_selected_root_in_hydration():
    root = langsmith_run(TRACE_ID)
    client = FakeLangSmithClient([root], [])

    with pytest.raises(LangSmithTraceLoadError, match="did not return Runs"):
        LangSmithTraceLoader(LangSmithTraceConfig(project_name="project"), client=client).load()


def test_loader_rejects_run_for_unselected_trace():
    root = langsmith_run(TRACE_ID)
    other_trace = UUID("60000000-0000-0000-0000-000000000000")
    other = langsmith_run(other_trace, trace_id=other_trace)

    class UnexpectedClient(FakeLangSmithClient):
        def list_runs(self, **kwargs):
            self.list_calls.append(kwargs)
            return iter([root]) if kwargs.get("is_root") else iter([root, other])

    with pytest.raises(LangSmithTraceLoadError, match="unselected trace"):
        LangSmithTraceLoader(
            LangSmithTraceConfig(project_name="project"),
            client=UnexpectedClient([root], [root, other]),
        ).load()


def test_loader_rejects_parent_cycle():
    root = langsmith_run(TRACE_ID)
    first = langsmith_run(LLM_ID, parent_id=TOOL_ID, run_type="llm", offset=1)
    second = langsmith_run(TOOL_ID, parent_id=LLM_ID, run_type="tool", offset=2)

    with pytest.raises(LangSmithTraceLoadError, match="contains a parent cycle"):
        LangSmithTraceLoader(
            LangSmithTraceConfig(project_name="project"),
            client=FakeLangSmithClient([root], [root, first, second]),
        ).load()


def test_describe_before_load_uses_unresolved_coordinates():
    loader = LangSmithTraceLoader(LangSmithTraceConfig(project_name="project"))

    description = loader.describe()

    assert description["source"] == "langsmith:<configured>#projects/<unresolved>"
    assert description["trace_count"] == 0
    assert description["run_count"] == 0
    assert description["max_traces"] == LANGSMITH_DEFAULT_MAX_TRACES


@pytest.mark.parametrize("max_traces", [0, -1])
def test_config_max_traces_must_be_positive(max_traces):
    with pytest.raises(ValueError, match="max_traces must be at least 1"):
        LangSmithTraceConfig(project_name="project", max_traces=max_traces)


def test_config_requires_project_name_and_timezone_aware_start():
    with pytest.raises(ValueError, match="project_name must not be empty"):
        LangSmithTraceConfig(project_name=" ")
    with pytest.raises(ValueError, match="start_time must include a timezone"):
        LangSmithTraceConfig(project_name="project", start_time=datetime(2026, 9, 2))


def test_live_source_link_uses_sdk_ui_url():
    runs = trace_runs()
    root = next(run for run in runs if run.parent_run_id is None)
    root._host_url = "https://ui.langsmith.test"
    client = FakeLangSmithClient([root], runs)
    loader = LangSmithTraceLoader(LangSmithTraceConfig(project_name="project"), client)
    assert root.app_path is not None
    trace = next(iter(loader.load()))
    url = "https://ui.langsmith.test" + root.app_path
    assert trace.source_url is None
    assert _records(trace)[0]["source_url"] == url
    assert trace.root_spans[0].children[0].source_url == url


@pytest.mark.parametrize("child_path", ["/o/acme/projects/p/r/child?trace_id=parent", None, ""])
def test_span_links_use_each_sdk_run_url_at_any_depth(child_path):
    runs = trace_runs()
    tool, llm, root = runs
    tool.parent_run_id = llm.id
    for run in runs:
        run._host_url = "https://regional.langsmith.test"
    llm.app_path = "/o/acme/projects/p/r/llm"
    tool.app_path = child_path
    loader = LangSmithTraceLoader(
        LangSmithTraceConfig(project_name="project"), client=FakeLangSmithClient([root], runs)
    )
    trace = next(iter(loader.load()))
    (root_span,) = trace.root_spans[0].children
    llm_span = root_span.children[0]
    tool_span = llm_span.children[0]
    assert root_span.source_url == root.url == _records(trace)[0]["source_url"]
    assert llm_span.id == f"{TRACE_ID}/{LLM_ID}"
    assert llm_span.source_url == llm.url
    assert tool_span.id == f"{TRACE_ID}/{TOOL_ID}"
    assert tool_span.source_url == (tool.url if child_path else None)
    assert llm_span.source_url != root_span.source_url


def test_loader_reads_the_model_tool_call_id_of_tool_runs():
    root = langsmith_run(TRACE_ID)
    tool = langsmith_run(
        TOOL_ID,
        parent_id=TRACE_ID,
        name="get_weather",
        run_type="tool",
        offset=1,
        extra={"tool_call_id": "toolu_01Jd7c27FT9pVshw55h7N6JB"},
    )
    loader = LangSmithTraceLoader(
        LangSmithTraceConfig(project_name="tools"),
        client=FakeLangSmithClient([root], [root, tool]),
    )

    (trace,) = loader.load()

    (tool_span,) = trace.root_spans[0].children
    assert tool_span.tool_call is not None
    assert tool_span.tool_call.call_id == "toolu_01Jd7c27FT9pVshw55h7N6JB"


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


def test_loader_uses_v2_query_api_when_the_server_answers_it():
    """LangSmith Cloud and SmithDB servers: /api/v2 runs and traces, an explicit window."""
    root = {
        "id": str(TRACE_ID),
        "trace_id": str(TRACE_ID),
        "name": "agent",
        "run_type": "CHAIN",
        "status": "SUCCESS",
        "start_time": START.isoformat(),
        "end_time": (START + timedelta(seconds=3)).isoformat(),
        "app_path": f"/o/t/projects/p/{PROJECT_ID}/r/{TRACE_ID}",
        "parent_run_ids": [],
        "inputs": {"messages": [{"role": "user", "content": "Weather in Paris?"}]},
        "outputs": {"messages": [{"role": "assistant", "content": "Sunny."}]},
        "extra": {},
        "metadata": {"thread_id": "t-1"},  # v2 returns metadata apart from extra
        "feedback_stats": {"quality": {"n": 1, "avg": 0.0, "comments": None}},
    }
    llm = {
        **root,
        "id": str(LLM_ID),
        "name": "ChatOpenAI",
        "run_type": "LLM",
        "start_time": (START + timedelta(seconds=1)).isoformat(),
        "parent_run_ids": [str(TRACE_ID)],
        "extra": {
            "invocation_params": {"model": "gpt", "tools": [WEATHER_TOOL]},
            "metadata": {"ls_model_name": "gpt"},
        },
        "prompt_tokens": 120,
        "completion_tokens": 7,
        "prompt_token_details": {"raw": {"cache_read": 100, "cache_write": 15}},
        "feedback_stats": None,
    }
    tool = {
        **root,
        "id": str(TOOL_ID),
        "name": "get_weather",
        "run_type": "TOOL",
        "start_time": (START + timedelta(seconds=2)).isoformat(),
        "parent_run_ids": [str(TRACE_ID), str(LLM_ID)],
        "outputs": {"output": {"type": "tool", "status": "error", "content": "unknown city"}},
        "feedback_stats": None,
    }
    requests = []

    class Handler(JSONHandler):
        def do_GET(self):
            if self.path.endswith("/info"):
                flags = {"ch_query_enabled": True, "sdb_query_enabled": True}
                return self.send_json({"version": "0.18.26", "instance_flags": flags})
            requests.append(("GET", self.path, None))
            if self.path.startswith("/api/v1/sessions"):
                project = {
                    "id": str(PROJECT_ID),
                    "tenant_id": str(PROJECT_ID),
                    "name": "p",
                    "reference_dataset_id": None,
                    "start_time": START.isoformat(),
                }
                return self.send_json([project])
            if self.path.startswith("/api/v1/feedback"):
                # v2 statistics carry no comments; the feedback API has them.
                return self.send_json(
                    [
                        {
                            "id": str(PROJECT_ID),
                            "run_id": str(TRACE_ID),
                            "trace_id": str(TRACE_ID),
                            "key": "quality",
                            "score": 0,
                            "comment": "wrong city",
                            "created_at": START.isoformat(),
                            "modified_at": START.isoformat(),
                        }
                    ]
                )
            self.send_error(404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(("POST", self.path, body))
            if not body.get("is_root"):
                return self.send_json({"items": [root, llm, tool], "next_cursor": None})
            window = [
                datetime.fromisoformat(body[key]) for key in ("min_start_time", "max_start_time")
            ]
            items = [root] if window[0] <= START <= window[1] else []
            self.send_json({"items": items, "next_cursor": None})

    with serving(Handler) as url:
        client = Client(api_url=f"{url}/api/v1", api_key="test", auto_batch_tracing=False)
        snapshot = LangSmithTraceLoader(
            LangSmithTraceConfig(project_name="p", filter='eq(status, "success")'),
            client=client,
        ).load()

    paths = [path.split("?")[0] for _, path, _ in requests]
    assert paths[0] == "/api/v1/sessions"
    assert set(paths[1:-1]) == {"/api/v2/runs/query"}
    assert paths[-1] == "/api/v1/feedback"
    *windows, hydration = [body for _, _, body in requests[1:-1]]
    # Root windows read back from now, newest first, each twice as long, to a day
    # before the project.
    bounds = [
        (datetime.fromisoformat(w["min_start_time"]), datetime.fromisoformat(w["max_start_time"]))
        for w in windows
    ]
    assert bounds[0][1] - bounds[0][0] == timedelta(hours=1)
    assert all(
        newer[0] - older[1] == timedelta(microseconds=1)
        for newer, older in itertools.pairwise(bounds)
    )
    assert bounds[-1][0] == START - timedelta(days=1)
    query = windows[0]
    assert query["project_ids"] == [str(PROJECT_ID)]
    assert query["is_root"] is True
    assert query["filter"] == 'eq(status, "success")'
    assert "EXTRA" in query["selects"]
    assert hydration["trace_filter"] == f'eq(id, "{TRACE_ID}")'
    assert datetime.fromisoformat(hydration["min_start_time"]) == START
    assert hydration["include_subtree_usage"] is True
    assert "PROMPT_TOKEN_DETAILS" in hydration["selects"]

    trace = next(iter(snapshot))
    assert trace.id == "session:t-1"
    (session,) = trace.root_spans
    (agent,) = session.children
    (llm_span,) = agent.children
    (tool_span,) = llm_span.children
    # The SDK's host URL for the app, as v1 runs get it.
    assert str(agent.source_url).endswith(root["app_path"])
    assert llm_span.kind is SpanKind.LLM
    assert llm_span.token_counts is not None
    assert llm_span.token_counts.model_dump(exclude_none=True) == {
        "input_total": 120,
        "cached_input_tokens": 100,
        "output_tokens": 7,
        "cache_write_tokens": 15,
        "source": "langsmith",
    }
    assert llm_span.tool_definitions is not UNSET
    assert [definition.name for definition in llm_span.tool_definitions] == ["get_weather"]
    assert tool_span.tool_call is not None
    assert tool_span.tool_call.results is not UNSET
    assert tool_span.tool_call.results[0].is_error is True
    assert trace.evaluator_results == {
        "langsmith.feedback_stats.quality": [
            {
                "trace_id": str(TRACE_ID),
                "value": {str(TRACE_ID): {"n": 1, "avg": 0.0, "comments": ["wrong city"]}},
            }
        ]
    }


def test_llm_runs_without_reported_usage_have_no_token_counts():
    """LangSmith stores zero tokens when a model call reported no usage."""
    root = langsmith_run(TRACE_ID)
    llm = langsmith_run(LLM_ID, parent_id=TRACE_ID, run_type="llm", offset=1)
    llm.prompt_tokens = 0
    llm.completion_tokens = 0
    loader = LangSmithTraceLoader(
        LangSmithTraceConfig(project_name="p"), client=FakeLangSmithClient([root], [root, llm])
    )
    (llm_span,) = next(iter(loader.load())).root_spans[0].children
    assert llm_span.token_counts is None
    assert llm_span.tool_definitions is UNSET


def _load_llm_runs(*llm_runs: Run) -> list:
    root = langsmith_run(TRACE_ID)
    client = FakeLangSmithClient([root], [root, *llm_runs])
    trace = next(iter(LangSmithTraceLoader(LangSmithTraceConfig(project_name="p"), client).load()))
    return [span for span in walk_spans(trace.root_spans) if span.kind is SpanKind.LLM]


def test_anthropic_cache_writes_split_by_ttl_count_as_cache_writes():
    """langchain-anthropic zeroes cache_creation when it reports writes per TTL."""
    llm = langsmith_run(LLM_ID, parent_id=TRACE_ID, run_type="llm", offset=1)
    llm.prompt_tokens = 2_100
    llm.completion_tokens = 9
    llm.prompt_token_details = {
        "cache_read": 0,
        "cache_creation": 0,
        "ephemeral_5m_input_tokens": 2_000,
        "ephemeral_1h_input_tokens": 0,
    }
    (span,) = _load_llm_runs(llm)
    assert span.token_counts is not None
    assert span.token_counts.cache_write_tokens == 2_000
    assert span.token_counts.cache_writes_by_ttl == {300: 2_000, 3600: 0}


def test_nested_model_calls_count_their_usage_once():
    """A model Run's usage includes model Runs nested in it, as its cost does."""
    outer = langsmith_run(LLM_ID, parent_id=TRACE_ID, run_type="llm", offset=1, duration=3)
    inner = langsmith_run(TOOL_ID, parent_id=LLM_ID, run_type="llm", offset=2)
    outer.prompt_tokens, outer.completion_tokens = 120, 7
    inner.prompt_tokens, inner.completion_tokens = 120, 7
    outer_span, inner_span = _load_llm_runs(outer, inner)
    assert outer_span.token_counts is None
    assert inner_span.token_counts is not None
    assert inner_span.token_counts.input_total == 120


def test_unreadable_server_info_is_an_error_not_a_silent_v1_load():
    root = langsmith_run(TRACE_ID)
    client = FakeLangSmithClient([root], [root])
    client._info = SimpleNamespace(version="", instance_flags={})  # the SDK's failed /info
    loader = LangSmithTraceLoader(LangSmithTraceConfig(project_name="p"), client)
    with pytest.raises(LangSmithTraceLoadError, match="Could not read LangSmith server info"):
        loader.load()
