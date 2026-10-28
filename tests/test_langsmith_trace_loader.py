# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
import threading
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
            return iter(self.roots[: kwargs.get("limit", len(self.roots))])
        selected = set(re.findall(r'"([0-9a-f-]{36})"', kwargs["trace_filter"]))
        return iter(item for item in self._fake_runs if str(item.trace_id) in selected)


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
    assert trace.aggregate.cost_usd == 0.0175
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

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def send_json(self, payload):
            contents = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(contents)))
            self.end_headers()
            self.wfile.write(contents)

        def do_GET(self):
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

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    api_url = f"http://127.0.0.1:{server.server_port}"
    try:
        client = Client(api_url=api_url, auto_batch_tracing=False)
        loader = LangSmithTraceLoader(
            LangSmithTraceConfig(project_name="glamr-ux"),
            client=client,
        )
        snapshot = loader.load()
    finally:
        server.shutdown()
        thread.join()

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


def test_loader_stops_selection_at_the_conversation_cap_and_marks_sessions_partial():
    roots = thread_roots()
    client = FakeLangSmithClient(roots, roots)
    loader = LangSmithTraceLoader(
        LangSmithTraceConfig(project_name="threads", max_traces=1),
        client=client,
    )

    (trace,) = loader.load()

    assert trace.id == "session:t1"
    assert [record["trace_id"] for record in _records(trace)] == [
        str(UUID(int=1)),
        str(UUID(int=2)),
    ]
    assert [(loss.code, loss.detail) for loss in trace.losses] == [
        ("merge_incomplete", {"reason": "max_traces", "loaded_record_count": 2})
    ]
    assert client.list_calls[1]["trace_filter"] == (f'in(id, ["{UUID(int=1)}", "{UUID(int=2)}"])')


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
