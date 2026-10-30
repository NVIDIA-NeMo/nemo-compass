# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo Platform Intake loader and normalization tests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import permutations

import httpx
import pytest

from compass.cli.main import (
    _configured_trace_loader,
)
from compass.config import TraceConfig
from compass.trace_loaders.intake import (
    IntakeLoadError,
    IntakeStatus,
    IntakeTraceLoader,
    IntakeTraceLoaderConfig,
    IntakeTraceQuery,
)
from compass.traces import UNSET, SpanKind, Trace
from trace_ingest.catalog import tool_catalog
from trace_ingest.models import TokenCounts, ToolResult

BASE_URL = "https://platform.example.test"
WORKSPACE = "example-workspace"
TRACE_ID = "trace-1"


def _empty_evaluator_page() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "data": [],
            "pagination": _pagination(page=1, total_pages=0, total_results=0, size=0),
        },
    )


def _mock_transport(handler) -> httpx.MockTransport:
    """Older fixtures have no separately recorded evaluation results."""

    def dispatch(request):
        if request.url.path.endswith("/evaluator-results"):
            return _empty_evaluator_page()
        return handler(request)

    return httpx.MockTransport(dispatch)


def _pagination(*, page: int, total_pages: int, total_results: int, size: int) -> dict:
    return {
        "page": page,
        "page_size": 1,
        "current_page_size": size,
        "total_pages": total_pages,
        "total_results": total_results,
    }


def _trace_record(trace_id: str = TRACE_ID, session_id: str = "session-1") -> dict:
    return {
        "id": trace_id,
        "root_span_id": "root",
        "session_id": session_id,
        "workspace": WORKSPACE,
        "input": '{"messages":[{"role":"user","content":"find it"}]}',
        "output": "null",
        "evaluation_context": {
            "evaluation_name": "eval-a",
            "test_case_name": "case-a",
        },
        "started_at": "2026-08-28T18:40:00",
        "ended_at": "2026-08-28T18:40:02",
        "status": "success",
        "cost_usd": 0.125,
    }


def _span_record(span_id: str, **overrides: object) -> dict:
    values = {
        "span_id": span_id,
        "session_id": "session-1",
        "workspace": WORKSPACE,
        "kind": "AGENT",
        "name": "agent",
        "source": "atif",
        "trace_id": TRACE_ID,
        "started_at": "2026-08-28T18:40:00",
        "ended_at": "2026-08-28T18:40:02",
        "status": "success",
        "raw_attributes": "{}",
    }
    values.update(overrides)
    return values


def _handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/evaluator-results"):
        return _empty_evaluator_page()
    page = int(request.url.params["page"])
    if request.url.path.endswith("/traces"):
        assert request.url.params["filter[started_at][gte]"] == "2026-08-28T18:40:00+00:00"
        assert request.url.params["filter[started_at][lte]"] == "2026-08-28T18:42:00+00:00"
        assert request.url.params["sort"] == "started_at"
        assert request.url.params["mode"] == "detailed"
        return httpx.Response(
            200,
            json={
                "data": [_trace_record()],
                "pagination": _pagination(page=page, total_pages=1, total_results=1, size=1),
            },
        )

    assert request.url.path.endswith("/spans")
    assert request.url.params["filter[trace_id]"] == TRACE_ID
    assert request.url.params["sort"] == "started_at"
    assert request.url.params["mode"] == "detailed"
    if page == 1:
        data = [
            _span_record(
                "tool",
                parent_span_id="root",
                kind="TOOL",
                name="search",
                tool_name="search",
                ended_at="2026-08-28T18:40:01",
                input=json.dumps(
                    {
                        "tool_call_id": "call-1",
                        "function_name": "search",
                        "arguments": {"query": "report"},
                    }
                ),
                output=json.dumps(
                    {"source_call_id": "call-1", "content": '{"documents":["report"]}'}
                ),
            )
        ]
    else:
        data = [
            _span_record(
                "root",
                raw_attributes=json.dumps(
                    {
                        "agent": {
                            "tool_definitions": [
                                {
                                    "name": "search",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {"query": {"type": "string"}},
                                        "required": ["query"],
                                    },
                                }
                            ]
                        }
                    }
                ),
            )
        ]
    return httpx.Response(
        200,
        json={
            "data": data,
            "pagination": _pagination(page=page, total_pages=2, total_results=2, size=1),
        },
    )


def _query() -> IntakeTraceQuery:
    return IntakeTraceQuery(
        started_at_gte=datetime(2026, 8, 28, 18, 40, tzinfo=timezone.utc),
        started_at_lte=datetime(2026, 8, 28, 18, 42, tzinfo=timezone.utc),
    )


def _config(**overrides) -> IntakeTraceLoaderConfig:
    values = {
        "base_url": BASE_URL,
        "workspace": WORKSPACE,
        "query": _query(),
    }
    values.update(overrides)
    return IntakeTraceLoaderConfig.model_validate(values)


def _record_roots(trace: Trace):
    """Every record's root spans, under the merged trace's root span."""
    (merged_root,) = trace.root_spans
    assert merged_root.id == trace.id == f"session:{trace.session_id}"
    return merged_root.children


def _records(trace: Trace) -> list[dict]:
    """The merged trace's per-record provenance entries."""
    records = trace.attributes["records"]
    assert isinstance(records, list)
    entries = [entry for entry in records if isinstance(entry, dict)]
    assert len(entries) == len(records)
    return entries


def test_intake_query_requires_timezone_aware_ordered_bounds():
    with pytest.raises(ValueError, match="started_at_gte must include a timezone"):
        IntakeTraceQuery(
            started_at_gte=datetime(2026, 8, 28, 18, 40),
            started_at_lte=datetime(2026, 8, 28, 18, 42, tzinfo=timezone.utc),
        )

    with pytest.raises(ValueError, match="must not follow"):
        IntakeTraceQuery(
            started_at_gte=datetime(2026, 8, 29, tzinfo=timezone.utc),
            started_at_lte=datetime(2026, 8, 28, tzinfo=timezone.utc),
        )


def test_intake_query_builds_filters_and_supports_newest_first():
    query = IntakeTraceQuery(
        started_at_gte=datetime(2026, 8, 28, 18, 40, tzinfo=timezone.utc),
        started_at_lte=datetime(2026, 8, 28, 18, 42, tzinfo=timezone.utc),
        agent_name="agent-a",
        status=IntakeStatus.ERROR,
        max_traces=500,
        sort="-started_at",
    )

    assert query.params()["filter[agent_name]"] == "agent-a"
    assert query.params()["filter[status]"] == "error"
    assert query.describe()["sort"] == "-started_at"


def test_intake_config_is_strict_and_validates_runtime_settings():
    config = IntakeTraceLoaderConfig(
        base_url=f" {BASE_URL}/ ",
        workspace=f" {WORKSPACE} ",
        query=_query(),
    )

    assert config.base_url == BASE_URL
    assert config.workspace == WORKSPACE
    assert config.page_size == 100
    assert config.timeout_seconds == 30.0

    with pytest.raises(ValueError, match="page_size"):
        _config(page_size=0)
    with pytest.raises(ValueError, match="absolute HTTP"):
        IntakeTraceLoaderConfig(
            base_url="platform.example.test",
            workspace=WORKSPACE,
            query=_query(),
        )
    with pytest.raises(ValueError, match="must not contain credentials"):
        IntakeTraceLoaderConfig(
            base_url="https://user:secret@platform.example.test",
            workspace=WORKSPACE,
            query=_query(),
        )
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        IntakeTraceLoaderConfig.model_validate(
            {
                "base_url": BASE_URL,
                "workspace": WORKSPACE,
                "query": _query(),
                "headers": {},
            }
        )


def test_intake_loader_returns_canonical_snapshot_without_writing_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loader = IntakeTraceLoader(
        config=_config(page_size=1),
        transport=_mock_transport(_handler),
    )

    snapshot = loader.load()

    assert loader.load() is snapshot
    assert snapshot.trace_count == 1
    trace = snapshot.get_trace_by_id("session:session-1")
    assert (trace.session_id, trace.session_basis) == ("session-1", "intake.session_id")
    (root,) = _record_roots(trace)
    assert root.id == f"{TRACE_ID}/root"
    assert (root.kind, root.attributes["subtype"]) == (SpanKind.CHAIN, "AGENT")
    assert [span.id for span in root.children] == [f"{TRACE_ID}/tool"]
    assert root.start_time is not None
    assert root.start_time.tzinfo == timezone.utc
    assert trace.attributes["task_text"] == {"messages": [{"role": "user", "content": "find it"}]}
    assert trace.case_id == "case-a"
    assert "tool_catalog" not in trace.attributes
    assert tool_catalog(trace) == {
        "search": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
    }
    (record,) = _records(trace)
    assert record["trace_id"] == TRACE_ID
    intake_attributes = record["attributes"]["intake"]
    assert isinstance(intake_attributes, dict)
    assert intake_attributes["output"] is None
    assert trace.aggregate.cost_usd == 0.125
    assert trace.aggregate.latency_ms == 2000.0
    tool = root.children[0]
    assert tool.kind is SpanKind.TOOL
    assert tool.input == {"query": "report"}
    assert tool.output == '{"documents":["report"]}'
    assert tool.tool_call is not None
    assert tool.tool_call.call_id == "call-1"
    assert tool.tool_call.result_id == "call-1"
    assert tool.tool_call.index == 0
    assert tool.attributes["explicit_error"] is False
    assert tool.tool_call.results == [
        ToolResult(content='{"documents":["report"]}', is_error=False)
    ]

    assert Trace.model_validate_json(trace.model_dump_json()) == trace
    dumped = trace.model_dump(exclude_unset=True)
    assert dumped["root_spans"][0]["children"][0]["children"][0]["id"] == f"{TRACE_ID}/tool"
    assert list(tmp_path.iterdir()) == []
    assert loader.describe()["source"].startswith("intake:https://")
    # The Intake root and tool spans, not the merged conversation root.
    assert loader.describe()["span_count"] == 2
    assert loader.describe()["call_count"] == 1
    assert loader.describe()["distinct_cases"] == 1
    assert loader.describe()["distinct_sessions"] == 1


@pytest.mark.parametrize("source", ["atif", "otel"])
def test_intake_preserves_missing_and_explicit_null_payloads(source):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/traces"):
            return httpx.Response(
                200,
                json={
                    "data": [_trace_record()],
                    "pagination": _pagination(page=1, total_pages=1, total_results=1, size=1),
                },
            )
        return httpx.Response(
            200,
            json={
                "data": [
                    _span_record("missing", kind="TOOL", tool_name="missing", source=source),
                    _span_record(
                        "null",
                        kind="TOOL",
                        tool_name="null",
                        source=source,
                        input="null",
                        output="null",
                    ),
                ],
                "pagination": {
                    **_pagination(page=1, total_pages=1, total_results=2, size=2),
                    "page_size": 100,
                },
            },
        )

    loader = IntakeTraceLoader(
        config=_config(),
        transport=_mock_transport(handler),
    )

    trace = next(iter(loader.load()))
    assert trace.id == "session:session-1"
    missing_span, null_span = _record_roots(trace)
    assert missing_span.input is UNSET
    assert missing_span.output is UNSET
    assert missing_span.tool_call is not None
    assert missing_span.tool_call.result_count == 0
    assert null_span.input is None
    assert null_span.output is None
    assert null_span.tool_call is not None
    assert null_span.tool_call.result_count == 1


def test_intake_loader_rejects_invalid_parent():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/traces"):
            data = [_trace_record()]
        else:
            data = [_span_record("child", parent_span_id="missing")]
        return httpx.Response(
            200,
            json={
                "data": data,
                "pagination": _pagination(
                    page=1, total_pages=1, total_results=len(data), size=len(data)
                ),
            },
        )

    loader = IntakeTraceLoader(
        config=_config(),
        transport=_mock_transport(handler),
    )

    with pytest.raises(IntakeLoadError, match="references missing parent"):
        loader.load()


def test_intake_loader_rejects_pagination_inconsistency():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [_trace_record()],
                "pagination": _pagination(page=2, total_pages=1, total_results=1, size=1),
            },
        )

    with pytest.raises(IntakeLoadError, match="returned page 2"):
        IntakeTraceLoader(
            config=_config(),
            transport=_mock_transport(handler),
        ).load()


def test_intake_max_traces_stops_before_requesting_another_page():
    selection_pages = []
    session_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if request.url.path.endswith("/traces"):
            if "filter[session_id]" in params:
                session_requests.append(dict(params))
                session = params["filter[session_id]"]
                data = [_trace_record(f"trace-{session[-1]}", session)]
                pagination = _pagination(page=1, total_pages=1, total_results=1, size=1)
            else:
                page = int(params["page"])
                selection_pages.append(page)
                data = [_trace_record(f"trace-{page}", f"session-{page}")]
                pagination = _pagination(page=page, total_pages=3, total_results=3, size=1)
        else:
            trace_id = params["filter[trace_id]"]
            data = [_span_record("root", trace_id=trace_id)]
            pagination = _pagination(page=1, total_pages=1, total_results=1, size=1)
        return httpx.Response(200, json={"data": data, "pagination": pagination})

    loader = _configured_trace_loader(
        TraceConfig(
            intake=_config(page_size=1),
            max_traces=1,
        )
    )
    assert isinstance(loader, IntakeTraceLoader)
    loader.transport = _mock_transport(handler)

    assert [trace.id for trace in loader.load()] == ["session:session-1"]
    assert selection_pages == [1]
    (session_request,) = session_requests
    assert session_request["filter[session_id]"] == "session-1"
    assert session_request["filter[started_at][gte]"] == "2026-08-28T18:40:00+00:00"
    assert session_request["filter[started_at][lte]"] == "2026-08-28T18:42:00+00:00"


def test_intake_loader_rejects_duplicate_trace_ids():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/traces"):
            data = [_trace_record(), _trace_record()]
            pagination = _pagination(page=1, total_pages=1, total_results=2, size=2)
        else:
            data = [_span_record("root")]
            pagination = _pagination(page=1, total_pages=1, total_results=1, size=1)
        return httpx.Response(200, json={"data": data, "pagination": pagination})

    loader = IntakeTraceLoader(
        config=_config(),
        transport=_mock_transport(handler),
    )

    with pytest.raises(IntakeLoadError, match="duplicate trace id"):
        loader.load()


def _conversation_handler(requests):
    """Two traces of session-1; the second re-sends the first exchange."""
    inputs = {
        "trace-1": [{"role": "user", "content": "find it"}],
        "trace-2": [
            {"role": "user", "content": "find it"},
            {"role": "assistant", "content": "found"},
            {"role": "user", "content": "thanks"},
        ],
    }
    outputs = {"trace-1": "found", "trace-2": "you're welcome"}
    started = {"trace-1": "2026-08-28T18:40:00", "trace-2": "2026-08-28T18:41:00"}

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/evaluator-results"):
            return _empty_evaluator_page()
        if request.url.path.endswith("/traces"):
            data = []
            for trace_id in inputs:
                record = _trace_record(trace_id)
                record["input"] = json.dumps({"messages": inputs[trace_id]})
                record["started_at"] = started[trace_id]
                record["ended_at"] = None
                data.append(record)
        else:
            trace_id = request.url.params["filter[trace_id]"]
            times = {"started_at": started[trace_id], "ended_at": started[trace_id]}
            data = [
                _span_record("root", trace_id=trace_id, **times),
                _span_record(
                    "llm",
                    trace_id=trace_id,
                    parent_span_id="root",
                    kind="LLM",
                    output=json.dumps(outputs[trace_id]),
                    **times,
                ),
            ]
        return httpx.Response(
            200,
            json={
                "data": data,
                "pagination": _pagination(
                    page=1, total_pages=1, total_results=len(data), size=len(data)
                ),
            },
        )

    return handler


def test_intake_session_traces_merge_into_one_conversation():
    requests = []
    loader = IntakeTraceLoader(
        config=_config(), transport=httpx.MockTransport(_conversation_handler(requests))
    )

    (trace,) = loader.load()

    assert trace.id == "session:session-1"
    roots = _record_roots(trace)
    assert [(root.id, root.kind, root.attributes["subtype"]) for root in roots] == [
        ("trace-1/root", SpanKind.CHAIN, "AGENT"),
        ("trace-2/root", SpanKind.CHAIN, "AGENT"),
    ]
    turns = trace.root_spans[0].turns
    assert turns is not UNSET
    assert [(turn.role.value, turn.content, turn.llm_span_ids) for turn in turns] == [
        ("user", "find it", []),
        ("agent", "found", ["trace-1/llm"]),
        ("user", "thanks", []),
        ("agent", "you're welcome", ["trace-2/llm"]),
    ]
    # Selection read the whole window, so the session is not fetched again.
    trace_requests = [r for r in requests if r.url.path.endswith("/traces")]
    assert [r.url.params.get("filter[session_id]") for r in trace_requests] == [None]
    # Evaluator results are per session, so they are fetched once for both traces.
    evaluator_requests = [r for r in requests if r.url.path.endswith("/evaluator-results")]
    assert [r.url.params["filter[session_id]"] for r in evaluator_requests] == ["session-1"]


def test_intake_loader_rejects_an_empty_selection():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [],
                "pagination": _pagination(page=1, total_pages=0, total_results=0, size=0),
            },
        )

    with pytest.raises(IntakeLoadError, match="returned no traces"):
        IntakeTraceLoader(
            config=_config(),
            transport=_mock_transport(handler),
        ).load()


def test_intake_loader_reports_http_failure():
    def fail(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unavailable")

    loader = IntakeTraceLoader(
        config=_config(),
        transport=_mock_transport(fail),
    )

    with pytest.raises(IntakeLoadError, match="503"):
        loader.load()


@pytest.mark.parametrize("token", [None, "", "test-token"])
def test_intake_optional_auth_is_sent_on_every_page_without_persisting_token(monkeypatch, token):
    if token is None:
        monkeypatch.delenv("NMP_ACCESS_TOKEN", raising=False)
    else:
        monkeypatch.setenv("NMP_ACCESS_TOKEN", token)
    requests = []

    def handler(request):
        requests.append(request)
        assert request.headers.get("Authorization") == (f"Bearer {token}" if token else None)
        return _handler(request)

    loader = IntakeTraceLoader(config=_config(), transport=httpx.MockTransport(handler))
    loader.load()
    assert len(requests) == 4
    if token:
        assert token not in repr(loader.config)
        assert token not in json.dumps(loader.describe())
        assert token not in next(iter(loader.load())).model_dump_json()


def _loader_for_records(spans, *, span_total=None):
    def handler(request):
        is_trace = request.url.path.endswith("/traces")
        data = [_trace_record()] if is_trace else spans
        total = len(data) if is_trace or span_total is None else span_total
        return httpx.Response(
            200,
            json={
                "data": data,
                "pagination": _pagination(
                    page=1, total_pages=1, total_results=total, size=len(data)
                ),
            },
        )

    return IntakeTraceLoader(config=_config(), transport=_mock_transport(handler))


@pytest.mark.parametrize("total", [0, 2])
def test_intake_rejects_incomplete_or_excess_span_pages(total):
    loader = _loader_for_records([_span_record("root")], span_total=total)
    with pytest.raises(IntakeLoadError, match=f"spans expected {total} records but received 1"):
        loader.load()


@pytest.mark.parametrize(
    ("spans", "message"),
    [
        ([_span_record("root"), _span_record("root")], "duplicate span id"),
        (
            [_span_record("a", parent_span_id="b"), _span_record("b", parent_span_id="a")],
            "parent cycle",
        ),
        ([_span_record("a", parent_span_id="a")], "parent cycle"),
        ([_span_record("root", trace_id="another-trace")], "belongs to trace"),
        ([_span_record("root", raw_attributes="[]")], "must be a JSON object"),
        ([_span_record("root", raw_attributes="{")], "invalid raw_attributes JSON"),
    ],
)
def test_intake_rejects_malformed_span_data(spans, message):
    with pytest.raises(IntakeLoadError, match=message):
        _loader_for_records(spans).load()


def test_intake_equal_time_span_order_is_independent_of_response_order():
    spans = [
        _span_record("root"),
        _span_record("a", kind="TOOL", parent_span_id="root"),
        _span_record("b", kind="TOOL", parent_span_id="root"),
    ]
    outputs = []
    for ordering in permutations(spans):
        loader = _loader_for_records(list(ordering))
        trace = next(iter(loader.load()))
        outputs.append(trace.model_dump_json())
        (root,) = _record_roots(trace)
        assert [(span.id, span.tool_call.index) for span in root.children] == [
            (f"{TRACE_ID}/a", 0),
            (f"{TRACE_ID}/b", 1),
        ]
    assert len(set(outputs)) == 1


def test_intake_preserves_span_model_and_cost():
    loader = _loader_for_records(
        [
            _span_record("root", kind="LLM", model="example-model", cost_total_usd=0.125),
        ],
    )
    (span,) = _record_roots(next(iter(loader.load())))
    assert span.model == "example-model"
    assert span.cost_usd == 0.125


def test_intake_maps_span_token_counts():
    loader = _loader_for_records(
        [
            _span_record(
                "root",
                kind="LLM",
                input_tokens=120,
                output_tokens=30,
                total_tokens=150,
                cached_tokens=100,
            ),
            _span_record("tool", kind="TOOL", parent_span_id="root"),
        ],
    )
    (span,) = _record_roots(next(iter(loader.load())))
    assert span.token_counts == TokenCounts(
        input_total=120, cached_input_tokens=100, output_tokens=30
    )
    # A span that records no counts has none.
    assert span.children[0].token_counts is None


def test_intake_langchain_tool_message_output_carries_the_call_id():
    message = {"type": "tool", "data": {"content": "42", "tool_call_id": "call-1"}}
    loader = _loader_for_records(
        [
            _span_record("root", source="otel"),
            _span_record(
                "a", source="otel", kind="TOOL", parent_span_id="root", output=json.dumps(message)
            ),
            _span_record(
                "b",
                source="otel",
                kind="TOOL",
                parent_span_id="root",
                started_at="2026-08-28T18:40:01",
                output='"42"',
            ),
        ],
    )
    (root,) = _record_roots(next(iter(loader.load())))
    calls = [span.tool_call for span in root.children]
    assert [(call.call_id, call.result_id) for call in calls if call] == [
        ("call-1", "call-1"),
        (None, None),
    ]
    assert calls[0] is not None and calls[0].results == [
        ToolResult(content=message, is_error=False)
    ]


def _agent_attributes(*names: str) -> str:
    definitions = [{"name": name, "parameters": {"type": "object"}} for name in names]
    return json.dumps({"agent": {"tool_definitions": definitions}})


@pytest.mark.parametrize("root_names", [("search",), ()])
def test_intake_atif_definitions_stay_on_the_agent_that_declared_them(root_names):
    trace = next(
        iter(
            _loader_for_records(
                [
                    _span_record("root", raw_attributes=_agent_attributes(*root_names)),
                    _span_record(
                        "sub",
                        parent_span_id="root",
                        raw_attributes=_agent_attributes("lookup"),
                    ),
                    _span_record(
                        "step",
                        kind="CHAIN",
                        parent_span_id="sub",
                        raw_attributes=_agent_attributes("lookup", "write"),
                    ),
                ]
            ).load()
        )
    )

    session = trace.root_spans[0]
    (root,) = session.children
    sub = root.children[0]
    # The record's root agent is the merged trace's run, so its tools move to its root.
    assert root.tool_definitions is UNSET
    declared = [] if session.tool_definitions is UNSET else session.tool_definitions
    assert [definition.name for definition in declared] == list(root_names)
    assert [definition.name for definition in sub.tool_definitions] == ["lookup", "lookup", "write"]
    assert sub.children[0].tool_definitions is UNSET
    assert "intake_loader_gaps" not in _records(trace)[0]["attributes"]


def test_intake_atif_definitions_without_owner_are_a_gap():
    trace = next(
        iter(
            _loader_for_records(
                [_span_record("root", kind="CHAIN", raw_attributes=_agent_attributes("search"))]
            ).load()
        )
    )

    assert tool_catalog(trace) is None
    assert _records(trace)[0]["attributes"]["intake_loader_gaps"] == [
        {"code": "tool_definitions_without_owner_span", "span_id": "root"}
    ]


@pytest.mark.parametrize(
    "status, payload, is_error, result_attributes, explicit_error, error",
    [
        ("success", {}, False, {}, False, None),
        (
            "success",
            {"is_error": True},
            None,
            {
                "is_error_status": "conflicting",
                "is_error_flags": {"is_error": True, "status": "success"},
            },
            True,
            "ATIF tool result reported is_error",
        ),
        (
            "error",
            {"is_error": False},
            None,
            {
                "is_error_status": "conflicting",
                "is_error_flags": {"is_error": False, "status": "error"},
            },
            True,
            "boom",
        ),
        ("error", {"is_error": True}, True, {}, True, "boom"),
        (
            "unknown",
            {"extra": {"is_error": True}},
            True,
            {},
            True,
            "ATIF tool result reported is_error",
        ),
        (
            "unknown",
            {"is_error": "yes", "extra": {"isError": True}},
            None,
            {
                "is_error_status": "invalid",
                "is_error_flags": {"is_error": "yes", "extra.isError": True},
            },
            None,
            None,
        ),
    ],
)
def test_intake_atif_tool_result_combines_status_and_payload_flag(
    status, payload, is_error, result_attributes, explicit_error, error
):
    trace = next(
        iter(
            _loader_for_records(
                [
                    _span_record("root"),
                    _span_record(
                        "tool",
                        parent_span_id="root",
                        kind="TOOL",
                        status=status,
                        error_message="boom" if status == "error" else None,
                        output=json.dumps({"source_call_id": "call-1", "content": "x", **payload}),
                    ),
                ]
            ).load()
        )
    )

    tool = _record_roots(trace)[0].children[0]
    assert tool.tool_call.results == [
        ToolResult(content="x", is_error=is_error, attributes=result_attributes)
    ]
    assert tool.attributes["explicit_error"] is explicit_error
    assert tool.error == error


@pytest.mark.parametrize("query_limit", [None, 5])
@pytest.mark.parametrize("shared_limit", [None, 1])
def test_intake_shared_limit_maps_to_query(query_limit, shared_limit):
    query = _query().model_copy(update={"max_traces": query_limit})
    config = _config(query=query)
    settings = {"intake": config}
    if shared_limit is not None:
        settings["max_traces"] = shared_limit
    loader = _configured_trace_loader(TraceConfig.model_validate(settings))
    assert isinstance(loader, IntakeTraceLoader)
    assert loader.config.query.max_traces == (
        shared_limit if shared_limit is not None else query_limit
    )
    assert config.query.max_traces == query_limit


def test_intake_failed_load_does_not_cache_partial_snapshot_and_can_retry():
    fail = True

    def handler(request):
        if request.url.path.endswith("/traces"):
            data = [_trace_record("trace-1"), _trace_record("trace-2")]
        else:
            trace_id = request.url.params["filter[trace_id]"]
            if fail and trace_id == "trace-2":
                return httpx.Response(503)
            data = [_span_record("root", trace_id=trace_id)]
        return httpx.Response(
            200,
            json={
                "data": data,
                "pagination": _pagination(
                    page=1, total_pages=1, total_results=len(data), size=len(data)
                ),
            },
        )

    loader = IntakeTraceLoader(config=_config(), transport=_mock_transport(handler))
    before = loader.describe()
    assert before["trace_count"] == before["span_count"] == before["call_count"] == 0
    with pytest.raises(IntakeLoadError, match="503"):
        loader.load()
    assert loader.describe() == before
    fail = False
    snapshot = loader.load()
    assert [trace.id for trace in snapshot] == ["session:session-1"]
    (trace,) = snapshot
    assert [record["trace_id"] for record in _records(trace)] == ["trace-1", "trace-2"]
    assert loader.load() is snapshot
    description = loader.describe()
    assert description["trace_count"] == 1
    # Both records' root spans; the merged conversation root is not an Intake span.
    assert description["span_count"] == 2
    assert description["distinct_cases"] == 1
    description["query"].clear()
    assert loader.describe()["query"] == _query().describe()


@pytest.mark.parametrize("sort", ["started_at", "-started_at"])
def test_experiment_selection_pages_evaluations_and_merges_traces_before_limit(sort):
    requests = []
    trace_pages = []
    records = {
        "eval-a": [("a1", "2026-08-28T18:40:01"), ("a3", "2026-08-28T18:40:03")],
        "eval-b": [("b2", "2026-08-28T18:40:02"), ("b4", "2026-08-28T18:40:04")],
    }

    def handler(request):
        requests.append(request)
        page = int(request.url.params["page"])
        if request.url.path.endswith("/evaluations"):
            assert request.url.params["filter[experiment_id]"] == "experiment-1"
            assert request.url.params["sort"] == "name"
            return httpx.Response(
                200,
                json={
                    "data": [{"name": ["eval-a", "eval-b"][page - 1]}],
                    "pagination": _pagination(page=page, total_pages=2, total_results=2, size=1),
                },
            )
        if request.url.path.endswith("/traces"):
            params = request.url.params
            assert "filter[experiment_id]" not in params
            assert params["filter[agent_name]"] == "agent-a"
            assert params["filter[status]"] == "success"
            assert params["filter[started_at][gte]"] == "2026-08-28T18:40:00+00:00"
            assert params["filter[started_at][lte]"] == "2026-08-28T18:42:00+00:00"
            assert params["sort"] == sort
            name = params["filter[evaluation_name]"]
            selected = sorted(records[name], reverse=sort.startswith("-"))
            session = params.get("filter[session_id]")
            if session is not None:
                selected = [entry for entry in selected if entry[0] == session]
                page_count = len(selected)
            else:
                trace_pages.append((name, page))
                page_count = 2
            data = []
            for trace_id, timestamp in selected[page - 1 : page]:
                record = _trace_record(trace_id, trace_id)
                record["started_at"] = timestamp
                record["ended_at"] = None
                record["evaluation_context"]["evaluation_name"] = name
                data.append(record)
            return httpx.Response(
                200,
                json={
                    "data": data,
                    "pagination": _pagination(
                        page=page, total_pages=page_count, total_results=page_count, size=len(data)
                    ),
                },
            )
        if request.url.path.endswith("/spans"):
            trace_id = request.url.params["filter[trace_id]"]
            return httpx.Response(
                200,
                json={
                    "data": [_span_record("root", trace_id=trace_id)],
                    "pagination": _pagination(page=1, total_pages=1, total_results=1, size=1),
                },
            )
        return _empty_evaluator_page()

    config = _config(
        page_size=1,
        query=_query().model_copy(
            update={
                "experiment_id": "experiment-1",
                "agent_name": "agent-a",
                "status": IntakeStatus.SUCCESS,
                "sort": sort,
            }
        ),
    )
    loader = _configured_trace_loader(TraceConfig(intake=config, max_traces=2))
    assert isinstance(loader, IntakeTraceLoader)
    loader.transport = httpx.MockTransport(handler)
    snapshot = loader.load()

    assert [trace.id for trace in snapshot] == (
        ["session:b4", "session:a3"] if sort.startswith("-") else ["session:a1", "session:b2"]
    )
    assert len(trace_pages) == 3
    assert len([r for r in requests if r.url.path.endswith("/spans")]) == 2
    assert loader.describe()["query"]["experiment_id"] == "experiment-1"
    assert loader.load() is snapshot


@pytest.mark.parametrize("evaluation_name", [None, "eval-a", "outside-experiment"])
@pytest.mark.parametrize("has_evaluations", [True, False])
def test_experiment_selection_intersects_evaluation_name_and_never_falls_back(
    evaluation_name, has_evaluations
):
    trace_requests = []

    def handler(request):
        if request.url.path.endswith("/evaluations"):
            data = [{"name": "eval-a"}] if has_evaluations else []
            return httpx.Response(
                200,
                json={
                    "data": data,
                    "pagination": _pagination(
                        page=1,
                        total_pages=int(has_evaluations),
                        total_results=len(data),
                        size=len(data),
                    ),
                },
            )
        if request.url.path.endswith("/traces"):
            trace_requests.append(request)
            assert request.url.params["filter[evaluation_name]"] == "eval-a"
        return _handler(request)

    loader = IntakeTraceLoader(
        _config(
            query=_query().model_copy(
                update={
                    "experiment_id": "experiment-1",
                    "evaluation_name": evaluation_name,
                }
            )
        ),
        transport=httpx.MockTransport(handler),
    )
    if has_evaluations and evaluation_name != "outside-experiment":
        assert loader.load().trace_count == 1
        assert len(trace_requests) == 1
    else:
        with pytest.raises(IntakeLoadError, match="returned no traces"):
            loader.load()
        assert trace_requests == []


def test_experiment_lookup_failure_does_not_load_unfiltered_traces(monkeypatch):
    monkeypatch.setenv("NMP_ACCESS_TOKEN", "test-token")

    def handler(request):
        assert request.url.path.endswith("/evaluations")
        assert request.headers["Authorization"] == "Bearer test-token"
        return httpx.Response(403)

    loader = IntakeTraceLoader(
        _config(query=_query().model_copy(update={"experiment_id": "experiment-1"})),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(IntakeLoadError, match="evaluations request failed.*403"):
        loader.load()


def test_experiment_id_cli_overrides_yaml(tmp_path):
    from compass.cli.main import get_config

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "trace:\n  intake:\n    base_url: https://platform.example.test\n"
        "    workspace: test\n    query:\n"
        "      started_at_gte: 2026-08-28T18:40:00Z\n"
        "      started_at_lte: 2026-08-28T18:42:00Z\n"
        "      experiment_id: from-yaml\n"
    )
    config = get_config(
        ["--config", str(config_path), "--trace.intake.query.experiment-id", "from-cli"]
    )
    loader = _configured_trace_loader(config.trace)
    assert isinstance(loader, IntakeTraceLoader)
    assert loader.config.query.experiment_id == "from-cli"
    assert "filter[experiment_id]" not in loader.config.query.params()
