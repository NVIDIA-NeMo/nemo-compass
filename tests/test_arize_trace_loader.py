# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AX REST fixtures exercise discovery, complete span retrieval and normalization."""

import copy
import json
from datetime import datetime, timezone

import httpx
import pytest

from compass.cli.main import _check_environment, _configured_trace_loader
from compass.config import RunConfig
from trace_ingest.loaders.arize import (
    ArizeTraceConfig,
    ArizeTraceLoader,
    ArizeTraceLoadError,
    _AXSpan,
    _normalize_trace,
    _record,
)
from trace_ingest.loaders.helper.merge import assemble
from trace_ingest.models import UNSET, SpanKind

TRACE = "1" * 32
OTHER = "2" * 32
START = datetime(2026, 11, 1, tzinfo=timezone.utc)
END = datetime(2026, 11, 2, tzinfo=timezone.utc)


def record(key, *, parent=None, kind="AGENT", attrs=None, trace=TRACE, **extra):
    return {
        "context": {"trace_id": trace, "span_id": key},
        "parent_id": parent,
        "kind": kind,
        "name": f"{kind.lower()}-{key}",
        "start_time": "2026-11-01T00:00:01Z",
        "end_time": "2026-11-01T00:00:02Z",
        "attributes": attrs or {},
        **extra,
    }


def fixture_spans():
    return [
        record("root", kind="CHAIN", attrs={"input.value": "Hello", "output.value": "Done"}),
        record(
            "llm",
            parent="root",
            kind="LLM",
            attrs={
                "output.value": "Looking it up",
                "llm.model_name": "test-model",
                "llm.token_count.prompt": "100",
                "llm.token_count.completion": 20,
                "llm.cost.total": "0.003",
                "llm.invocation_parameters": '{"temperature":0.2,"model":"test-model","messages":[]}',
                "llm.tools": [
                    {
                        "tool.json_schema": '{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}'
                    }
                ],
            },
        ),
        record(
            "tool-null",
            parent="root",
            kind="TOOL",
            attrs={
                "tool.name": "lookup",
                "input.value": '{"id":1}',
                "input.mime_type": "application/json",
                "output.value": "null",
                "output.mime_type": "application/json",
            },
            status_code="ERROR",
            status_message="not found",
            evaluations=[{"name": "quality", "score": 0.0, "label": "bad"}],
        ),
        record("tool-missing", parent="root", kind="TOOL", status_code="UNSET"),
    ]


def normalize(rows):
    trace = _normalize_trace(TRACE, [_AXSpan.model_validate(row) for row in rows])
    return assemble([_record(trace)])[0]


def config(*, from_timestamp=START, **kwargs):
    return ArizeTraceConfig(
        project_id="project-id", from_timestamp=from_timestamp, to_timestamp=END, **kwargs
    )


def test_native_openinference_mapping_is_deterministic_and_preserves_results():
    original = fixture_spans()
    trace = normalize(list(reversed(original)))
    assert trace.model_dump() == normalize(original).model_dump()
    (root,) = trace.root_spans
    assert root.kind is SpanKind.AGENT and root.attributes["subtype"] == "CHAIN"
    assert root.attributes["arize"]["kind"] == "CHAIN"
    assert [s.id for s in root.children] == ["llm", "tool-missing", "tool-null"]
    llm, missing, null = root.children
    assert llm.kind is SpanKind.LLM
    assert llm.model == "test-model"
    assert llm.token_counts.input_total == 100
    assert llm.token_counts.output_tokens == 20
    assert llm.cost_usd == 0.003
    assert llm.llm_params == {"temperature": 0.2}
    assert llm.tool_definitions[0].name == "lookup"
    assert llm.tool_definitions[0].raw == original[1]["attributes"]["llm.tools"][0]
    assert null.input == {"id": 1} and null.output is None
    assert null.tool_name == "lookup"
    assert null.tool_call.results[0].content is None
    assert null.tool_call.results[0].is_error is True
    assert missing.output is UNSET
    assert missing.tool_call.results == [] and missing.tool_call.result_count == 0
    assert missing.error is None
    assert root.turns[0].content == "Hello"
    assert root.turns[1].llm_span_ids == ["llm"]
    assert root.turns[-1].content == "Done"
    assert trace.aggregate.latency_ms == 1000
    assert list(trace.evaluator_results.values()) == [
        {"name": "quality", "score": 0.0, "label": "bad"}
    ]
    assert null.attributes["arize"]["status_message"] == "not found"


@pytest.mark.parametrize(
    "corruption,match",
    [
        ("duplicate", "Duplicate"),
        ("missing-parent", "missing parent"),
        ("cycle", "parent cycle"),
        ("wrong-trace", "different trace"),
        ("empty", "no spans"),
    ],
)
def test_invalid_graphs_fail_instead_of_returning_partial_traces(corruption, match):
    rows = fixture_spans()
    if corruption == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif corruption == "missing-parent":
        rows[1]["parent_id"] = "absent"
    elif corruption == "cycle":
        rows[1]["parent_id"] = "tool-null"
        rows[2]["parent_id"] = "llm"
    elif corruption == "wrong-trace":
        rows[1]["context"]["trace_id"] = OTHER
    else:
        rows = []
    with pytest.raises(ArizeTraceLoadError, match=match):
        normalize(rows)


def test_load_pages_complete_spans_and_drops_discovery_filter(monkeypatch):
    monkeypatch.setenv("ARIZE_AX_API_KEY", "test-secret")
    monkeypatch.setenv("ARIZE_AX_API_URL", "https://overridden.example.com")
    requests = []
    spans = fixture_spans()

    def respond(request):
        body = json.loads(request.content)
        requests.append((request.url.path, dict(request.url.params), body))
        assert request.url.host == "private-arize.example.com"
        assert request.headers["Authorization"] == "Bearer test-secret"
        if request.url.path == "/v2/traces":
            assert body["filter"] == "status_code = 'ERROR'"
            summary = {
                "trace_id": TRACE,
                "root_span_id": "root",
                "start_time": "2026-10-31T23:59:59Z",
                "end_time": "2026-11-02T00:00:01Z",
                "spans_truncated": False,
                "spans": [spans[0]],
            }
            return httpx.Response(
                200, json={"traces": [summary], "pagination": {"has_more": False}}
            )
        assert body["filter"] == f"context.trace_id = '{TRACE}'"
        assert datetime.fromisoformat(body["start_time"]) < START
        assert datetime.fromisoformat(body["end_time"]) > END
        if "cursor" not in request.url.params:
            return httpx.Response(
                200,
                json={
                    "spans": spans[2:],
                    "pagination": {"has_more": True, "next_cursor": "span-page-2"},
                },
            )
        return httpx.Response(200, json={"spans": spans[:2], "pagination": {"has_more": False}})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        loader = ArizeTraceLoader(
            config(
                max_traces=1,
                filter="status_code = 'ERROR'",
                api_url="https://private-arize.example.com",
            ),
            client,
        )
        snapshot = loader.load()
    assert [r[0] for r in requests] == ["/v2/traces", "/v2/spans", "/v2/spans"]
    assert requests[0][1]["limit"] == "2"
    assert snapshot.trace_count == 1
    assert next(iter(snapshot)).model_dump() == normalize(spans).model_dump()
    assert loader.describe() == {
        "source": "arize",
        "trace_count": 1,
        "span_count": 4,
        "call_count": 2,
        "distinct_cases": 1,
        "distinct_sessions": 0,
        "api_url": "https://private-arize.example.com",
        "project_id": "project-id",
        "from_timestamp": START.isoformat(),
        "to_timestamp": END.isoformat(),
        "filter": "status_code = 'ERROR'",
        "max_traces": 1,
    }


def test_discovery_pagination_keeps_multiple_roots_and_stops_at_trace_limit(monkeypatch):
    monkeypatch.setenv("ARIZE_AX_API_KEY", "test-secret")
    reads = []

    def respond(request):
        reads.append((request.url.path, request.url.params.get("cursor")))
        if request.url.path == "/v2/traces":
            cursor = request.url.params.get("cursor")
            trace_id = {None: TRACE, "discovery-2": OTHER, "discovery-3": "3" * 32}[cursor]
            return httpx.Response(
                200,
                json={
                    "traces": [
                        {
                            "trace_id": trace_id,
                            "root_span_id": "root",
                            "start_time": START.isoformat(),
                            "end_time": END.isoformat(),
                        }
                    ],
                    "pagination": {
                        "has_more": True,
                        "next_cursor": "discovery-2"
                        if trace_id == TRACE
                        else "discovery-3"
                        if trace_id == OTHER
                        else "do-not-fetch",
                    },
                },
            )
        body = json.loads(request.content)
        trace_id = (
            TRACE if TRACE in body["filter"] else OTHER if OTHER in body["filter"] else "3" * 32
        )
        return httpx.Response(
            200,
            json={
                "spans": [record("root", trace=trace_id), record("other-root", trace=trace_id)],
                "pagination": {"has_more": False},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        snapshot = ArizeTraceLoader(config(max_traces=2), client).load()
    assert [t.id for t in snapshot] == [TRACE, OTHER]
    assert all(len(t.root_spans) == 2 for t in snapshot)
    assert [read for read in reads if read[0] == "/v2/traces"] == [
        ("/v2/traces", None),
        ("/v2/traces", "discovery-2"),
        ("/v2/traces", "discovery-3"),
    ]
    assert len(reads) == 6  # One complete trace of lookahead decides the next conversation key.


@pytest.mark.parametrize("missing_on_reread", [False, True])
def test_session_limit_rereads_and_merges_records_in_order(monkeypatch, missing_on_reread):
    monkeypatch.setenv("ARIZE_AX_API_KEY", "test-secret")
    older = "original-request-with-uuid-id"
    session = "customer's session"
    query_filter = "status_code = 'ERROR' OR name = 'agent'"
    fetches = []
    selections = []

    def spans(trace_id):
        offset = "00:00:05" if trace_id == TRACE else "00:00:01"
        # Only a child needs to carry the OpenInference session ID.
        attrs = {"session.id": session} if trace_id != OTHER else {}
        return [
            record(
                "root",
                trace=trace_id,
                attrs={"input.value": f"question {trace_id}", "output.value": f"answer {trace_id}"},
                start_time=f"2026-11-01T{offset}Z",
                end_time="2026-11-01T00:00:06Z",
            ),
            record(
                "tool",
                parent="root",
                trace=trace_id,
                kind="TOOL",
                attrs={"tool.name": "lookup", "output.value": "ok", **attrs},
                start_time=f"2026-11-01T{offset}Z",
                end_time="2026-11-01T00:00:06Z",
            ),
        ]

    def summary(trace_id):
        return {
            "trace_id": trace_id,
            "root_span_id": "root",
            "start_time": START.isoformat(),
            "end_time": END.isoformat(),
        }

    def respond(request):
        body = json.loads(request.content)
        if request.url.path == "/v2/traces":
            selections.append(body)
            if body["filter"] == query_filter:
                # A different conversation interrupts the selected session; its
                # older record is beyond the limit and must be read by session.
                ids = [TRACE, OTHER]
            else:
                assert (
                    body["filter"]
                    == f"({query_filter}) AND (attributes.session.id = 'customer''s session')"
                )
                assert body["start_time"] == START.isoformat()
                assert body["end_time"] == END.isoformat()
                ids = [older] if missing_on_reread else [TRACE, older]
            return httpx.Response(
                200, json={"traces": [summary(t) for t in ids], "pagination": {"has_more": False}}
            )
        trace_id = next(t for t in [TRACE, OTHER, older] if t in body["filter"])
        fetches.append(trace_id)
        return httpx.Response(
            200, json={"spans": spans(trace_id), "pagination": {"has_more": False}}
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        loader = ArizeTraceLoader(config(max_traces=1, filter=query_filter), client)
        snapshot = loader.load()
    (conversation,) = snapshot
    assert conversation.id == "session:" + session
    assert conversation.session_id == session
    assert conversation.session_basis == "arize.session.id"
    assert [r["trace_id"] for r in conversation.attributes["records"]] == [older, TRACE]
    (root,) = conversation.root_spans
    assert root.turns is not UNSET
    assert [s.id for s in root.children] == [older + "/root", TRACE + "/root"]
    assert root.turns[0].content == "question " + older
    assert root.turns[-1].content == "answer " + TRACE
    assert root.turns[1].tool_call_span_ids == [older + "/tool"]
    assert [loss.code for loss in conversation.losses] == (
        ["merge_incomplete"] if missing_on_reread else []
    )
    assert fetches == [TRACE, OTHER, older]  # Re-queries reuse each trace's complete spans.
    assert len(selections) == 2
    assert loader.describe()["trace_count"] == 1
    assert loader.describe()["distinct_sessions"] == 1
    assert loader.describe()["span_count"] == 4  # Excludes the synthesized conversation root.


def test_conflicting_span_sessions_do_not_merge_or_choose_arbitrarily():
    rows = fixture_spans()
    rows[0]["attributes"]["session.id"] = "session-a"
    rows[1]["attributes"]["session.id"] = "session-b"
    trace = normalize(rows)
    assert trace.session_id is None and trace.id == TRACE
    assert trace.losses[0].code == "conflicting_session_ids"
    assert trace.losses[0].detail == {"session_ids": ["session-a", "session-b"]}


@pytest.mark.parametrize("failure", ["cursor", "http", "malformed", "empty"])
def test_pagination_errors_empty_selection_and_http_failures(monkeypatch, failure):
    monkeypatch.setenv("ARIZE_AX_API_KEY", "test-secret")

    def respond(request):
        if failure == "http":
            return httpx.Response(403, text="test-secret")
        if failure == "malformed":
            return httpx.Response(200, json={"traces": None})
        return httpx.Response(
            200,
            json={
                "traces": [],
                "pagination": {"has_more": failure == "cursor", "next_cursor": "repeated"},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        loader = ArizeTraceLoader(config(), client)
        if failure == "empty":
            assert loader.load().trace_count == 0
        else:
            with pytest.raises(ArizeTraceLoadError) as caught:
                loader.load()
            assert "test-secret" not in str(caught.value)


def test_cli_configuration_credentials_and_loader_selection(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COMPASS_API_KEY", "model-key")
    monkeypatch.setenv("ARIZE_AX_API_KEY", "test-secret")
    source = {"project_id": "project-id", "from_timestamp": START, "to_timestamp": END}
    run = RunConfig(trace={"arize": source, "max_traces": 3}, compass_signals={"tool_issues": {}})
    assert _check_environment(run) == "model-key"
    loader = _configured_trace_loader(run.trace)
    assert isinstance(loader, ArizeTraceLoader)
    assert loader.config.max_traces == 3
    monkeypatch.delenv("ARIZE_AX_API_KEY")
    with pytest.raises(ValueError, match="ARIZE_AX_API_KEY"):
        _check_environment(run)
    with pytest.raises(ValueError, match="timezone"):
        config(from_timestamp=START.replace(tzinfo=None))
