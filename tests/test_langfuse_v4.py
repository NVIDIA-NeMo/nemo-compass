# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real SDK's HTTP parsing and both Langfuse API generations."""

import json
from datetime import datetime, timedelta, timezone
from itertools import count

import httpx
import pytest
from langfuse import Langfuse
from langfuse.api.core.api_error import ApiError

from compass.cli.main import _configured_trace_loader
from compass.config import LangfuseConfig, TraceConfig
from trace_ingest.loaders.langfuse import (
    LangfuseTraceConfig,
    LangfuseTraceLoader,
    LangfuseTraceLoadError,
    LangfuseV3TraceLoader,
    LangfuseV4TraceLoader,
)
from trace_ingest.models import UNSET, SpanKind

START = datetime(2026, 8, 1, tzinfo=timezone.utc)
END = START + timedelta(hours=1)
CLIENT_IDS = count()


def observation(id, *, trace_id="t", parent=None, start=START, **extra):
    return dict(
        id=id,
        traceId=trace_id,
        projectId="p",
        type="AGENT" if parent is None else "TOOL",
        parentObservationId=parent,
        startTime=start.isoformat(),
        endTime=(start + timedelta(seconds=1)).isoformat(),
        name="lookup",
        level="DEFAULT",
        sessionId="s",
        traceName="assistant",
        **extra,
    )


def page(data, cursor=None):
    return {"data": data, "meta": {"cursor": cursor, "limit": 100}}


def loader(handler, *, server_version="4.0.0", **config):
    def route(request):
        if server_version is not None and request.url.path == "/api/public/health":
            return httpx.Response(200, json={"status": "OK", "version": server_version})
        return handler(request)

    client = Langfuse(
        public_key=f"pk-lf-v4-test-{next(CLIENT_IDS)}",
        secret_key="sk-lf-v4-test",
        base_url="https://langfuse.test",
        tracing_enabled=False,
        httpx_client=httpx.Client(transport=httpx.MockTransport(route)),
    )
    return LangfuseTraceLoader(
        LangfuseTraceConfig(START, END, base_url="https://langfuse.test", **config), client=client
    )


@pytest.mark.parametrize("direct", [False, True])
def test_v4_pagination_full_tree_scores_and_context(direct):
    requests = []
    root = observation("root", start=START, input='{"question":"where?"}')
    child = observation(
        "child", parent="root", start=START + timedelta(seconds=10), output="null", totalCost=0.12
    )

    def handler(request):
        requests.append(request)
        q = request.url.params
        if request.url.path.endswith("/v3/scores"):
            common = dict(
                projectId="p",
                source="API",
                timestamp=END.isoformat(),
                environment="default",
                createdAt=END.isoformat(),
                updatedAt=END.isoformat(),
                name="quality",
                subject={"kind": "observation", "id": "child", "traceId": "t"},
            )
            if q.get("cursor"):
                return httpx.Response(
                    200,
                    json=page([dict(common, id="score-b", dataType="CATEGORICAL", value="bad")]),
                )
            return httpx.Response(
                200,
                json=page(
                    [dict(common, id="score-a", dataType="BOOLEAN", value=False)], "scores-next"
                ),
            )
        assert request.url.path.endswith("/v2/observations")
        if q.get("traceId"):
            assert q["traceId"] == "t"
            assert "filter" not in q
            assert datetime.fromisoformat(q["fromStartTime"]) == START
            assert datetime.fromisoformat(q["toStartTime"]) == END
            assert "io" in q["fields"] and "trace_context" in q["fields"]
            return httpx.Response(
                200,
                json=page(
                    [root] if q.get("cursor") else [child],
                    None if q.get("cursor") else "details-next",
                ),
            )
        conditions = json.loads(q["filter"])
        assert conditions[-2]["column"] == "startTime"
        assert conditions[-1]["operator"] == "<"
        return httpx.Response(
            200,
            json=page([observation("match")], "selection-next")
            if not q.get("cursor")
            else page([]),
        )

    instance = loader(
        handler,
        filter_string='[{"type":"string","column":"environment","operator":"=","value":"production"}]',
    )
    if direct:
        instance = LangfuseV4TraceLoader(instance.config, instance.client, "https://langfuse.test")
    trace = list(instance.load())[0]
    assert instance.describe()["api_version"] == "v4"
    assert trace.source_url == "https://langfuse.test/project/p/traces/t"
    assert trace.root_spans[0].id == "root"
    tool = trace.root_spans[0].children[0]
    assert tool.kind == SpanKind.TOOL and tool.output is None
    assert trace.root_spans[0].output is UNSET
    assert tool.tool_call is not None
    assert tool.tool_call.result_count == 1
    assert trace.aggregate.cost_usd == 0.12
    assert trace.aggregate.latency_ms == 11000
    assert (trace.session_id, trace.session_basis) == ("s", "langfuse.sessionId")
    context = trace.attributes["langfuse"]
    assert isinstance(context, dict)
    assert context["input"] == {"question": "where?"}
    scores = trace.evaluator_results["quality"]
    assert isinstance(scores, list)
    assert all(isinstance(score, dict) for score in scores)
    assert [score["value"] for score in scores if isinstance(score, dict)] == [False, "bad"]
    first_score = scores[0]
    assert isinstance(first_score, dict)
    subject = first_score["subject"]
    assert isinstance(subject, dict)
    assert subject["traceId"] == "t"
    assert len(requests) == 6


@pytest.mark.parametrize("status", [401, 403, 404, 405, 429, 500, 503])
def test_health_does_not_hide_errors(status):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, json={"message": "failure"})

    with pytest.raises(ApiError):
        loader(handler, server_version=None).load()
    assert requests
    assert all(r.url.path == "/api/public/health" for r in requests)


@pytest.mark.parametrize(
    ("version", "api_version", "endpoint"),
    [
        ("3.155.1", "v3", "/api/public/traces"),
        ("v3.0.0", "v3", "/api/public/traces"),
        ("4.0.0", "v4", "/api/public/v2/observations"),
        ("4.17.0-rc.1+build.2", "v4", "/api/public/v2/observations"),
    ],
)
def test_server_version_selects_api(version, api_version, endpoint):
    paths = []

    def handler(request):
        paths.append(request.url.path)
        if request.url.path == "/api/public/health":
            assert not request.url.params
            return httpx.Response(200, json={"status": "OK", "version": version})
        assert request.url.path == endpoint
        return httpx.Response(
            200,
            json=page([])
            if api_version == "v4"
            else {
                "data": [],
                "meta": {"page": 1, "limit": 100, "totalItems": 0, "totalPages": 0},
            },
        )

    instance = loader(handler, server_version=None)
    assert instance.describe()["api_version"] is None
    assert paths == []
    assert list(instance.load()) == []
    assert instance.describe()["api_version"] == api_version
    assert paths == ["/api/public/health", endpoint]


@pytest.mark.parametrize("version", ["", "unknown", "4", "4.invalid", "2.55.1", "5.0.0"])
def test_unsupported_server_version_fails_before_loading(version):
    def handler(request):
        pytest.fail(f"Unexpected data request: {request.url}")

    with pytest.raises(LangfuseTraceLoadError, match="Unsupported Langfuse server version"):
        loader(handler, server_version=version).load()


@pytest.mark.parametrize("status", [404, 405])
def test_v4_data_errors_do_not_fallback(status):
    paths = []

    def handler(request):
        paths.append(request.url.path)
        return httpx.Response(status, json={"message": "Not found"})

    with pytest.raises(ApiError):
        loader(handler).load()
    assert paths == ["/api/public/v2/observations"]


@pytest.mark.parametrize("kind", ["selection", "details", "scores"])
def test_nonprogressing_cursors_fail(kind):
    def handler(request):
        q = request.url.params
        if request.url.path.endswith("/v3/scores"):
            return httpx.Response(200, json=page([], "stuck"))
        if q.get("traceId"):
            return httpx.Response(
                200, json=page([observation("root")], "stuck" if kind == "details" else None)
            )
        return httpx.Response(
            200, json=page([observation("root")], "stuck" if kind == "selection" else None)
        )

    with pytest.raises(LangfuseTraceLoadError, match="cursor"):
        loader(handler).load()


@pytest.mark.parametrize("case", ["duplicate", "cycle", "wrong_trace", "empty"])
def test_malformed_tree_rejected(case):
    def handler(request):
        if request.url.path.endswith("/v3/scores"):
            return httpx.Response(200, json=page([]))
        if request.url.params.get("traceId"):
            data = {
                "duplicate": [observation("root"), observation("root")],
                "cycle": [observation("root", parent="root")],
                "wrong_trace": [observation("root", trace_id="other")],
                "empty": [],
            }[case]
            return httpx.Response(200, json=page(data))
        return httpx.Response(200, json=page([observation("root")]))

    with pytest.raises(LangfuseTraceLoadError):
        loader(handler).load()


def test_unique_trace_limit_and_empty_result():
    def handler(request):
        if request.url.path.endswith("/v3/scores"):
            return httpx.Response(200, json=page([]))
        q = request.url.params
        if q.get("traceId"):
            return httpx.Response(200, json=page([observation("root", trace_id=q["traceId"])]))
        return httpx.Response(
            200,
            json=page(
                [observation("a"), observation("b"), observation("c", trace_id="other")], "unused"
            ),
        )

    result = list(loader(handler, max_traces=1).load())
    assert [t.id for t in result] == ["t"]
    assert list(loader(lambda r: httpx.Response(200, json=page([]))).load()) == []


def test_config_wires_langfuse_source():
    source = LangfuseConfig(from_timestamp=START, to_timestamp=END)
    instance = _configured_trace_loader(TraceConfig(langfuse=source))
    assert isinstance(instance, LangfuseTraceLoader)
    assert instance.config.from_timestamp == START
    assert instance.config.to_timestamp == END


@pytest.mark.parametrize("offline", [False, True])
def test_v4_cli_runs_evidence(tmp_path, monkeypatch, select_signals, offline):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from nooa.unifiedllm import FakeLLMClient

    import compass.cli.main as cli
    import trace_ingest.loaders.langfuse.loader as adapter

    tool = observation(
        "tool", parent="root", input='{"query":"missing"}', output='{"error":"not found"}'
    )
    tool.update(level="ERROR", statusMessage="not found")

    def handler(request):
        if request.url.path.endswith("/v3/scores"):
            return httpx.Response(200, json=page([]))
        return httpx.Response(200, json=page([tool, observation("root")]))

    client = loader(handler).client
    monkeypatch.setattr(adapter, "_new_langfuse_client", lambda base_url: client)
    compilation = SimpleNamespace(compile_insights=AsyncMock(return_value=[]))
    monkeypatch.setenv("COMPASS_API_KEY", "test-key")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setattr(cli, "_build_llm", lambda *a: FakeLLMClient())
    monkeypatch.setattr(cli, "InsightCompilation", lambda llm: compilation)
    source_args = [
        "--trace.langfuse.base-url",
        "https://langfuse.test",
        "--trace.langfuse.from-timestamp",
        START.isoformat(),
        "--trace.langfuse.to-timestamp",
        END.isoformat(),
    ]
    if offline:
        path, scores_path = write_exports(tmp_path, [tool, observation("root")])
        source_args = [
            "--trace.langfuse-trace-export-file.path",
            str(path),
            "--trace.langfuse-trace-export-file.scores-path",
            str(scores_path),
            "--trace.langfuse-trace-export-file.api-version",
            "v4",
        ]
        monkeypatch.setattr(
            adapter, "_new_langfuse_client", lambda *a: pytest.fail("offline contacted Langfuse")
        )
    assert (
        cli.main(
            [
                *source_args,
                "--compass-signals",
                json.dumps(
                    select_signals(
                        tool_issues={"include_audit_problems": True}, anomaly_and_patterns={}
                    )
                ),
                "--output-path",
                str(tmp_path / "insights.yml"),
            ]
        )
        == cli.EXIT_OK
    )
    evidence = compilation.compile_insights.await_args.args[0]
    assert {item.signal_name for item in evidence} == {"tool-issues", "anomaly-and-patterns"}
    tool_evidence = next(item for item in evidence if item.signal_name == "tool-issues")
    assert tool_evidence.problems


def test_v4_missing_parent_and_conflicting_sessions_remain_visible():
    orphan = observation("orphan", parent="absent")
    other = observation("other")
    other["sessionId"] = "another-session"

    def handler(request):
        if request.url.path.endswith("/v3/scores"):
            return httpx.Response(200, json=page([]))
        return httpx.Response(200, json=page([orphan, other]))

    instance = loader(handler)
    trace = list(instance.load())[0]
    assert instance.describe()["unresolved_parent_count"] == 1
    assert trace.session_id is None and trace.session_basis is None
    assert len(trace.root_spans) == 2
    assert trace.root_spans[0].attributes["langfuse"]["session_id"] in {"s", "another-session"}


def test_v4_trace_details_include_time_bounds_to_avoid_server_timeout():
    """Self-hosted v4 can time out on trace-ID-only observation reads."""

    def handler(request):
        if request.url.path.endswith("/v3/scores"):
            return httpx.Response(200, json=page([]))
        q = request.url.params
        if q.get("traceId") and not (q.get("fromStartTime") and q.get("toStartTime")):
            return httpx.Response(
                422,
                json={"message": "Please narrow your request", "error": "Request timed out"},
            )
        return httpx.Response(200, json=page([observation("root")]))

    assert [trace.id for trace in loader(handler).load()] == ["t"]


def test_v4_window_excludes_outside_steps_and_records_partial_scope():
    records = [
        observation("root", start=START - timedelta(seconds=1)),
        observation("child", parent="root", start=START),
        observation("later", parent="root", start=END),
    ]

    def handler(request):
        if request.url.path.endswith("/v3/scores"):
            return httpx.Response(200, json=page([]))
        q = request.url.params
        lower = datetime.fromisoformat(q["fromStartTime"])
        upper = datetime.fromisoformat(q["toStartTime"])
        return httpx.Response(
            200,
            json=page(
                [
                    row
                    for row in records
                    if lower <= datetime.fromisoformat(row["startTime"]) < upper
                ]
            ),
        )

    instance = loader(handler)
    trace = list(instance.load())[0]
    assert [span.id for span in trace.root_spans] == ["child"]
    assert instance.describe()["unresolved_parent_count"] == 1
    assert trace.attributes["observation_window"] == {
        "from_timestamp": START.isoformat(),
        "to_timestamp": END.isoformat(),
        "may_exclude_trace_steps": True,
    }


@pytest.mark.parametrize("loader_type", [LangfuseV3TraceLoader, LangfuseV4TraceLoader])
def test_version_specific_loaders_do_not_detect_again(loader_type):
    requests = []

    def handler(request):
        requests.append(request.url.path)
        endpoint = (
            "/api/public/traces"
            if loader_type is LangfuseV3TraceLoader
            else "/api/public/v2/observations"
        )
        assert request.url.path == endpoint
        return httpx.Response(
            200,
            json={
                "data": [],
                "meta": {"page": 1, "limit": 100, "totalItems": 0, "totalPages": 0, "cursor": None},
            },
        )

    entry = loader(handler, server_version=None)
    instance = (
        loader_type(entry.config, entry.client)
        if loader_type is LangfuseV3TraceLoader
        else loader_type(entry.config, entry.client, "https://langfuse.test")
    )
    assert requests == []
    assert list(instance.load()) == []
    assert len(requests) == 1
    assert instance.describe()["trace_count"] == 0


def exported_score(id, subject, **extra):
    return dict(
        id=id,
        projectId="p",
        name="quality",
        dataType="BOOLEAN",
        value=False,
        source="API",
        timestamp=END.isoformat(),
        createdAt=END.isoformat(),
        updatedAt=END.isoformat(),
        environment="default",
        subject=subject,
        **extra,
    )


def test_v4_details_fetch_four_traces_concurrently_in_stable_order():
    from threading import Barrier, Lock

    barrier = Barrier(4, timeout=5)
    lock = Lock()
    active = peak = 0
    trace_ids = [f"t{i}" for i in range(8)]
    calls = []

    def handler(request):
        nonlocal active, peak
        q = request.url.params
        if request.url.path.endswith("/v3/scores"):
            return httpx.Response(200, json=page([]))
        trace_id = q.get("traceId")
        if trace_id is None:
            return httpx.Response(
                200, json=page([observation("match", trace_id=t) for t in reversed(trace_ids)])
            )
        with lock:
            calls.append((trace_id, q.get("cursor")))
        if q.get("cursor"):
            return httpx.Response(
                200, json=page([observation("child", trace_id=trace_id, parent="root")])
            )
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            barrier.wait()
        finally:
            with lock:
                active -= 1
        return httpx.Response(200, json=page([observation("root", trace_id=trace_id)], "next"))

    snapshot = loader(handler).load()
    assert peak == 4
    assert [trace.id for trace in snapshot] == trace_ids
    for trace_id in trace_ids:
        assert [cursor for tid, cursor in calls if tid == trace_id] == [None, "next"]


def test_v4_batched_scores_paginate_and_associate():
    trace_ids = [f"t{i:03}" for i in range(101)]
    items = [observation("root", trace_id=tid) for tid in trace_ids]
    scores = [
        exported_score(f"s-{tid}", {"kind": "observation", "id": "root", "traceId": tid})
        for tid in trace_ids
    ]
    score_requests = []

    def handler(request):
        q = request.url.params
        if request.url.path.endswith("/v3/scores"):
            batch = q["traceId"].split(",")
            score_requests.append((batch, q.get("cursor")))
            assert "fromTimestamp" not in q and "toTimestamp" not in q
            matching = [s for s in scores if s["subject"]["traceId"] in batch]
            return httpx.Response(
                200,
                json=page(matching[50:])
                if q.get("cursor")
                else page(matching[:50], "next" if len(matching) > 50 else None),
            )
        if q.get("traceId"):
            return httpx.Response(200, json=page([observation("root", trace_id=q["traceId"])]))
        return httpx.Response(
            200, json=page(items[100:]) if q.get("cursor") else page(items[:100], "selection-next")
        )

    live = list(loader(handler, max_traces=101).load())
    assert score_requests == [
        (trace_ids[:100], None),
        (trace_ids[:100], "next"),
        (trace_ids[100:], None),
    ]
    for trace in live:
        assert trace.evaluator_results["quality"][0]["id"] == f"s-{trace.id}"


def test_v4_concurrent_detail_failure_does_not_return_partial_snapshot():
    def handler(request):
        q = request.url.params
        if q.get("traceId") == "bad":
            return httpx.Response(400, json={"message": "failed detail request"})
        return httpx.Response(
            200,
            json=page(
                [observation("root", trace_id=q["traceId"])]
                if q.get("traceId")
                else [observation("root", trace_id=t) for t in ("good", "bad")]
            ),
        )

    with pytest.raises(ApiError):
        loader(handler).load()


def cli_export(data):
    return {
        "status": 200,
        "headers": {},
        "body": {
            "data": list(data),
            "meta": {
                "pages": 1,
                "fetchedItems": len(data),
                "truncated": False,
            },
        },
    }


def write_exports(tmp_path, observations=(), scores=()):
    paths = (tmp_path / "observations.json", tmp_path / "scores.json")
    for path, data in zip(paths, (observations, scores), strict=True):
        path.write_text(json.dumps(cli_export(data)))
    return paths


def file_loader(tmp_path, observations=(), scores=(), **kwargs):
    from trace_ingest.loaders.langfuse import LangfuseV4FileTraceConfig, LangfuseV4FileTraceLoader

    return LangfuseV4FileTraceLoader(
        LangfuseV4FileTraceConfig(*write_exports(tmp_path, observations, scores), **kwargs)
    )


def test_v4_cli_exports_match_online_normalization(tmp_path, monkeypatch):
    items = [
        observation("child", parent="root", output="null", totalCost=0.12),
        observation("root"),
    ]
    scores = [
        exported_score("s1", {"kind": "observation", "id": "child", "traceId": "t"}),
        exported_score("s2", {"kind": "trace", "id": "t"}),
    ]

    def handler(request):
        data = scores if request.url.path.endswith("/v3/scores") else items
        return httpx.Response(
            200, json=page(data[1:]) if request.url.params.get("cursor") else page(data[:1], "next")
        )

    live = next(iter(loader(handler).load()))
    monkeypatch.setattr(
        Langfuse, "__init__", lambda *a, **kw: pytest.fail("offline constructed a client")
    )
    instance = file_loader(tmp_path, items, scores, base_url="https://langfuse.test")
    offline = next(iter(instance.load()))

    live.attributes.pop("observation_window")
    offline.attributes.pop("export_path")
    assert offline.model_dump() == live.model_dump()
    assert offline.root_spans[0].input is UNSET
    assert offline.root_spans[0].children[0].output is None
    assert instance.describe()["call_count"] == 1


def test_v4_file_limits_and_partial_trees(tmp_path):
    instance = file_loader(
        tmp_path,
        [
            observation("orphan", trace_id="new", parent="missing", start=END),
            observation("old", trace_id="old"),
        ],
        max_traces=1,
    )
    trace = next(iter(instance.load()))
    assert trace.id == "new"
    assert trace.source_url == (tmp_path / "observations.json").as_uri()
    assert trace.evaluator_results == {}
    assert instance.describe()["unresolved_parent_count"] == 1
    assert instance.describe()["export_trace_count"] == 2
    assert instance.describe()["truncated"] is True


@pytest.mark.parametrize("which", ["observations", "scores"])
@pytest.mark.parametrize(
    "case",
    [
        "status",
        "truncated",
        "cursor",
        "missing_meta",
        "missing_truncated",
        "invalid_data",
        "bad_json",
    ],
)
def test_v4_exports_require_successful_complete_cli_responses(tmp_path, which, case):
    instance = file_loader(tmp_path, [observation("root")])
    path = tmp_path / f"{which}.json"
    payload = json.loads(path.read_text())
    if case == "status":
        payload["status"] = 401
    elif case == "truncated":
        payload["body"]["meta"]["truncated"] = True
    elif case == "cursor":
        payload["body"]["meta"]["cursor"] = "next"
    elif case == "missing_meta":
        payload["body"].pop("meta")
    elif case == "missing_truncated":
        payload["body"]["meta"].pop("truncated")
    elif case == "invalid_data":
        payload["body"]["data"] = {}
    path.write_text("bad json" if case == "bad_json" else json.dumps(payload))
    with pytest.raises(LangfuseTraceLoadError):
        instance.load()


@pytest.mark.parametrize(
    "case,match",
    [
        ("missing_trace", "trace"),
        ("duplicate_observation", "duplicate"),
        ("multiple_projects", "project"),
        ("cycle", "cycle"),
        ("naive_time", "timezone"),
    ],
)
def test_v4_file_rejects_invalid_observations(tmp_path, case, match):
    items = [observation("root")]
    if case == "missing_trace":
        items[0].pop("traceId")
    elif case == "duplicate_observation":
        items.append(items[0])
    elif case == "multiple_projects":
        other = observation("other")
        other["projectId"] = "other-project"
        items.append(other)
    elif case == "cycle":
        items[0]["parentObservationId"] = "root"
    elif case == "naive_time":
        items[0]["startTime"] = "2026-08-01T00:00:00"
    with pytest.raises(LangfuseTraceLoadError, match=match):
        file_loader(tmp_path, items).load()


def test_v4_file_config_and_successful_empty_exports(tmp_path):
    from compass.config import LangfuseTraceExportFileConfig
    from trace_ingest.loaders.langfuse import LangfuseV4FileTraceConfig, LangfuseV4FileTraceLoader

    observations, scores = write_exports(tmp_path)
    instance = _configured_trace_loader(
        TraceConfig(
            langfuse_trace_export_file=LangfuseTraceExportFileConfig(
                path=observations, scores_path=scores, api_version="v4"
            )
        )
    )
    assert isinstance(instance, LangfuseV4FileTraceLoader)
    assert list(instance.load()) == []
    assert instance.describe()["export_trace_count"] == 0
    assert instance.describe()["path"] == str(observations.resolve())
    with pytest.raises(ValueError, match="max_traces"):
        LangfuseV4FileTraceConfig(observations, scores, max_traces=0)


@pytest.mark.parametrize("case", ["duplicate", "wrong_project"])
def test_v4_file_rejects_conflicting_scores(tmp_path, case):
    score = exported_score("s", {"kind": "trace", "id": "t"})
    scores = [score, score] if case == "duplicate" else [score]
    if case == "wrong_project":
        score["projectId"] = "other"
    with pytest.raises(LangfuseTraceLoadError, match="score"):
        file_loader(tmp_path, [observation("root")], scores).load()


@pytest.mark.parametrize(
    "subject",
    [
        {"kind": "trace", "id": "other"},
        {"kind": "observation", "id": "root", "traceId": "other"},
        {"kind": "observation", "id": "missing"},
        {"kind": "session", "id": "s"},
        None,
    ],
)
def test_v4_file_counts_scores_outside_loaded_subjects(tmp_path, subject):
    instance = file_loader(tmp_path, [observation("root")], [exported_score("s", subject)])
    assert next(iter(instance.load())).evaluator_results == {}
    assert instance.describe()["ignored_score_count"] == 1


def test_v4_file_observation_ids_are_scoped_to_traces(tmp_path):
    items = [observation("root", trace_id="a"), observation("root", trace_id="b")]
    scores = [
        exported_score("assigned", {"kind": "observation", "id": "root", "traceId": "b"}),
        exported_score("ambiguous", {"kind": "observation", "id": "root"}),
    ]
    instance = file_loader(tmp_path, items, scores)
    a, b = list(instance.load())
    assert a.evaluator_results == {}
    assert b.evaluator_results["quality"][0]["id"] == "assigned"
    assert instance.describe()["ignored_score_count"] == 1


def test_v4_file_infers_unique_observation_score_trace(tmp_path):
    instance = file_loader(
        tmp_path,
        [observation("root")],
        [exported_score("s", {"kind": "observation", "id": "root"})],
    )
    trace = next(iter(instance.load()))
    assert trace.evaluator_results["quality"][0]["id"] == "s"
    assert instance.describe()["ignored_score_count"] == 0


@pytest.mark.parametrize(
    "data_type,value",
    [
        ("NUMERIC", 0.25),
        ("BOOLEAN", False),
        ("CATEGORICAL", "bad"),
        ("TEXT", "Needs clarification"),
    ],
)
def test_v4_file_preserves_score_types(tmp_path, data_type, value):
    score = exported_score("s", {"kind": "trace", "id": "t"})
    score.update(dataType=data_type, value=value)
    trace = next(iter(file_loader(tmp_path, [observation("root")], [score]).load()))
    record = trace.evaluator_results["quality"][0]
    assert record["value"] == value
    assert record["dataType"] == data_type


def test_v4_without_score_file_skips_requested_evaluation(tmp_path, select_signals):
    import asyncio

    from nooa.unifiedllm import FakeLLMClient

    from compass.cli.main import _run_compass_signals
    from compass.config import CompassSignalsConfig
    from trace_ingest.loaders.langfuse import LangfuseV4FileTraceLoader

    observations = tmp_path / "observations.json"
    observations.write_text(json.dumps(cli_export([observation("root")])))
    instance = _configured_trace_loader(
        TraceConfig.model_validate(
            {"langfuse_trace_export_file": {"api_version": "v4", "path": observations}}
        )
    )
    assert isinstance(instance, LangfuseV4FileTraceLoader)
    snapshot = instance.load()
    assert snapshot.trace_count == 1
    assert next(iter(snapshot)).evaluator_results == {}
    assert instance.describe()["scores_path"] is None
    assert instance.describe()["ignored_score_count"] == 0
    llm = FakeLLMClient()
    signals = CompassSignalsConfig.model_validate(select_signals(grader_failure_patterns={}))
    results = asyncio.run(_run_compass_signals(signals, snapshot, lambda: llm))
    assert results[0].signal_name == "grader-failure-patterns"
    assert results[0].skip_reason == "No evaluator results"
    assert llm.call_count == 0


def test_v4_supplied_missing_score_file_is_an_error(tmp_path):
    instance = file_loader(tmp_path, [observation("root")])
    instance.config.scores_path.unlink()
    with pytest.raises(LangfuseTraceLoadError, match="scores.json"):
        instance.load()
