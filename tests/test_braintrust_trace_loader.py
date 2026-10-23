# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from nooa.unifiedllm import FakeLLMClient
from pydantic import ValidationError

from compass.config import RunConfig, TraceConfig
from compass.trace_loaders.braintrust import (
    BraintrustTraceConfig,
    BraintrustTraceLoader,
    BraintrustTraceLoadError,
)
from compass.traces import UNSET

START = datetime(2026, 9, 1, tzinfo=timezone.utc)
END = datetime(2026, 9, 2, tzinfo=timezone.utc)
FIXTURE = Path(__file__).parent / "fixtures" / "braintrust" / "spans.json"


@pytest.fixture(autouse=True)
def isolate_link_environment(monkeypatch):
    monkeypatch.delenv("BRAINTRUST_ORG_NAME", raising=False)
    monkeypatch.delenv("BRAINTRUST_APP_URL", raising=False)


@pytest.fixture
def rows():
    return json.loads(FIXTURE.read_text())


def config(**kwargs):
    kwargs.setdefault("project_id", "project")
    return BraintrustTraceConfig(START, END, **kwargs)


def client_for(rows, requests=None):
    def handle(request):
        query = json.loads(request.content)["query"]
        if requests is not None:
            requests.append(query)
        data = (
            [{"root_span_id": "root", "created": START.isoformat()}]
            if "WHERE is_root" in query
            else rows
        )
        return httpx.Response(200, json={"data": data})

    return httpx.Client(transport=httpx.MockTransport(handle))


@pytest.mark.parametrize("root_scores", [{"accuracy": 0}, None])
def test_normalized_tree_and_description_are_deterministic(rows, root_scores):
    rows[1]["scores"] = root_scores
    with client_for(rows) as client:
        loader = BraintrustTraceLoader(config(), client=client)
        assert loader.describe()["trace_count"] == 0
        snapshot = loader.load()
    trace = next(iter(snapshot))
    expected = json.loads(FIXTURE.with_name("normalized.json").read_text())
    expected["root_spans"][0]["attributes"]["braintrust"]["scores"] = root_scores
    expected["evaluator_results"] = root_scores or {}
    assert json.loads(trace.model_dump_json()) == expected
    first, second = trace.root_spans[0].children[0].children
    # JSON equality cannot prove the in-memory missing-value sentinel survives.
    assert first.input is None and first.output is None
    assert second.output is UNSET
    assert loader.describe() == {
        "source": "braintrust:https://api.braintrust.dev#project_logs/project",
        "api_url": "https://api.braintrust.dev",
        "project_id": "project",
        "experiment_id": None,
        "from_timestamp": START.isoformat(),
        "to_timestamp": END.isoformat(),
        "max_traces": 100,
        "span_count": 4,
        "trace_count": 1,
        "call_count": 2,
        "distinct_logical_cases": 1,
    }
    with client_for(list(reversed(rows))) as client:
        repeated = BraintrustTraceLoader(config(), client=client).load()
    assert [t.model_dump() for t in repeated] == [t.model_dump() for t in snapshot]


def test_selection_limit_and_complete_paginated_details(rows):
    queries = []

    def handle(request):
        query = json.loads(request.content)["query"]
        queries.append(query)
        if "WHERE is_root" in query:
            assert "created >= '2026-09-01T00:00:00+00:00'" in query
            assert "created < '2026-09-02T00:00:00+00:00'" in query
            return httpx.Response(
                200,
                json={"data": [{"root_span_id": "root", "created": START.isoformat()}]},
                headers={"x-bt-cursor": "unused-root-page"},
            )
        assert "created" not in query
        if "OFFSET" not in query:
            return httpx.Response(
                200, json={"data": rows[:2]}, headers={"x-bt-cursor": "span-page"}
            )
        assert "OFFSET 'span-page'" in query
        return httpx.Response(200, json={"data": rows[2:]})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        loader = BraintrustTraceLoader(config(max_traces=1), client)
        assert loader.load().trace_count == 1
    assert len(queries) == 3
    assert loader.describe()["span_count"] == 4


def test_root_pagination_order_and_experiment_source(rows):
    def handle(request):
        query = json.loads(request.content)["query"]
        assert "FROM experiment('experiment')" in query
        if "WHERE is_root" in query:
            if "OFFSET" not in query:
                return httpx.Response(
                    200,
                    json={"data": [{"root_span_id": "root", "created": "2026-09-01T01:00:00Z"}]},
                    headers={"x-bt-cursor": "next"},
                )
            return httpx.Response(
                200, json={"data": [{"root_span_id": "older", "created": START.isoformat()}]}
            )
        data = (
            rows
            if "= 'root'" in query
            else [{"id": "row-older", "span_id": "older", "root_span_id": "older"}]
        )
        return httpx.Response(200, json={"data": data})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        snapshot = BraintrustTraceLoader(
            config(project_id=None, experiment_id="experiment"), client
        ).load()
    assert [t.id for t in snapshot] == ["older", "root"]


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda r: r.append(deepcopy(r[0])), "duplicate"),
        (lambda r: r[0].update(root_span_id="other"), "different trace"),
        (lambda r: r[0].update(span_parents=["missing"]), "missing parent"),
        (lambda r: r[0].update(span_parents=["root-span", "llm"]), "exactly one parent"),
        (lambda r: r[0].update(span_parents=[]), "exactly one parent"),
        (lambda r: r[0].update(span_parents=None), "exactly one parent"),
        (lambda r: r[0].update(is_root=False, span_parents=None), "exactly one parent"),
        (lambda r: r[1].update(span_parents=["llm"], is_root=True), "root span has parents"),
        (lambda r: r[3].update(span_parents=["tool-b"]), "parent cycle"),
        (lambda r: r.pop(1), "missing its root"),
        (lambda r: r[0].update(metrics={"start": "yesterday"}), "must be numeric"),
        (lambda r: r[0].update(metrics={"start": 10, "end": 1}), "ends before"),
        (lambda r: r[0].update(metrics={"start": float("inf")}), "Invalid Braintrust trace"),
        (lambda r: r[0].update(span_id=""), "Invalid Braintrust trace"),
    ],
)
def test_rejects_malformed_traces(rows, change, match):
    change(rows)
    # Mock the JSON decoding for the non-JSON infinite-value test separately.
    if any(s.get("metrics", {}).get("start") == float("inf") for s in rows):
        from compass.trace_loaders.braintrust import _normalize_trace

        with pytest.raises(BraintrustTraceLoadError, match=match):
            _normalize_trace("root", rows, {})
        return
    with client_for(rows) as client, pytest.raises(BraintrustTraceLoadError, match=match):
        BraintrustTraceLoader(config(), client).load()


@pytest.mark.parametrize("payload", [{}, {"data": {}}, {"data": [None]}])
def test_rejects_malformed_query_responses(payload):
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
    ) as client:
        with pytest.raises(BraintrustTraceLoadError):
            BraintrustTraceLoader(config(), client).load()


@pytest.mark.parametrize("empty", [True, False])
def test_pagination_must_progress(empty):
    def handle(request):
        query = json.loads(request.content)["query"]
        row = {"root_span_id": "b" if "OFFSET" in query else "a", "created": START.isoformat()}
        return httpx.Response(
            200, json={"data": [] if empty else [row]}, headers={"x-bt-cursor": "same"}
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(BraintrustTraceLoadError, match="no progress"):
            BraintrustTraceLoader(config(), client).load()


def test_empty_selection():
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"data": []}))
    ) as client:
        loader = BraintrustTraceLoader(config(), client)
        assert len(loader.load()) == loader.describe()["span_count"] == 0


def test_http_errors_are_source_specific():
    status = 429
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(status, text="private details"))
    ) as client:
        with pytest.raises(BraintrustTraceLoadError, match=f"HTTP {status}") as error:
            BraintrustTraceLoader(config(), client).load()
    assert "private details" not in str(error.value)


def test_credentials_and_configured_endpoint(rows, monkeypatch):
    monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)
    with pytest.raises(BraintrustTraceLoadError, match="BRAINTRUST_API_KEY"):
        BraintrustTraceLoader(config()).load()
    monkeypatch.setenv("BRAINTRUST_API_KEY", "secret")
    monkeypatch.setenv("BRAINTRUST_API_URL", "https://eu.example.test/")
    original = httpx.Client
    requests = []

    def handle(request):
        assert request.headers["Authorization"] == "Bearer secret"
        assert str(request.url) == "https://eu.example.test/btql"
        requests.append(request)
        return httpx.Response(200, json={"data": []})

    monkeypatch.setattr(
        httpx, "Client", lambda **kw: original(transport=httpx.MockTransport(handle), **kw)
    )
    loader = BraintrustTraceLoader(config())
    loader.load()
    assert len(requests) == 1
    assert "secret" not in repr(loader) + json.dumps(loader.describe())


@pytest.mark.parametrize(
    "overrides",
    [
        {"project_id": None},
        {"experiment_id": "both"},
        {"project_id": " "},
        {"max_traces": 0},
    ],
)
def test_invalid_selection(overrides):
    with pytest.raises(ValueError):
        config(**overrides)


@pytest.mark.parametrize(
    "start,end", [(START.replace(tzinfo=None), END), (START, START), (END, START)]
)
def test_invalid_time_window(start, end):
    with pytest.raises(ValueError):
        BraintrustTraceConfig(start, end, project_id="project")


def test_config_cli_and_source_exclusivity():
    from compass.cli.main import _configured_trace_loader

    parsed = RunConfig(
        _cli_parse_args=[
            "--trace.braintrust.project-id",
            "project",
            "--trace.braintrust.from-timestamp",
            START.isoformat(),
            "--trace.braintrust.to-timestamp",
            END.isoformat(),
            "--trace.max-traces",
            "3",
        ]
    )
    loader = _configured_trace_loader(parsed.trace)
    assert isinstance(loader, BraintrustTraceLoader)
    assert loader.config.max_traces == 3
    assert parsed.trace.braintrust is not None
    with pytest.raises(ValidationError, match="exactly one loader"):
        TraceConfig.model_validate(
            {"braintrust": parsed.trace.braintrust.model_dump(), "filesystem": {"path": "a.jsonl"}}
        )


def test_cli_runs_compass_signals(rows, tmp_path, monkeypatch, select_signals):
    import compass.cli.main as cli

    compilation = SimpleNamespace(compile_insights=AsyncMock(return_value=[]))
    monkeypatch.setenv("COMPASS_API_KEY", "test-key")
    monkeypatch.setenv("BRAINTRUST_API_KEY", "test-key")
    monkeypatch.setattr(cli, "_build_llm", lambda *a: FakeLLMClient())
    monkeypatch.setattr(cli, "InsightCompilation", lambda llm: compilation)
    with client_for(rows) as client:
        monkeypatch.setattr(
            cli, "BraintrustTraceLoader", lambda cfg: BraintrustTraceLoader(cfg, client)
        )
        assert (
            cli.main(
                [
                    "--trace.braintrust.project-id",
                    "project",
                    "--trace.braintrust.from-timestamp",
                    START.isoformat(),
                    "--trace.braintrust.to-timestamp",
                    END.isoformat(),
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
    assert {e.signal_name for e in evidence} == {"tool-issues", "anomaly-and-patterns"}
    findings = next(e for e in evidence if e.signal_name == "tool-issues").artifacts.findings
    assert any(
        f["trace_id"] == "root"
        and f["tool_name"] == "lookup"
        and f["issue_type"] == "explicit_tool_failure"
        for f in findings
    )
    assert compilation.compile_insights.await_count == 1


def test_transport_errors_are_source_specific():
    def fail(request):
        raise httpx.ConnectError("private connection details", request=request)

    with httpx.Client(transport=httpx.MockTransport(fail)) as client:
        with pytest.raises(BraintrustTraceLoadError, match="Braintrust") as error:
            BraintrustTraceLoader(config(), client).load()
    assert "private connection details" not in str(error.value)


@pytest.mark.parametrize("experiment", [False, True])
def test_source_link_uses_root_event_id_and_app_settings(rows, monkeypatch, experiment):
    from urllib.parse import parse_qs, urlsplit

    monkeypatch.setenv("BRAINTRUST_ORG_NAME", "my org/name")
    monkeypatch.setenv("BRAINTRUST_APP_URL", "https://wrong.test")
    settings = {"project_id": None, "experiment_id": "experiment"} if experiment else {}
    with client_for(rows) as client:
        snapshot = BraintrustTraceLoader(
            config(app_url="https://ui.test/base", **settings), client
        ).load()
    source_url = next(iter(snapshot)).source_url
    assert source_url is not None
    url = urlsplit(source_url)
    assert url.netloc == "ui.test"
    assert url.path == "/base/app/my%20org%2Fname/object"
    assert parse_qs(url.query) == {
        "object_type": ["experiment" if experiment else "project_logs"],
        "object_id": ["experiment" if experiment else "project"],
        "id": ["row-root"],
    }


def test_nested_span_links_use_each_row_id(rows):
    from urllib.parse import parse_qs, urlsplit

    with client_for(rows) as client:
        trace = next(iter(BraintrustTraceLoader(config(org_name="org"), client).load()))
    pending = list(trace.root_spans)
    seen = {}
    while pending:
        span = pending.pop()
        pending.extend(span.children)
        assert span.source_url is not None
        seen[span.id] = parse_qs(urlsplit(span.source_url).query)["id"][0]
    assert seen == {row["span_id"]: row["id"] for row in rows}
    assert trace.source_url == trace.root_spans[0].source_url


def test_tool_results_match_result_counts(rows):
    with client_for(rows) as client:
        trace = next(iter(BraintrustTraceLoader(config(), client=client).load()))
    first, second = trace.root_spans[0].children[0].children
    assert first.tool_call is not None and second.tool_call is not None
    assert first.tool_call.results is not UNSET
    assert len(first.tool_call.results) == first.tool_call.result_count == 1
    assert first.tool_call.results[0].is_error is (True if first.error else None)
    assert second.tool_call.results == [] and second.tool_call.result_count == 0
