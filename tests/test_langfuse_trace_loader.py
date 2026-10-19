# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import builtins
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langfuse import Langfuse
from langfuse.api.commons.types.observation_level import ObservationLevel
from langfuse.api.commons.types.observations_view import ObservationsView
from langfuse.api.commons.types.trace_with_full_details import TraceWithFullDetails
from langfuse.api.commons.types.usage import Usage
from langfuse.api.core.pydantic_utilities import parse_obj_as

from compass.compass_signals.tool_issues.signal import (
    detect_trace,
    to_tool_issue_trace,
)
from compass.config import LangfuseConfig
from compass.trace_loaders.langfuse import (
    LangfuseFileTraceConfig,
    LangfuseTraceConfig,
    LangfuseTraceLoadError,
    LangfuseV3FileTraceLoader,
    LangfuseV3TraceLoader,
)
from compass.traces import UNSET, SpanKind

START = datetime(2026, 8, 1, 12, tzinfo=timezone.utc)
END = START + timedelta(hours=1)


@pytest.mark.parametrize("envelope", [False, True])
def test_native_export_matches_live_normalization(tmp_path, monkeypatch, envelope):
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        Langfuse, "__init__", lambda *args, **kwargs: pytest.fail("offline constructed a client")
    )
    native = provider_trace(
        observations=[
            observation("tool", parent_id="root", observation_type="TOOL", output_value="null"),
            observation("root", observation_type="AGENT"),
        ]
    )
    record = json.loads(native.json(exclude_unset=True))
    exported = {"status": 200, "headers": {}, "body": record} if envelope else record
    export_path = tmp_path / "trace.json"
    export_path.write_text(json.dumps(exported))
    loader = LangfuseV3FileTraceLoader(LangfuseFileTraceConfig(export_path))
    offline = list(loader.load())
    live = list(
        LangfuseV3TraceLoader(
            LangfuseTraceConfig(START, END, base_url="https://langfuse.test"),
            client=FakeClient([([native], 1)], {native.id: native}),
        ).load()
    )

    # Provenance intentionally distinguishes a frozen file from a live query.
    def without_pointers(value):
        if isinstance(value, dict):
            return {
                k: without_pointers(v)
                for k, v in value.items()
                if k not in {"source_pointer", "source_url"}
            }
        if isinstance(value, list):
            return [without_pointers(v) for v in value]
        return value

    assert without_pointers(offline[0].model_dump()) == without_pointers(live[0].model_dump())
    assert offline[0].source_url == export_path.as_uri()
    assert live[0].source_url == "https://langfuse.test" + native.html_path
    assert offline[0].root_spans[0].children[0].output is None
    assert offline[0].root_spans[0].output is UNSET
    assert loader.describe()["call_count"] == 1
    assert loader.describe()["export_trace_count"] == 1
    pointer = offline[0].attributes["source_pointer"]
    assert isinstance(pointer, dict)
    assert pointer["export_path"] == str(export_path)


def test_export_directory_limits_and_ordering(tmp_path):
    (tmp_path / "b.json").write_text(
        provider_trace("old", timestamp=START).json(exclude_unset=True)
    )
    (tmp_path / "a.jsonl").write_text(
        "\n".join(provider_trace(id, timestamp=END).json(exclude_unset=True) for id in ("z", "a"))
    )
    loader = LangfuseV3FileTraceLoader(LangfuseFileTraceConfig(tmp_path, max_traces=2))
    assert [t.id for t in loader.load()] == ["a", "z"]
    assert loader.describe()["truncated"] is True
    assert loader.describe()["export_trace_count"] == 3
    assert [t.id for t in loader.load()] == ["a", "z"]


@pytest.mark.parametrize(
    "case,match",
    [
        ("duplicate_trace", "duplicate trace"),
        ("duplicate_observation", "duplicate observation"),
        ("missing_parent", "missing parent"),
        ("wrong_trace", "traceId"),
        ("cycle", "parent cycle"),
        ("summary", "complete v3"),
        ("list_response", "complete v3"),
        ("failed_response", "unsuccessful CLI"),
        ("missing_field", "validation error"),
    ],
)
def test_export_rejects_invalid_records_before_limit(tmp_path, case, match):
    obs = observation("child")
    if case == "missing_parent":
        obs = observation("child", parent_id="missing")
    if case == "wrong_trace":
        obs = observation("child", trace_id="other")
    if case == "cycle":
        obs = observation("child", parent_id="child")
    native = provider_trace(observations=[obs, obs] if case == "duplicate_observation" else [obs])
    record = json.loads(native.json(exclude_unset=True))
    if case == "summary":
        record["observations"] = ["child"]
    if case == "list_response":
        record = {"data": [record], "meta": {"totalPages": 2}}
    if case == "failed_response":
        record = {"status": 401, "body": record}
    if case == "missing_field":
        del record["timestamp"]
    newest_id = "trace-1" if case == "duplicate_trace" else "newest"
    export_path = tmp_path / "traces.jsonl"
    export_path.write_text(
        provider_trace(newest_id, timestamp=END).json(exclude_unset=True)
        + "\n"
        + json.dumps(record)
    )
    with pytest.raises(LangfuseTraceLoadError, match=match):
        LangfuseV3FileTraceLoader(LangfuseFileTraceConfig(export_path, max_traces=1)).load()


@pytest.mark.parametrize(
    "filename,contents",
    [("a.json", b"{"), ("a.jsonl", b"\n"), ("a.json", b"\xff"), ("a.json", b"[]")],
)
def test_export_rejects_bad_files(tmp_path, filename, contents):
    export_path = tmp_path / filename
    export_path.write_bytes(contents)
    with pytest.raises(LangfuseTraceLoadError):
        LangfuseV3FileTraceLoader(LangfuseFileTraceConfig(export_path)).load()


def test_export_empty_directory_and_invalid_limit(tmp_path):
    with pytest.raises(LangfuseTraceLoadError, match="no JSON"):
        LangfuseV3FileTraceLoader(LangfuseFileTraceConfig(tmp_path)).load()
    with pytest.raises(ValueError, match="at least 1"):
        LangfuseFileTraceConfig(Path("unused"), max_traces=0)


def test_export_cli_runs_evidence_without_langfuse_client(tmp_path, monkeypatch, select_signals):
    from unittest.mock import AsyncMock

    from nooa.unifiedllm import FakeLLMClient

    import compass.cli.main as cli

    native = provider_trace(
        observations=[
            observation(
                "tool",
                name="lookup",
                observation_type="TOOL",
                level="ERROR",
                input_value={"query": "missing"},
                output_value={"error": "not found"},
            )
        ]
    )
    export_path = tmp_path / "trace.json"
    export_path.write_text(native.json(exclude_unset=True))
    compilation = SimpleNamespace(compile_insights=AsyncMock(return_value=[]))
    monkeypatch.setenv("COMPASS_API_KEY", "test-key")
    monkeypatch.setattr(Langfuse, "__init__", lambda *a, **kw: pytest.fail("offline client"))
    monkeypatch.setattr(cli, "_build_llm", lambda *a: FakeLLMClient())
    monkeypatch.setattr(cli, "InsightCompilation", lambda llm: compilation)
    assert (
        cli.main(
            [
                "--trace.langfuse-trace-export-file.api-version",
                "v3",
                "--trace.langfuse-trace-export-file.path",
                str(export_path),
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
    findings = tool_evidence.artifacts.findings
    assert len(findings) == 1
    assert findings[0]["trace_id"] == native.id
    assert findings[0]["tool_name"] == "lookup"
    assert findings[0]["issue_type"] == "explicit_tool_failure"


@pytest.mark.parametrize("offline", [False, True])
def test_corpus_reports_count_only_selected_traces(tmp_path, offline):
    newest = provider_trace(
        "newest",
        timestamp=END,
        session_id=None,
        observations=[observation("tool", trace_id="newest", observation_type="TOOL")],
    )
    middle = provider_trace("middle", timestamp=START, session_id="shared")
    oldest = provider_trace(
        "oldest",
        timestamp=START - timedelta(seconds=1),
        session_id="excluded",
        observations=[observation("span", trace_id="oldest")],
    )
    if offline:
        export_path = tmp_path / "traces.jsonl"
        export_path.write_text(
            "\n".join(t.json(exclude_unset=True) for t in (newest, oldest, middle))
        )
        loader = LangfuseV3FileTraceLoader(LangfuseFileTraceConfig(export_path, max_traces=2))
    else:
        loader = LangfuseV3TraceLoader(
            LangfuseTraceConfig(
                START, END + timedelta(seconds=1), base_url="https://example.test", max_traces=2
            ),
            client=FakeClient([([newest, middle], 1)], {t.id: t for t in (newest, middle)}),
        )
    assert [t.id for t in loader.load()] == ["middle", "newest"]
    description = loader.describe()
    expected = {
        "trace_count": 2,
        "call_count": 1,
        "observation_count": 1,
        "distinct_logical_cases": 2,
        "unresolved_parent_count": 0,
    }
    assert {key: value for key, value in description.items() if key in expected} == expected


def test_live_order_uses_query_summary_timestamps():
    summaries = [provider_trace("b", timestamp=END), provider_trace("a", timestamp=START)]
    details = {"a": provider_trace("a", timestamp=END), "b": provider_trace("b", timestamp=START)}
    loader = LangfuseV3TraceLoader(
        LangfuseTraceConfig(START, END + timedelta(seconds=1), base_url="https://example.test"),
        client=FakeClient([(summaries, 1)], details),
    )
    assert [t.id for t in loader.load()] == ["a", "b"]


@pytest.mark.parametrize("offline", [False, True])
def test_scores_reach_evaluation_index_without_losing_repeated_names(tmp_path, offline):
    from compass.compass_signals.grader_failure_patterns import _trace_index

    scores = [
        {"id": "numeric", "name": "quality", "dataType": "NUMERIC", "value": 0.25},
        {
            "id": "boolean",
            "name": "quality",
            "dataType": "BOOLEAN",
            "value": 0.0,
            "stringValue": "False",
            "observationId": "tool",
        },
        {
            "id": "category",
            "name": "category",
            "dataType": "CATEGORICAL",
            "value": 1.0,
            "stringValue": "incorrect",
        },
        {"id": "text", "name": "review", "dataType": "TEXT", "stringValue": "Needs work"},
    ]
    records = [
        {
            "traceId": "trace-1",
            "source": "API",
            "timestamp": START.isoformat(),
            "createdAt": START.isoformat(),
            "updatedAt": END.isoformat(),
            "environment": "test",
            "metadata": {"evaluator": "fixture"},
            "comment": "Review",
            **score,
        }
        for score in scores
    ]
    payload = provider_trace(observations=[observation("tool", observation_type="TOOL")]).dict()
    payload["scores"] = records
    native = parse_obj_as(TraceWithFullDetails, payload)
    if offline:
        export_path = tmp_path / "trace.json"
        export_path.write_text(native.json())
        loader = LangfuseV3FileTraceLoader(LangfuseFileTraceConfig(export_path))
    else:
        loader = LangfuseV3TraceLoader(
            LangfuseTraceConfig(START, END, base_url="https://example.test"),
            client=FakeClient([([native], 1)], {native.id: native}),
        )
    snapshot = loader.load()
    evaluations = next(iter(snapshot)).evaluator_results
    expected = {}
    for score in sorted(records, key=lambda score: (score["name"], score["id"])):
        expected.setdefault(score["name"], []).append(
            {
                **score,
                "timestamp": START.isoformat().replace("+00:00", "Z"),
                "createdAt": START.isoformat().replace("+00:00", "Z"),
                "updatedAt": END.isoformat().replace("+00:00", "Z"),
            }
        )
    assert evaluations == expected
    assert _trace_index(snapshot)[0]["evaluator_results"] == expected


def test_export_config_is_exclusive():
    from pydantic import ValidationError

    from compass.config import TraceConfig

    with pytest.raises(ValidationError, match="exactly one loader"):
        TraceConfig.model_validate(
            {
                "langfuse_trace_export_file": {"api_version": "v3", "path": "a.json"},
                "filesystem": {"path": "b.jsonl"},
            }
        )


class FakeTraceAPI:
    def __init__(
        self,
        pages: builtins.list[tuple[builtins.list[TraceWithFullDetails], int]],
        details: dict[str, TraceWithFullDetails],
    ) -> None:
        self.pages = list(pages)
        self.details = details
        self.list_calls: list[dict[str, object]] = []
        self.get_calls: list[str] = []

    def list(self, **kwargs: object) -> SimpleNamespace:
        self.list_calls.append(kwargs)
        data, total_pages = self.pages.pop(0)
        return SimpleNamespace(
            data=data,
            meta=SimpleNamespace(page=kwargs["page"], total_pages=total_pages),
        )

    def get(self, trace_id: str) -> TraceWithFullDetails:
        self.get_calls.append(trace_id)
        return self.details[trace_id]


class FakeClient(Langfuse):
    def __init__(
        self,
        pages: list[tuple[list[TraceWithFullDetails], int]],
        details: dict[str, TraceWithFullDetails],
    ) -> None:
        self.trace = FakeTraceAPI(pages, details)
        self._fake_api = SimpleNamespace(trace=self.trace, health=self)

    @property
    def api(self) -> Any:
        return self._fake_api

    def health(self):
        return SimpleNamespace(status="OK", version="3.155.1")


def observation(
    observation_id: str,
    *,
    trace_id: str = "trace-1",
    parent_id: str | None = None,
    observation_type: str = "SPAN",
    start_time: datetime = START,
    end_time: datetime | None = None,
    name: str | None = None,
    level: str = "DEFAULT",
    status_message: str | None = None,
    input_value: object = None,
    output_value: object = None,
    total_cost: float | None = None,
    model: str | None = None,
) -> ObservationsView:
    return ObservationsView(
        id=observation_id,
        trace_id=trace_id,
        type=observation_type,
        name=name,
        start_time=start_time,
        end_time=end_time,
        completion_start_time=None,
        model=model,
        model_parameters={"temperature": 0},
        input=input_value,
        version="v2",
        metadata={"source": "test"},
        output=output_value,
        usage=Usage(
            input=10,
            output=4,
            total=14,
            total_cost=total_cost,
        ),
        level=ObservationLevel(level),
        status_message=status_message,
        parent_observation_id=parent_id,
        prompt_id="prompt-1",
        usage_details={"input": 10, "output": 4},
        cost_details={"total": total_cost} if total_cost is not None else {},
        environment="test",
        prompt_name="support",
        prompt_version=3,
        model_id=None,
        input_price=None,
        output_price=None,
        total_price=None,
        calculated_input_cost=None,
        calculated_output_cost=None,
        calculated_total_cost=total_cost,
        latency=None,
        time_to_first_token=None,
    )


def provider_trace(
    trace_id: str = "trace-1",
    *,
    timestamp: datetime = START,
    observations: list[ObservationsView] | None = None,
    session_id: str | None = "session-7",
    latency: float = 5.0,
    total_cost: float = 0.03,
) -> TraceWithFullDetails:
    return TraceWithFullDetails(
        id=trace_id,
        timestamp=timestamp,
        name="support-agent",
        input={"messages": [{"role": "user", "content": "find it"}]},
        output={"answer": "done"},
        session_id=session_id,
        release="2026.08",
        version="v2",
        user_id="user-1",
        metadata={"source": "fixture"},
        tags=["test"],
        public=False,
        environment="test",
        html_path=f"/project/project-1/traces/{trace_id}",
        latency=latency,
        total_cost=total_cost,
        observations=observations or [],
        scores=[],
    )


def test_loader_selects_then_fetches_complete_v3_traces() -> None:
    root = observation(
        "root",
        observation_type="AGENT",
        end_time=START + timedelta(seconds=5),
        input_value={"messages": [{"role": "user", "content": "find it"}]},
        output_value={"answer": "done"},
    )
    generation = observation(
        "llm",
        parent_id="root",
        observation_type="GENERATION",
        start_time=START + timedelta(seconds=1),
        end_time=START + timedelta(seconds=2),
        level="ERROR",
        status_message="rate_limit",
        input_value=["one", "two"],
        output_value="null",
        total_cost=0.03,
        model="gpt-test",
    )
    tool = observation(
        "tool",
        parent_id="root",
        observation_type="TOOL",
        start_time=START + timedelta(seconds=2),
        end_time=START + timedelta(seconds=4),
        name="search",
        input_value={"query": "x"},
    )
    full_trace = provider_trace(observations=[tool, generation, root])
    client = FakeClient(pages=[([full_trace], 1)], details={"trace-1": full_trace})

    loader = LangfuseV3TraceLoader(
        LangfuseTraceConfig(
            from_timestamp=START - timedelta(days=1),
            to_timestamp=END,
            base_url="https://langfuse.example.com",
            max_traces=2,
        ),
        client=client,
    )
    normalized = next(iter(loader.load()))

    assert client.trace.list_calls == [
        {
            "page": 1,
            "limit": 2,
            "from_timestamp": START - timedelta(days=1),
            "to_timestamp": END,
            "order_by": "timestamp.desc",
            "fields": "core",
            "filter": None,
        }
    ]
    assert client.trace.get_calls == ["trace-1"]
    assert normalized.id == "trace-1"
    assert normalized.attributes["logical_case_id"] == "session-7"
    assert normalized.aggregate.cost_usd == 0.03
    assert normalized.aggregate.latency_ms == 5_000
    assert [span.id for span in normalized.root_spans] == ["root"]
    assert [span.id for span in normalized.root_spans[0].children] == ["llm", "tool"]

    root_span = normalized.root_spans[0]
    llm_span, tool_span = root_span.children
    assert root_span.kind is SpanKind.AGENT
    assert llm_span.kind is SpanKind.LLM
    assert llm_span.input == ["one", "two"]
    assert llm_span.output is None
    assert llm_span.error == "rate_limit"
    assert llm_span.model == "gpt-test"
    assert tool_span.kind is SpanKind.TOOL
    assert tool_span.tool_name == "search"
    assert tool_span.output is UNSET
    assert tool_span.tool_call is not None
    assert tool_span.tool_call.index == 0
    assert tool_span.tool_call.result_count == 0
    assert tool_span.attributes["source_pointer"] == {
        "provider": "langfuse",
        "base_url": "https://langfuse.example.com",
        "trace_id": "trace-1",
        "observation_id": "tool",
    }


@pytest.mark.parametrize(
    ("observation_type", "definition"),
    [
        (
            "AGENT",
            {
                "name": "search",
                "description": "Search the corpus.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        ),
        (
            "GENERATION",
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the corpus.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            },
        ),
    ],
)
def test_loader_extracts_tool_catalog_from_supported_langfuse_input(
    observation_type: str, definition: dict[str, object]
) -> None:
    catalog_input = {"messages": [{"role": "user", "content": "find it"}], "tools": [definition]}
    root = observation(
        "root",
        observation_type="AGENT",
        input_value=catalog_input if observation_type == "AGENT" else {"task": "find it"},
    )
    observations = [root]
    if observation_type == "GENERATION":
        observations.append(
            observation(
                "generation",
                parent_id="root",
                observation_type="GENERATION",
                input_value=catalog_input,
            )
        )
    full_trace = provider_trace(observations=observations)
    client = FakeClient(pages=[([full_trace], 1)], details={"trace-1": full_trace})

    normalized = next(
        iter(
            LangfuseV3TraceLoader(
                LangfuseTraceConfig(START, END, base_url="https://langfuse.example.com"),
                client=client,
            ).load()
        )
    )

    assert normalized.attributes["tool_catalog"] == {
        "search": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
    }
    catalog_span = (
        normalized.root_spans[0]
        if observation_type == "AGENT"
        else normalized.root_spans[0].children[0]
    )
    assert catalog_span.input == catalog_input


@pytest.mark.parametrize(
    "generation_inputs,expected",
    [
        pytest.param(
            [{"tools": [{"name": "model_tool", "parameters": {"type": "object"}}]}],
            {"model_tool": {"type": "object"}},
            id="prefer-generation",
        ),
        pytest.param(
            [{"messages": [{"role": "user", "content": "find it"}]}],
            None,
            id="missing-generation-catalog",
        ),
        pytest.param(
            [
                {"tools": [{"name": "root_only", "parameters": {"type": "object"}}]},
                {"tools": [{"name": "write", "parameters": {"type": "object"}}]},
            ],
            None,
            id="conflicting-generation-catalogs",
        ),
    ],
)
def test_generation_catalogs_override_root_fallback(generation_inputs, expected) -> None:
    root = observation(
        "root",
        observation_type="AGENT",
        input_value={"tools": [{"name": "root_only", "parameters": {"type": "object"}}]},
    )
    generations = [
        observation(
            f"generation-{index}",
            parent_id="root",
            observation_type="GENERATION",
            start_time=START + timedelta(seconds=index + 1),
            input_value=value,
        )
        for index, value in enumerate(generation_inputs)
    ]
    full_trace = provider_trace(observations=[*reversed(generations), root])
    client = FakeClient(pages=[([full_trace], 1)], details={"trace-1": full_trace})

    normalized = next(
        iter(
            LangfuseV3TraceLoader(
                LangfuseTraceConfig(START, END, base_url="https://langfuse.example.com"),
                client=client,
            ).load()
        )
    )

    if expected is None:
        assert "tool_catalog" not in normalized.attributes
    else:
        assert normalized.attributes["tool_catalog"] == expected


@pytest.mark.parametrize(
    "input_value,catalog_kind,expected",
    [
        pytest.param(None, "empty", [], id="missing-zero-argument-input"),
        pytest.param(None, "required", [], id="missing-required-input"),
        pytest.param(None, "absent", [], id="missing-input-without-catalog"),
        pytest.param(None, "unknown", ["unknown_tool"], id="unknown-tool-with-missing-input"),
        pytest.param("null", "empty", ["malformed_tool_call"], id="explicit-null-input"),
        pytest.param("null", "absent", ["malformed_tool_call"], id="null-without-catalog"),
        pytest.param("{}", "empty", [], id="recorded-empty-arguments"),
        pytest.param("{}", "required", ["missing_required_argument"], id="recorded-missing-key"),
    ],
)
def test_export_argument_analysis_distinguishes_missing_input(
    tmp_path, input_value, catalog_kind, expected
):
    schema = {"type": "object", "properties": {}}
    if catalog_kind == "required":
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
    generation_input = (
        {}
        if catalog_kind == "absent"
        else {"tools": [{"type": "function", "function": {"name": "search", "parameters": schema}}]}
    )
    native = provider_trace(
        observations=[
            observation("generation", observation_type="GENERATION", input_value=generation_input),
            observation(
                "tool",
                parent_id="generation",
                observation_type="TOOL",
                name="unknown" if catalog_kind == "unknown" else "search",
                input_value=input_value,
                output_value={"ok": True},
            ),
        ]
    )
    export_path = tmp_path / "trace.json"
    export_path.write_text(native.json(exclude_unset=True))
    trace = next(iter(LangfuseV3FileTraceLoader(LangfuseFileTraceConfig(export_path)).load()))
    call = to_tool_issue_trace(trace)

    if input_value is None:
        assert call.calls[0].arguments is UNSET
    elif input_value == "null":
        assert call.calls[0].arguments is None
    else:
        assert call.calls[0].arguments == {}
    assert [finding["issue_type"] for finding in detect_trace(call)] == expected


def test_loader_catalog_enables_unknown_tool_and_argument_schema_findings() -> None:
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["query"],
        "additionalProperties": False,
    }
    root = observation(
        "root",
        observation_type="AGENT",
        input_value={"task": "find it", "tools": [{"name": "search", "parameters": schema}]},
    )
    invalid = observation(
        "invalid",
        parent_id="root",
        observation_type="TOOL",
        start_time=START + timedelta(seconds=1),
        name="search",
        input_value={"limit": "many"},
        output_value={"error": "invalid arguments"},
    )
    unknown = observation(
        "unknown",
        parent_id="root",
        observation_type="TOOL",
        start_time=START + timedelta(seconds=2),
        name="invented_search",
        input_value={"query": "x"},
        output_value={"error": "tool not found"},
    )
    full_trace = provider_trace(observations=[unknown, invalid, root])
    client = FakeClient(pages=[([full_trace], 1)], details={"trace-1": full_trace})
    normalized = next(
        iter(
            LangfuseV3TraceLoader(
                LangfuseTraceConfig(START, END, base_url="https://langfuse.example.com"),
                client=client,
            ).load()
        )
    )

    findings = detect_trace(to_tool_issue_trace(normalized))

    assert {finding["issue_type"] for finding in findings} >= {
        "unknown_tool",
        "missing_required_argument",
        "argument_type_mismatch",
    }


def test_loader_pages_with_a_stable_page_size_and_returns_oldest_first() -> None:
    newest = provider_trace("trace-2", timestamp=START)
    oldest = provider_trace("trace-1", timestamp=START - timedelta(minutes=1))
    client = FakeClient(
        pages=[([newest], 2), ([oldest], 2)],
        details={"trace-1": oldest, "trace-2": newest},
    )

    snapshot = LangfuseV3TraceLoader(
        LangfuseTraceConfig(
            from_timestamp=START - timedelta(days=1),
            to_timestamp=END,
            base_url="https://langfuse.example.com",
            max_traces=150,
        ),
        client=client,
    ).load()

    assert [trace.id for trace in snapshot] == ["trace-1", "trace-2"]
    assert [call["page"] for call in client.trace.list_calls] == [1, 2]
    assert [call["limit"] for call in client.trace.list_calls] == [100, 100]


def test_advanced_filter_is_combined_with_required_time_bounds() -> None:
    full_trace = provider_trace()
    user_filter = json.dumps(
        [{"type": "string", "column": "environment", "operator": "=", "value": "prod"}]
    )
    client = FakeClient(pages=[([full_trace], 1)], details={"trace-1": full_trace})

    LangfuseV3TraceLoader(
        LangfuseTraceConfig(
            from_timestamp=START,
            to_timestamp=END,
            base_url="https://langfuse.example.com",
            filter_string=user_filter,
        ),
        client=client,
    ).load()

    assert json.loads(str(client.trace.list_calls[0]["filter"])) == [
        {
            "type": "string",
            "column": "environment",
            "operator": "=",
            "value": "prod",
        },
        {
            "type": "datetime",
            "column": "timestamp",
            "operator": ">=",
            "value": START.isoformat(),
        },
        {
            "type": "datetime",
            "column": "timestamp",
            "operator": "<",
            "value": END.isoformat(),
        },
    ]


def test_loader_rejects_invalid_filters_before_calling_langfuse() -> None:
    client = FakeClient(pages=[], details={})
    loader = LangfuseV3TraceLoader(
        LangfuseTraceConfig(
            from_timestamp=START,
            to_timestamp=END,
            base_url="https://langfuse.example.com",
            filter_string='{"environment":"prod"}',
        ),
        client=client,
    )

    with pytest.raises(LangfuseTraceLoadError, match="JSON array of conditions"):
        loader.load()

    assert client.trace.list_calls == []


def test_loader_reports_missing_credentials_before_creating_a_client(monkeypatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    loader = LangfuseV3TraceLoader(
        LangfuseTraceConfig(
            from_timestamp=START,
            to_timestamp=END,
            base_url="https://langfuse.example.com",
        )
    )

    with pytest.raises(
        LangfuseTraceLoadError,
        match="LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY",
    ):
        loader.load()


def test_loader_requires_a_configured_or_environment_base_url(monkeypatch) -> None:
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    loader = LangfuseV3TraceLoader(
        LangfuseTraceConfig(
            from_timestamp=START,
            to_timestamp=END,
        ),
        client=FakeClient(pages=[], details={}),
    )

    with pytest.raises(
        LangfuseTraceLoadError,
        match="trace.langfuse.base_url or LANGFUSE_BASE_URL",
    ):
        loader.load()


def test_loader_reports_unresolved_parents_and_rejects_cycles() -> None:
    root = observation("root")
    orphan = observation("orphan", parent_id="outside", start_time=START + timedelta(seconds=1))
    unresolved_trace = provider_trace(observations=[root, orphan])
    client = FakeClient(pages=[([unresolved_trace], 1)], details={"trace-1": unresolved_trace})
    loader = LangfuseV3TraceLoader(
        LangfuseTraceConfig(
            from_timestamp=START,
            to_timestamp=END,
            base_url="https://langfuse.example.com",
        ),
        client=client,
    )

    normalized = next(iter(loader.load()))

    assert [span.id for span in normalized.root_spans] == ["root", "orphan"]
    assert loader.describe()["unresolved_parent_count"] == 1
    pointer = normalized.root_spans[1].attributes["source_pointer"]
    assert isinstance(pointer, dict)
    assert pointer["unresolved_parent_observation_id"] == "outside"

    first = observation("first", parent_id="second")
    second = observation("second", parent_id="first")
    cycle_trace = provider_trace(observations=[first, second])
    cycle_client = FakeClient(pages=[([cycle_trace], 1)], details={"trace-1": cycle_trace})
    cycle_loader = LangfuseV3TraceLoader(
        LangfuseTraceConfig(
            from_timestamp=START,
            to_timestamp=END,
            base_url="https://langfuse.example.com",
        ),
        client=cycle_client,
    )
    with pytest.raises(LangfuseTraceLoadError, match="parent cycle"):
        cycle_loader.load()


@pytest.mark.parametrize("config_type", [LangfuseTraceConfig, LangfuseConfig])
@pytest.mark.parametrize(
    ("from_timestamp", "to_timestamp", "message"),
    [
        (START.replace(tzinfo=None), END, "from_timestamp must include a timezone"),
        (START, END.replace(tzinfo=None), "to_timestamp must include a timezone"),
        (END, START, "to_timestamp must be after from_timestamp"),
        (START, START, "to_timestamp must be after from_timestamp"),
    ],
)
def test_config_requires_a_valid_bounded_time_window(
    config_type: type[LangfuseTraceConfig] | type[LangfuseConfig],
    from_timestamp: datetime,
    to_timestamp: datetime,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        config_type(
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
        )
