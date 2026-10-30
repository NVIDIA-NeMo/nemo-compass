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
from trace_ingest.catalog import tool_catalog
from trace_ingest.loaders.helper.llm_span import SpanData, nemo_relay
from trace_ingest.loaders.langfuse.normalization import _normalize_trace

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
    def native(trace_id, timestamp):
        step = observation("step", trace_id=trace_id, start_time=timestamp)
        return provider_trace(trace_id, timestamp=timestamp, observations=[step])

    (tmp_path / "b.json").write_text(native("old", START).json(exclude_unset=True))
    (tmp_path / "a.jsonl").write_text(
        "\n".join(native(id, END).json(exclude_unset=True) for id in ("z", "a"))
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
        observations=[
            observation("tool", trace_id="newest", observation_type="TOOL", start_time=END)
        ],
    )
    # Two records of one session count as one conversation toward max_traces.
    middle = provider_trace(
        "middle",
        timestamp=START,
        session_id="shared",
        observations=[observation("span", trace_id="middle", start_time=START)],
    )
    later = provider_trace(
        "later",
        timestamp=START + timedelta(seconds=1),
        session_id="shared",
        observations=[
            observation("span", trace_id="later", start_time=START + timedelta(seconds=1))
        ],
    )
    oldest = provider_trace(
        "oldest",
        timestamp=START - timedelta(seconds=1),
        session_id="excluded",
        observations=[
            observation("span", trace_id="oldest", start_time=START - timedelta(seconds=1))
        ],
    )
    if offline:
        export_path = tmp_path / "traces.jsonl"
        export_path.write_text(
            "\n".join(t.json(exclude_unset=True) for t in (newest, oldest, later, middle))
        )
        loader = LangfuseV3FileTraceLoader(LangfuseFileTraceConfig(export_path, max_traces=2))
    else:
        selected = (newest, later, middle)
        loader = LangfuseV3TraceLoader(
            LangfuseTraceConfig(
                START, END + timedelta(seconds=1), base_url="https://example.test", max_traces=2
            ),
            client=FakeClient([([*selected, oldest], 1)], {t.id: t for t in selected}),
        )
    traces = list(loader.load())
    assert [t.id for t in traces] == ["session:shared", "newest"]
    records = traces[0].attributes["records"]
    assert isinstance(records, list)
    assert [record["trace_id"] for record in records if isinstance(record, dict)] == [
        "middle",
        "later",
    ]
    assert [loss.code for trace in traces for loss in trace.losses] == (
        [] if offline else ["merge_incomplete"]
    )
    description = loader.describe()
    expected = {
        "trace_count": 2,
        "call_count": 1,
        "observation_count": 3,
        "distinct_cases": 2,
        "distinct_sessions": 1,
        "unresolved_parent_count": 0,
    }
    assert {key: value for key, value in description.items() if key in expected} == expected
    if isinstance(loader, LangfuseV3FileTraceLoader):
        file_description = loader.describe()
        assert file_description["export_trace_count"] == 4
        assert file_description["truncated"] is True


@pytest.mark.parametrize("offline", [False, True])
def test_export_keeps_the_conversations_live_selection_keeps(tmp_path, offline):
    # Live lists traces by timestamp; a trace without observations is still newest.
    newer = provider_trace("newer-no-obs", timestamp=START + timedelta(seconds=100))
    older = provider_trace(
        "older",
        timestamp=START,
        observations=[
            observation("span", trace_id="older", start_time=START + timedelta(seconds=200))
        ],
    )
    if offline:
        export_path = tmp_path / "traces.jsonl"
        export_path.write_text("\n".join(t.json(exclude_unset=True) for t in (older, newer)))
        loader = LangfuseV3FileTraceLoader(LangfuseFileTraceConfig(export_path, max_traces=1))
    else:
        loader = LangfuseV3TraceLoader(
            LangfuseTraceConfig(START, END, base_url="https://example.test", max_traces=1),
            client=FakeClient([([newer, older], 1)], {newer.id: newer}),
        )
    assert [t.id for t in loader.load()] == ["newer-no-obs"]


def load_live(*traces: TraceWithFullDetails) -> list:
    loader = LangfuseV3TraceLoader(
        LangfuseTraceConfig(START, END, base_url="https://langfuse.test"),
        client=FakeClient([(list(traces), 1)], {t.id: t for t in traces}),
    )
    return list(loader.load())


def conversation(trace_id, *, root_type, offset, messages, reply, session_id=None):
    start = START + timedelta(seconds=offset)
    return provider_trace(
        trace_id,
        timestamp=start,
        session_id=session_id,
        input_value={"messages": messages},
        output_value=reply,
        observations=[
            observation("root", trace_id=trace_id, observation_type=root_type, start_time=start),
            observation(
                "llm",
                trace_id=trace_id,
                parent_id="root",
                observation_type="GENERATION",
                start_time=start + timedelta(milliseconds=1),
                output_value=reply,
            ),
        ],
    )


def test_v3_traces_sharing_a_session_merge_into_one_conversation():
    first = conversation(
        "t1",
        root_type="AGENT",
        offset=0,
        messages=[{"role": "user", "content": "find it"}],
        reply="found",
        session_id="session-7",
    )
    second = conversation(
        "t2",
        root_type="AGENT",
        offset=1,
        messages=[
            {"role": "user", "content": "find it"},
            {"role": "assistant", "content": "found"},
            {"role": "user", "content": "thanks"},
        ],
        reply="welcome",
        session_id="session-7",
    )

    (trace,) = load_live(second, first)

    assert trace.id == "session:session-7"
    assert (trace.session_id, trace.session_basis) == ("session-7", "langfuse.sessionId")
    assert trace.losses == []
    (root,) = trace.root_spans
    assert [(span.id, span.kind, span.attributes["subtype"]) for span in root.children] == [
        ("t1/root", SpanKind.CHAIN, "AGENT"),
        ("t2/root", SpanKind.CHAIN, "AGENT"),
    ]
    assert [span.source_url for span in root.children] == [
        "https://langfuse.test/project/project-1/traces/t1",
        "https://langfuse.test/project/project-1/traces/t2",
    ]
    # The second input re-sends the first exchange; only its new message is kept.
    assert [(turn.role.value, turn.content, turn.llm_span_ids) for turn in root.turns] == [
        ("user", "find it", []),
        ("agent", "found", ["t1/llm"]),
        ("user", "thanks", []),
        ("agent", "welcome", ["t2/llm"]),
    ]


def test_v3_span_root_without_session_owns_its_turns():
    (trace,) = load_live(
        conversation(
            "t1",
            root_type="SPAN",
            offset=0,
            messages=[{"role": "user", "content": "find it"}],
            reply="found",
        )
    )

    assert trace.id == "t1" and trace.session_id is None
    (root,) = trace.root_spans
    assert (root.kind, root.attributes["subtype"]) == (SpanKind.AGENT, "SPAN")
    assert [(turn.id, turn.role.value, turn.content, turn.llm_span_ids) for turn in root.turns] == [
        ("root/turn/1", "user", "find it", []),
        ("root/turn/2", "agent", "found", ["llm"]),
    ]


def test_steps_in_one_millisecond_follow_their_causal_order():
    # Langfuse keeps milliseconds: a call, its tool and the answer can all start
    # and end in the same one. Ids sort them backwards.
    step = START + timedelta(milliseconds=1)

    def child(observation_id, observation_type, output, end=step):
        return observation(
            observation_id,
            trace_id="t1",
            parent_id="root",
            observation_type=observation_type,
            start_time=step,
            end_time=end,
            output_value=output,
        )

    call = {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function"}]}
    native = provider_trace(
        "t1",
        input_value={"messages": [{"role": "user", "content": "weather?"}]},
        output_value="rain",
        observations=[
            observation("root", trace_id="t1", observation_type="AGENT", end_time=END),
            child("a-answer", "GENERATION", {"role": "assistant", "content": "rain"}),
            child("b-tool", "TOOL", {"forecast": "rain"}),
            child("c-call", "GENERATION", json.dumps(call)),
            # Started in the same millisecond, but ended after the others.
            child("0-late", "GENERATION", {"role": "assistant", "content": "rain"}, END),
        ],
    )
    (trace,) = load_live(native)
    (root,) = trace.root_spans
    assert [span.id for span in root.children] == ["c-call", "b-tool", "a-answer", "0-late"]
    assert root.turns is not UNSET
    assert [(turn.llm_span_ids, turn.tool_call_span_ids) for turn in root.turns[1:3]] == [
        (["c-call"], ["b-tool"]),
        (["a-answer"], []),
    ]


def test_v3_generation_root_without_session_is_not_a_run():
    native = provider_trace(
        "t1",
        observations=[observation("llm", trace_id="t1", observation_type="GENERATION")],
    )

    (trace,) = load_live(native)

    (root,) = trace.root_spans
    assert root.kind is SpanKind.LLM
    assert root.turns is UNSET


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
    metadata: dict | None = None,
    usage_details: dict | None = None,
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
        metadata=metadata if metadata is not None else {"source": "test"},
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
        usage_details=usage_details if usage_details is not None else {"input": 10, "output": 4},
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
    session_id: str | None = None,
    latency: float = 5.0,
    total_cost: float = 0.03,
    input_value: object = None,
    output_value: object = None,
) -> TraceWithFullDetails:
    return TraceWithFullDetails(
        id=trace_id,
        timestamp=timestamp,
        name="support-agent",
        input=input_value or {"messages": [{"role": "user", "content": "find it"}]},
        output=output_value or {"answer": "done"},
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
            "limit": 100,
            "from_timestamp": START - timedelta(days=1),
            "to_timestamp": END,
            "order_by": "timestamp.desc",
            "fields": "core",
            "filter": None,
        }
    ]
    assert client.trace.get_calls == ["trace-1"]
    assert normalized.id == "trace-1"
    assert normalized.session_id is None
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

    assert tool_catalog(normalized) == {
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
    definitions = catalog_span.tool_definitions
    assert definitions is not UNSET
    assert definitions[0].raw == definition


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
            {"root_only": {"type": "object"}},
            id="agent-declaration-without-call-lists",
        ),
        pytest.param(
            [
                {"tools": [{"name": "root_only", "parameters": {"type": "object"}}]},
                {"tools": [{"name": "write", "parameters": {"type": "object"}}]},
            ],
            None,
            id="differing-generation-catalogs",
        ),
        pytest.param(
            [
                {"tools": [{"name": "write", "parameters": {"type": "object"}}]},
                {"tools": [{"function_declarations": [{"name": "write"}]}]},
            ],
            None,
            id="unparseable-generation-catalog",
        ),
        pytest.param(
            [
                {"tools": [{"name": "write", "parameters": {"type": "object"}}]},
                {"tools": [{"name": "write", "parameters": {"type": "string"}}]},
            ],
            None,
            id="conflicting-generation-schemas",
        ),
        pytest.param([{"tools": []}], {}, id="empty-generation-tools"),
    ],
)
def test_catalog_prefers_calls_without_discarding_agent_definitions(
    generation_inputs, expected
) -> None:
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

    assert "tool_catalog" not in normalized.attributes
    assert tool_catalog(normalized) == expected
    root_span = normalized.root_spans[0]
    assert root_span.tool_definitions is not UNSET
    assert [entry.name for entry in root_span.tool_definitions] == ["root_only"]
    for span, recorded in zip(root_span.children, generation_inputs, strict=True):
        if "tools" in recorded:
            assert span.tool_definitions is not UNSET
            assert [entry.raw for entry in span.tool_definitions] == recorded["tools"]
        else:
            assert span.tool_definitions is UNSET


def test_each_generation_carries_its_own_offered_tools() -> None:
    root = observation("root", observation_type="AGENT", input_value={"task": "find it"})
    first = {"name": "search", "description": "first", "parameters": {"type": "object"}}
    second = {"name": "search", "description": "second", "parameters": {"type": "object"}}
    generations = [
        observation(
            f"generation-{index}",
            parent_id="root",
            observation_type="GENERATION",
            start_time=START + timedelta(seconds=index + 1),
            input_value={"tools": [tool]},
        )
        for index, tool in enumerate([first, second])
    ]
    full_trace = provider_trace(observations=[root, *generations])
    client = FakeClient(pages=[([full_trace], 1)], details={"trace-1": full_trace})

    normalized = next(
        iter(
            LangfuseV3TraceLoader(
                LangfuseTraceConfig(START, END, base_url="https://langfuse.example.com"),
                client=client,
            ).load()
        )
    )

    root_span = normalized.root_spans[0]
    assert root_span.tool_definitions is UNSET
    offered = []
    for span in root_span.children:
        assert span.tool_definitions is not UNSET
        offered.append([definition.raw for definition in span.tool_definitions])
    assert offered == [[first], [second]]
    assert tool_catalog(normalized) == {"search": {"type": "object"}}


def test_empty_root_agent_tools_declare_an_empty_catalog() -> None:
    full_trace = provider_trace(
        observations=[observation("root", observation_type="AGENT", input_value={"tools": []})]
    )
    client = FakeClient(pages=[([full_trace], 1)], details={"trace-1": full_trace})

    normalized = next(
        iter(
            LangfuseV3TraceLoader(
                LangfuseTraceConfig(START, END, base_url="https://langfuse.example.com"),
                client=client,
            ).load()
        )
    )

    assert tool_catalog(normalized) == {}


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


def test_generation_root_carries_definitions_and_tool_results() -> None:
    generation = observation(
        "generation",
        observation_type="GENERATION",
        input_value={"tools": [{"name": "search", "parameters": {"type": "object"}}]},
    )
    tool = observation(
        "tool",
        parent_id="generation",
        observation_type="TOOL",
        start_time=START + timedelta(seconds=1),
        name="search",
        input_value={"query": "x"},
        output_value={"ok": True},
    )
    full_trace = provider_trace(observations=[tool, generation])
    client = FakeClient(pages=[([full_trace], 1)], details={"trace-1": full_trace})
    trace = next(
        iter(
            LangfuseV3TraceLoader(
                LangfuseTraceConfig(START, END, base_url="https://langfuse.example.com"),
                client=client,
            ).load()
        )
    )
    root = trace.root_spans[0]
    assert root.kind is SpanKind.LLM
    assert root.tool_definitions is not UNSET
    assert [definition.name for definition in root.tool_definitions] == ["search"]
    assert tool_catalog(trace) == {"search": {"type": "object"}}
    span = root.children[0]
    assert span.tool_call is not None
    assert span.tool_call.results is not UNSET
    assert len(span.tool_call.results) == span.tool_call.result_count == 1
    assert span.tool_call.results[0].is_error is (True if span.error else None)


def relay_metadata(**attributes: object) -> dict:
    return {
        "attributes": {
            key: value if isinstance(value, str) else json.dumps(value)
            for key, value in attributes.items()
        },
        "scope": {"name": "opentelemetry", "attributes": {}},
    }


def test_relay_spans_become_canonical_llm_calls() -> None:
    canary = "CANARY-relay-content"
    request = {
        "model": "aws/anthropic/bedrock-claude-sonnet-5-5",
        "max_tokens": 32000,
        "thinking": {"type": "enabled", "budget_tokens": 2048},
        "system": [
            {"type": "text", "text": canary, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
        ],
        "messages": [{"role": "user", "content": canary}],
        "tools": [{"name": "read", "description": canary, "input_schema": {"type": "object"}}],
        "stream": True,
    }
    usage = {
        "input_tokens": 4,
        "cache_read_input_tokens": 2170,
        "cache_creation_input_tokens": 119,
        "output_tokens": 48,
        "output_tokens_details": {"thinking_tokens": 0},
    }
    session = observation(
        "session",
        observation_type="AGENT",
        name="pi",
        end_time=START + timedelta(seconds=9),
        metadata=relay_metadata(
            **{
                "nemo_relay.scope_type": "agent",
                "nemo_relay.start.metadata.nemo_relay_scope_role": "session",
                "nemo_relay.start.metadata.agent_kind": "pi",
            }
        ),
    )
    # Relay's full export: Langfuse types the call a plain span.
    llm = observation(
        "llm",
        parent_id="session",
        observation_type="SPAN",
        name="anthropic.messages",
        start_time=START + timedelta(seconds=1),
        end_time=START + timedelta(seconds=2),
        metadata=relay_metadata(
            **{
                "nemo_relay.scope_type": "llm",
                "nemo_relay.start.data.content": request,
                "nemo_relay.start.data.headers": {"x-claude-code-agent-id": "a8af"},
                "nemo_relay.end.data.model": "aws/anthropic/bedrock-claude-sonnet-5-5",
                "nemo_relay.end.data.stop_reason": "tool_use",
                "nemo_relay.end.data.content": [
                    {"type": "thinking", "thinking": canary},
                    {"type": "text", "text": "ok"},
                    {"type": "tool_use", "input": {"path": "a"}},
                ],
                "nemo_relay.end.data.usage": usage,
            }
        ),
    )
    # Langfuse types count_tokens a generation too; it is kept as one, without usage.
    count_tokens = observation(
        "count",
        parent_id="session",
        observation_type="GENERATION",
        name="anthropic.count_tokens",
        start_time=START + timedelta(seconds=3),
        metadata=relay_metadata(**{"nemo_relay.scope_type": "llm"}),
    )
    # A turn span carries a copy of its last call's usage; it is not a call.
    turn = observation(
        "turn",
        parent_id="session",
        name="claude-code-turn",
        start_time=START + timedelta(seconds=4),
        metadata=relay_metadata(
            **{"nemo_relay.scope_type": "custom", "nemo_relay.end.data.usage": usage}
        ),
    )
    normalized = _normalize_trace(
        provider_trace(observations=[session, llm, count_tokens, turn], session_id="session-7"),
        source_pointer={"provider": "langfuse"},
    )

    assert (normalized.session_id, normalized.session_basis) == ("session-7", "langfuse.sessionId")
    (agent,) = normalized.root_spans
    assert agent.kind is SpanKind.AGENT
    # Relay records no agent ID on the session span, and none is made up.
    assert agent.agent_id is None
    call, counted, step = agent.children
    assert counted.kind is SpanKind.LLM
    assert counted.token_counts is None
    assert (step.kind, step.token_counts) == (SpanKind.UNKNOWN, None)
    assert call.kind is SpanKind.LLM
    assert call.attributes["subtype"] == "SPAN"
    assert call.token_counts is not None


def test_relay_agent_ids_in_both_layouts():
    def agent_id(**attributes: str) -> str | None:
        span = SpanData(None, "SPAN", None, {"nemo_relay.scope_type": "agent", **attributes}, {})
        return nemo_relay.agent_id(span)

    assert (
        agent_id(
            **{
                "nemo_relay.start.metadata.nemo_relay_scope_role": "subagent",
                "nemo_relay.start.metadata.agent_id": "a8af",
            }
        )
        == "a8af"
    )
    assert (
        agent_id(
            **{
                "openinference.metadata.nemo_relay_scope_role": "session",
                "openinference.metadata.agent_kind": "pi",
            }
        )
        is None
    )
    assert agent_id(**{"openinference.metadata.nemo_relay_scope_role": "subagent"}) is None
    # An LLM call carries the subagent Relay attributes it to, in either layout.
    assert agent_id(**{"nemo_relay.start.metadata.llm_correlation_subagent_id": "a8af"}) == "a8af"
    assert agent_id(**{"openinference.metadata.llm_correlation_subagent_id": "a8af"}) == "a8af"
    # Only spans Relay exported.
    assert (
        nemo_relay.agent_id(
            SpanData(None, "SPAN", None, {"openinference.metadata.agent_id": "a8af"}, {})
        )
        is None
    )


def test_openinference_spans_without_relay() -> None:
    llm = observation(
        "oi",
        observation_type="GENERATION",
        name="ChatOpenAI",
        start_time=START + timedelta(seconds=1),
        end_time=START + timedelta(seconds=2),
        model="gpt-5.5",
        metadata=relay_metadata(
            **{
                "openinference.span.kind": "LLM",
                "llm.model_name": "gpt-5.5",
                "llm.invocation_parameters": {"model": "gpt-5.5", "temperature": 0},
                "llm.tools.0.tool.json_schema": {"name": "Read"},
                "llm.token_count.prompt": "11409",
                "llm.token_count.prompt_details.cache_read": "3712",
                "llm.token_count.prompt_details.cache_write": {"intValue": 0},
                "llm.token_count.completion": "81",
            }
        ),
    )
    chain = observation(
        "chain",
        observation_type="CHAIN",
        start_time=START + timedelta(seconds=3),
        metadata=relay_metadata(**{"openinference.span.kind": "CHAIN"}),
    )
    normalized = _normalize_trace(
        provider_trace(observations=[llm, chain]),
        source_pointer={"provider": "langfuse"},
    )

    call, step = normalized.root_spans
    assert (call.kind, step.kind) == (SpanKind.LLM, SpanKind.CHAIN)
    assert call.model == "gpt-5.5"
    assert call.token_counts is not None
    # The spec counts cache inside llm.token_count.prompt.
    assert call.token_counts.model_dump(exclude_none=True, exclude={"source"}) == {
        "input_total": 11409,
        "cached_input_tokens": 3712,
        "output_tokens": 81,
        "cache_write_tokens": 0,
    }
    assert call.tool_definitions is not UNSET
    assert [definition.raw for definition in call.tool_definitions] == [{"name": "Read"}]
    assert call.llm_params == {"temperature": 0}


# One LLM observation per layout found in Langfuse, with the same keys and value
# shapes but made-up values. Relay's usage objects keep the provider's own keys,
# extras such as cost included.
ANTHROPIC_USAGE = {
    "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 300},
    "cache_creation_input_tokens": 300,
    "cache_read_input_tokens": 20000,
    "cost": 0.01,
    "input_tokens": 2,
    "output_tokens": 50,
    "output_tokens_details": {"thinking_tokens": 0},
    "service_tier": "standard",
}
ANTHROPIC_COUNTS = {
    "input_tokens": 2,
    "cached_input_tokens": 20000,
    "output_tokens": 50,
    "cache_write_tokens": 300,
    "cache_writes_by_ttl": {300: 300, 3600: 0},
    "source": "nemo_relay",
}
OPENAI_USAGE = {
    "cost": 0.01,
    "input_tokens": 12000,
    "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 11000},
    "output_tokens": 100,
    "output_tokens_details": {"reasoning_tokens": 30},
    "total_tokens": 12100,
}
OPENAI_COUNTS = {
    # input_total includes the cache read.
    "input_total": 12000,
    "cached_input_tokens": 11000,
    "output_tokens": 100,
    "cache_write_tokens": 0,
    "source": "nemo_relay",
}
OI_TOOLS = {
    "llm.tools.1.tool.json_schema": {"name": "Write"},
    "llm.tools.0.tool.json_schema": {"name": "Read"},
}

LAYOUTS = {
    # Relay full export: Langfuse types the call a plain span and records no model;
    # the request body holds the tools and parameters.
    "relay-full-anthropic": {
        "type": "SPAN",
        "name": "anthropic.messages",
        "metadata": relay_metadata(
            **{
                "nemo_relay.scope_type": "llm",
                "nemo_relay.start.data.content": {
                    "model": "claude-alias",
                    "max_tokens": 32000,
                    "thinking": {"type": "adaptive"},
                    "system": [{"type": "text", "text": "s"}],
                    "messages": [{"role": "user", "content": "m"}],
                    "tools": [{"name": "Read"}, {"name": "Write"}],
                    "stream": True,
                },
                "nemo_relay.end.data.model": "claude-response",
                "nemo_relay.end.data.usage": ANTHROPIC_USAGE,
            }
        ),
        "token_counts": ANTHROPIC_COUNTS,
        "model": "claude-response",
        "tools": ["Read", "Write"],
        "params": ["max_tokens", "stream", "thinking"],
    },
    # The model is the one the response names, which can differ from the request's.
    "relay-full-openai": {
        "type": "SPAN",
        "name": "openai.responses",
        "metadata": relay_metadata(
            **{
                "nemo_relay.scope_type": "llm",
                "nemo_relay.start.data.content": {
                    "model": "gpt-alias",
                    "instructions": "i",
                    "input": [{"role": "user", "content": "m"}],
                    "tools": [{"type": "function", "name": "exec_command"}],
                    "reasoning": {"effort": "medium"},
                    "tool_choice": "auto",
                    "prompt_cache_key": "k",
                    "stream": True,
                },
                "nemo_relay.end.data.model": "gpt-alias-global",
                "nemo_relay.end.data.usage": OPENAI_USAGE,
            }
        ),
        "token_counts": OPENAI_COUNTS,
        "model": "gpt-alias-global",
        "tools": ["exec_command"],
        "params": ["prompt_cache_key", "reasoning", "stream", "tool_choice"],
    },
    # Relay openinference export: OpenInference attributes beside Relay's usage.
    # The counts are Relay's, not llm.token_count.*, and Langfuse's own model wins.
    "relay-oi-anthropic": {
        "type": "GENERATION",
        "name": "anthropic.messages",
        "langfuse_model": "claude-alias",
        "metadata": relay_metadata(
            **{
                "nemo_relay.scope_type": "llm",
                "openinference.span.kind": "LLM",
                "llm.model_name": "claude-alias",
                "llm.invocation_parameters": {"model": "claude-alias", "max_tokens": 64000},
                **OI_TOOLS,
                "llm.token_count.prompt": "20302",
                "llm.token_count.prompt_details.cache_read": '{"intValue":20000}',
                "llm.token_count.prompt_details.cache_write": "300",
                "llm.token_count.completion": "50",
                "nemo_relay.end.data.model": "claude-response",
                "nemo_relay.end.data.usage": ANTHROPIC_USAGE,
            }
        ),
        "token_counts": ANTHROPIC_COUNTS,
        "model": "claude-alias",
        "tools": ["Read", "Write"],
        "params": ["max_tokens"],
    },
    # Codex's openinference export records tools but no invocation parameters.
    "relay-oi-openai-no-params": {
        "type": "GENERATION",
        "name": "openai.responses",
        "langfuse_model": "gpt-alias",
        "metadata": relay_metadata(
            **{
                "nemo_relay.scope_type": "llm",
                "openinference.span.kind": "LLM",
                "llm.model_name": "gpt-alias",
                "llm.tools.0.tool.json_schema": {"name": "exec_command"},
                "llm.token_count.prompt": "12000",
                "llm.token_count.prompt_details.cache_read": "11000",
                "llm.token_count.completion": "100",
                "nemo_relay.end.data.model": "gpt-alias-global",
                "nemo_relay.end.data.usage": OPENAI_USAGE,
            }
        ),
        "token_counts": OPENAI_COUNTS,
        "model": "gpt-alias",
        "tools": ["exec_command"],
        "params": None,
    },
    # A call without Relay's usage has no counts: llm.token_count.prompt excluded
    # cache in some Relay builds, against the spec, so it is not read.
    "relay-oi-no-usage": {
        "type": "GENERATION",
        "name": "anthropic.messages",
        "langfuse_model": "claude-alias",
        "metadata": relay_metadata(
            **{
                "nemo_relay.scope_type": "llm",
                "openinference.span.kind": "LLM",
                "llm.model_name": "claude-alias",
                "llm.invocation_parameters": {"model": "claude-alias", "max_tokens": 64000},
                **OI_TOOLS,
                "llm.token_count.prompt": "2",
                "llm.token_count.prompt_details.cache_read": "20000",
                "llm.token_count.completion": "50",
            }
        ),
        "token_counts": None,
        "model": "claude-alias",
        "tools": ["Read", "Write"],
        "params": ["max_tokens"],
    },
    # Claude Code's Langfuse integration: no OTel attributes, Anthropic's cache keys.
    "langfuse-sdk-claude": {
        "type": "GENERATION",
        "name": "LLM Call",
        "langfuse_model": "claude-alias",
        "metadata": {"scope": {"name": "langfuse-sdk"}},
        "usage_details": {
            "input": 2,
            "output": 50,
            "cache_read_input_tokens": 20000,
            "input_cache_creation_5m": 300,
            "total": 20352,
        },
        "token_counts": {
            "input_tokens": 2,
            "cached_input_tokens": 20000,
            "output_tokens": 50,
            "cache_writes_by_ttl": {300: 300},
            "source": "langfuse_sdk",
        },
        "model": "claude-alias",
        "tools": None,
        "params": None,
    },
    # Codex's Langfuse integration: Langfuse splits cache out of input and reasoning
    # out of output; the buckets add up to total.
    "langfuse-sdk-codex": {
        "type": "GENERATION",
        "name": "LLM",
        "langfuse_model": "gpt-alias",
        "metadata": {"scope": {"name": "langfuse-sdk"}},
        "usage_details": {
            "input": 1000,
            "output": 70,
            "total": 12100,
            "input_cached_tokens": 11000,
            "output_reasoning_tokens": 30,
        },
        "token_counts": {
            "input_tokens": 1000,
            "cached_input_tokens": 11000,
            "output_tokens": 100,
            "source": "langfuse_sdk",
        },
        "model": "gpt-alias",
        "tools": None,
        "params": None,
    },
}


@pytest.mark.parametrize("layout", list(LAYOUTS.values()), ids=list(LAYOUTS))
def test_each_layout_reaches_the_span(layout: dict) -> None:
    llm = observation(
        "llm",
        observation_type=layout["type"],
        name=layout["name"],
        model=layout.get("langfuse_model"),
        metadata=layout["metadata"],
        usage_details=layout.get("usage_details", {}),
    )
    normalized = _normalize_trace(
        provider_trace(observations=[llm]), source_pointer={"provider": "langfuse"}
    )

    (span,) = normalized.root_spans
    assert span.kind is SpanKind.LLM
    counts = span.token_counts
    assert (counts.model_dump(exclude_none=True) if counts else None) == layout["token_counts"]
    assert span.model == layout["model"]
    tools = span.tool_definitions
    assert ([tool.name for tool in tools] if tools is not UNSET else None) == layout["tools"]
    params = span.llm_params
    assert (sorted(params) if params is not None else None) == layout["params"]


@pytest.mark.parametrize("request_tools", [None, [], [{"name": "search", "input_schema": False}]])
def test_request_tools_take_precedence_over_platform_input(request_tools):
    request = {} if request_tools is None else {"tools": request_tools}
    platform_tools = [{"name": "platform", "parameters": {"type": "object"}}]
    llm = observation(
        "llm",
        observation_type="SPAN",
        name="anthropic.messages",
        input_value={"tools": platform_tools},
        metadata=relay_metadata(
            **{
                "nemo_relay.scope_type": "llm",
                "nemo_relay.start.data.content": json.dumps(request),
            }
        ),
    )
    normalized = _normalize_trace(provider_trace(observations=[llm]), source_pointer={})
    (span,) = normalized.root_spans
    assert span.kind is SpanKind.LLM
    expected = platform_tools if request_tools is None else request_tools
    assert span.tool_definitions is not UNSET
    assert [entry.raw for entry in span.tool_definitions] == expected
    assert tool_catalog(normalized) == (
        {"platform": {"type": "object"}}
        if request_tools is None
        else {"search": False}
        if request_tools
        else {}
    )
