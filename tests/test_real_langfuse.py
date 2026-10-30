# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the Langfuse loaders on data exported from real self-hosted Langfuse servers.

See tests/fixtures/real/langfuse/README.md for how the data was produced. The live
loaders are driven through a real ``langfuse.Langfuse`` client whose transport
replays the recorded HTTP responses byte for byte.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from itertools import count
from pathlib import Path
from urllib.parse import parse_qsl

import httpx
import pytest
from langfuse import Langfuse

from trace_ingest.loaders.langfuse import (
    LangfuseFileTraceConfig,
    LangfuseTraceConfig,
    LangfuseTraceLoader,
    LangfuseV3FileTraceLoader,
    LangfuseV4FileTraceConfig,
    LangfuseV4FileTraceLoader,
)
from trace_ingest.models import UNSET, Span, SpanKind, Trace, TraceSnapshot

FIXTURES = Path(__file__).parent / "fixtures" / "real" / "langfuse"
SESSION = "weather-session-001"
SESSION_TRACE = f"session:{SESSION}"
CLIENT_IDS = count()

GET_WEATHER_SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "day": {"type": "string", "enum": ["today", "tomorrow"]},
    },
    "required": ["city", "day"],
}
TODAY = {"city": "Paris", "day": "today", "condition": "light rain", "high_c": 14}
TOMORROW = {"city": "Paris", "day": "tomorrow", "condition": "sunny", "high_c": 17}


@dataclass(frozen=True)
class Record:
    """Observation ids of one weather-chat trace: agent -> generation, tool, generation."""

    trace: str
    agent: str
    call_llm: str
    tool: str
    answer_llm: str

    def span(self, observation: str) -> str:
        return f"{self.trace}/{observation}"


@dataclass(frozen=True)
class Run:
    """Ids of one server run, read from the exported records."""

    turn1: Record
    turn2: Record
    single_trace: str
    single_agent: str
    single_llm: str
    session_latency_ms: float
    single_latency_ms: float
    scores: dict[str, tuple[str, str]]  # name -> (score id, scored trace id)
    tool_score_observation: str


V3 = Run(
    turn1=Record(
        "640ff068b666c1e9cebbe8e99ac8a4c2",
        "f462aaf22cd53b82",
        "ae9d87cc681c9b47",
        "b87ff78e3f94c3f9",
        "75e874b31f699b0b",
    ),
    turn2=Record(
        "f32f6a56a5b4fe791b5e084f6a91ba9a",
        "1cacd69a9d087b2f",
        "0f24506435de5ee0",
        "4e3d215756c18133",
        "cd1a5cba5ab8d659",
    ),
    single_trace="735a8d34fd28a78771ccaf393e7b6dd1",
    single_agent="b6a721bf5f1f25ef",
    single_llm="e493acecd6029a09",
    session_latency_ms=236.0,  # 16:04:49.698 -> 16:04:49.934
    single_latency_ms=46.0,  # 16:04:49.934 -> 16:04:49.980
    scores={
        "helpfulness": ("72dc545cee86de9f", "640ff068b666c1e9cebbe8e99ac8a4c2"),
        "tool_args_valid": ("f8abedaed172dc50", "640ff068b666c1e9cebbe8e99ac8a4c2"),
        "resolution": ("0c19e2408912b7b7", "f32f6a56a5b4fe791b5e084f6a91ba9a"),
    },
    tool_score_observation="b87ff78e3f94c3f9",
)
V4 = Run(
    turn1=Record(
        "2d1908d80ab2c59b39f565d3257dac03",
        "44e7cc1c4e06e705",
        "7be612ac38b545ce",
        "f1e726c6498cd564",
        "5524361b377526d2",
    ),
    turn2=Record(
        "d32e24d409099b7d42a3ecca69f6e2f3",
        "3358e4883d2223d2",
        "587d7d66d2791577",
        "8993d695036d6420",
        "65e3a6e1f32aded6",
    ),
    single_trace="9c8a7a9e69ac8069fdf78ef0a5a96f4f",
    single_agent="b4549734ddbcdfff",
    single_llm="6c0c22b033016e87",
    session_latency_ms=295.0,  # 16:05:56.492 -> 16:05:56.787
    single_latency_ms=47.0,  # 16:05:56.787 -> 16:05:56.834
    scores={
        "helpfulness": ("8ee9bb4b38e4eab2", "2d1908d80ab2c59b39f565d3257dac03"),
        "tool_args_valid": ("83c66e2c68fec8f9", "2d1908d80ab2c59b39f565d3257dac03"),
        "resolution": ("345e470e0faa92c2", "d32e24d409099b7d42a3ecca69f6e2f3"),
    },
    tool_score_observation="f1e726c6498cd564",
)


def replay(cassette: Path) -> tuple[TraceSnapshot, dict]:
    """Run the live loader against recorded Langfuse HTTP responses."""
    data = json.loads(cassette.read_text())
    exchanges = list(data["exchanges"])

    def handler(request: httpx.Request) -> httpx.Response:
        query = [
            list(pair) for pair in parse_qsl(request.url.query.decode(), keep_blank_values=True)
        ]
        for index, exchange in enumerate(exchanges):
            if exchange["path"] == request.url.path and exchange["query"] == query:
                del exchanges[index]
                return httpx.Response(
                    exchange["status"],
                    content=exchange["body"].encode(),
                    headers={"content-type": exchange["content_type"]},
                )
        raise AssertionError(f"unrecorded request {request.method} {request.url}")

    # The SDK shares one resource manager per public key, so each client needs its own.
    client = Langfuse(
        public_key=f"pk-lf-real-replay-{next(CLIENT_IDS)}",
        secret_key="sk-lf-real-replay",
        base_url=data["base_url"],
        tracing_enabled=False,
        httpx_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    loader = LangfuseTraceLoader(
        LangfuseTraceConfig(
            datetime.fromisoformat(data["from_timestamp"]),
            datetime.fromisoformat(data["to_timestamp"]),
            base_url=data["base_url"],
        ),
        client=client,
    )
    snapshot = loader.load()
    assert exchanges == []
    return snapshot, dict(loader.describe())


def load_v3_file() -> tuple[TraceSnapshot, dict]:
    loader = LangfuseV3FileTraceLoader(
        LangfuseFileTraceConfig(FIXTURES / "v3" / "langfuse-v3.jsonl")
    )
    return loader.load(), dict(loader.describe())


def load_v4_file() -> tuple[TraceSnapshot, dict]:
    loader = LangfuseV4FileTraceLoader(
        LangfuseV4FileTraceConfig(
            FIXTURES / "v4" / "observations.json", scores_path=FIXTURES / "v4" / "scores.json"
        )
    )
    return loader.load(), dict(loader.describe())


SOURCES = {
    "v3-file": (load_v3_file, V3, "v3"),
    "v3-live": (lambda: replay(FIXTURES / "v3" / "live-v3-http.json"), V3, "v3"),
    "v4-file": (load_v4_file, V4, "v4"),
    "v4-live": (lambda: replay(FIXTURES / "v4" / "live-v4-http.json"), V4, "v4"),
}


@pytest.fixture(params=list(SOURCES), ids=list(SOURCES))
def source(request) -> tuple[dict[str, Trace], dict, Run, str]:
    load, run, api = SOURCES[request.param]
    snapshot, description = load()
    return {trace.id: trace for trace in snapshot}, description, run, api


def walk(spans: list[Span]):
    for span in spans:
        yield span
        yield from walk(span.children)


def test_two_traces_of_one_session_merge_and_the_sessionless_trace_stands_alone(source):
    traces, description, run, _ = source

    assert set(traces) == {SESSION_TRACE, run.single_trace}
    session = traces[SESSION_TRACE]
    assert (session.session_id, session.session_basis) == (SESSION, "langfuse.sessionId")
    assert [
        (record["trace_id"], record["root_span_ids"]) for record in session.attributes["records"]
    ] == [
        (run.turn1.trace, [run.turn1.span(run.turn1.agent)]),
        (run.turn2.trace, [run.turn2.span(run.turn2.agent)]),
    ]
    (root,) = session.root_spans
    assert (root.id, root.kind) == (SESSION_TRACE, SpanKind.AGENT)
    assert [(span.id, span.kind, span.attributes["subtype"]) for span in root.children] == [
        (run.turn1.span(run.turn1.agent), SpanKind.CHAIN, "AGENT"),
        (run.turn2.span(run.turn2.agent), SpanKind.CHAIN, "AGENT"),
    ]
    for record, agent in zip((run.turn1, run.turn2), root.children, strict=True):
        assert [(span.id, span.kind) for span in agent.children] == [
            (record.span(record.call_llm), SpanKind.LLM),
            (record.span(record.tool), SpanKind.TOOL),
            (record.span(record.answer_llm), SpanKind.LLM),
        ]

    single = traces[run.single_trace]
    assert (single.session_id, single.session_basis) == (None, None)
    assert "records" not in single.attributes
    (agent,) = single.root_spans
    assert (agent.id, agent.kind) == (run.single_agent, SpanKind.AGENT)
    assert [(span.id, span.kind) for span in agent.children] == [(run.single_llm, SpanKind.LLM)]

    assert session.losses == [] and single.losses == []
    assert description["observation_count"] == 10
    assert description["unresolved_parent_count"] == 0
    assert description["distinct_sessions"] == 1


def test_session_turns_follow_the_conversation_across_both_traces(source):
    traces, _, run, _ = source
    (root,) = traces[SESSION_TRACE].root_spans
    t1, t2 = run.turn1, run.turn2

    assert [
        (turn.index, turn.role.value, turn.content, turn.llm_span_ids, turn.tool_call_span_ids)
        for turn in root.turns
    ] == [
        (1, "user", "What's the weather in Paris today?", [], []),
        # The model answered with only a tool call, so the message has no content.
        (2, "agent", UNSET, [t1.span(t1.call_llm)], [t1.span(t1.tool)]),
        (3, "agent", "Paris today: light rain, high of 14°C.", [t1.span(t1.answer_llm)], []),
        # Turn 2 re-sends turn 1's history; only the new user message is a new turn.
        (4, "user", "And tomorrow?", [], []),
        (5, "agent", UNSET, [t2.span(t2.call_llm)], [t2.span(t2.tool)]),
        (6, "agent", "Paris tomorrow: sunny, high of 17°C.", [t2.span(t2.answer_llm)], []),
    ]

    (agent,) = traces[run.single_trace].root_spans
    assert [
        (turn.id, turn.role.value, turn.content, turn.llm_span_ids) for turn in agent.turns
    ] == [
        (f"{run.single_agent}/turn/1", "user", "What is 2 + 2?", []),
        (f"{run.single_agent}/turn/2", "agent", "2 + 2 = 4.", [run.single_llm]),
    ]


def test_tool_observations_carry_the_arguments_and_results_the_model_asked_for(source):
    traces, _, run, _ = source
    spans = {span.id: span for span in walk(traces[SESSION_TRACE].root_spans)}

    for record, result in ((run.turn1, TODAY), (run.turn2, TOMORROW)):
        tool = spans[record.span(record.tool)]
        assert tool.tool_name == "get_weather"
        # The arguments the generation emitted in its tool_calls are the tool's input.
        (call,) = spans[record.span(record.call_llm)].output["tool_calls"]
        assert call["function"]["name"] == "get_weather"
        assert tool.input == json.loads(call["function"]["arguments"])
        assert tool.input == {"city": result["city"], "day": result["day"]}
        assert tool.tool_call is not None
        assert [item.content for item in tool.tool_call.results] == [result]


def test_generations_keep_model_cost_and_offered_tools(source):
    traces, _, run, _ = source
    spans = {span.id: span for trace in traces.values() for span in walk(trace.root_spans)}
    # (span id, Langfuse calculated cost, input tokens, output tokens) from the export.
    generations = [
        (run.turn1.span(run.turn1.call_llm), 2.685e-05, 95, 21),
        (run.turn1.span(run.turn1.answer_llm), 2.88e-05, 120, 18),
        (run.turn2.span(run.turn2.call_llm), 2.685e-05, 95, 21),
        (run.turn2.span(run.turn2.answer_llm), 2.88e-05, 120, 18),
        (run.single_llm, 9.299999e-06, 30, 8),
    ]
    for span_id, cost, input_tokens, output_tokens in generations:
        span = spans[span_id]
        assert span.model == "gpt-4o-mini-2024-07-18"
        assert span.cost_usd == pytest.approx(cost)
        assert [
            (tool.name, tool.description, tool.input_schema) for tool in span.tool_definitions
        ] == [
            (
                "get_weather",
                "Get the weather forecast for a city on a given day.",
                GET_WEATHER_SCHEMA,
            )
        ]
        assert span.token_counts is not None
        assert (span.token_counts.input_tokens, span.token_counts.output_tokens) == (
            input_tokens,
            output_tokens,
        )

    session, single = traces[SESSION_TRACE], traces[run.single_trace]
    assert session.aggregate.cost_usd == pytest.approx(2 * (2.685e-05 + 2.88e-05))
    assert session.aggregate.latency_ms == run.session_latency_ms
    assert single.aggregate.cost_usd == pytest.approx(9.299999e-06)
    assert single.aggregate.latency_ms == run.single_latency_ms


def test_scores_attach_to_the_session_under_their_own_traces(source):
    traces, _, run, api = source
    results = traces[SESSION_TRACE].evaluator_results

    assert {
        name: [(entry["trace_id"], [score["id"] for score in entry["value"]]) for entry in entries]
        for name, entries in results.items()
    } == {name: [(trace_id, [score_id])] for name, (score_id, trace_id) in run.scores.items()}

    def score(name: str) -> dict:
        ((entry,),) = [results[name]]
        (value,) = entry["value"]
        return value

    assert (score("helpfulness")["dataType"], score("helpfulness")["value"]) == ("NUMERIC", 0.9)
    assert score("helpfulness")["comment"] == "fixture grader"
    assert score("tool_args_valid")["dataType"] == "BOOLEAN"
    assert score("resolution")["dataType"] == "CATEGORICAL"
    if api == "v3":
        assert score("tool_args_valid")["observationId"] == run.tool_score_observation
        assert (score("tool_args_valid")["value"], score("tool_args_valid")["stringValue"]) == (
            1.0,
            "True",
        )
        assert score("resolution")["stringValue"] == "resolved"
    else:
        assert score("tool_args_valid")["subject"] == {
            "kind": "observation",
            "id": run.tool_score_observation,
            "traceId": run.turn1.trace,
        }
        assert score("tool_args_valid")["value"] is True
        assert score("resolution")["value"] == "resolved"
    assert traces[run.single_trace].evaluator_results == {}


def test_v3_cli_envelope_loads_the_same_trace_as_its_body():
    loader = LangfuseV3FileTraceLoader(
        LangfuseFileTraceConfig(FIXTURES / "v3-envelope" / "single-trace.json")
    )
    (trace,) = loader.load()

    assert (trace.id, trace.session_id) == (V3.single_trace, None)
    (agent,) = trace.root_spans
    assert (agent.id, agent.kind) == (V3.single_agent, SpanKind.AGENT)
    assert [(span.id, span.kind) for span in agent.children] == [(V3.single_llm, SpanKind.LLM)]
    assert agent.turns is not UNSET
    assert [(turn.role.value, turn.content, turn.llm_span_ids) for turn in agent.turns] == [
        ("user", "What is 2 + 2?", []),
        ("agent", "2 + 2 = 4.", [V3.single_llm]),
    ]
    assert trace.aggregate.cost_usd == pytest.approx(9.299999e-06)
    assert trace.aggregate.latency_ms == V3.single_latency_ms
    assert trace.losses == []
