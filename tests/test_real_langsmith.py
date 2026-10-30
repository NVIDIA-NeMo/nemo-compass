# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the LangSmith loaders on real runs LangChain publishes from its docs CI.

See tests/fixtures/real/langsmith/README.md for how the data was produced. The export
loader reads the real CLI's files; the API loader is driven through a ``langsmith.Client``
that answers ``list_runs`` from the saved API responses, keeping only the selected fields.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langsmith import Client
from langsmith.schemas import Run

from trace_ingest.loaders.langsmith import (
    LangSmithTraceConfig,
    LangSmithTraceExportFileConfig,
    LangSmithTraceExportFileLoader,
    LangSmithTraceLoader,
)
from trace_ingest.loaders.langsmith._normalization import walk_spans
from trace_ingest.models import UNSET, Span, SpanKind, Trace, TraceSnapshot, TurnRole

FIXTURES = Path(__file__).parent / "fixtures" / "real" / "langsmith"
MEMORY = "01a189c7-63c9-72b2-a6c6-7dfb9b6bd47d"  # no thread key
ACCOUNT = "01a189c9-3719-7fe0-ac21-95a03442a355"
ACCOUNT_THREAD = "01a189c9-3716-7070-b983-2f22abbb0864"
# The four roots with thread_id "1", oldest first.
THREAD_1 = [
    "01a13c55-2f6c-72de-aeaa-c9803ea1a291",
    "01a189c8-ee2b-7c03-8f38-5c94a37a9eca",
    "01a189c9-5b35-7873-83f6-e1f120fcf1a6",
    "01a189c9-6b2a-7531-bc88-dd109dec4218",
]


class ReplayClient(Client):
    """Answer the loader's root and hydration queries from the saved API responses."""

    api_url = "https://api.smith.langchain.com"

    def __init__(self) -> None:
        self._raw = [
            raw
            for path in sorted((FIXTURES / "api").glob("*.runs-query.json"))
            for raw in json.loads(path.read_bytes())["runs"]
        ]

    def read_project(self, **kwargs):
        assert kwargs == {"project_name": "docs-code-samples"}
        return SimpleNamespace(id="00000000-0000-4000-8000-000000000001")

    def list_runs(self, **kwargs):
        select = kwargs["select"]
        if kwargs.get("is_root"):
            # /runs/query orders by start_time, newest first.
            raws = sorted(
                (raw for raw in self._raw if raw["parent_run_id"] is None),
                key=lambda raw: raw["start_time"],
                reverse=True,
            )
        else:
            trace_filter = kwargs["trace_filter"]
            if trace_filter.startswith("in(id, "):
                wanted = json.loads(trace_filter.removeprefix("in(id, ").removesuffix(")"))
            else:
                wanted = [json.loads(trace_filter.removeprefix("eq(id, ").removesuffix(")"))]
            raws = [raw for raw in self._raw if raw["trace_id"] in wanted]
        return iter(
            Run(
                **{key: raw[key] for key in select if key in raw},
                _host_url="https://smith.langchain.com",
            )
            for raw in raws
        )


def _export(max_traces: int = 100) -> LangSmithTraceExportFileLoader:
    return LangSmithTraceExportFileLoader(
        LangSmithTraceExportFileConfig(path=FIXTURES / "export", max_traces=max_traces)
    )


def _api(max_traces: int = 100) -> LangSmithTraceLoader:
    return LangSmithTraceLoader(
        LangSmithTraceConfig(project_name="docs-code-samples", max_traces=max_traces),
        client=ReplayClient(),
    )


@pytest.fixture(params=["export", "api"])
def snapshot(request) -> TraceSnapshot:
    loader = _export() if request.param == "export" else _api()
    return loader.load()


def _trace(snapshot: TraceSnapshot, trace_id: str) -> Trace:
    (trace,) = [trace for trace in snapshot if trace.id == trace_id]
    return trace


def _spans(trace: Trace) -> list[Span]:
    return list(walk_spans(trace.root_spans))


def _named(trace: Trace, name: str) -> list[Span]:
    return [span for span in _spans(trace) if span.attributes.get("name") == name]


def _records(trace: Trace) -> list[dict]:
    records = trace.attributes["records"]
    assert isinstance(records, list)
    return [record for record in records if isinstance(record, dict)]


def test_both_loaders_describe_the_same_corpus():
    export, api = _export(), _api()
    export.load()
    api.load()
    shared = {
        "trace_count": 3,
        "call_count": 3,
        "distinct_cases": 3,
        "distinct_sessions": 2,
        "run_count": 30,
        "unresolved_parent_count": 0,
    }
    for description in (dict(export.describe()), dict(api.describe())):
        assert {key: description[key] for key in shared} == shared
    assert export.describe()["export_trace_count"] == 6
    assert export.describe()["truncated"] is False


def test_loaders_build_the_same_traces(snapshot):
    assert [trace.id for trace in snapshot] == [
        "session:1",
        MEMORY,
        f"session:{ACCOUNT_THREAD}",
    ]
    shapes = {
        trace.id: [(span.kind, span.attributes.get("name")) for span in _spans(trace)]
        for trace in snapshot
    }
    assert shapes[MEMORY] == [
        (SpanKind.AGENT, "LangGraph"),
        (SpanKind.CHAIN, "model"),
        (SpanKind.LLM, "ChatAnthropic"),
        (SpanKind.CHAIN, "tools"),
        (SpanKind.TOOL, "get_user_info"),
        (SpanKind.CHAIN, "model"),
        (SpanKind.LLM, "ChatAnthropic"),
    ]
    assert shapes[f"session:{ACCOUNT_THREAD}"] == [
        (SpanKind.AGENT, "conversation"),
        (SpanKind.CHAIN, "LangGraph"),
        (SpanKind.CHAIN, "model"),
        (SpanKind.LLM, "ChatOpenAI"),
        (SpanKind.CHAIN, "tools"),
        (SpanKind.TOOL, "get_account_info"),
        (SpanKind.CHAIN, "model"),
        (SpanKind.LLM, "ChatOpenAI"),
    ]
    assert len(shapes["session:1"]) == 1 + 5 + 3 + 3 + 5


def test_thread_id_collision_merges_four_roots(snapshot):
    trace = _trace(snapshot, "session:1")
    assert (trace.session_id, trace.session_basis) == ("1", "langsmith.metadata.thread_id")
    records = trace.attributes["records"]
    assert isinstance(records, list)
    assert [record["trace_id"] for record in records if isinstance(record, dict)] == THREAD_1
    # The colliding roots span 2026-10-14T21:32:58.093001Z to 2026-10-29T22:30:43.558089Z.
    assert trace.aggregate.latency_ms == pytest.approx(1_299_465_088.093)
    (root,) = trace.root_spans
    assert [child.id for child in root.children] == [f"{run}/{run}" for run in THREAD_1]
    assert root.turns is not UNSET
    # Records join in start order: the JS and Python memory samples (each greets as Bob),
    # the interrupt sample whose state is {"vals": []}, then the subgraph sample whose model
    # only requests fruit_info.
    assert [(turn.role, turn.content) for turn in root.turns] == [
        (TurnRole.USER, "Hi! My name is Bob."),
        (TurnRole.AGENT, "Hi Bob! Nice to meet you. How can I help today?"),
        (TurnRole.USER, "Hi! My name is Bob."),
        (TurnRole.AGENT, "Hi Bob! Nice to meet you."),
        (TurnRole.USER, '{"vals": []}'),
        (TurnRole.AGENT, '{"vals": []}'),
        (TurnRole.USER, "Tell me about apples"),
        (TurnRole.AGENT, UNSET),
    ]
    (fruit,) = _named(trace, "fruit_info")
    assert root.turns[-1].tool_call_span_ids == [fruit.id]


def test_sessionless_trace_keeps_its_own_conversation(snapshot):
    trace = _trace(snapshot, MEMORY)
    assert (trace.session_id, trace.session_basis) == (None, None)
    # Root LangGraph run: 22:28:28.233485 to 22:28:30.987597.
    assert trace.aggregate.latency_ms == pytest.approx(2754.112)
    (root,) = trace.root_spans
    assert root.kind is SpanKind.AGENT
    assert root.turns is not UNSET
    assert [(turn.role, turn.content) for turn in root.turns] == [
        (TurnRole.USER, "look up user information"),
        (TurnRole.AGENT, "Sure! Let me look up the user information right away."),
        (
            TurnRole.AGENT,
            "Here is the user information that was retrieved:\n\n"
            "- **Name:** John Smith\n- **Language:** English\n\n"
            "Let me know if there's anything else you'd like to know!",
        ),
    ]
    (tool,) = _named(trace, "get_user_info")
    assert tool.tool_name == "get_user_info"
    # LangSmith records inputs: null for a tool whose only argument is the runtime; the
    # SDK's Run model turns that into {}.
    assert tool.input in (UNSET, {})
    assert tool.tool_call is not None
    assert tool.tool_call.results is not UNSET
    (result,) = tool.tool_call.results
    assert isinstance(result.content, dict)
    output = result.content["output"]
    assert isinstance(output, dict)
    assert output["content"] == "{'name': 'John Smith', 'language': 'English'}"


def test_threaded_single_record_answers_from_its_tool(snapshot):
    trace = _trace(snapshot, f"session:{ACCOUNT_THREAD}")
    assert (trace.session_id, trace.session_basis) == (
        ACCOUNT_THREAD,
        "langsmith.metadata.thread_id",
    )
    records = trace.attributes["records"]
    assert isinstance(records, list)
    assert [record["trace_id"] for record in records if isinstance(record, dict)] == [ACCOUNT]
    (root,) = trace.root_spans
    assert root.turns is not UNSET
    turns = root.turns
    assert [turn.role for turn in turns] == [TurnRole.USER, TurnRole.AGENT, TurnRole.AGENT]
    assert turns[0].content == "What's my current balance?"
    # The model's first reply only calls get_account_info.
    assert turns[1].content is UNSET
    assert turns[2].content == "Your current balance is **$5,000**."
    (tool,) = _named(trace, "get_account_info")
    assert turns[1].tool_call_span_ids == [tool.id]


def test_bound_keeps_the_conversation_with_the_newest_root():
    # Thread "1" holds the newest root (01a189c9-6b2a, 22:30:41) and also the oldest
    # (01a13c55, 2026-10-14), so both loaders select it first.
    (exported,) = _export(max_traces=1).load()
    (live,) = _api(max_traces=1).load()
    assert exported.id == live.id == "session:1"
    # The export holds the whole thread; the live selection stops at the third
    # conversation's root and marks the thread partial.
    assert [record["trace_id"] for record in _records(exported)] == THREAD_1
    assert exported.losses == []
    assert [record["trace_id"] for record in _records(live)] == THREAD_1[2:]
    assert [(loss.code, loss.detail) for loss in live.losses] == [
        ("merge_incomplete", {"reason": "max_traces", "loaded_record_count": 2})
    ]


def test_cost_counts_each_model_call_once(snapshot):
    # Each Run's total_cost includes its descendants: the root holds the trace total
    # and every `model` chain repeats its ChatAnthropic/ChatOpenAI child's cost.
    memory = _trace(snapshot, MEMORY)
    assert memory.aggregate.cost_usd == pytest.approx(0.002409 + 0.002481)
    assert [span.cost_usd for span in _spans(memory) if span.cost_usd is not None] == [
        0.002409,
        0.002481,
    ]
    assert [span.cost_usd for span in _named(memory, "ChatAnthropic")] == [0.002409, 0.002481]
    account = _trace(snapshot, f"session:{ACCOUNT_THREAD}")
    assert account.aggregate.cost_usd == pytest.approx(0.00109 + 0.001295)
    # 01a189c9-5b35 calls no model and reports no cost, so the thread total is unknown.
    assert _trace(snapshot, "session:1").aggregate.cost_usd is None
