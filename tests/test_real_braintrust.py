# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BraintrustTraceLoader on real Braintrust /btql responses (see fixtures/real/braintrust)."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from trace_ingest.loaders.braintrust import BraintrustTraceConfig, BraintrustTraceLoader
from trace_ingest.models import UNSET, SpanKind

FIXTURES = Path(__file__).parent / "fixtures" / "real" / "braintrust"
START = datetime(2026, 1, 1, tzinfo=timezone.utc)
END = datetime(2027, 1, 1, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def isolate_link_environment(monkeypatch):
    monkeypatch.delenv("BRAINTRUST_ORG_NAME", raising=False)
    monkeypatch.delenv("BRAINTRUST_APP_URL", raising=False)
    monkeypatch.delenv("BRAINTRUST_API_URL", raising=False)


def java_rows(name: str) -> list[dict]:
    return json.loads((FIXTURES / f"{name}.json").read_text())["data"]


def ruby_rows() -> list[dict]:
    text = (FIXTURES / "trace_spans.btql-response.jsonl").read_text()
    return [json.loads(line) for line in text.splitlines()]


def harbor_rows(name: str) -> list[dict]:
    return json.loads((FIXTURES / f"{name}.fetch-response.json").read_text())["events"]


def load(rows: list[dict], *, project_id: str | None = None, experiment_id: str | None = None):
    """Replay the recorded rows as Braintrust would answer the loader's two BTQL queries."""

    def handle(request: httpx.Request) -> httpx.Response:
        query = json.loads(request.content)["query"]
        if " WHERE is_root " in query:
            data = [
                {"root_span_id": row["root_span_id"], "created": row["created"]}
                for row in rows
                if row["is_root"]
            ]
        else:
            match = re.search(r"root_span_id = '([^']+)'", query)
            assert match is not None
            data = [row for row in rows if row["root_span_id"] == match.group(1)]
        return httpx.Response(200, json={"data": data})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        config = BraintrustTraceConfig(
            START, END, project_id=project_id, experiment_id=experiment_id
        )
        loader = BraintrustTraceLoader(config, client=client)
        traces = list(loader.load())
    return loader, traces


def at(seconds: float) -> datetime:
    return datetime.fromtimestamp(seconds, timezone.utc)


def test_java_openai_responses_web_search_tool_call():
    rows = java_rows("btql-0196bdc4536f")
    _, traces = load(rows, project_id="f1e858a4-58e3-408f-983f-016760d7fa25")

    [trace] = traces
    assert trace.id == "ff642aef149e2798f70003a539499a49"
    # Braintrust has no session key: each trace stands alone.
    assert trace.session_id is None and trace.session_basis is None
    assert trace.evaluator_results == {}  # root scores: null
    assert trace.aggregate.latency_ms == pytest.approx(
        (1788197836.7153447 - 1788197827.907864) * 1000
    )

    [root] = trace.root_spans
    assert root.id == "245e64ea156b947a"
    assert root.kind is SpanKind.AGENT
    assert (root.attributes["name"], root.attributes["subtype"]) == ("web_search", "task")
    assert (root.start_time, root.end_time) == (at(1788197827.907864), at(1788197836.7153447))
    # The Java SDK leaves the root's input/output null, so no conversation is recorded.
    assert root.input is None and root.output is None
    assert root.turns is UNSET

    [llm] = root.children
    assert llm.id == "e35d169233af09e2"
    assert llm.kind is SpanKind.LLM
    assert llm.attributes["name"] == "responses"
    assert llm.model == "gpt-4o"
    assert llm.cost_usd == 0.00221
    assert llm.input == [
        {
            "content": [
                {
                    "text": "Search the web for a recent AI news headline "
                    "and answer in one sentence.",
                    "type": "input_text",
                }
            ],
            "role": "user",
            "type": "message",
        }
    ]
    web_search_item, message = llm.output
    assert message["content"][0]["text"].startswith(
        "An AI advocacy group has launched a $50 million campaign"
    )

    # The OpenAI built-in web_search call is a child TOOL span of the LLM call that made it.
    [tool] = llm.children
    assert tool.id == "c5e6cee6671b782a"
    assert tool.kind is SpanKind.TOOL
    assert tool.tool_name == "web_search_call" == web_search_item["type"]
    assert tool.input == {"action": web_search_item["action"]}
    assert tool.input == {
        "action": {"queries": ["recent AI news"], "query": "recent AI news", "type": "search"}
    }
    assert tool.output == {"status": "completed"}
    assert tool.tool_call is not None
    assert tool.tool_call.index == 0
    assert tool.tool_call.result_count == 1
    [result] = tool.tool_call.results
    assert result.content == {"status": "completed"}
    assert result.is_error is None
    assert tool.children == []


def test_java_o4_mini_two_llm_calls_in_start_order():
    rows = java_rows("btql-a4a38e186326")
    _, traces = load(rows, project_id="f1e858a4-58e3-408f-983f-016760d7fa25")

    [trace] = traces
    assert trace.id == "1f605acceb8152d2e62d9ed8f9f79fe0"
    assert trace.session_id is None
    assert trace.aggregate.latency_ms == pytest.approx(
        (1788197855.0829535 - 1788197840.683676) * 1000
    )
    [root] = trace.root_spans
    assert root.id == "8d6f13c58b832744"
    assert root.kind is SpanKind.AGENT
    assert root.attributes["name"] == "reasoning"
    assert root.turns is UNSET

    first, second = root.children
    assert [first.id, second.id] == ["bc69c6ca41587922", "be11264b4ec3002f"]
    assert first.end_time is not None and second.start_time is not None
    assert first.end_time < second.start_time
    assert [span.kind for span in root.children] == [SpanKind.LLM, SpanKind.LLM]
    assert [span.model for span in root.children] == ["o4-mini", "o4-mini"]
    assert [span.cost_usd for span in root.children] == [
        0.0023683000000000003,
        0.0026268000000000003,
    ]
    assert all(span.tool_call is None and span.children == [] for span in root.children)

    # The second call continues the first: its input replays the first answer, then asks more.
    assert [item["type"] for item in first.output] == ["reasoning", "message"]
    assert [(item["type"], item.get("role")) for item in second.input] == [
        ("message", "user"),
        ("reasoning", None),
        ("message", "assistant"),
        ("message", "user"),
    ]
    assert second.input[2]["content"][0]["text"] == first.output[1]["content"][0]["text"]
    assert second.input[3]["content"][0]["text"] == (
        "Using the pattern you discovered, what would be the 10th term? "
        "And can you find the sum of the first 10 terms?"
    )
    assert second.output[1]["content"][0]["text"].startswith(
        "The 10th term is  \n  a₁₀ = 10·(10 + 1) = 10·11 = 110"
    )


def test_java_project_with_two_traces_loads_each_separately_in_created_order():
    rows = java_rows("btql-a4a38e186326") + java_rows("btql-0196bdc4536f")
    loader, traces = load(rows, project_id="f1e858a4-58e3-408f-983f-016760d7fa25")

    # Roots created 17:37:07.907Z (web_search) and 17:37:20.683Z (reasoning).
    assert [trace.id for trace in traces] == [
        "ff642aef149e2798f70003a539499a49",
        "1f605acceb8152d2e62d9ed8f9f79fe0",
    ]
    assert all(trace.session_id is None for trace in traces)
    description = loader.describe()
    assert {
        key: description[key]
        for key in (
            "span_count",
            "trace_count",
            "call_count",
            "distinct_cases",
            "distinct_sessions",
        )
    } == {
        "span_count": 6,
        "trace_count": 2,
        "call_count": 1,
        "distinct_cases": 2,
        "distinct_sessions": 0,
    }


def test_ruby_experiment_eval_and_task_spans():
    _, traces = load(ruby_rows(), experiment_id="110afc87-6974-4976-84a7-a5fa8b800ca8")

    [trace] = traces
    # root_span_id names the trace; it differs from the root span's own span_id.
    assert trace.id == "4a5fdda0f881776e093341bdb782f5cc"
    assert trace.session_id is None
    assert trace.evaluator_results == {}
    # Span times are datetimes, so latency has microsecond resolution.
    assert trace.aggregate.latency_ms == pytest.approx(
        (1772766230.701538 - 1772766230.701321) * 1000, abs=1e-3
    )

    [root] = trace.root_spans
    assert root.id == "5d5f8d377f9c3dec"
    assert root.kind is SpanKind.AGENT
    assert (root.attributes["name"], root.attributes["subtype"]) == ("eval", "task")
    assert root.input is None and root.output is None
    assert root.turns is UNSET

    [task] = root.children
    assert task.id == "fd7af2f8afc12c98"
    assert task.kind is SpanKind.CHAIN
    assert task.attributes["name"] == "task"
    assert task.input == {"text": "hello"}
    assert task.output == {"result": "5"}
    assert (task.start_time, task.end_time) == (at(1772766230.701499), at(1772766230.7015128))
    assert task.children == []


def test_python_harbor_rows_with_null_metadata():
    rows = harbor_rows("tool_calls_per_turn")
    # The root and the trajectory step span were logged without metadata.
    assert [row["span_id"] for row in rows if row["metadata"] is None] == [
        "072a770c29393da6",
        "e18806b7b5dc6cc3",
    ]
    _, traces = load(rows, experiment_id="5338e98e-ec96-47bd-8aab-99dc1aefcc74")

    [trace] = traces
    assert trace.id == "72e7027c0bca3e9e21011705849951f7"
    assert trace.aggregate.latency_ms == 4000
    [root] = trace.root_spans
    assert root.id == "e18806b7b5dc6cc3"
    assert root.kind is SpanKind.AGENT
    assert root.attributes["braintrust"]["metadata"] is None
    assert root.model is None
    assert root.turns is UNSET  # root input is null
    # Starts are whole seconds: each LLM call starts with the tool call it requested.
    # exec_counter (6..10) keeps the request before the tool, as it was created.
    assert [(child.id, child.kind, child.model) for child in root.children] == [
        ("e85ba18b067a54df", SpanKind.LLM, "gpt-4o-mini"),
        ("0da5cd06d84547d0", SpanKind.TOOL, None),
        ("efe8b93dc41a3c81", SpanKind.LLM, "gpt-4o-mini"),
        ("942ad65273083ced", SpanKind.TOOL, None),
        ("072a770c29393da6", SpanKind.CHAIN, None),
    ]
    first_call, first_tool, second_call, second_tool, _ = root.children
    assert first_call.start_time == first_tool.start_time == at(1767225601)
    assert second_call.start_time == second_tool.start_time == at(1767225602)
    assert [
        (tool.tool_name, tool.input, tool.output, tool.tool_call and tool.tool_call.index)
        for tool in (first_tool, second_tool)
    ] == [
        ("read_file", {"path": "/app/answer.txt"}, "42", 0),
        ("read_file", {"path": "/app/notes.txt"}, "none", 1),
    ]


def test_python_harbor_skips_rows_of_other_traces():
    rows = harbor_rows("round_trips")
    _, traces = load(rows, experiment_id="7aba8758-862e-41ef-9e77-6247c4b71287")

    # One stale tool row belongs to a root that is not in the page; it is never selected.
    [trace] = traces
    assert trace.id == "756e122672cb74dd5a53964ea59f2853"
    assert trace.aggregate.latency_ms == 3000
    [root] = trace.root_spans
    assert root.attributes["braintrust"]["metadata"] is None
    assert [(child.id, child.kind) for child in root.children] == [
        ("f8e4e029c001460e", SpanKind.LLM),
        ("942655bdec6ae4dd", SpanKind.TOOL),
        ("e55e28b6572e18a1", SpanKind.LLM),
    ]
