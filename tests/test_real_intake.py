# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""IntakeTraceLoader on responses recorded from a real NeMo Helix 0.8.1 Intake server.

See tests/fixtures/real/intake/README.md for how the recording was made.

Not asserted, because the loader is known to get them wrong on this data: token counts
(Intake span and trace token fields are dropped); evaluator result `status`; and losses,
since the dropped tokens are not reported as one.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from compass.trace_loaders.intake import (
    IntakeTraceLoader,
    IntakeTraceLoaderConfig,
    IntakeTraceQuery,
)
from compass.traces import Span, Trace, TraceSnapshot
from trace_ingest.models import UNSET

RECORDING = Path(__file__).parent / "fixtures" / "real" / "intake" / "all-sessions.json"
BASE_URL = "http://127.0.0.1:18731"

LANGCHAIN_SESSION = "langchain-openinference-smoke"
LANGCHAIN_TRACES = ("05ade7af32f75cd66154d56abc8959ce", "e859c721c71a3b6934fe8caa492eadd1")
ATIF_SESSION = "d7d4e19e-608d-44ef-b166-cd050ef274ba"
ATIF_ROOT = f"{ATIF_SESSION}/span-e299ea5f59c8bbcf0982e2aba19e4839"
QUESTION = "What is 6 times 7? Use the multiply tool."


def _replay() -> httpx.MockTransport:
    """Serve each recorded body unchanged for the exact URL and query it was recorded for."""
    exchanges = json.loads(RECORDING.read_text())

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url.copy_with(query=None))
        params = dict(request.url.params)
        for exchange in exchanges:
            if exchange["url"] == url and exchange["params"] == params:
                return httpx.Response(
                    exchange["status_code"],
                    content=exchange["body"].encode(),
                    headers={"content-type": exchange["content_type"]},
                )
        raise AssertionError(f"unrecorded request {url} {params}")

    return httpx.MockTransport(handler)


@pytest.fixture(scope="module")
def loaded() -> tuple[IntakeTraceLoader, TraceSnapshot]:
    loader = IntakeTraceLoader(
        config=IntakeTraceLoaderConfig(
            base_url=BASE_URL,
            workspace="default",
            query=IntakeTraceQuery(
                started_at_gte=datetime(2026, 8, 1, tzinfo=timezone.utc),
                started_at_lte=datetime(2026, 10, 31, tzinfo=timezone.utc),
            ),
        ),
        transport=_replay(),
    )
    return loader, loader.load()


def _trace(snapshot: TraceSnapshot, session_id: str) -> Trace:
    (trace,) = [trace for trace in snapshot if trace.session_id == session_id]
    return trace


def _walk(spans: list[Span]) -> Iterator[Span]:
    for span in spans:
        yield span
        yield from _walk(span.children)


def _turns(trace: Trace):
    turns = trace.root_spans[0].turns
    assert isinstance(turns, list)
    return turns


def _result_fields(results: object, *fields: str) -> list[tuple[str, dict]]:
    assert isinstance(results, list)
    return [
        (entry["trace_id"], {field: entry["value"][field] for field in fields}) for entry in results
    ]


def test_real_intake_selects_both_sessions_and_counts_raw_spans(loaded):
    loader, snapshot = loaded

    # Intake returned 3 trace records for 2 session_ids; the ATIF run started first.
    assert [trace.id for trace in snapshot] == [
        f"session:{ATIF_SESSION}",
        f"session:{LANGCHAIN_SESSION}",
    ]
    assert {(trace.session_id, trace.session_basis) for trace in snapshot} == {
        (ATIF_SESSION, "intake.session_id"),
        (LANGCHAIN_SESSION, "intake.session_id"),
    }
    description = loader.describe()
    # 5 + 7 + 7 Intake spans, without the merged conversation roots; one tool span per trace.
    assert description["span_count"] == 19
    assert description["call_count"] == 3
    assert description["trace_count"] == 2
    assert description["distinct_sessions"] == 2


def test_real_intake_langchain_session_merges_both_otlp_traces(loaded):
    _, snapshot = loaded
    trace = _trace(snapshot, LANGCHAIN_SESSION)

    records = trace.attributes["records"]
    assert isinstance(records, list)
    assert [(r["trace_id"], r["root_span_ids"]) for r in records] == [  # type: ignore[index]
        (LANGCHAIN_TRACES[0], [f"{LANGCHAIN_TRACES[0]}/3871b4b2a9ab1db6"]),
        (LANGCHAIN_TRACES[1], [f"{LANGCHAIN_TRACES[1]}/e95366268936a416"]),
    ]
    (root,) = trace.root_spans
    assert root.id == trace.id
    assert [(child.id, child.kind.value, child.attributes["name"]) for child in root.children] == [
        (f"{LANGCHAIN_TRACES[0]}/3871b4b2a9ab1db6", "CHAIN", "LangGraph"),
        (f"{LANGCHAIN_TRACES[1]}/e95366268936a416", "CHAIN", "LangGraph"),
    ]
    # Intake sends naive timestamps; they are UTC. The session spans both runs.
    assert root.start_time == datetime(2026, 10, 30, 16, 2, 21, 989228, tzinfo=timezone.utc)
    assert root.end_time == datetime(2026, 10, 30, 16, 2, 27, 714825, tzinfo=timezone.utc)
    assert trace.aggregate.latency_ms == pytest.approx(5725.597)
    assert trace.aggregate.cost_usd is None  # Intake reports no cost for these traces.

    # Per run: LangGraph, model, tools, model chains; 2 ChatOpenAI calls; 1 multiply tool.
    kinds = Counter(span.kind.value for span in _walk(root.children))
    assert kinds == {"CHAIN": 8, "LLM": 4, "TOOL": 2}
    llm_models = {span.model for span in _walk(root.children) if span.kind.value == "LLM"}
    assert llm_models == {"openai/openai/gpt-5.5"}

    tools = [span for span in _walk(root.children) if span.kind.value == "TOOL"]
    assert [span.id for span in tools] == [
        f"{LANGCHAIN_TRACES[0]}/f6d5a997d7abe1b6",
        f"{LANGCHAIN_TRACES[1]}/9e09992d0cfd838c",
    ]
    for span in tools:
        assert span.tool_name == "multiply"
        assert span.input == {"a": 6, "b": 7}
        assert span.tool_call is not None
        assert isinstance(span.tool_call.results, list)
        (result,) = span.tool_call.results
        assert result.is_error is False
        assert isinstance(result.content, dict)
        # LangChain ToolMessage as OpenInference recorded it.
        assert result.content["data"]["content"] == "42"  # type: ignore[index]
        assert result.content["data"]["tool_call_id"] == "call_multiply_1"  # type: ignore[index]

    assert set(trace.evaluator_results) == {"answer_correct"}
    assert _result_fields(
        trace.evaluator_results["answer_correct"], "span_id", "data_type", "value", "comment"
    ) == [
        (
            LANGCHAIN_TRACES[1],
            {
                "span_id": "e95366268936a416",
                "data_type": "BOOLEAN",
                "value": 1.0,
                "comment": "6 * 7 == 42",
            },
        )
    ]


def test_real_intake_langchain_turns_link_model_calls_and_tools(loaded):
    _, snapshot = loaded
    turns = _turns(_trace(snapshot, LANGCHAIN_SESSION))

    # Each run sends the same single user question, then a ChatOpenAI call requests
    # multiply (OpenInference output without content) and one answers "42". The
    # LangGraph state output ends with that answer, so it adds no closing turn.
    first, second = LANGCHAIN_TRACES
    assert [
        (
            turn.index,
            turn.role.value,
            turn.authorship.value,
            turn.content,
            turn.llm_span_ids,
            turn.tool_call_span_ids,
        )
        for turn in turns
    ] == [
        (1, "user", "unknown", QUESTION, [], []),
        (2, "agent", "model", UNSET, [f"{first}/f89cc07cce0f4531"], [f"{first}/f6d5a997d7abe1b6"]),
        (3, "agent", "model", "42", [f"{first}/54e0f5e3ed699b36"], []),
        (4, "user", "unknown", QUESTION, [], []),
        (
            5,
            "agent",
            "model",
            UNSET,
            [f"{second}/94900f84ec51a089"],
            [f"{second}/9e09992d0cfd838c"],
        ),
        (6, "agent", "model", "42", [f"{second}/b73f9acf0e1e9b4d"], []),
    ]


def test_real_intake_atif_session_is_a_single_record(loaded):
    _, snapshot = loaded
    trace = _trace(snapshot, ATIF_SESSION)

    records = trace.attributes["records"]
    assert isinstance(records, list)
    assert [(r["trace_id"], r["root_span_ids"]) for r in records] == [  # type: ignore[index]
        (ATIF_SESSION, [ATIF_ROOT]),
    ]
    (root,) = trace.root_spans
    (agent,) = root.children
    assert (agent.id, agent.attributes["subtype"], agent.model) == (
        ATIF_ROOT,
        "AGENT",
        "claude-opus-5",
    )
    assert agent.cost_usd == pytest.approx(0.051767)
    assert trace.aggregate.cost_usd == pytest.approx(0.051767)
    assert trace.aggregate.latency_ms == pytest.approx(5439.0)
    assert root.start_time == datetime(2026, 8, 5, 18, 59, 15, 639000, tzinfo=timezone.utc)
    assert root.end_time == datetime(2026, 8, 5, 18, 59, 21, 78000, tzinfo=timezone.utc)

    # ATIF steps: user-1 (AGENT), two claude-code LLM steps, the Write call under the first.
    assert [(span.kind.value, span.attributes["name"]) for span in agent.children] == [
        ("AGENT", "user-1"),
        ("LLM", "claude-code"),
        ("LLM", "claude-code"),
    ]
    (tool,) = agent.children[1].children
    assert tool.id == f"{ATIF_SESSION}/span-15726b0b0e0d93510e7d0772c0b5f823"
    assert tool.tool_name == "Write"
    assert tool.input == {"file_path": "/app/hello.txt", "content": "Hello, world!\n"}
    assert tool.tool_call is not None
    assert tool.tool_call.call_id == "toolu_01Rmm2GPxpGq9Er31axjfYRv"
    assert tool.tool_call.result_id == "toolu_01Rmm2GPxpGq9Er31axjfYRv"
    assert isinstance(tool.tool_call.results, list)
    (result,) = tool.tool_call.results
    assert result.is_error is False
    assert isinstance(result.content, str)
    assert result.content.startswith("File created successfully at: /app/hello.txt")

    assert set(trace.evaluator_results) == {"outcome"}
    assert _result_fields(
        trace.evaluator_results["outcome"], "span_id", "data_type", "string_value"
    ) == [
        (
            ATIF_SESSION,
            {
                "span_id": "span-e299ea5f59c8bbcf0982e2aba19e4839",
                "data_type": "CATEGORICAL",
                "string_value": "pass",
            },
        )
    ]


def test_real_intake_atif_turns(loaded):
    _, snapshot = loaded
    turns = _turns(_trace(snapshot, ATIF_SESSION))

    llm = [f"{ATIF_SESSION}/span-11a09e329b85ad83a694ede94b637d0f"]
    final_llm = [f"{ATIF_SESSION}/span-4c65a2bf0c3499c0243195b417348ad1"]
    tool = [f"{ATIF_SESSION}/span-15726b0b0e0d93510e7d0772c0b5f823"]
    assert [
        (
            turn.index,
            turn.role.value,
            turn.authorship.value,
            turn.llm_span_ids,
            turn.tool_call_span_ids,
        )
        for turn in turns
    ] == [
        (1, "user", "unknown", [], []),
        (2, "agent", "model", llm, tool),
        (3, "agent", "model", final_llm, []),
    ]
    assert (
        turns[0].content == 'Create a file called hello.txt with "Hello, world!" as the content.\n'
    )
    # The ATIF step that only calls Write records no message.
    assert turns[1].content is UNSET
    assert turns[2].content == "Created `/app/hello.txt` containing `Hello, world!`."
