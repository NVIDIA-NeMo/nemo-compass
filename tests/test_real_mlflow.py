# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MLflow file loader on real ``mlflow traces search`` / ``traces get`` exports.

See ``tests/fixtures/real/mlflow/README.md`` for how the exports were produced. Span ids
below are the hex form of the base64 OTLP ids in the export.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from compass.trace_loaders.mlflow import MLflowFileTraceConfig, MLflowFileTraceLoader
from compass.traces import UNSET, SpanKind
from trace_ingest.models import Authorship, TurnRole

FIXTURES = Path(__file__).parent / "fixtures" / "real" / "mlflow"
SESSION_EXPORT = FIXTURES / "openai_agent_session.json"
SINGLE_GET = FIXTURES / "openai_agent_single_get.json"

TURN_1 = "tr-da94d9e34d258bb29b8963aac45c1c2e"
TURN_2 = "tr-4d16959a8f191e47e3692e8770745c76"
SESSIONLESS = "tr-7a33c505e9fb2de5e6af3c337c9e68c7"

QUESTION_1 = "Where is my order A-1001?"
ANSWER_1 = "Order A-1001 has shipped via UPS and should arrive on 2026-11-03."
QUESTION_2 = "Can you change its delivery address?"
ANSWER_2 = (
    "Order A-1001 has already shipped, so I can't change its address. "
    "You can request a redirect from UPS with the tracking number."
)


def load(path):
    loader = MLflowFileTraceLoader(MLflowFileTraceConfig(path=path))
    return list(loader.load()), loader.describe()


def turn_rows(turns):
    return [
        (turn.role, turn.authorship, turn.content, turn.llm_span_ids, turn.tool_call_span_ids)
        for turn in turns
    ]


def test_session_records_merge_and_sessionless_trace_stays_alone():
    traces, description = load(SESSION_EXPORT)

    assert [(t.id, t.session_id, t.session_basis) for t in traces] == [
        ("session:support-chat-42", "support-chat-42", "mlflow.trace.session"),
        (SESSIONLESS, None, None),
    ]
    assert {
        key: description[key]
        for key in (
            "trace_count",
            "export_trace_count",
            "distinct_sessions",
            "span_count",
            "unresolved_parent_count",
            "continuation_token_present",
            "truncated",
        )
    } == {
        "trace_count": 2,
        "export_trace_count": 3,
        "distinct_sessions": 1,
        "span_count": 8,
        "unresolved_parent_count": 0,
        "continuation_token_present": False,
        "truncated": False,
    }
    for trace in traces:
        assert trace.losses == []
        # The export carries no assessments.
        assert trace.evaluator_results == {}

    merged = traces[0]
    records = merged.attributes["records"]
    assert isinstance(records, list)
    # Records are ordered by request time (turn 1 at .128Z, turn 2 at .185Z).
    assert [(r["trace_id"], r["root_span_ids"]) for r in records] == [
        (TURN_1, [f"{TURN_1}/ddca2c8f07cc3585"]),
        (TURN_2, [f"{TURN_2}/f5014414715409f2"]),
    ]
    metadata = [r["attributes"]["mlflow"]["trace_metadata"] for r in records]
    assert [m["mlflow.trace.user"] for m in metadata] == ["user-7", "user-7"]
    assert {r["source_url"] for r in records} == {SESSION_EXPORT.resolve().as_uri()}


def test_session_spans_keep_the_agent_tool_loop_under_one_conversation():
    merged = load(SESSION_EXPORT)[0][0]

    (conversation,) = merged.root_spans
    assert conversation.kind is SpanKind.AGENT
    assert conversation.start_time == datetime(2026, 10, 30, 15, 59, 2, 128276, tzinfo=timezone.utc)
    assert conversation.end_time == datetime(2026, 10, 30, 15, 59, 2, 186626, tzinfo=timezone.utc)
    assert merged.aggregate.latency_ms == 58.35

    first, second = conversation.children
    # Each record's AGENT root is kept as an orchestration step of the conversation.
    assert [(s.id, s.kind, s.attributes["subtype"]) for s in (first, second)] == [
        (f"{TURN_1}/ddca2c8f07cc3585", SpanKind.CHAIN, "AGENT"),
        (f"{TURN_2}/f5014414715409f2", SpanKind.CHAIN, "AGENT"),
    ]
    assert first.input == {"messages": [{"role": "user", "content": QUESTION_1}]}
    assert first.output == ANSWER_1
    assert second.output == ANSWER_2

    assert [(s.id, s.kind, s.attributes["name"]) for s in first.children] == [
        (f"{TURN_1}/22dcbc8f1853738b", SpanKind.LLM, "Completions"),
        (f"{TURN_1}/162e8c41c02fc2a2", SpanKind.TOOL, "get_order_status"),
        (f"{TURN_1}/0016a943bf3bc0f0", SpanKind.LLM, "Completions"),
    ]
    assert [(s.id, s.kind) for s in second.children] == [
        (f"{TURN_2}/8587c96d0dd9f95c", SpanKind.LLM),
    ]

    tool = first.children[1]
    assert tool.tool_name == "get_order_status"
    assert tool.input == {"order_id": "A-1001"}
    assert tool.tool_call is not None
    assert [result.content for result in tool.tool_call.results] == [
        {"status": "shipped", "carrier": "UPS", "eta": "2026-11-03"}
    ]
    assert tool.attributes["status"] == "OK"

    request = first.children[0]
    assert request.output["choices"][0]["finish_reason"] == "tool_calls"
    assert request.output["choices"][0]["message"]["tool_calls"][0]["function"] == {
        "arguments": '{"order_id": "A-1001"}',
        "name": "get_order_status",
    }


def test_session_turns_follow_the_conversation_across_records():
    merged = load(SESSION_EXPORT)[0][0]

    (conversation,) = merged.root_spans
    assert conversation.turns is not UNSET
    user, model = TurnRole.USER, TurnRole.AGENT
    unknown, authored = Authorship.UNKNOWN, Authorship.MODEL
    # Turn 2's history repeats turn 1, which appears once. The tool-requesting model
    # call has no text (``content: null``) and owns the tool call it made.
    assert turn_rows(conversation.turns) == [
        (user, unknown, QUESTION_1, [], []),
        (
            model,
            authored,
            UNSET,
            [f"{TURN_1}/22dcbc8f1853738b"],
            [f"{TURN_1}/162e8c41c02fc2a2"],
        ),
        (model, authored, ANSWER_1, [f"{TURN_1}/0016a943bf3bc0f0"], []),
        (user, unknown, QUESTION_2, [], []),
        (model, authored, ANSWER_2, [f"{TURN_2}/8587c96d0dd9f95c"], []),
    ]
    assert [turn.id for turn in conversation.turns] == [
        f"session:support-chat-42/turn/{index}" for index in range(1, 6)
    ]
    assert all(turn.unlinked_results == [] for turn in conversation.turns)


def test_session_of_non_orchestration_roots_has_no_turns():
    """MLflow's documented ``process_chat`` session: default-typed roots own no turns."""
    (merged,), _ = load(FIXTURES / "docs_process_chat_session.json")

    assert (merged.id, merged.losses) == ("session:web-session-7", [])
    (conversation,) = merged.root_spans
    # A lone UNKNOWN root owns no turns (docs/sources/mlflow.md); a session changes nothing.
    assert [(s.attributes["name"], s.kind) for s in conversation.children] == [
        ("process_chat", SpanKind.UNKNOWN),
        ("process_chat", SpanKind.UNKNOWN),
    ]
    assert conversation.turns is UNSET


def test_sessionless_agent_trace_matches_its_traces_get_export():
    searched = load(SESSION_EXPORT)[0][1]
    (fetched,), description = load(SINGLE_GET)

    assert description["trace_count"] == description["export_trace_count"] == 1
    assert description["distinct_sessions"] == 0
    for trace in (searched, fetched):
        assert (trace.id, trace.session_id, trace.losses) == (SESSIONLESS, None, [])
        assert trace.aggregate.latency_ms == 0.623
        (root,) = trace.root_spans
        assert (root.id, root.kind, root.attributes["name"]) == (
            "438ffce0ba01fab0",
            SpanKind.AGENT,
            "support_agent",
        )
        assert root.input == {
            "messages": [{"role": "user", "content": "How long does standard shipping take?"}]
        }
        assert root.output == "Standard shipping takes 3-5 business days."
        assert [(s.id, s.kind) for s in root.children] == [("6fab034efb413192", SpanKind.LLM)]
        assert turn_rows(root.turns) == [
            (
                TurnRole.USER,
                Authorship.UNKNOWN,
                "How long does standard shipping take?",
                [],
                [],
            ),
            (
                TurnRole.AGENT,
                Authorship.MODEL,
                "Standard shipping takes 3-5 business days.",
                ["6fab034efb413192"],
                [],
            ),
        ]
    assert fetched.attributes["mlflow"] == searched.attributes["mlflow"]
