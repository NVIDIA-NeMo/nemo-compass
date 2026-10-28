# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tests.test_trace_contract import agent, turn
from trace_ingest import (
    UNSET,
    Authorship,
    Loss,
    Span,
    SpanKind,
    TokenCounts,
    ToolCall,
    ToolDefinition,
    Trace,
    TraceAggregate,
    TurnRole,
)
from trace_ingest.loaders.helper.merge import (
    Record,
    assemble,
    join_turns,
    merge_records,
    prepare_record,
)

U = TurnRole.USER
A = TurnRole.AGENT
COPY = Authorship.COPIED_CONTEXT
MODEL = Authorship.MODEL


def at(second):
    return datetime(2026, 1, 1, 0, 0, second, tzinfo=UTC)


def llm(span_id, second, **values):
    return Span(
        id=span_id, kind=SpanKind.LLM, start_time=at(second), end_time=at(second + 1), **values
    )


def record(trace_id, *roots, **values):
    return Trace(id=trace_id, root_spans=list(roots), aggregate=TraceAggregate(), **values)


def chain(span_id, *children, **values):
    return Span(id=span_id, kind=SpanKind.CHAIN, children=list(children), **values)


def t(role, content, authorship=Authorship.UNKNOWN, **values):
    return turn("x", 1, role=role, content=content, authorship=authorship, **values)


def two_records():
    r1 = record("r1", chain("c1", llm("l1", 1)), evaluator_results={"score": 1}, case_id="task")
    r2 = record(
        "r2",
        Span(id="a2", kind=SpanKind.AGENT, children=[llm("l2", 5)]),
        evaluator_results={"score": 0},
        case_id="task",
    )
    return [prepare_record(r)[0] for r in (r1, r2)]


def merge_two(records, partial_reason=None):
    turns = {
        "r1": [t(U, "U1"), t(A, "A1", MODEL, llm_span_ids=["r1/l1"])],
        "r2": [
            t(U, "U1", COPY),
            t(A, "A1", COPY),
            t(U, "U2"),
            t(A, "A2", MODEL, llm_span_ids=["r2/l2"]),
        ],
    }
    return merge_records(
        records,
        session_id="s",
        session_basis="test.session",
        record_turns=[turns[r.id] for r in records],
        partial_reason=partial_reason,
    )


def test_two_records_merge_into_one_conversation():
    merged = merge_two(two_records())

    assert merged.id == "session:s"
    assert (merged.session_id, merged.session_basis) == ("s", "test.session")
    (root,) = merged.root_spans
    assert (root.id, root.kind, root.run.session_id) == ("session:s", SpanKind.AGENT, "s")
    assert [child.id for child in root.children] == ["r1/c1", "r2/a2"]
    demoted = root.children[1]
    assert (demoted.kind, demoted.attributes["subtype"]) == (SpanKind.CHAIN, "AGENT")
    assert [(turn.id, turn.index, turn.content) for turn in root.turns] == [
        ("session:s/turn/1", 1, "U1"),
        ("session:s/turn/2", 2, "A1"),
        ("session:s/turn/3", 3, "U2"),
        ("session:s/turn/4", 4, "A2"),
    ]
    assert [turn.llm_span_ids for turn in root.turns] == [[], ["r1/l1"], [], ["r2/l2"]]
    assert (root.start_time, root.end_time) == (at(1), at(6))
    assert merged.aggregate.latency_ms == 5000
    assert merged.case_id == "task"
    assert merged.evaluator_results == {
        "score": [{"trace_id": "r1", "value": 1}, {"trace_id": "r2", "value": 0}]
    }
    assert [entry["trace_id"] for entry in merged.attributes["records"]] == ["r1", "r2"]
    assert merged.attributes["records"][1]["root_span_ids"] == ["r2/a2"]


def test_records_are_ordered_by_earliest_start():
    merged = merge_two(list(reversed(two_records())))

    (root,) = merged.root_spans
    assert [child.id for child in root.children] == ["r1/c1", "r2/a2"]
    assert [turn.content for turn in root.turns] == ["U1", "A1", "U2", "A2"]


def test_copied_history_fills_turns_of_records_not_loaded():
    turns = join_turns([[t(U, "U1", COPY), t(A, "A1", COPY), t(U, "U2"), t(A, "A2", MODEL)]])

    assert [turn.content for turn in turns] == ["U1", "A1", "U2", "A2"]


def test_records_without_resent_history_concatenate():
    turns = join_turns([[t(U, "U1"), t(A, "A1", MODEL)], [t(U, "U2"), t(A, "A2", MODEL)]])

    assert [turn.content for turn in turns] == ["U1", "A1", "U2", "A2"]


def test_copied_history_keeps_its_place_among_a_records_turns():
    system = t(TurnRole.SYSTEM, "reminder", COPY)
    record = [t(U, "U1", COPY), t(A, "A1", COPY), t(U, "U2"), system, t(A, "A2", MODEL)]

    assert join_turns([record]) == record


def test_edited_resent_history_is_matched_by_role_and_content():
    reminder = t(TurnRole.SYSTEM, "<reminder>", COPY)
    history = [t(U, "U1", COPY), t(A, "A1", COPY), t(U, "U2", COPY), t(A, "A2", COPY)]
    turns = join_turns(
        [
            [t(U, "U1"), t(A, "A1", MODEL)],
            [t(U, "U1", COPY), reminder, t(A, "A1", COPY), t(U, "U2"), t(A, "A2", MODEL)],
            # A record that was not loaded sent U3 and got A3.
            [*history, t(U, "U3", COPY), t(A, "A3", COPY), t(U, "U4"), t(A, "A4", MODEL)],
        ]
    )

    assert [turn.content for turn in turns] == ["U1", "A1", "U2", "A2", "U3", "A3", "U4", "A4"]


def test_resent_history_matching_nothing_is_skipped_by_position():
    turns = join_turns(
        [
            [t(U, "U1"), t(A, "A1", MODEL)],
            [t(U, "summary", COPY), t(A, "ok", COPY), t(A, "kept", COPY), t(U, "U2")],
        ]
    )

    assert [turn.content for turn in turns] == ["U1", "A1", "kept", "U2"]


def test_agent_root_tools_move_to_the_merged_root():
    definition = ToolDefinition(name="search", raw={"name": "search"})
    prepared, definitions = prepare_record(
        record("r", Span(id="a", kind=SpanKind.AGENT, tool_definitions=[definition]))
    )

    (root,) = prepared.root_spans
    assert (root.id, root.kind, root.tool_definitions) == ("r/a", SpanKind.CHAIN, UNSET)
    assert definitions == [definition]
    changed = ToolDefinition(name="search", raw={"name": "search", "v": 2})
    merged = merge_records(
        [prepared], session_id="s", session_basis="b", definitions=[*definitions * 2, changed]
    )
    assert merged.root_spans[0].tool_definitions == [definition, changed]
    assert merged.root_spans[0].turns is UNSET


def test_merged_totals_are_unknown_when_any_record_does_not_report_them():
    r1, r2 = two_records()
    reported = TraceAggregate(
        cost_usd=0.4, token_counts=TokenCounts(input_total=10, output_tokens=2)
    )
    partial = TraceAggregate(token_counts=TokenCounts(input_total=5))
    merged = merge_two(
        [
            r1.model_copy(update={"aggregate": reported}),
            r2.model_copy(update={"aggregate": partial}),
        ]
    )

    counts = merged.aggregate.token_counts
    assert merged.aggregate.cost_usd is None
    assert (counts.input_total, counts.output_tokens) == (15, None)


def test_record_root_that_owns_a_run_is_rejected():
    with pytest.raises(ValueError, match="record r root a already owns a run"):
        prepare_record(record("r", agent("a")))


def test_nested_references_follow_renamed_spans():
    nested = agent("sub", turns=[turn("sub", 1, llm_span_ids=["l"])], children=[llm("l", 1)])
    prepared, _ = prepare_record(
        record(
            "r",
            chain("c", nested),
            losses=[Loss(code="gap", span_id="l", turn_id="sub/turn/1")],
        )
    )

    merged = merge_records([prepared], session_id="s", session_basis="b")
    sub = merged.root_spans[0].children[0].children[0]
    assert sub.turns is not UNSET
    assert (sub.id, sub.turns[0].id, sub.turns[0].llm_span_ids) == (
        "r/sub",
        "r/sub/turn/1",
        ["r/l"],
    )
    assert (merged.losses[0].span_id, merged.losses[0].turn_id) == ("r/l", "r/sub/turn/1")


def tool(span_id, second, index):
    return Span(
        id=span_id,
        kind=SpanKind.TOOL,
        start_time=at(second),
        end_time=at(second + 1),
        tool_name="search",
        tool_call=ToolCall(call_id=span_id, index=index),
    )


def test_tool_call_indexes_run_across_the_conversation():
    records = [
        prepare_record(record("r2", chain("c", tool("t0", 5, 0), tool("t1", 6, 1))))[0],
        prepare_record(record("r3", chain("c", tool("t0", 9, None))))[0],
        prepare_record(record("r1", chain("c", tool("t0", 1, 0), tool("t2", 3, 2))))[0],
    ]

    merged = merge_records(records, session_id="s", session_basis="b")
    calls = [
        (span.id, span.tool_call.index)
        for root in merged.root_spans[0].children
        for span in root.children
        if span.tool_call is not None
    ]
    assert calls == [("r1/t0", 0), ("r1/t2", 2), ("r2/t0", 3), ("r2/t1", 4), ("r3/t0", None)]


def test_demoted_agent_root_keeps_its_agent_id_on_its_calls():
    """Calls under a record's agent stay that agent's once its root is no longer an AGENT."""
    nested = Span(id="n", kind=SpanKind.AGENT, children=[Span(id="nl", kind=SpanKind.LLM)])
    root = Span(
        id="a",
        kind=SpanKind.AGENT,
        agent_id="triage",
        children=[chain("c", llm("l", 1)), Span(id="o", kind=SpanKind.LLM, agent_id="sub"), nested],
    )
    prepared, _ = prepare_record(record("r", root))

    def walk(spans):
        for span in spans:
            yield span
            yield from walk(span.children)

    assert [(span.id, span.agent_id) for span in walk(prepared.root_spans)] == [
        ("r/a", "triage"),
        ("r/c", None),
        ("r/l", "triage"),
        ("r/o", "sub"),
        ("r/n", None),
        ("r/nl", None),
    ]


def test_incomplete_merge_is_recorded_as_a_loss():
    merged = merge_two(two_records(), partial_reason="max_traces")

    assert merged.losses[-1] == Loss(
        code="merge_incomplete",
        detail={"reason": "max_traces", "loaded_record_count": 2},
        span_id="session:s",
    )


def test_case_id_is_dropped_when_records_disagree():
    records = [
        prepare_record(record(trace_id, chain("c"), case_id=case))[0]
        for trace_id, case in (("r1", "a"), ("r2", "b"))
    ]

    assert merge_records(records, session_id="s", session_basis="b").case_id is None


def session_record(trace_id, second, conversation_input, session_id="s"):
    trace = record(
        trace_id,
        chain("c", llm("l", second, output="reply")),
        session_id=session_id,
        session_basis="test.session",
    )
    return Record(trace, conversation_input=conversation_input)


def test_assemble_groups_by_session_id_and_keeps_records_without_one():
    loose = Record(
        record("loose", chain("c", llm("l", 9, output="hi"))), conversation_input="hello"
    )
    traces = assemble(
        [session_record("r1", 1, "q1"), loose, session_record("r2", 5, "q2")], capped=True
    )

    merged, single = traces
    assert (merged.id, single.id) == ("session:s", "loose")
    merged_turns = merged.root_spans[0].turns
    assert merged_turns is not UNSET
    assert [turn.content for turn in merged_turns] == ["q1", "reply", "q2", "reply"]
    assert merged.losses[-1].code == "merge_incomplete"
    (root,) = single.root_spans
    assert (root.id, root.kind, root.attributes["subtype"]) == ("c", SpanKind.AGENT, "CHAIN")
    assert root.turns is not UNSET
    assert [turn.content for turn in root.turns] == ["hello", "hi"]
    assert single.losses == []


def test_session_without_a_recorded_input_has_no_turns():
    (merged,) = assemble([session_record("r1", 1, "q1"), session_record("r2", 5, None)])

    assert merged.root_spans[0].turns is UNSET


def test_session_basis_requires_session_id():
    with pytest.raises(ValidationError, match="session_basis is required"):
        record("r", session_basis="b")
    with pytest.raises(ValidationError, match="session_basis is required"):
        record("r", session_id="s")
