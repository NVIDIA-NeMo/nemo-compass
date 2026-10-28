# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from trace_ingest import (
    UNSET,
    Authorship,
    Loss,
    Span,
    SpanKind,
    ToolCall,
    Trace,
    TraceAggregate,
    TurnRole,
)
from trace_ingest.loaders.helper.turns import (
    NoTurns,
    agent_turns,
    input_turns,
    message_records,
    message_text,
    promote_run_root,
)

COPY, UNKNOWN, MODEL = Authorship.COPIED_CONTEXT, Authorship.UNKNOWN, Authorship.MODEL


def lc(cls, **kwargs):
    return {
        "lc": 1,
        "type": "constructor",
        "id": ["langchain", "schema", "messages", cls],
        "kwargs": kwargs,
    }


def llm(span_id, output=UNSET, children=()):
    return Span(id=span_id, kind=SpanKind.LLM, output=output, children=list(children))


def tool(span_id):
    return Span(id=span_id, kind=SpanKind.TOOL, tool_call=ToolCall())


def rows(turns):
    return [
        (
            turn.id,
            turn.role,
            turn.authorship,
            turn.content,
            turn.llm_span_ids,
            turn.tool_call_span_ids,
        )
        for turn in turns
    ]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("plain", "plain"),
        ({"answer": "first str key"}, "first str key"),
        ({"message": {"content": "nested"}}, "nested"),
        ({"choices": [{"message": {"role": "assistant", "content": "choice"}}]}, "choice"),
        ({"generations": [[{"text": "generation"}], [{"text": "other"}]]}, "generation"),
        ([{"content": "first item"}, {"content": "second"}], "first item"),
        ({"n": 1}, '{"n": 1}'),
    ],
)
def test_message_text(value, expected):
    assert message_text(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]},
        [{"type": "system", "content": "s"}, {"type": "human", "content": "u"}],
        {"messages": [[lc("SystemMessage", content="s"), lc("HumanMessage", content="u")]]},
    ],
    ids=["openai", "langchain-type", "langchain-constructor-batch"],
)
def test_message_records_resolve_roles(value):
    records = message_records(value)

    assert records is not None
    assert [record["content"] for record in records] == ["s", "u"]
    assert [record["role"] for record in records] in (["system", "user"], ["system", "human"])


def test_message_records_reads_constructor_roles():
    records = message_records(
        [
            lc("AIMessage", content="a"),
            lc("ToolMessage", content="r", tool_call_id="c1"),
            lc("ChatMessage", content="x", role="developer"),
        ]
    )

    assert records == [
        {"role": "assistant", "content": "a"},
        {"role": "tool", "content": "r", "tool_call_id": "c1"},
        {"role": "developer", "content": "x"},
    ]
    assert message_records([lc("UnknownMessage", content="x")]) is None


def test_input_turns_copy_history_and_fold_tool_results():
    turns = input_turns(
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": None, "tool_calls": []},
            {"role": "tool", "tool_call_id": "c1", "content": '{"n": 1}'},
            {"role": "user", "content": "q2"},
        ]
    )

    assert [(turn.role, turn.authorship, turn.content) for turn in turns] == [
        (TurnRole.SYSTEM, COPY, "s"),
        (TurnRole.USER, COPY, "q1"),
        (TurnRole.AGENT, COPY, UNSET),
        (TurnRole.USER, UNKNOWN, "q2"),
    ]
    (result,) = turns[2].unlinked_results
    assert (result.content, result.extra) == ({"n": 1}, {"tool_call_id": "c1"})


def test_tool_result_without_an_agent_turn_stands_alone():
    turns = input_turns([{"role": "tool", "content": "r"}, {"role": "user", "content": "q"}])

    assert [(turn.role, turn.authorship, turn.content) for turn in turns] == [
        (TurnRole.AGENT, COPY, UNSET),
        (TurnRole.USER, UNKNOWN, "q"),
    ]
    assert turns[0].unlinked_results[0].content == "r"


@pytest.mark.parametrize(
    "value",
    [[{"role": "system", "content": "s"}], [{"role": "critic", "content": "c"}, {"role": "user"}]],
    ids=["no-user", "unknown-role"],
)
def test_input_turns_reject_unmappable_messages(value):
    with pytest.raises(NoTurns):
        input_turns(value)


def test_agent_turns_fold_tools_into_the_latest_model_turn():
    spans = [
        tool("early"),
        llm("l1", {"content": "Looking."}, children=[tool("t1")]),
        tool("t2"),
        Span(id="sub", kind=SpanKind.AGENT, children=[llm("sub-llm"), tool("sub-tool")]),
        llm("l2", "Done."),
    ]

    turns, losses = agent_turns(
        spans, owner_id="root", conversation_input="Go", conversation_output="Finished."
    )

    assert rows(turns) == [
        ("root/turn/1", TurnRole.USER, UNKNOWN, "Go", [], []),
        ("root/turn/2", TurnRole.AGENT, UNKNOWN, UNSET, [], ["early"]),
        ("root/turn/3", TurnRole.AGENT, MODEL, "Looking.", ["l1"], ["t1", "t2"]),
        ("root/turn/4", TurnRole.AGENT, MODEL, "Done.", ["l2"], []),
        ("root/turn/5", TurnRole.AGENT, UNKNOWN, "Finished.", [], []),
    ]
    assert losses == [
        Loss(
            code="steps_outside_agent_not_in_turns",
            detail={"agent_span_id": "root", "span_ids": ["sub-llm", "sub-tool"]},
            span_id="sub",
        )
    ]


def test_agent_turns_skip_closing_turn_matching_last_agent_text():
    turns, _ = agent_turns(
        [llm("l", "Done.")], owner_id="r", conversation_input="Go", conversation_output="Done."
    )

    assert [turn.content for turn in turns] == ["Go", "Done."]
    with pytest.raises(NoTurns):
        agent_turns([], owner_id="r", conversation_input=UNSET, conversation_output="x")


def trace_of(*roots):
    return Trace(id="t", root_spans=list(roots), aggregate=TraceAggregate())


def test_promote_run_root():
    chain = trace_of(Span(id="c", kind=SpanKind.CHAIN))
    root = promote_run_root(chain)
    assert root is not None and (root.kind, root.attributes["subtype"]) == (SpanKind.AGENT, "CHAIN")

    task = trace_of(Span(id="u", kind=SpanKind.UNKNOWN, attributes={"subtype": "TASK"}))
    assert promote_run_root(task) is None
    root = promote_run_root(task, orchestration_subtypes=frozenset({"TASK"}))
    assert root is not None and (root.kind, root.attributes["subtype"]) == (SpanKind.AGENT, "TASK")

    two = trace_of(Span(id="a", kind=SpanKind.CHAIN), Span(id="b", kind=SpanKind.CHAIN))
    assert promote_run_root(two) is None
    assert promote_run_root(trace_of(tool("t"))) is None
