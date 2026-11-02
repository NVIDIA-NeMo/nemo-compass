# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

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
from trace_ingest.loaders.helper.merge import join_turns
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


# A tool-call-only reply, as OpenAI's chat completions return it.
TOOL_CALL = {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
NULL_REPLY = {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]}
# LangChain ``output_version="v1"`` AIMessage that only calls a tool (LangSmith).
V1_TOOL_CALL = {"type": "tool_call", "id": "call_1", "name": "fruit_info", "args": {"x": 1}}


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
        # OpenInference LLM output (Intake).
        ({"messages": [{"role": "assistant", "content": "42"}]}, "42"),
        # LangChain messages_to_dict (LangGraph state in Intake).
        ({"type": "ai", "data": {"content": "42", "tool_calls": []}}, "42"),
        # Content blocks: the text blocks' text, as LangChain's AIMessage.text.
        (
            {"content": [{"type": "text", "text": "Let me look."}, {"type": "tool_use"}]},
            "Let me look.",
        ),
        (
            {"generations": [[{"text": "Hi", "message": lc("AIMessage", content="Hi")}]]},
            "Hi",
        ),
    ],
)
def test_message_text(value, expected):
    assert message_text(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        NULL_REPLY,
        {"choices": [{"message": NULL_REPLY}]},
        # Braintrust wrap_openai output: the choices list.
        [{"finish_reason": "tool_calls", "index": 0, "message": {**NULL_REPLY, "audio": None}}],
        # OpenInference (Intake): no content key at all.
        {"messages": [{"role": "assistant", "tool_calls": [TOOL_CALL]}]},
        # ATIF step (Intake): tool calls without a message.
        {"tool_calls": [{"tool_call_id": "c1", "function_name": "Write", "arguments": {}}]},
        # LangChain generation with empty content (LangSmith).
        {
            "generations": [
                [{"text": "", "message": lc("AIMessage", content="", tool_calls=[V1_TOOL_CALL])}]
            ]
        },
        # LangChain v1 content blocks that only call a tool (LangSmith).
        {"type": "ai", "content": [V1_TOOL_CALL], "tool_calls": [V1_TOOL_CALL]},
    ],
    ids=["openai", "choices", "choices-list", "openinference", "atif", "generation", "v1-blocks"],
)
def test_message_text_is_none_for_a_reply_without_text(value):
    assert message_text(value) is None


@pytest.mark.parametrize(
    "value",
    [
        {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]},
        [{"type": "system", "content": "s"}, {"type": "human", "content": "u"}],
        {"messages": [[lc("SystemMessage", content="s"), lc("HumanMessage", content="u")]]},
        {
            "messages": [
                {"type": "system", "data": {"content": "s"}},
                {"type": "human", "data": {"content": "u", "type": "human"}},
            ]
        },
    ],
    ids=["openai", "langchain-type", "langchain-constructor-batch", "langchain-messages-to-dict"],
)
def test_message_records_resolve_roles(value):
    records = message_records(value)

    assert records is not None
    assert [record["content"] for record in records] == ["s", "u"]
    assert [record["role"] for record in records] in (["system", "user"], ["system", "human"])


def test_message_records_reads_constructor_roles():
    records = message_records(
        [
            lc("AIMessage", content="", tool_calls=[V1_TOOL_CALL]),
            lc("ToolMessage", content="r", tool_call_id="c1"),
            lc("ChatMessage", content="x", role="developer"),
        ]
    )

    assert records == [
        {"role": "assistant", "content": "", "tool_calls": [V1_TOOL_CALL]},
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
    ("tail", "user"),
    [
        ([NULL_REPLY, {"role": "tool", "tool_call_id": "call_1", "content": "r"}], COPY),
        ([{"role": "assistant", "content": "a"}], COPY),
        ([{"role": "system", "content": "reminder"}], UNKNOWN),
    ],
    ids=["tool-loop", "answered", "system-after"],
)
def test_input_turns_copy_a_user_message_an_agent_already_answered(tail, user):
    # A per-call trace in a tool loop re-sends the user message the first call owned.
    turns = input_turns([{"role": "user", "content": "q"}, *tail])

    assert turns[0].authorship is user
    assert {turn.authorship for turn in turns[1:]} == {COPY}


def test_input_turns_keep_unmappable_part_lists_whole():
    parts = [
        {"type": "text", "text": "Describe"},
        {"type": "image_url", "image_url": {"url": "https://[::1/x.png"}},
        {"type": "text", "text": "in detail"},
    ]

    (turn,) = input_turns([{"role": "user", "content": parts}])

    assert isinstance(turn.content, str)
    assert json.loads(turn.content) == parts


@pytest.mark.parametrize(
    "value",
    [
        [{"role": "system", "content": "s"}],
        [{"role": "critic", "content": "c"}, {"role": "user"}],
        [{"role": "user", "parts": [{"type": "text", "content": "Hi"}]}],
    ],
    ids=["no-user", "unknown-role", "otel-parts"],
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


@pytest.mark.parametrize(
    ("spans", "output"),
    [
        ([llm("l", "Done.")], "Done."),
        ([llm("l", "Done."), tool("t")], "Done."),
        ([llm("l", "Done.")], {"messages": [{"role": "user"}, {"role": "ai", "content": "Done."}]}),
    ],
    ids=["same-text", "after-folded-tool", "message-list-output"],
)
def test_agent_turns_skip_closing_turn_matching_last_agent_text(spans, output):
    turns, _ = agent_turns(spans, owner_id="r", conversation_input="Go", conversation_output=output)

    assert [turn.content for turn in turns] == ["Go", "Done."]
    with pytest.raises(NoTurns):
        agent_turns([], owner_id="r", conversation_input=UNSET, conversation_output="x")


def test_agent_turns_leave_null_reply_content_unset():
    reply = {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]}
    turns, _ = agent_turns(
        [llm("l", {"choices": [{"message": reply}]}), tool("t")],
        owner_id="r",
        conversation_input="Go",
        conversation_output={"messages": [{"role": "ai", "content": "Done."}]},
    )

    assert [turn.content for turn in turns] == ["Go", UNSET, "Done."]


@pytest.mark.parametrize(
    "output",
    [
        NULL_REPLY,
        {
            "messages": [
                {"type": "human", "content": "Go"},
                {"type": "ai", "content": [V1_TOOL_CALL]},
            ]
        },
        {"tool_calls": [{"tool_call_id": "c1", "function_name": "f", "arguments": {}}]},
    ],
    ids=["openai", "langchain-v1-state", "atif"],
)
def test_agent_turns_add_no_closing_turn_for_a_reply_without_text(output):
    turns, _ = agent_turns(
        [llm("l", "Looking."), llm("l2", {"choices": [{"message": NULL_REPLY}]})],
        owner_id="r",
        conversation_input="Go",
        conversation_output=output,
    )

    assert [turn.content for turn in turns] == ["Go", "Looking.", UNSET]


def test_agent_turns_close_with_langchain_dict_state_text():
    # LangGraph state as messages_to_dict (Intake): the answer matches the last model turn.
    state = {
        "messages": [
            {"type": "human", "data": {"content": "Go", "type": "human"}},
            {"type": "ai", "data": {"content": "42", "type": "ai", "tool_calls": []}},
        ]
    }
    turns, _ = agent_turns(
        [llm("l", {"messages": [{"role": "assistant", "content": "42"}]})],
        owner_id="r",
        conversation_input="Go",
        conversation_output=state,
    )

    assert [turn.content for turn in turns] == ["Go", "42"]


def test_per_call_records_of_a_tool_loop_join_into_one_conversation():
    # One record per LLM call (e.g. Langfuse's OpenAI integration): the trace output is
    # the generation output, and the second call re-sends the first call's exchange.
    request = [{"role": "system", "content": "S"}, {"role": "user", "content": "U1"}]
    result = {"role": "tool", "tool_call_id": "call_1", "content": "R"}
    records = [
        agent_turns(
            [llm(span_id, output)],
            owner_id="session",
            conversation_input=conversation,
            conversation_output=output,
        )[0]
        for span_id, conversation, output in (
            ("g1", request, NULL_REPLY),
            ("g2", [*request, NULL_REPLY, result], "Answer"),
        )
    ]

    turns = join_turns(records)

    assert [(turn.role, turn.authorship, turn.content, turn.llm_span_ids) for turn in turns] == [
        (TurnRole.SYSTEM, COPY, "S", []),
        (TurnRole.USER, UNKNOWN, "U1", []),
        (TurnRole.AGENT, MODEL, UNSET, ["g1"]),
        (TurnRole.AGENT, MODEL, "Answer", ["g2"]),
    ]


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
