# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from trace_ingest import (
    AgentInfo,
    LLMCallInfo,
    Loss,
    Span,
    SpanKind,
    SubagentRef,
    ToolCall,
    ToolResult,
    Trace,
    TraceAggregate,
    Turn,
    TurnRole,
)


def tool(span_id, tool_call=None, **values):
    return Span(id=span_id, kind=SpanKind.TOOL, tool_call=tool_call or ToolCall(), **values)


def turn(agent_id, index, role=TurnRole.AGENT, **values):
    return Turn(id=f"{agent_id}/turn/{index}", index=index, role=role, **values)


def agent(span_id, turns=None, children=(), **values):
    return Span(
        id=span_id,
        kind=SpanKind.AGENT,
        turns=turns if turns is not None else [],
        children=list(children),
        **values,
    )


def trace(*roots, losses=()):
    return Trace(id="t", root_spans=list(roots), aggregate=TraceAggregate(), losses=list(losses))


def test_agent_fields_require_agent_spans():
    with pytest.raises(ValidationError, match="require kind=AGENT"):
        Span(id="s", kind=SpanKind.LLM, turns=[])
    with pytest.raises(ValidationError, match="require kind=AGENT"):
        Span(id="s", kind=SpanKind.TOOL, agent=AgentInfo(name="a"))
    with pytest.raises(ValidationError, match="llm_call requires"):
        Span(id="s", kind=SpanKind.CHAIN, llm_call=LLMCallInfo(call_count=1))
    Span(id="s", kind=SpanKind.CHAIN, llm_call=LLMCallInfo(call_count=0))
    Span(id="s", kind=SpanKind.LLM, llm_call=LLMCallInfo())


def test_turn_indexes_are_contiguous():
    agent("a", turns=[turn("a", 3), turn("a", 4)])
    with pytest.raises(ValidationError, match="contiguous"):
        agent("a", turns=[turn("a", 1), turn("a", 3)])
    with pytest.raises(ValidationError):
        turn("a", 0)


def test_turn_ids_are_unique():
    with pytest.raises(ValidationError, match="duplicate turn id"):
        trace(agent("a", turns=[turn("a", 1)]), agent("b", turns=[turn("a", 1)]))


@pytest.mark.parametrize(
    "references, match",
    [
        ({"llm_span_ids": ["missing"]}, "must reference a LLM span"),
        ({"llm_span_ids": ["a/tool"]}, "must reference a LLM span"),
        ({"tool_call_span_ids": ["a/llm"]}, "must reference a TOOL span"),
        ({"tool_call_span_ids": ["a/sub/tool"]}, "under agent 'a'"),
    ],
)
def test_turn_references_must_resolve_under_the_owning_agent(references, match):
    sub = agent("a/sub", children=[tool("a/sub/tool")])
    children = [Span(id="a/llm", kind=SpanKind.LLM), tool("a/tool", children=[sub])]
    with pytest.raises(ValidationError, match=match):
        trace(agent("a", turns=[turn("a", 1, **references)], children=children))


def test_a_tool_span_belongs_to_one_turn():
    with pytest.raises(ValidationError, match="more than one turn"):
        trace(
            agent(
                "a",
                turns=[
                    turn("a", 1, tool_call_span_ids=["a/tool"]),
                    turn("a", 2, tool_call_span_ids=["a/tool"]),
                ],
                children=[tool("a/tool")],
            )
        )


def test_subagent_refs_and_losses_must_resolve():
    with pytest.raises(ValidationError, match="not an AGENT span"):
        trace(
            agent(
                "a",
                children=[
                    tool(
                        "a/tool",
                        tool_call=ToolCall(
                            results=[ToolResult(subagent_refs=[SubagentRef(span_id="a/tool")])]
                        ),
                    )
                ],
            )
        )
    with pytest.raises(ValidationError, match="unknown span"):
        trace(agent("a"), losses=[Loss(code="x", span_id="missing")])
    with pytest.raises(ValidationError, match="unknown turn"):
        trace(agent("a"), losses=[Loss(code="x", turn_id="missing")])
    trace(
        agent("a", turns=[turn("a", 1)]), losses=[Loss(code="x", span_id="a", turn_id="a/turn/1")]
    )
