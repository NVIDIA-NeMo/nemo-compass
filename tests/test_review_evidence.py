# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json

import pytest
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from compass.insight import Insight
from compass.insights_generation.confidence import InsightConfidence
from compass.insights_generation.evidence import TraceEvidence
from compass.traces import Span, SpanKind, Trace, TraceAggregate


def test_summary_omits_large_prompt_but_original_span_is_available():
    trace = Trace(
        id="t",
        aggregate=TraceAggregate(),
        root_spans=[
            Span(id="model", kind=SpanKind.LLM, input={"system": "large prompt " * 1000}),
            Span(
                id="tool",
                kind=SpanKind.TOOL,
                tool_name="terminal",
                input="curl url",
                output="not found",
            ),
            Span(
                id="recovery",
                kind=SpanKind.TOOL,
                tool_name="python",
                output="Recovered successfully",
            ),
        ],
    )
    evidence = TraceEvidence((trace,))
    summary = evidence.overview()
    assert "large prompt" not in summary
    assert "not found" in summary
    assert "Recovered successfully" in summary
    assert "large prompt" in evidence.read_span("t", "model")
    with pytest.raises(KeyError):
        evidence.read_span("other", "model")
    with pytest.raises(KeyError):
        evidence.read_span("t", "missing")


def test_span_detail_avoids_recursive_children_and_marks_truncation():
    child = Span(id="child", kind=SpanKind.TOOL, output="child-result")
    trace = Trace(
        id="t",
        aggregate=TraceAggregate(),
        root_spans=[Span(id="parent", kind=SpanKind.AGENT, children=[child])],
    )
    evidence = TraceEvidence((trace,))
    assert "child-result" not in evidence.read_span("t", "parent")
    assert "child-result" in evidence.read_span("t", "child")
    huge = Trace(
        id="huge",
        aggregate=TraceAggregate(),
        root_spans=[Span(id="s", kind=SpanKind.TOOL, output="x" * 30_000)],
    )
    text = TraceEvidence((huge,)).read_span("huge", "s")
    assert len(text) < 17_000
    assert "len=30000" in text or "truncat" in text


def test_confidence_can_read_original_span_through_nooa(tmp_path):
    def response(name, args):
        return LLMResponse(
            raw_response=None,
            content="",
            tool_calls=[ToolCall(id=name, name=name, arguments=json.dumps(args))],
            finish_reason="tool_calls",
            assistant_message={},
        )

    llm = FakeLLMClient(
        [
            response("execute_python", {"code": "print(self.traces.read_span('t', 's'))"}),
            response("return_result", {"result": {"confidence": "med"}}),
        ]
    )
    trace = Trace(
        id="t",
        aggregate=TraceAggregate(),
        root_spans=[Span(id="s", kind=SpanKind.TOOL, output="Recovered")],
    )
    insight = Insight(
        name="Issue", description="Claim", evidence=[{"trace_id": "t"}, {"trace_id": "u"}]
    )
    assert asyncio.run(InsightConfidence(tmp_path, llm).rate(insight, (trace,))) == "med"
    assert "Recovered" in str(llm.last_messages)
