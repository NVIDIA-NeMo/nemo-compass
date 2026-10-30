# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
from nooa.unifiedllm import FakeLLMClient
from pydantic import BaseModel, ValidationError

from compass.compass_signals._trace import walk_spans
from compass.compass_signals.cost_efficiency.evidence import CacheEvidence, enrich_trace
from compass.compass_signals.cost_efficiency.signal import CostEfficiencyCompassSignal
from compass.field_extraction import _Extraction
from compass.traces import UNSET, Span, SpanKind, Trace, TraceAggregate, TraceSnapshot


class DurationEvidence(BaseModel):
    seconds: float


def enrich_duration(trace, evidence):
    for span, value in evidence:
        span.attributes["seconds"] = value.seconds


def test_neutral_contract_complete_scan_isolated_sources_and_atomic_failure():
    trace = Trace(
        id="t",
        aggregate=TraceAggregate(),
        attributes={"native": {"duration_ms": 500}},
        root_spans=[
            Span(
                id="parent",
                kind=SpanKind.AGENT,
                children=[
                    Span(id="child", kind=SpanKind.TOOL),
                ],
            )
        ],
    )
    snapshot = TraceSnapshot([trace])
    before = trace.model_dump_json()
    extraction = _Extraction(snapshot, DurationEvidence, enrich_duration)

    def mapper(context, span):
        context["attributes"]["native"]["duration_ms"] = 1000
        return {"seconds": 0.5} if span["id"] == "child" else None

    assert extraction.apply(mapper) == {"source_records": 2, "extracted": 1, "unavailable": 1}
    assert extraction.result is not None
    result = extraction.result.get_trace_by_id("t")
    assert result.root_spans[0].children[0].attributes["seconds"] == 0.5
    assert result.attributes == trace.attributes
    assert trace.model_dump_json() == before
    with pytest.raises(ValidationError):
        extraction.apply(lambda context, span: {"seconds": "invalid"})
    assert extraction.result is None


def test_cost_pipeline_native_evidence_matches_detector_and_preserves_snapshot(monkeypatch):
    from test_cost_efficiency import tool_change_session

    from compass.compass_signals.builtins import registered_builtin_signals
    from compass.compass_signals.cost_efficiency import signal
    from compass.compass_signals.cost_efficiency.signal import CostEfficiencyConfig

    canonical = TraceSnapshot([tool_change_session(i) for i in range(4)])
    expected = CostEfficiencyCompassSignal().analyze_snapshot(canonical)
    native_traces = list(canonical)
    fields = set(CacheEvidence.model_fields) - {"session_id"}
    for trace in native_traces:
        for visit in walk_spans(trace):
            span = visit.span
            span.attributes["foreign_schema"] = span.model_dump(mode="json", include=fields)
            span.token_counts = None
            if span.kind is SpanKind.LLM:
                span.kind = SpanKind.UNKNOWN
                span.tool_definitions = UNSET
                span.llm_params = None
    native = TraceSnapshot(native_traces)
    before = [trace.model_dump_json() for trace in native]

    async def extract(snapshot, schema, instructions, llm, project):
        assert schema is CacheEvidence
        extraction = _Extraction(snapshot, schema, project)
        extraction.apply(lambda context, span: span["attributes"]["foreign_schema"])
        return extraction.result

    monkeypatch.setattr(signal, "extract_evidence", extract)
    registry = registered_builtin_signals(
        cost_efficiency=CostEfficiencyConfig(), llm_factory=FakeLLMClient
    )
    actual = asyncio.run(registry.analyze("cost-efficiency", native))
    assert actual.problems == expected.problems
    assert actual.artifacts == expected.artifacts
    assert actual.finding_count > 0
    assert [trace.model_dump_json() for trace in native] == before


def test_cost_contract_rejects_contradictions_but_keeps_unknowns():
    evidence = CacheEvidence(token_counts={"input_total": 100, "cached_input_tokens": 40})
    assert evidence.token_counts is not None
    assert evidence.token_counts.input_tokens is None
    assert evidence.token_counts.cache_write_tokens is None
    assert evidence.tool_definitions is None
    assert CacheEvidence(tool_definitions=[]).tool_definitions == []
    with pytest.raises(ValidationError, match="Input total"):
        CacheEvidence(
            token_counts={
                "input_total": 100,
                "input_tokens": 60,
                "cached_input_tokens": 40,
                "cache_write_tokens": 20,
            }
        )


def test_nooa_requires_complete_application_before_returning():
    import json

    from nooa.unifiedllm import LLMResponse, ToolCall

    from compass.field_extraction import extract_evidence

    def response(name, arguments):
        return LLMResponse(
            raw_response=None,
            content="",
            tool_calls=[ToolCall(id=name, name=name, arguments=json.dumps(arguments))],
            finish_reason="tool_calls",
            assistant_message={},
        )

    llm = FakeLLMClient(
        [
            response("return_result", {"result": None}),
            response(
                "execute_python",
                {
                    "code": "assert len(list(self.records())) == 1\nself.apply(lambda context, span: {'seconds': 0.5})"
                },
            ),
            response("return_result", {"result": None}),
        ]
    )
    snapshot = TraceSnapshot(
        [
            Trace(
                id="t",
                root_spans=[Span(id="s", kind=SpanKind.TOOL)],
                aggregate=TraceAggregate(),
            )
        ]
    )
    evidence = asyncio.run(
        extract_evidence(snapshot, DurationEvidence, "Read duration.", llm, enrich_duration)
    )
    assert llm.call_count == 3
    assert evidence.get_trace_by_id("t").root_spans[0].attributes["seconds"] == 0.5


@pytest.mark.parametrize("completed", [False, True])
def test_cost_extraction_budget_preserves_completed_work_or_skips(completed):
    import json

    from nooa.unifiedllm import LLMResponse, ToolCall
    from test_cost_efficiency import tool_change_session

    snapshot = TraceSnapshot(tool_change_session(i) for i in range(4))
    llm = FakeLLMClient(
        [
            LLMResponse(
                raw_response=None,
                content="",
                tool_calls=[
                    ToolCall(
                        id=str(turn),
                        name="execute_python",
                        arguments=json.dumps(
                            {
                                "code": "self.apply(lambda trace, span: {})"
                                if completed and turn == 24
                                else "pass"
                            }
                        ),
                    )
                ],
                finish_reason="tool_calls",
                assistant_message={},
            )
            for turn in range(25)
        ]
    )
    result = asyncio.run(CostEfficiencyCompassSignal(llm=llm).analyze(snapshot))
    assert llm.call_count == 25
    if completed:
        assert result == CostEfficiencyCompassSignal().analyze_snapshot(snapshot)
        assert result.finding_count > 0
    else:
        assert result.skip_reason == "Could not extract the required evidence."
        assert result.problems == () and result.artifacts is None


@pytest.mark.parametrize("mode", ["abstain", "missing", "conflicting"])
def test_enrichment_preserves_recorded_facts_and_tool_ownership(mode):
    from test_cost_efficiency import tool_change_session

    from compass.traces import TokenCounts

    trace = tool_change_session(1)
    parent = trace.root_spans[0]
    first, second, failed = parent.children
    parent.tool_definitions = first.tool_definitions
    first.tool_definitions = []  # Explicit no tools, never replaced by native/parent tools.
    second.tool_definitions = UNSET
    second.token_counts = TokenCounts(cached_input_tokens=0)
    failed.error = "provider failed"
    snapshot = TraceSnapshot([trace])
    before = snapshot.get_trace_by_id(trace.id).model_dump_json()
    extraction = _Extraction(snapshot, CacheEvidence, enrich_trace)

    def extract(context, span):
        if mode == "abstain":
            return None
        if mode == "missing":
            return {}
        return {
            "kind": "LLM",
            "agent_id": "other-agent",
            "session_id": "other-session",
            "error": "different error",
            "model": "different model",
            "token_counts": {"cached_input_tokens": 99, "output_tokens": 7},
            "tool_definitions": [{"name": "native", "raw": {"name": "native"}}],
        }

    extraction.apply(extract)
    assert extraction.result is not None
    result = extraction.result.get_trace_by_id(trace.id)
    agent = result.root_spans[0]
    a, b, c = agent.children
    assert agent.kind is SpanKind.AGENT and agent.agent_id == parent.agent_id
    assert agent.tool_definitions == parent.tool_definitions
    assert result.attributes["logical_case_id"] == trace.attributes["logical_case_id"]
    assert c.error == "provider failed"
    assert c.start_time == failed.start_time and c.end_time == failed.end_time
    assert c.model == failed.model
    assert a.tool_definitions == []
    assert a.token_counts == first.token_counts
    assert b.token_counts is not None
    assert b.token_counts.cached_input_tokens == 0
    if mode == "conflicting":
        assert b.token_counts.output_tokens == 7
        assert b.tool_definitions is not UNSET
        assert b.tool_definitions[0].name == "native"
    else:
        assert b.token_counts.output_tokens is None
        assert b.tool_definitions is UNSET
    assert snapshot.get_trace_by_id(trace.id).model_dump_json() == before


def test_native_context_is_shared_once_per_trace_and_output_is_disk_backed():
    traces = [
        Trace(
            id=str(i),
            aggregate=TraceAggregate(),
            attributes={"gym": {"steps": [{"text": "x" * 1000}] * 80}},
            root_spans=[Span(id=str(n), kind=SpanKind.LLM) for n in range(80)],
        )
        for i in range(2)
    ]
    snapshot = TraceSnapshot(traces)
    extraction = _Extraction(snapshot, DurationEvidence, enrich_duration)
    previous = None
    contexts = 0

    def extract(context, span):
        nonlocal previous, contexts
        if context is not previous:
            contexts += 1
            previous = context
        return {"seconds": 1}

    assert extraction.apply(extract)["extracted"] == 160
    assert contexts == 2
    assert extraction.result is not None
    assert isinstance(extraction.result, TraceSnapshot)
    assert sum(len(trace.root_spans) for trace in extraction.result) == 160
    assert sum(len(trace.root_spans) for trace in extraction.result) == 160
