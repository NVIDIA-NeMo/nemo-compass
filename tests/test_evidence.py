# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
from nooa.unifiedllm import FakeLLMClient
from pydantic import BaseModel, ValidationError

from compass.compass_signals._trace import walk_spans
from compass.compass_signals.cost_efficiency.evidence import CacheEvidence
from compass.compass_signals.cost_efficiency.signal import CostEfficiencyCompassSignal
from compass.evidence import _Extraction
from compass.traces import UNSET, Span, SpanKind, Trace, TraceAggregate, TraceSnapshot


class DurationEvidence(BaseModel):
    seconds: float


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
    extraction = _Extraction(snapshot, DurationEvidence)

    def mapper(source):
        source["trace"]["attributes"]["native"]["duration_ms"] = 1000
        return {"seconds": 0.5} if source["span"]["id"] == "child" else None

    assert extraction.apply(mapper) == {"source_records": 2, "extracted": 1, "unavailable": 1}
    assert extraction.result is not None
    assert extraction.result.records[0].span_path == (0, 0)
    assert extraction.result.records[0].value.seconds == 0.5
    assert trace.model_dump_json() == before
    with pytest.raises(ValidationError):
        extraction.apply(lambda source: {"seconds": "invalid"})
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

    async def extract(snapshot, schema, instructions, llm):
        assert schema is CacheEvidence
        extraction = _Extraction(snapshot, schema)
        extraction.apply(lambda source: source["span"]["attributes"]["foreign_schema"])
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

    from compass.evidence import extract_evidence

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
                    "code": "assert len(list(self.records())) == 1\nself.apply(lambda source: {'seconds': 0.5})"
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
    evidence = asyncio.run(extract_evidence(snapshot, DurationEvidence, "Read duration.", llm))
    assert llm.call_count == 3
    assert evidence.source_count == 1
    assert evidence.records[0].trace_id == "t"
    assert evidence.records[0].value.seconds == 0.5
