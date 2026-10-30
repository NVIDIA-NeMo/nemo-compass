# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
import json
from datetime import datetime, timezone

import pytest
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall
from openinference.instrumentation import OITracer, TraceConfig
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from compass import telemetry
from compass.insight import Insight, SpanEvidence, TraceEvidence
from compass.insights_generation.insight_compilation import (
    EvidenceCompletion,
    EvidenceDecision,
    InsightCompilation,
)
from compass.traces import UNSET, Span, SpanKind, Trace, TraceAggregate, TraceSnapshot

RUN_TIMESTAMP = datetime(2026, 10, 2, tzinfo=timezone.utc)


def trace(trace_id, stage="enrich", output='{"title":"unfinished'):
    return Trace(
        id=trace_id,
        root_spans=[
            Span(
                id=f"{trace_id}-span", kind=SpanKind.LLM, attributes={"stage": stage}, output=output
            )
        ],
        aggregate=TraceAggregate(),
    )


def truncated_enrich(trace):
    span = trace.root_spans[0]
    # The claim does not apply to another stage, even when its output is missing.
    if span.attributes.get("stage") != "enrich":
        return {"status": "no_match"}
    if span.output is UNSET:
        return {"status": "unknown"}
    try:
        json.loads(span.output)
    except ValueError:
        return {"status": "match", "witness_span_ids": (span.id,)}
    return {"status": "no_match"}


@pytest.fixture
def completion_case():
    positive = trace("seed")
    other_stage = trace("other-stage", stage="summarize")
    valid_json = trace("valid-json", output='{"title":"complete"}')
    missing = trace("missing", output=UNSET)
    insight = Insight(
        id="saved",
        name="Truncated enrich JSON",
        description="Original claim",
        evidence=[
            TraceEvidence(
                trace_id="seed",
                spans=[SpanEvidence(span_id="seed-span", url="https://example.com/seed-span")],
            ),
            TraceEvidence(
                trace_id="historical",
                url="https://example.com/historical",
                spans=[SpanEvidence(span_id="historical-span")],
            ),
        ],
    )
    checks = [
        (positive, "match"),
        (other_stage, "no_match"),
        (valid_json, "no_match"),
        (missing, "unknown"),
    ]
    snapshot = TraceSnapshot(
        [
            positive,
            trace("additional"),
            other_stage,
            valid_json,
            missing,
            trace("irrelevant-missing", stage="summarize", output=UNSET),
        ]
    )
    return insight, checks, snapshot


@pytest.mark.parametrize(
    "fault",
    [
        "runtime",
        "invalid_status",
        "span_from_another_trace",
        "missing_span_ids",
        "no_snapshot_matches",
    ],
)
def test_invalid_completion_retains_citations_and_reports_failure(
    completion_case, fault, monkeypatch
):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        telemetry, "tracer", OITracer(provider.get_tracer(__name__), config=TraceConfig())
    )
    insight, checks, snapshot = completion_case

    def predicate(item):
        if fault == "no_snapshot_matches" and item.id != "synthetic-positive":
            return {"status": "unknown" if item.id == "missing" else "no_match"}
        if item.id == "additional":
            if fault == "runtime":
                raise RuntimeError("Predicate failed after an earlier match")
            if fault == "invalid_status":
                return EvidenceDecision.model_construct(status="maybe")
            if fault == "span_from_another_trace":
                return {"status": "match", "witness_span_ids": ("seed-span",)}
            if fault == "missing_span_ids":
                return {"status": "match"}
        return truncated_enrich(item)

    if fault == "no_snapshot_matches":
        checks[0] = (trace("synthetic-positive"), "match")
    completion = EvidenceCompletion(insight=insight, predicate=predicate, checks=checks)
    with pytest.warns(
        UserWarning, match="(?s)The evidence scan did not complete.*added no new references"
    ):
        result = completion.apply(snapshot, RUN_TIMESTAMP)
    assert result == insight
    (scan,) = exporter.get_finished_spans()
    assert scan.name == "evidence_scan"
    assert scan.status.status_code == StatusCode.ERROR
    assert any(event.name == "exception" for event in scan.events)


@pytest.mark.parametrize("updated_date", [None, datetime(2026, 9, 29, tzinfo=timezone.utc)])
def test_already_complete_claim_gains_no_traces(completion_case, updated_date):
    insight, checks, snapshot = completion_case
    insight.updated_date = updated_date
    insight.evidence.append(TraceEvidence(trace_id="additional"))
    completion = EvidenceCompletion(insight=insight, predicate=truncated_enrich, checks=checks)
    with pytest.warns(UserWarning, match="1 of 6 traces"):
        result = completion.apply(snapshot, RUN_TIMESTAMP)
    assert result == insight.model_copy(
        update={
            "evidence": [
                *insight.evidence[:2],
                TraceEvidence(
                    trace_id="additional", spans=[SpanEvidence(span_id="additional-span")]
                ),
            ]
        }
    )
    assert insight.evidence[-1].spans == []


def response(name, arguments, call_id):
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=json.dumps(arguments))],
        finish_reason="tool_calls",
        assistant_message={},
    )


@pytest.mark.parametrize(
    "invalid_return, diagnostic",
    [
        (None, "Without a predicate"),
        ("EvidenceCompletion(insight=completion.insight)", "Without a predicate"),
        (
            "completion.model_copy(update={'unresolved_reason': 'partial'})",
            "unresolved_reason=None",
        ),
        (
            "completion.model_copy(update={'predicate': None, 'unresolved_reason': 'cannot check'})",
            "Without a predicate",
        ),
        ("completion.model_copy(update={'checks': []})", "Checks must include"),
        (
            "completion.model_copy(update={'predicate': lambda t: {'status': 'match', "
            "'witness_span_ids': (t.root_spans[0].id,)}})",
            "expected no_match, got match",
        ),
        ("completion.model_copy(update={'predicate': lambda t: 1 / 0})", "division by zero"),
        (
            "completion.model_copy(update={'predicate': lambda t: {'status': 'match', "
            "'witness_span_ids': ('invented',)}})",
            "Cited span ID does not exist",
        ),
    ],
)
def test_invalid_return_can_be_fixed_in_same_python_session(
    completion_case, invalid_return, diagnostic
):
    insight, checks, snapshot = completion_case
    prepare = (
        "import json\nfrom compass.traces import UNSET\n"
        + inspect.getsource(truncated_enrich)
        + "\nchecks = [(trace_snapshot.get_trace_by_id(i), expected) for i, expected in "
        + repr([(t.id, expected) for t, expected in checks])
        + "]\ncompletion = EvidenceCompletion(insight=existing_insights[0], "
        "predicate=truncated_enrich, checks=checks)\noriginal_predicate = completion.predicate"
    )
    invalid = (
        response("return_result", {"result": [{"insight": insight.model_dump()}]}, "invalid")
        if invalid_return is None
        else response("execute_python", {"code": f"return_result([{invalid_return}])"}, "invalid")
    )
    llm = FakeLLMClient(
        [
            response("execute_python", {"code": prepare}, "prepare"),
            invalid,
            response(
                "execute_python",
                {
                    "code": "assert completion.predicate is original_predicate\n"
                    "return_result([completion])"
                },
                "correct",
            ),
        ]
    )
    with pytest.warns(UserWarning, match="1 of 6 traces"):
        result = asyncio.run(
            InsightCompilation(llm=llm).compile_insights([], snapshot, [insight], RUN_TIMESTAMP)
        )
    assert result == [
        insight.model_copy(
            update={
                "evidence": [
                    *insight.evidence,
                    TraceEvidence(
                        trace_id="additional", spans=[SpanEvidence(span_id="additional-span")]
                    ),
                ],
                "updated_date": RUN_TIMESTAMP,
            }
        )
    ]
    assert llm.call_count == 3
    assert diagnostic in str(llm.last_messages)
    assert "JSON cannot preserve their functions" in str(llm.last_messages)


def test_skipping_the_scan_requires_nonblank_reason(completion_case):
    insight, _, snapshot = completion_case
    valid_reason = "The traces do not record the user intent"
    llm = FakeLLMClient(
        [
            response(
                "return_result",
                {"result": [{"insight": insight.model_dump(), "unresolved_reason": "   "}]},
                "abstain",
            ),
            response(
                "return_result",
                {"result": [{"insight": insight.model_dump(), "unresolved_reason": valid_reason}]},
                "correct",
            ),
        ]
    )
    with pytest.warns(UserWarning, match=valid_reason):
        assert asyncio.run(
            InsightCompilation(llm=llm).compile_insights([], snapshot, [insight], RUN_TIMESTAMP)
        ) == [insight]
    assert llm.call_count == 2


@pytest.mark.parametrize("ethos", [None, "Prefer reliable delivery over decorative formatting."])
def test_compilation_exposes_ethos_to_the_agent(completion_case, ethos):
    _, _, snapshot = completion_case
    llm = FakeLLMClient(
        [
            response(
                "execute_python",
                {"code": f"assert ethos == {ethos!r}\nreturn_result([])"},
                "check_context",
            )
        ]
    )
    assert (
        asyncio.run(
            InsightCompilation(llm=llm).compile_insights(
                [], snapshot, [], RUN_TIMESTAMP, ethos=ethos
            )
        )
        == []
    )
    assert llm.call_count == 1
