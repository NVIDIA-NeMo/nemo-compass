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


def test_agent_and_application_scans_are_traced(completion_case, monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        telemetry, "tracer", OITracer(provider.get_tracer(__name__), config=TraceConfig())
    )
    insight, checks, snapshot = completion_case
    completion = EvidenceCompletion(insight=insight, predicate=truncated_enrich, checks=checks)

    matches, unknown = completion.scan(snapshot)
    with pytest.warns(UserWarning, match="1 of 6 traces"):
        completion.apply(snapshot, RUN_TIMESTAMP)

    scans = exporter.get_finished_spans()
    assert len(scans) == 2
    for scan in scans:
        assert scan.name == "evidence_scan"
        assert scan.attributes is not None
        output = scan.attributes["output.value"]
        assert isinstance(output, str)
        assert json.loads(output) == {
            "status": "incomplete",
            "match_count": len(matches),
            "unknown_count": unknown,
        }
    completion.predicate = None
    completion.checks = []
    completion.unresolved_reason = "Required evidence is absent"
    with pytest.raises(ValueError, match="Required evidence is absent"):
        completion.scan(snapshot)
    skipped = exporter.get_finished_spans()[-1]
    assert skipped.status.status_code != StatusCode.ERROR
    assert skipped.attributes is not None
    output = skipped.attributes["output.value"]
    assert isinstance(output, str)
    assert json.loads(output)["status"] == "skipped"


def test_unregistered_handle_can_be_fixed_in_same_python_session(completion_case):
    insight, checks, snapshot = completion_case
    prepare = (
        "import json\nfrom compass.traces import UNSET\n"
        + inspect.getsource(truncated_enrich)
        + "\nchecks = [(tools.trace(i), expected) for i, expected in "
        + repr([(t.id, expected) for t, expected in checks])
        + "]\nhandle = tools.register({'id': 'saved'}, truncated_enrich, checks)"
    )
    llm = FakeLLMClient(
        [
            response("execute_python", {"code": prepare}, "prepare"),
            response("return_result", {"result": ["unregistered"]}, "invalid"),
            response("execute_python", {"code": "return_result([handle])"}, "correct"),
        ]
    )
    with pytest.warns(UserWarning, match="1 of 6 traces"):
        result = asyncio.run(
            InsightCompilation(llm=llm).compile_insights([], snapshot, [insight], RUN_TIMESTAMP)
        )
    assert {item.trace_id for item in result[0].evidence} == {"seed", "historical", "additional"}
    assert result[0].updated_date == RUN_TIMESTAMP
    assert llm.call_count == 3
    assert "Return unique handles" in str(llm.last_messages)


def test_skipping_the_scan_requires_nonblank_reason(completion_case):
    insight, _, snapshot = completion_case
    valid_reason = "The traces do not record the user intent"
    llm = FakeLLMClient(
        [
            response(
                "execute_python",
                {"code": "tools.register({'id': 'saved'}, unresolved_reason='   ')"},
                "invalid",
            ),
            response(
                "execute_python",
                {
                    "code": f"handle = tools.register({{'id': 'saved'}}, unresolved_reason={valid_reason!r})\nreturn_result([handle])"
                },
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


def test_batch_scans_load_once_and_isolate_predicates(completion_case, monkeypatch):
    from compass.insights_generation.evidence import scan_completions

    insight, checks, snapshot = completion_case
    passes = []
    original = TraceSnapshot.iter_json

    def counted(self):
        passes.append(self)
        return original(self)

    def mutating(item):
        decision = truncated_enrich(item)
        item.attributes["changed"] = True
        return decision

    def isolated(item):
        assert "changed" not in item.attributes
        return truncated_enrich(item)

    completions = [
        EvidenceCompletion(insight=insight, predicate=p, checks=checks)
        for p in (mutating, isolated)
    ]
    monkeypatch.setattr(TraceSnapshot, "iter_json", counted)
    results = scan_completions(completions, snapshot)
    assert len(passes) == 1
    assert all(not isinstance(result, Exception) for result in results)
    for result in results:
        assert not isinstance(result, Exception)
        matches, unknown = result
        assert (len(matches), unknown) == (2, 1)
    assert all("changed" not in item.attributes for item, _ in checks)


@pytest.mark.parametrize("failure_target", ["seed", "additional"])
def test_registration_and_fresh_final_scan_preserve_existing(completion_case, failure_target):
    from compass.insights_generation.compilation_tools import CompilationTools

    insight, checks, snapshot = completion_case
    insight.severity = "med"
    unchanged = insight.model_copy(update={"id": "untouched", "name": "Another existing claim"})
    tools = CompilationTools([], snapshot, [insight, unchanged], RUN_TIMESTAMP)
    state = {"fail": False}

    def predicate(item):
        if state["fail"] and item.id == failure_target:
            raise ValueError("changed predicate state")
        return truncated_enrich(item)

    handle = tools.register(
        {"id": "saved", "severity_reason": "The recorded response is incomplete"}, predicate, checks
    )
    assert tools.scan([handle])[0]["match_count"] == 2
    duplicate = tools.register({"id": "saved"}, predicate, checks)
    state["fail"] = True
    with pytest.warns(UserWarning, match="changed predicate state"):
        first, second = tools._finish([handle])
    assert first.evidence == insight.evidence
    assert first.id == insight.id and first.name == insight.name
    assert first.updated_date == insight.updated_date
    assert first.severity == "med"
    assert first.severity_reason == "The recorded response is incomplete"
    assert second == unchanged
    with pytest.raises(ValueError, match="one final update"):
        tools._finish([handle, duplicate])
    for handles in ([handle, handle], ["made-up"]):
        with pytest.raises(ValueError, match="unique handles"):
            tools._finish(handles)


@pytest.mark.parametrize(
    "fields, diagnostic",
    [
        ({"id": "invented"}, "ID was not supplied"),
        ({"id": "saved", "severity_reason": ""}, "at least 1 character"),
        ({"id": "saved", "confidence": "high"}, "Application-owned"),
        ({"id": "saved", "updated_date": RUN_TIMESTAMP}, "Application-owned"),
        ({"id": "saved", "evidence": [{"trace_id": "made-up"}]}, "Unknown evidence trace"),
        (
            {"id": "saved", "evidence": [{"trace_id": "seed", "spans": [{"span_id": "made-up"}]}]},
            "unknown span",
        ),
        (
            {"id": "saved", "evidence": [{"trace_id": "seed", "url": "https://invented.example"}]},
            "Omit evidence URLs",
        ),
    ],
)
def test_registration_rejects_untrusted_fields(completion_case, fields, diagnostic):
    from compass.insights_generation.compilation_tools import CompilationTools

    insight, checks, snapshot = completion_case
    insight.severity = "med"
    tools = CompilationTools([], snapshot, [insight], RUN_TIMESTAMP)
    with pytest.raises(ValueError, match=diagnostic):
        tools.register(fields, truncated_enrich, checks)


def test_inspection_is_bounded_and_raw_fields_remain_readable(completion_case):
    from compass.insights_generation.compilation_tools import CompilationTools

    insight, _, snapshot = completion_case
    tools = CompilationTools([], snapshot, [insight], RUN_TIMESTAMP)
    page = tools.inspect_trace("seed", characters=3, limit=1)
    assert page["items"][0]["output"] == {"text": '{"t', "characters": 20, "has_more": True}
    assert "no global chronology" in page["ordering"]
    first = tools.read_span("seed", "seed-span", characters=3)
    second = tools.read_span("seed", "seed-span", offset=first["next_offset"], characters=100)
    assert first["text"] + second["text"] == tools.trace("seed").root_spans[0].output
    assert second["next_offset"] is None
    with pytest.raises(KeyError):
        tools.read_span("seed", "not-a-span")


def test_source_helpers_preserve_message_authorship_and_local_order():
    from compass.insights_generation.compilation_tools import recorded_messages, tool_calls

    item = Trace.model_validate(
        {
            "id": "recorded",
            "aggregate": {},
            "root_spans": [
                {
                    "id": "agent",
                    "kind": "AGENT",
                    "turns": [
                        {
                            "id": "turn-1",
                            "index": 1,
                            "role": "user",
                            "authorship": "harness",
                            "content": "generated task",
                        },
                        {
                            "id": "turn-2",
                            "index": 2,
                            "role": "user",
                            "content": "authorship not recorded",
                        },
                    ],
                    "children": [
                        {
                            "id": "call",
                            "kind": "TOOL",
                            "tool_name": "lookup",
                            "tool_call": {"index": 7},
                            "input": {"key": "value"},
                        },
                    ],
                },
            ],
        }
    )
    turns = recorded_messages(item)
    assert [row["turn"]["authorship"] for row in turns] == ["harness", "unknown"]
    assert [row["turn"]["index"] for row in turns] == [1, 2]
    assert all(row["agent_span_id"] == "agent" for row in turns)
    assert [span.id for span in tool_calls(item, "lookup")] == ["call"]
    assert tool_calls(item, "different") == []


def test_batch_telemetry_reports_shared_pass_without_nested_claim_durations(
    completion_case, monkeypatch
):
    from compass.insights_generation.evidence import scan_completions

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        telemetry, "tracer", OITracer(provider.get_tracer(__name__), config=TraceConfig())
    )
    insight, checks, snapshot = completion_case
    valid = EvidenceCompletion(insight=insight, predicate=truncated_enrich, checks=checks)
    invalid = EvidenceCompletion(insight=insight, predicate=None, checks=[])
    results = scan_completions([valid, invalid], snapshot)
    assert not isinstance(results[0], Exception)
    assert isinstance(results[1], Exception)
    (batch,) = exporter.get_finished_spans()
    assert batch.name == "evidence_scan_batch"
    assert batch.status.status_code == StatusCode.ERROR
    assert batch.attributes is not None
    output = batch.attributes["output.value"]
    assert isinstance(output, str)
    statuses = [row["status"] for row in json.loads(output)["results"]]
    assert statuses == ["incomplete", "failed"]


def test_signal_inventory_preserves_capability_limits(completion_case):
    from compass.compass_signals.compass_signals import CompassSignalResult
    from compass.insights_generation.compilation_tools import CompilationTools

    _, _, snapshot = completion_case
    signal = CompassSignalResult(
        signal_name="tool_issues",
        problems=(),
        finding_count=7,
        skip_reason="Schema unavailable",
        limitations=("Cannot validate argument types",),
    )
    tools = CompilationTools([signal], snapshot, [], RUN_TIMESTAMP)
    row = tools.signals()["items"][0]
    assert row["finding_count"] == 7 and row["problem_count"] == 0
    assert row["skip_reason"]["text"] == "Schema unavailable"
    assert "Cannot validate argument types" in row["limitations"]["text"]
    assert tools.signal(0) is signal


def test_existing_indices_update_anonymous_insights_without_duplicates(completion_case):
    from compass.insights_generation.compilation_tools import CompilationTools

    insight, checks, snapshot = completion_case
    first = insight.model_copy(update={"id": None, "name": "First anonymous claim"})
    second = insight.model_copy(update={"id": None, "name": "Second anonymous claim"})
    tools = CompilationTools([], snapshot, [first, second], RUN_TIMESTAMP)
    assert [row["existing_index"] for row in tools.existing()["items"]] == [0, 1]
    handle = tools.register({}, truncated_enrich, checks, existing_index=1)
    with pytest.warns(UserWarning, match="1 of 6 traces"):
        result = tools._finish([handle])
    assert len(result) == 2
    assert result[0] == first
    assert result[1].id is None and result[1].name == second.name
    assert result[1].updated_date == RUN_TIMESTAMP
    assert {e.trace_id for e in result[1].evidence} == {"seed", "historical", "additional"}
    duplicate = tools.register({}, truncated_enrich, checks, existing_index=1)
    with pytest.raises(ValueError, match="one final update"):
        tools._finish([handle, duplicate])
    with pytest.raises(ValueError, match="duplicate persistent IDs"):
        CompilationTools([], snapshot, [insight, insight], RUN_TIMESTAMP)


def test_long_existing_claim_is_recoverable_without_mutating_authority(completion_case):
    from compass.insights_generation.compilation_tools import CompilationTools

    insight, _, snapshot = completion_case
    insight.description = "Important full claim. " * 100
    tools = CompilationTools([], snapshot, [insight], RUN_TIMESTAMP)
    preview = tools.existing()["items"][0]["description"]
    assert preview["has_more"] is True
    full = tools.existing_insight(0)
    assert full.description == insight.description
    full.description = "Changed detached copy"
    full.evidence[0].spans.clear()
    assert tools.existing_insight(0) == insight
    assert tools._finish([]) == [insight]
