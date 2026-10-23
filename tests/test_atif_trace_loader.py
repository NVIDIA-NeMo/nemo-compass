# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from nooa.unifiedllm import FakeLLMClient
from pydantic import BaseModel, ValidationError

import compass.cli.main as cli
from compass.compass_signals.anomaly_and_patterns.signal import to_anomaly_and_patterns_trace
from compass.compass_signals.tool_issues.coverage import corpus_coverage
from compass.compass_signals.tool_issues.signal import to_tool_issue_trace
from compass.config import RunConfig, TraceConfig
from compass.trace_loaders import ATIFTraceConfig, ATIFTraceLoader, ATIFTraceLoadError
from trace_ingest.catalog import tool_catalog
from trace_ingest.loaders import atif
from trace_ingest.loaders.atif import tool_definitions_from_atif
from trace_ingest.models import (
    UNSET,
    AgentInfo,
    AudioPart,
    Authorship,
    ImagePart,
    Initiator,
    LLMCallInfo,
    Loss,
    RunInfo,
    RunTotals,
    Span,
    SpanKind,
    SubagentRef,
    TextPart,
    TokenCounts,
    ToolCall,
    ToolDefinition,
    ToolResult,
    Trace,
    TraceAggregate,
    Turn,
    TurnRole,
)

FIXTURE = Path(__file__).parent / "data" / "atif_trajectories.jsonl"
FIXTURES = Path(__file__).parent / "fixtures" / "atif"

# Fields of harbor.models.trajectories (harbor 0.24.0, ATIF-v1.8). harbor is not a dev
# dependency: adding it downgrades websockets, a runtime dependency of the langsmith
# extra, so the lists are mirrored here. Every fixture under fixtures/atif was validated
# with harbor's Trajectory model when it was written.
HARBOR_FIELDS = {
    "Trajectory": {
        "schema_version",
        "session_id",
        "trajectory_id",
        "agent",
        "steps",
        "notes",
        "final_metrics",
        "continued_trajectory_ref",
        "extra",
        "subagent_trajectories",
    },
    "Agent": {"name", "version", "model_name", "tool_definitions", "extra"},
    "Step": {
        "step_id",
        "timestamp",
        "source",
        "model_name",
        "reasoning_effort",
        "message",
        "reasoning_content",
        "tool_calls",
        "observation",
        "metrics",
        "is_copied_context",
        "llm_call_count",
        "extra",
    },
    "ToolCall": {"tool_call_id", "function_name", "arguments", "extra"},
    "Observation": {"results"},
    "ObservationResult": {"source_call_id", "content", "subagent_trajectory_ref", "extra"},
    "Metrics": {
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens",
        "cost_usd",
        "prompt_token_ids",
        "completion_token_ids",
        "logprobs",
        "extra",
    },
    "FinalMetrics": {
        "total_prompt_tokens",
        "total_completion_tokens",
        "total_cached_tokens",
        "total_cost_usd",
        "total_steps",
        "extra",
    },
    "ContentPart": {"type", "text", "source"},
    "ImageSource": {"media_type", "path"},
    "AudioSource": {"media_type", "path", "duration_sec"},
    "SubagentTrajectoryRef": {"trajectory_id", "session_id", "trajectory_path", "extra"},
}
SOURCE_MODELS: dict[str, type[BaseModel]] = {
    "Trajectory": atif._Trajectory,
    "Agent": atif._Agent,
    "Step": atif._Step,
    "ToolCall": atif._Call,
    "Observation": atif._Observation,
    "ObservationResult": atif._Result,
    "Metrics": atif._Metrics,
    "FinalMetrics": atif._FinalMetrics,
    "ContentPart": atif._ContentPart,
    "ImageSource": atif._ImageSource,
    "AudioSource": atif._AudioSource,
    "SubagentTrajectoryRef": atif._Reference,
}
# Nested Harbor models by field, to walk a fixture with types.
NESTED = {
    ("Trajectory", "agent"): "Agent",
    ("Trajectory", "steps"): "Step",
    ("Trajectory", "final_metrics"): "FinalMetrics",
    ("Trajectory", "subagent_trajectories"): "Trajectory",
    ("Step", "message"): "ContentPart",
    ("Step", "tool_calls"): "ToolCall",
    ("Step", "observation"): "Observation",
    ("Step", "metrics"): "Metrics",
    ("Observation", "results"): "ObservationResult",
    ("ObservationResult", "content"): "ContentPart",
    ("ObservationResult", "subagent_trajectory_ref"): "SubagentTrajectoryRef",
    ("ContentPart", "source"): "ImageSource",
}


def minimal():
    return {
        "schema_version": "ATIF-v1.5",
        "session_id": "run",
        "agent": {"name": "agent", "version": "1"},
        "steps": [{"step_id": 1, "source": "user", "message": "Help."}],
    }


def loader_for(tmp_path, *records):
    path = tmp_path / "traces.jsonl"
    path.write_text("\n" + "\n\n".join(json.dumps(row) for row in records) + "\n")
    return ATIFTraceLoader(ATIFTraceConfig(path=path))


def load(path):
    """Load a pretty-printed fixture trajectory as a one-line JSONL corpus."""
    with TemporaryDirectory() as directory:
        corpus = Path(directory) / f"{path.stem}.jsonl"
        corpus.write_text(json.dumps(json.loads(path.read_text())) + "\n")
        (trace,) = ATIFTraceLoader(ATIFTraceConfig(path=corpus)).load()
    return trace


def load_record(tmp_path, record):
    return next(iter(loader_for(tmp_path, record).load()))


def spans_by_id(trace):
    found = {}
    pending = list(trace.root_spans)
    while pending:
        span = pending.pop()
        found[span.id] = span
        pending.extend(span.children)
    return found


def with_agent_step(step):
    source = minimal()
    source["steps"].append({"step_id": 2, "source": "agent", "message": "", **step})
    return source


def test_exact_normalization_and_description(tmp_path):
    loader = loader_for(tmp_path, minimal())
    expected = Trace(
        id="run",
        source_url=loader.config.path.resolve().as_uri() + "#L2",
        root_spans=[
            Span(
                id="trajectory",
                kind=SpanKind.AGENT,
                agent=AgentInfo(name="agent", version="1"),
                run=RunInfo(schema_version="ATIF-v1.5", session_id="run"),
                turns=[
                    Turn(
                        id="trajectory/turn/1",
                        index=1,
                        role=TurnRole.USER,
                        content="Help.",
                    )
                ],
            )
        ],
        aggregate=TraceAggregate(),
        attributes={"logical_case_id": "run"},
    )
    snapshot = loader.load()
    assert list(snapshot) == [expected]
    assert list(snapshot) == list(loader.load())
    assert loader.load() is snapshot
    assert loader.describe() == {
        "source": f"atif:{loader.config.path.resolve()}",
        "path": str(loader.config.path.resolve()),
        "trace_count": 1,
        "call_count": 0,
        "distinct_logical_cases": 1,
    }


def _walk_fixture(model, value, seen):
    if isinstance(value, list):
        for item in value:
            _walk_fixture(model, item, seen)
        return
    if not isinstance(value, dict):
        return
    for key, item in value.items():
        if item is not None:
            seen.add((model, key))
        nested = NESTED.get((model, key))
        if (model, key) == ("ContentPart", "source"):
            nested = "AudioSource" if value.get("type") == "audio" else "ImageSource"
        if nested is not None:
            _walk_fixture(nested, item, seen)


def _leaves(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _leaves(item)
    elif isinstance(value, list):
        for item in value:
            yield from _leaves(item)
    else:
        yield type(value).__name__, value


def test_source_models_mirror_harbor_fields():
    for name, fields in HARBOR_FIELDS.items():
        declared = set(SOURCE_MODELS[name].model_fields)
        # Root error flags are tolerated in addition to Harbor's fields.
        assert declared - {"isError", "is_error"} == fields, name
        assert SOURCE_MODELS[name].model_config["extra"] == "forbid"


def test_maximal_fixture_sets_every_harbor_field():
    seen = set()
    _walk_fixture("Trajectory", json.loads((FIXTURES / "maximal.json").read_text()), seen)
    expected = {(name, key) for name, fields in HARBOR_FIELDS.items() for key in fields}
    assert expected - seen == set()


def test_every_harbor_field_lands_in_a_typed_field():
    source = json.loads((FIXTURES / "maximal.json").read_text())
    trace = load(FIXTURES / "maximal.json")
    spans = spans_by_id(trace)

    # The case key is the session id; ATIF metadata is not copied into attributes.
    assert trace.attributes == {"logical_case_id": "max-session"}
    for span in spans.values():
        expected = {"explicit_error": False} if span.id == "trajectory/step/3/tool/0" else {}
        assert span.attributes == expected, span.id
        for result in span.tool_call.results if span.tool_call else []:
            assert result.attributes == {}
    for span in spans.values():
        for turn in span.turns if span.kind is SpanKind.AGENT else []:
            for result in turn.unlinked_results:
                assert result.attributes == {}

    # Every source value appears in the trace, except values a typed field interprets:
    # prompt_tokens 500 becomes input_tokens 380 (500 - 120 cached), and
    # is_copied_context becomes authorship.
    source_leaves = set(_leaves(source))
    trace_leaves = set(_leaves(trace.model_dump(mode="json")))
    assert source_leaves - trace_leaves == {("int", 500), ("bool", True)}

    root = spans["trajectory"]
    assert root.agent == AgentInfo(
        name="max-agent",
        version="9.1",
        model_name="max-agent-model",
        extra={"agent_extra": "agent-extra-value"},
    )
    assert root.tool_definitions == tool_definitions_from_atif(source["agent"]["tool_definitions"])
    assert root.tool_definitions[0].server == "fs-server"
    assert root.run == RunInfo(
        schema_version="ATIF-v1.8",
        session_id="max-session",
        trajectory_id="max-trajectory",
        notes="max-notes",
        continued_trajectory_ref="max-continued.json",
        final_metrics=RunTotals(
            prompt_tokens=900,
            completion_tokens=140,
            cached_tokens=240,
            cost_usd=1.25,
            steps=5,
            extra={"final_extra": "final-extra-value"},
        ),
        extra={"trajectory_extra": "trajectory-extra-value"},
    )
    assert trace.id == "max-trajectory"
    assert trace.aggregate == TraceAggregate(
        cost_usd=1.25,
        token_counts=TokenCounts(input_tokens=660, cached_input_tokens=240, output_tokens=140),
    )

    system, user, agent = root.turns
    assert system == Turn(
        id="trajectory/turn/1",
        index=1,
        role=TurnRole.SYSTEM,
        authorship=Authorship.HARNESS,
        content="max-system-message",
        timestamp="2026-10-01T09:00:00Z",
        extra={"system_extra": "system-extra-value"},
    )
    assert user == Turn(
        id="trajectory/turn/2",
        index=2,
        role=TurnRole.USER,
        authorship=Authorship.UNKNOWN,
        content=[
            TextPart(text="max-user-text"),
            ImagePart(media_type="image/png", path="images/max-screen.png"),
            AudioPart(media_type="audio/wav", path="audio/max-voice.wav", duration_sec=2.5),
        ],
        timestamp="2026-10-01T09:00:01",
        unlinked_results=[ToolResult(content="max-user-observation")],
        extra={"user_extra": "user-extra-value"},
    )
    assert agent == Turn(
        id="trajectory/turn/3",
        index=3,
        role=TurnRole.AGENT,
        authorship=Authorship.COPIED_CONTEXT,
        content="max-agent-message",
        reasoning="max-reasoning",
        timestamp="2026-10-01T09:00:02+02:00",
        llm_span_ids=["trajectory/step/3"],
        tool_call_span_ids=["trajectory/step/3/tool/0", "trajectory/step/3/tool/1"],
        extra={"agent_step_extra": "agent-step-extra-value"},
    )

    step = spans["trajectory/step/3"]
    assert step.kind is SpanKind.LLM
    assert step.start_time == datetime(2026, 10, 1, 9, 0, 2, tzinfo=timezone(timedelta(hours=2)))
    assert step.input is UNSET
    assert step.output == {"role": "assistant", "content": "max-agent-message"}
    assert step.model == "max-step-model"
    assert step.cost_usd == 0.75
    assert step.token_counts == TokenCounts(
        input_tokens=380, cached_input_tokens=120, output_tokens=70
    )
    assert step.llm_call == LLMCallInfo(
        call_count=3,
        reasoning_effort="max-effort",
        prompt_token_ids=[11, 12, 13],
        completion_token_ids=[21, 22],
        logprobs=[-0.25, -1.5],
        metrics_extra={"metrics_extra": "metrics-extra-value"},
    )

    read = spans["trajectory/step/3/tool/0"]
    assert read.input == {"path": "max-argument-path"}
    assert read.tool_name == "read_file"
    assert read.tool_call == ToolCall(
        call_id="max-call-read",
        result_id="max-call-read",
        result_count=1,
        prior_user_text="max-user-text",
        results=[
            ToolResult(
                content=[
                    {"type": "text", "text": "max-result-text"},
                    {
                        "type": "image",
                        "media_type": "image/jpeg",
                        "path": "https://example.com/max.jpg",
                    },
                ],
                is_error=False,
                extra={"result_extra": "result-extra-value", "isError": False},
            )
        ],
        server="max-mcp-server",
        initiator=Initiator.MODEL,
        extra={"tool_scope": {"transport": "max-transport"}, "call_extra": "call-extra-value"},
    )
    delegate = spans["trajectory/step/3/tool/1"]
    assert delegate.tool_call.results == [
        ToolResult(
            content="max-delegated",
            subagent_refs=[
                SubagentRef(
                    span_id="trajectory/subagent/0",
                    trajectory_id="max-child",
                    session_id="max-ref-session",
                    trajectory_path="max-child.json",
                    extra={"ref_extra": "ref-extra-value"},
                )
            ],
        )
    ]
    child = delegate.children[0]
    assert child.id == "trajectory/subagent/0"
    assert child.agent == AgentInfo(name="max-child-agent", version="2.0")
    assert child.run == RunInfo(
        schema_version="ATIF-v1.6", session_id="max-child-session", trajectory_id="max-child"
    )
    assert [turn.content for turn in child.turns] == ["max-child-task", "max-child-answer"]
    assert trace.losses == [
        Loss(
            code="continued_trajectory_not_followed",
            detail={"continued_trajectory_ref": "max-continued.json"},
            span_id="trajectory",
        ),
        Loss(
            code="llm_calls_aggregated",
            detail={"llm_call_count": 3},
            span_id="trajectory/step/3",
            turn_id="trajectory/turn/3",
        ),
    ]


def test_same_bytes_produce_identical_ids(tmp_path):
    for path in sorted(FIXTURES.glob("*.json")):
        first, second = load(path), load(path)
        assert first.model_dump(mode="json", exclude={"source_url"}) == second.model_dump(
            mode="json", exclude={"source_url"}
        )


def test_subagents_embedded_and_unresolved():
    trace = load(FIXTURES / "subagents.json")
    spans = spans_by_id(trace)
    root = spans["trajectory"]
    step = spans["trajectory/step/2"]
    first, second = (child for child in step.children if child.kind is SpanKind.TOOL)
    assert [child.id for child in first.children] == ["trajectory/subagent/0"]
    assert first.tool_call.results[0].subagent_refs == [
        SubagentRef(span_id="trajectory/subagent/0", trajectory_id="researcher")
    ]
    # External trajectory files are not read.
    assert second.children == []
    assert second.tool_call.results[0].subagent_refs[0].span_id is None
    # An unlinked result's subagent hangs off the step.
    (turn,) = [turn for turn in root.turns if turn.index == 2]
    (unlinked,) = turn.unlinked_results
    assert unlinked.subagent_refs == [
        SubagentRef(span_id="trajectory/subagent/1", trajectory_id="background"),
        SubagentRef(trajectory_path="missing-subagent.json"),
    ]
    assert spans["trajectory/subagent/1"] in step.children
    # Unreferenced subagents follow the steps.
    assert [child.id for child in root.children] == [
        "trajectory/step/2",
        "trajectory/step/3",
        "trajectory/subagent/2",
    ]
    assert trace.losses == [
        Loss(
            code="unresolved_subagent_ref",
            detail={"ref": {"trajectory_path": "subagent-checker.json"}},
            span_id="trajectory/step/2/tool/1",
            turn_id="trajectory/turn/2",
        ),
        Loss(
            code="unresolved_subagent_ref",
            detail={"ref": {"trajectory_path": "missing-subagent.json"}},
            span_id="trajectory/step/2",
            turn_id="trajectory/turn/2",
        ),
    ]
    # The subagent's tool call belongs to the subagent's own turn.
    subagent_turn = spans["trajectory/subagent/0"].turns[1]
    assert subagent_turn.tool_call_span_ids == ["trajectory/subagent/0/step/2/tool/0"]


def test_harbor_terminus_system_step_ref(tmp_path):
    # Terminus records a summarization subagent on a system step's observation,
    # with only trajectory_path: no content and no source_call_id.
    source = minimal()
    source["steps"].append(
        {
            "step_id": 2,
            "source": "system",
            "message": "Context summarized.",
            "observation": {
                "results": [
                    {
                        "subagent_trajectory_ref": [
                            {"trajectory_path": "trajectory.summarization-1-summary.json"}
                        ]
                    }
                ]
            },
        }
    )
    (tmp_path / "trajectory.json").write_text(json.dumps(source))
    trace = load(tmp_path / "trajectory.json")
    root = trace.root_spans[0]
    (unlinked,) = root.turns[1].unlinked_results
    assert unlinked.content is UNSET
    (ref,) = unlinked.subagent_refs
    assert ref == SubagentRef(trajectory_path="trajectory.summarization-1-summary.json")
    assert [span.kind for span in root.children] == []
    assert [loss.code for loss in trace.losses] == ["unresolved_subagent_ref"]


def test_skipped_tool_definitions_are_losses(tmp_path):
    source = minimal()
    source["agent"]["tool_definitions"] = [{"name": "ok"}, {"description": "nameless"}]
    trace = load_record(tmp_path, source)
    assert [tool.name for tool in trace.root_spans[0].tool_definitions] == ["ok"]
    assert trace.losses == [
        Loss(
            code="tool_definition_skipped",
            detail={"definition": {"description": "nameless"}},
            span_id="trajectory",
        )
    ]


def test_unknown_subagent_id_and_repeated_references(tmp_path):
    source = json.loads((FIXTURES / "subagents.json").read_text())
    results = source["steps"][1]["observation"]["results"]
    nobody = {"trajectory_id": "nobody", "session_id": "s", "extra": {"k": 1}}
    results[0]["subagent_trajectory_ref"].append(nobody)
    results[1]["subagent_trajectory_ref"] = [{"trajectory_id": "researcher"}]
    trace = load_record(tmp_path, source)
    spans = spans_by_id(trace)
    first, second = (
        spans[f"trajectory/step/2/tool/{index}"].tool_call.results[0] for index in (0, 1)
    )
    assert [ref.span_id for ref in first.subagent_refs] == ["trajectory/subagent/0", None]
    # A second reference to the same subagent links to the span attached first.
    assert [ref.span_id for ref in second.subagent_refs] == ["trajectory/subagent/0"]
    assert spans["trajectory/step/2/tool/1"].children == []
    # The whole reference is kept as recorded under "ref", never encoded as a string.
    assert [(loss.code, loss.detail) for loss in trace.losses] == [
        ("unresolved_subagent_ref", {"ref": nobody}),
        ("unresolved_subagent_ref", {"ref": {"trajectory_path": "missing-subagent.json"}}),
    ]


def test_continued_trajectory_loads_with_a_loss():
    trace = load(FIXTURES / "continued.json")
    root = trace.root_spans[0]
    assert root.run.continued_trajectory_ref == "long-run-part-2.json"
    assert len(root.turns) == 2
    assert trace.losses == [
        Loss(
            code="continued_trajectory_not_followed",
            detail={"continued_trajectory_ref": "long-run-part-2.json"},
            span_id="trajectory",
        )
    ]


def test_parallel_and_unlinked_results_keep_source_order():
    trace = load(FIXTURES / "parallel_unlinked.json")
    spans = spans_by_id(trace)
    a, b, c = (spans[f"trajectory/step/2/tool/{index}"] for index in range(3))
    (turn,) = [turn for turn in trace.root_spans[0].turns if turn.index == 2]
    assert turn.tool_call_span_ids == [a.id, b.id, c.id]
    assert [result.content for result in a.tool_call.results] == ["A, part 1", UNSET, None]
    assert a.tool_call.result_count == 3
    assert a.output == ["A, part 1", None, None]
    assert b.output == "B" and b.tool_call.result_count == 1
    assert c.output is UNSET and c.tool_call.result_count == 0
    assert c.tool_call.result_id is None
    assert [result.content for result in turn.unlinked_results] == [
        "Progress: 50%",
        "Progress: 100%",
    ]
    # Unlinked results no longer become spans.
    assert {span.kind for span in spans.values()} == {
        SpanKind.AGENT,
        SpanKind.LLM,
        SpanKind.TOOL,
    }


def test_images_in_messages_and_results():
    trace = load(FIXTURES / "images.json")
    image_only, mixed, agent = trace.root_spans[0].turns
    assert image_only.content == [ImagePart(media_type="image/png", path="screenshots/1.png")]
    assert mixed.content == [
        TextPart(text="What is shown?"),
        ImagePart(media_type="image/webp", path="https://example.com/2.webp"),
        TextPart(text="Be brief."),
    ]
    tool = spans_by_id(trace)["trajectory/step/3/tool/0"]
    assert tool.tool_call.results[0].content == [
        {"type": "image", "media_type": "image/gif", "path": "shots/3.gif"}
    ]
    assert agent.content == "A chart."


def test_llm_call_count_variants():
    trace = load(FIXTURES / "llm_call_counts.json")
    spans = spans_by_id(trace)
    unrecorded, one, merged, dispatch = (spans[f"trajectory/step/{n}"] for n in (2, 3, 4, 5))
    assert unrecorded.kind is SpanKind.LLM
    assert unrecorded.model == "default-model"
    assert unrecorded.llm_call is UNSET
    assert unrecorded.token_counts == TokenCounts(
        input_tokens=10, cached_input_tokens=0, output_tokens=2
    )
    assert one.kind is SpanKind.LLM
    assert one.model == "step-model"
    # Partial accounting is kept as recorded rather than zero-filled.
    assert one.token_counts is None
    assert one.llm_call == LLMCallInfo(call_count=1, prompt_tokens=12)
    assert merged.kind is SpanKind.LLM
    assert merged.llm_call == LLMCallInfo(call_count=3)
    assert merged.cost_usd == 0.03
    assert dispatch.kind is SpanKind.CHAIN
    assert dispatch.model is None
    assert dispatch.llm_call == LLMCallInfo(call_count=0, reasoning_effort="low")
    assert dispatch.children[0].tool_call.initiator is Initiator.HARNESS
    turns = {turn.index: turn for turn in trace.root_spans[0].turns}
    assert turns[4].llm_span_ids == ["trajectory/step/4"]
    assert turns[5].llm_span_ids == []
    assert turns[5].tool_call_span_ids == ["trajectory/step/5/tool/0"]
    assert trace.losses == [
        Loss(
            code="llm_calls_aggregated",
            detail={"llm_call_count": 3},
            span_id="trajectory/step/4",
            turn_id="trajectory/turn/4",
        )
    ]


def test_error_flag_fixture():
    trace = load(FIXTURES / "error_flags.json")
    spans = spans_by_id(trace)
    results = [spans[f"trajectory/step/2/tool/{index}"].tool_call.results[0] for index in range(6)]
    assert [(result.is_error, result.attributes) for result in results] == [
        (None, {}),
        (True, {}),
        (False, {}),
        (True, {}),
        (None, {"is_error_status": "invalid"}),
        (None, {"is_error_status": "conflicting"}),
    ]
    errors = [spans[f"trajectory/step/2/tool/{index}"].error for index in range(6)]
    assert errors == [None, atif.TOOL_ERROR, None, atif.TOOL_ERROR, None, None]
    calls = to_tool_issue_trace(trace).calls
    assert [call.explicit_error for call in calls] == [None, True, False, True, None, None]


@pytest.mark.parametrize(
    "flags, is_error, attributes",
    [
        ({}, None, {}),
        ({"is_error": False}, False, {}),
        ({"isError": True, "is_error": True}, True, {}),
        ({"isError": True, "extra": {"isError": True, "k": 1}}, True, {}),
        ({"is_error": "yes"}, None, {"is_error_status": "invalid"}),
        ({"is_error": None}, None, {"is_error_status": "invalid"}),
        (
            {"is_error": True, "extra": {"is_error": False}},
            None,
            {"is_error_status": "conflicting"},
        ),
    ],
)
def test_root_error_flags(tmp_path, flags, is_error, attributes):
    source = with_agent_step(
        {
            "tool_calls": [{"tool_call_id": "a", "function_name": "read", "arguments": {}}],
            "observation": {"results": [{"source_call_id": "a", "content": "x", **flags}]},
        }
    )
    tool = spans_by_id(load_record(tmp_path, source))["trajectory/step/2/tool/0"]
    assert tool.tool_call.results == [
        ToolResult(
            content="x", is_error=is_error, extra=flags.get("extra", {}), attributes=attributes
        )
    ]
    assert (tool.error is not None) is (is_error is True)


def test_realistic_nested_trajectory_and_signal_readers():
    loader = ATIFTraceLoader(ATIFTraceConfig(path=FIXTURE))
    trace = next(iter(loader.load()))
    root = trace.root_spans[0]
    assert trace.id == "trace-1"
    assert [span.id for span in root.children] == [
        "trajectory/step/3",
        "trajectory/step/4",
        "trajectory/subagent/1",
    ]
    agent_step = root.children[0]
    assert agent_step.start_time == datetime(2026, 9, 1, 10, 0, 1, tzinfo=timezone.utc)
    assert agent_step.model == "test-model"
    search, delegate, missing = agent_step.children
    assert search.output is None
    assert missing.output is UNSET
    assert missing.tool_call is not None
    assert missing.tool_call.result_count == 0
    assert delegate.children[0].id == "trajectory/subagent/0"
    assert root.children[1].kind is SpanKind.CHAIN
    assert trace.aggregate.token_counts == TokenCounts(
        input_tokens=100, cached_input_tokens=20, output_tokens=15
    )

    # Each tool call carries the nearest preceding user text of its own agent.
    calls = to_tool_issue_trace(trace).calls
    assert [call.tool_name for call in calls] == ["search", "delegate", "lookup", "save"]
    assert [call.call_index for call in calls] == [0, 1, 2, 3]
    assert [call.prior_user_text for call in calls] == [
        "Find the report.",
        "Find the report.",
        "Find a source.",
        "Find the report.",
    ]
    steps = {
        step.source_pointer["span_id"]: step for step in to_anomaly_and_patterns_trace(trace).steps
    }
    assert steps["trajectory/step/3"].content == "Checking."
    # A dispatch step (llm_call_count 0) keeps its message.
    assert steps["trajectory/step/4"].content == "Done."
    # AGENT spans carry no text, so a subagent's answer never stands in for the trace's.
    assert steps["trajectory"].content == ""
    assert steps["trajectory/subagent/0/step/2"].content == "Found it."
    assert loader.describe()["call_count"] == 4
    # Coverage counts the prior user text the loader records on each call.
    presence = corpus_coverage(loader)["field_presence"]
    assert presence["prior_user_text"] >= 4
    assert presence["logical_case_id"] == len(list(loader.load()))


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda t: t.update(schema_version="ATIF-v1.9"), "schema_version"),
        (lambda t: t.pop("session_id"), "stable trace identity"),
        (lambda t: t.pop("agent"), "agent"),
        (lambda t: t["agent"].pop("version"), "version"),
        (lambda t: t["agent"].update(name=123), "name"),
        (lambda t: t.update(unknown="x"), "unknown"),
        (lambda t: t.update(steps=[]), "steps"),
        (lambda t: t["steps"][0].update(step_id=True), "step_id"),
        (lambda t: t["steps"][0].update(message={}), "message"),
        (lambda t: t["steps"][0].update(observation={"results": {}}), "results"),
        (lambda t: t["steps"][0].update(step_id=2), "sequential"),
        (lambda t: t["steps"].append(t["steps"][0]), "sequential"),
        (lambda t: t["steps"][0].update(source="tool"), "source"),
        (lambda t: t["steps"][0].update(timestamp="not-a-date"), "isoformat"),
        (lambda t: t["steps"][0].update(tool_calls=[]), "require source"),
        (
            lambda t: t["steps"][0].update(message=[{"type": "text", "source": None}]),
            "text part",
        ),
        (
            lambda t: t["steps"].append(
                {
                    "step_id": 2,
                    "source": "agent",
                    "message": "",
                    "llm_call_count": 0,
                    "reasoning_content": "x",
                }
            ),
            "llm_call_count=0",
        ),
        (
            lambda t: t["steps"].append(
                {
                    "step_id": 2,
                    "source": "agent",
                    "message": "",
                    "tool_calls": [{"tool_call_id": "a", "function_name": "f"}],
                }
            ),
            "arguments",
        ),
        (
            lambda t: t["steps"][0].update(
                observation={"results": [{"subagent_trajectory_ref": [{"session_id": "s"}]}]}
            ),
            "trajectory_id or trajectory_path",
        ),
        (
            lambda t: t.update(final_metrics={"total_prompt_tokens": 2, "total_cached_tokens": 3}),
            "exceeds",
        ),
        (lambda t: t.update(final_metrics={"total_prompt_tokens": -1}), "non-negative"),
        (lambda t: t.update(final_metrics={"total_cost_usd": -1}), "greater than or equal"),
        (lambda t: t.update(final_metrics={"total_cost_usd": "1"}), "total_cost_usd"),
    ],
)
def test_invalid_source_has_line_context(tmp_path, mutation, match):
    source = minimal()
    mutation(source)
    with pytest.raises(ATIFTraceLoadError, match=match) as caught:
        loader_for(tmp_path, source).load()
    assert "traces.jsonl:2:" in str(caught.value)


@pytest.mark.parametrize(
    "case, match",
    [
        ("unknown_call", "unknown source_call_id"),
        ("duplicate_subagent", "unique trajectory_id"),
        ("anonymous_subagent", "unique trajectory_id"),
    ],
)
def test_invalid_relationships(tmp_path, case, match):
    source = json.loads(FIXTURE.read_text())
    step = source["steps"][2]
    if case == "unknown_call":
        step["observation"]["results"][1]["source_call_id"] = "unknown"
    elif case == "duplicate_subagent":
        source["subagent_trajectories"].append(source["subagent_trajectories"][0])
    else:
        source["subagent_trajectories"][1].pop("trajectory_id")
    with pytest.raises(ATIFTraceLoadError, match=match):
        loader_for(tmp_path, source).load()


def test_duplicate_tool_call_ids_are_a_loss(tmp_path):
    source = with_agent_step(
        {
            "tool_calls": [
                {"tool_call_id": "a", "function_name": "read", "arguments": {}},
                {"tool_call_id": "a", "function_name": "read", "arguments": {"again": True}},
            ],
            "observation": {"results": [{"source_call_id": "a", "content": "x"}]},
        }
    )
    trace = load_record(tmp_path, source)
    spans = spans_by_id(trace)
    assert spans["trajectory/step/2/tool/0"].tool_call.result_count == 1
    assert spans["trajectory/step/2/tool/1"].tool_call.result_count == 0
    assert [(loss.code, loss.detail, loss.span_id) for loss in trace.losses] == [
        ("duplicate_tool_call_id", {"tool_call_id": "a"}, "trajectory/step/2/tool/1")
    ]


def test_missing_and_null_result_content_are_not_conflated(tmp_path):
    source = with_agent_step(
        {
            "tool_calls": [
                {"tool_call_id": "a", "function_name": "read", "arguments": {}},
                {"tool_call_id": "b", "function_name": "read", "arguments": {}},
            ],
            "observation": {
                "results": [{"source_call_id": "a", "content": None}, {"source_call_id": "b"}]
            },
        }
    )
    spans = spans_by_id(load_record(tmp_path, source))
    a, b = spans["trajectory/step/2/tool/0"], spans["trajectory/step/2/tool/1"]
    assert a.output is None and a.tool_call.results[0].content is None
    assert b.output is UNSET and b.tool_call.results[0].content is UNSET
    assert a.tool_call.result_count == b.tool_call.result_count == 1


def test_naive_timestamps_stay_on_the_turn(tmp_path):
    source = with_agent_step({"timestamp": "2026-09-01T10:00:00"})
    trace = load_record(tmp_path, source)
    assert trace.root_spans[0].turns[1].timestamp == "2026-09-01T10:00:00"
    assert spans_by_id(trace)["trajectory/step/2"].start_time is None


def test_loads_all_records_and_rejects_duplicates(tmp_path):
    source = minimal()
    with pytest.raises(ATIFTraceLoadError, match="duplicate trace id"):
        loader_for(tmp_path, source, source).load()
    second = {**source, "trajectory_id": "another"}
    loader = loader_for(tmp_path, source, second)
    assert [trace.id for trace in loader.load()] == ["run", "another"]
    assert loader.describe()["distinct_logical_cases"] == 1
    assert {to_tool_issue_trace(trace).logical_case_id for trace in loader.load()} == {"run"}


@pytest.mark.parametrize(
    "raw, match", [(" \n\t", "no ATIF trajectories"), ("{", "invalid ATIF"), ("[]", "invalid ATIF")]
)
def test_malformed_and_empty_files(tmp_path, raw, match):
    path = tmp_path / "traces.jsonl"
    path.write_text(raw)
    with pytest.raises(ATIFTraceLoadError, match=match):
        ATIFTraceLoader(ATIFTraceConfig(path=path)).load()


def test_missing_file(tmp_path):
    config = ATIFTraceConfig(path=tmp_path / "absent")
    with pytest.raises(ATIFTraceLoadError, match="absent"):
        ATIFTraceLoader(config).load()


def test_failed_load_can_be_retried_and_success_is_cached(tmp_path):
    path = tmp_path / "traces.jsonl"
    loader = ATIFTraceLoader(ATIFTraceConfig(path=path))
    path.write_text("{malformed")
    with pytest.raises(ATIFTraceLoadError):
        loader.describe()
    path.write_text(json.dumps(minimal()))
    description = loader.describe()
    snapshot = loader.load()
    path.unlink()
    assert loader.load() is snapshot
    assert loader.describe() is description
    assert len(snapshot) == 1


def test_yaml_cli_and_exclusive_source_selection(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        f"trace:\n  atif:\n    path: {FIXTURE}\ncompass_signals:\n  tool_issues: {{}}\n"
    )
    config = RunConfig(_cli_parse_args=["--config", str(path)])
    loader = cli._configured_trace_loader(config.trace)
    assert isinstance(loader, ATIFTraceLoader)
    assert loader.config.path == FIXTURE
    with pytest.raises(ValidationError, match="max_traces is not supported by the ATIF loader"):
        RunConfig(_cli_parse_args=["--config", str(path), "--trace.max-traces", "2"])
    with pytest.raises(ValidationError, match="exactly one"):
        TraceConfig(atif={"path": FIXTURE}, filesystem={"path": FIXTURE})


def test_complete_cli_runs_real_compass_signal(tmp_path, monkeypatch, capsys, select_signals):
    compilation = SimpleNamespace(compile_insights=AsyncMock(return_value=[]))
    monkeypatch.setenv("COMPASS_API_KEY", "not-real")
    monkeypatch.setattr(cli, "_build_llm", lambda config, api_key: FakeLLMClient())
    monkeypatch.setattr(cli, "InsightCompilation", lambda llm: compilation)
    output = tmp_path / "insights.yml"
    assert (
        cli.main(
            [
                "--trace.atif.path",
                str(FIXTURE),
                "--compass-signals",
                json.dumps(select_signals(tool_issues={"include_audit_problems": True})),
                "--output-path",
                str(output),
            ]
        )
        == cli.EXIT_OK
    )
    evidence, snapshot, existing, run_timestamp = compilation.compile_insights.await_args.args
    assert isinstance(run_timestamp, datetime)
    assert len(snapshot) == 1
    assert existing == []
    assert evidence[0].signal_name == "tool-issues"
    artifacts = evidence[0].artifacts
    assert artifacts.catalog_coverage["missing_tool_result"] == 1
    assert any(finding["call_id"] == "missing" for finding in artifacts.findings)
    assert not output.exists()
    assert capsys.readouterr().out == "[]\n"


def test_tool_definitions_and_catalog(tmp_path):
    read_schema = {"type": "object", "properties": {"path": {"type": "string"}}}
    write_schema = {"type": "object"}
    items = [
        {
            "name": "read",
            "description": "Read a file.",
            "inputSchema": read_schema,
            "extra": {"tool_scope": {"server": "fs"}},
        },
        {
            "type": "function",
            "function": {"name": "write", "description": "Write.", "parameters": write_schema},
        },
        {"description": "nameless"},
    ]
    assert tool_definitions_from_atif([*items, "not a definition"]) == [
        ToolDefinition(
            name="read",
            server="fs",
            description="Read a file.",
            input_schema=read_schema,
            raw=items[0],
        ),
        ToolDefinition(name="write", description="Write.", input_schema=write_schema, raw=items[1]),
    ]
    assert tool_definitions_from_atif(None) == []
    source = minimal()
    source["agent"]["tool_definitions"] = items
    trace = load_record(tmp_path, source)
    assert trace.root_spans[0].tool_definitions == tool_definitions_from_atif(items)
    assert tool_catalog(trace) == {"read": read_schema, "write": write_schema}

    source["agent"]["tool_definitions"] = [
        {"name": "read", "parameters": read_schema},
        {"name": "read", "parameters": write_schema},
    ]
    assert tool_catalog(load_record(tmp_path, source)) is None

    trace = load_record(tmp_path, minimal())
    assert trace.root_spans[0].tool_definitions is UNSET
    assert tool_catalog(trace) is None


def test_subagent_declarations_stay_on_the_subagent(tmp_path):
    source = json.loads(FIXTURE.read_text())
    definitions = [{"name": "lookup", "parameters": {"type": "object"}}]
    source["subagent_trajectories"][0]["agent"]["tool_definitions"] = definitions
    trace = load_record(tmp_path, source)
    spans = spans_by_id(trace)
    assert spans["trajectory"].tool_definitions is UNSET
    assert spans["trajectory/subagent/0"].tool_definitions == tool_definitions_from_atif(
        definitions
    )
    assert tool_catalog(trace) == {"lookup": {"type": "object"}}


def test_every_fixture_round_trips_through_the_canonical_json(tmp_path):
    for path in sorted(FIXTURES.glob("*.json")):
        trace = load(path)
        assert Trace.model_validate_json(trace.model_dump_json()) == trace
