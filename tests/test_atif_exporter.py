# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from trace_ingest.exporters import atif
from trace_ingest.exporters.atif import EXPORTER, ATIFExportError
from trace_ingest.loaders.atif import ATIFTraceConfig, ATIFTraceLoader
from trace_ingest.loaders.mlflow import MLflowFileTraceConfig, MLflowFileTraceLoader
from trace_ingest.models import (
    Span,
    SpanKind,
    TokenCounts,
    ToolCall,
    ToolDefinition,
    ToolResult,
    Trace,
    TraceAggregate,
)

DATA = Path(__file__).parent / "data"
START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def to_atif(trace, **options) -> Any:
    return atif.to_atif(trace, **options)


def tool(span_id, *results, error=None, call_id=None, tool_input=None):
    return Span(
        id=span_id,
        kind=SpanKind.TOOL,
        start_time=START,
        tool_name="lookup",
        input={"q": "x"} if tool_input is None else tool_input,
        error=error,
        tool_call=ToolCall(call_id=call_id, result_count=len(results), results=list(results)),
    )


def trace_of(*roots, **attributes):
    return Trace(
        id="t-1", root_spans=list(roots), aggregate=TraceAggregate(), attributes=attributes
    )


def agent(*children, user_input="Find x.", output=None, **fields):
    span = Span(id="root", kind=SpanKind.AGENT, input=user_input, children=list(children), **fields)
    if output is not None:
        span.output = output
    return span


def test_mlflow_trace_exports_sequential_steps_with_compass_metadata():
    trace = next(
        iter(
            MLflowFileTraceLoader(MLflowFileTraceConfig(path=DATA / "mlflow_trace_v3.json")).load()
        )
    )
    exported = to_atif(trace, agent_name="x")
    assert [step["step_id"] for step in exported["steps"]] == list(
        range(1, len(exported["steps"]) + 1)
    )
    assert any(step["source"] == "user" for step in exported["steps"])
    assert exported["extra"]["compass"]["exporter"] == EXPORTER
    assert exported["extra"]["compass"]["trace_schema_version"] == "2"
    assert exported["agent"] == {"name": "x", "version": "unknown"}
    assert exported["trajectory_id"] == exported["session_id"] == trace.id


def test_non_atif_trace_requires_agent_name():
    with pytest.raises(ATIFExportError, match="agent_name is required for non-ATIF traces"):
        to_atif(trace_of(agent()))


@pytest.mark.parametrize("user_input", ["__unset__", None])
def test_missing_instruction_is_rejected(user_input):
    root = Span(id="root", kind=SpanKind.AGENT)
    if user_input is None:
        root.input = None
    with pytest.raises(ATIFExportError, match="trace has no recoverable user instruction"):
        to_atif(trace_of(root), agent_name="x")


def test_message_list_instruction_marks_prior_context_as_copied():
    messages = [
        {"role": "developer", "content": "Be brief."},
        {"role": "human", "content": "Earlier question."},
        {"role": "assistant", "content": "Earlier answer."},
        {"source": "user", "text": "Current question."},
        {"role": "critic", "message": "Unknown role."},
    ]
    steps = to_atif(trace_of(agent(user_input={"messages": messages})), agent_name="x")["steps"]
    assert [(s["source"], s["message"], s["is_copied_context"]) for s in steps] == [
        ("system", "Be brief.", True),
        ("user", "Earlier question.", True),
        ("agent", "Earlier answer.", True),
        ("user", "Current question.", False),
        ("system", "Unknown role.", True),
    ]


def test_message_list_without_user_is_rejected():
    messages = [{"role": "system", "content": "Only system."}]
    with pytest.raises(ATIFExportError, match="no recoverable user instruction"):
        to_atif(trace_of(agent(user_input=messages)), agent_name="x")


def test_structured_input_becomes_one_user_step():
    steps = to_atif(trace_of(agent(user_input={"b": 1, "a": 2})), agent_name="x")["steps"]
    assert steps == [{"step_id": 1, "source": "user", "message": '{"a": 2, "b": 1}'}]


def test_tool_results_carry_error_flags_and_missing_results_are_lossy():
    root = agent(
        tool("ok", ToolResult(content="fine", is_error=False), call_id="c1"),
        tool("failed", ToolResult(content={"e": 1}), error="boom"),
        tool("unknown", ToolResult(content="?"), tool_input="raw"),
        tool("missing", tool_input=None),
    )
    root.children[-1].input = None
    exported = to_atif(trace_of(root), agent_name="x")
    steps = exported["steps"][1:]
    assert [step["tool_calls"][0]["tool_call_id"] for step in steps] == [
        "c1",
        "failed",
        "unknown",
        "missing",
    ]
    assert steps[0]["observation"]["results"] == [
        {"source_call_id": "c1", "content": "fine", "extra": {"is_error": False}}
    ]
    assert steps[1]["observation"]["results"] == [
        {"source_call_id": "failed", "content": '{"e": 1}', "extra": {"is_error": True}}
    ]
    assert steps[2]["observation"]["results"] == [{"source_call_id": "unknown", "content": "?"}]
    assert steps[2]["tool_calls"][0]["arguments"] == {"value": "raw"}
    assert "observation" not in steps[3]
    assert steps[3]["tool_calls"][0]["arguments"] == {}
    assert all(step["llm_call_count"] == 0 and step["message"] == "" for step in steps)
    assert exported["extra"]["compass"]["loss_codes"] == ["tool_result_missing"]


def test_llm_steps_carry_model_and_metrics():
    llm = Span(
        id="llm",
        kind=SpanKind.LLM,
        model="m-1",
        output={"choices": [{"message": {"content": "Answer."}}]},
        token_counts=TokenCounts(input_tokens=5, cached_input_tokens=2, output_tokens=3),
        cost_usd=0.5,
    )
    exported = to_atif(trace_of(agent(llm, output="Answer.")), agent_name="x")
    assert exported["agent"]["model_name"] == "m-1"
    assert exported["steps"][1] == {
        "step_id": 2,
        "source": "agent",
        "message": "Answer.",
        "llm_call_count": 1,
        "model_name": "m-1",
        "metrics": {
            "prompt_tokens": 7,
            "cached_tokens": 2,
            "completion_tokens": 3,
            "cost_usd": 0.5,
        },
    }
    assert len(exported["steps"]) == 2


def test_linearization_records_losses_and_appends_final_answer():
    nested = Span(
        id="planner",
        kind=SpanKind.CHAIN,
        children=[
            Span(id="search", kind=SpanKind.RETRIEVER),
            tool("deep", ToolResult(content="r")),
        ],
    )
    leaf = Span(id="note", kind=SpanKind.CHAIN, output="Leaf output.")
    second_root = Span(id="other", kind=SpanKind.GUARDRAIL)
    exported = to_atif(
        trace_of(
            agent(nested, leaf, output={"answer": "Final."}), second_root, logical_case_id="case"
        ),
        agent_name="x",
        agent_version="2",
    )
    assert [step["message"] for step in exported["steps"]] == [
        "Find x.",
        "",
        "Leaf output.",
        "Final.",
    ]
    assert exported["session_id"] == "case"
    assert exported["agent"]["version"] == "2"
    assert exported["extra"]["compass"]["loss_codes"] == [
        "guardrail_span_not_emitted",
        "multiple_root_spans_linearized",
        "orchestration_span_not_emitted",
        "retriever_span_not_emitted",
        "span_tree_linearized",
    ]


def test_root_agent_tool_definitions_are_exported_raw():
    raw = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
    definitions = [ToolDefinition(name="lookup", input_schema={"type": "object"}, raw=raw)]
    exported = to_atif(trace_of(agent(tool_definitions=definitions)), agent_name="x")
    assert exported["agent"]["tool_definitions"] == [raw]


def test_generic_export_reloads_through_the_atif_loader(tmp_path):
    root = agent(
        tool("failed", ToolResult(content="bad", is_error=True), call_id="c1"),
        tool("ok", ToolResult(content="good", is_error=False), call_id="c2"),
        output="Done.",
    )
    root.start_time = START
    path = tmp_path / "exported.json"
    path.write_text(json.dumps(to_atif(trace_of(root), agent_name="x"), indent=2))
    reloaded = next(iter(ATIFTraceLoader(ATIFTraceConfig(path=path)).load()))
    tools = [
        span
        for step in reloaded.root_spans[0].children
        for span in step.children
        if span.kind is SpanKind.TOOL and span.tool_call is not None
    ]
    assert [
        (span.error, span.tool_call.results[0].is_error) for span in tools if span.tool_call
    ] == [
        ("ATIF tool result reported is_error", True),
        (None, False),
    ]


def test_atif_origin_trace_ignores_agent_name(tmp_path):
    record = {
        "schema_version": "ATIF-v1.7",
        "session_id": "s",
        "agent": {"name": "original", "version": "1"},
        "steps": [{"step_id": 1, "source": "user", "message": "Hi."}],
    }
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(record))
    trace = next(iter(ATIFTraceLoader(ATIFTraceConfig(path=path)).load()))
    assert to_atif(trace) == record
    assert to_atif(trace, agent_name="other") == record


@pytest.mark.parametrize("user_input", ["", "   ", [], {}])
def test_empty_instruction_is_rejected(user_input):
    with pytest.raises(ATIFExportError, match="no recoverable user instruction"):
        to_atif(trace_of(agent(user_input=user_input)), agent_name="x")


def test_responses_output_items_keep_only_assistant_content():
    llm = Span(
        id="llm",
        kind=SpanKind.LLM,
        output=[
            {"type": "reasoning", "encrypted_content": "opaque", "summary": []},
            {"type": "function_call", "call_id": "c1", "name": "lookup", "arguments": "{}"},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Done."}],
            },
        ],
    )
    exported = to_atif(trace_of(agent(llm, output="Done.")), agent_name="x")
    assert [step["message"] for step in exported["steps"]] == ["Find x.", "Done."]
    assert exported["extra"]["compass"]["loss_codes"] == ["reasoning_not_exported"]


def test_image_parts_become_atif_images_when_their_media_type_is_recorded():
    typed = "data:image/png;base64,AAAA"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe these."},
                {"type": "input_image", "image_url": typed},
                {"type": "input_image", "image_url": "https://example.test/photo"},
            ],
        }
    ]
    failed = tool("t", ToolResult(content=[{"type": "input_image", "image_url": "x.webp"}]))
    exported = to_atif(trace_of(agent(failed, user_input=messages)), agent_name="x")
    assert exported["steps"][0]["message"] == [
        {"type": "text", "text": "Describe these."},
        {"type": "image", "source": {"media_type": "image/png", "path": typed}},
        {"type": "text", "text": "[image omitted: media type not recorded]"},
    ]
    assert exported["steps"][1]["observation"]["results"][0]["content"] == [
        {"type": "image", "source": {"media_type": "image/webp", "path": "x.webp"}}
    ]
    assert exported["extra"]["compass"]["loss_codes"] == ["image_without_media_type_omitted"]
