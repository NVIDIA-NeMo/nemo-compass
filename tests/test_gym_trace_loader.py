# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic records from Gym's TrajectoryRecord schema at 399e6783e0e8."""

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from nooa.unifiedllm import FakeLLMClient
from pydantic import ValidationError

import compass.cli.main as cli
from compass.compass_signals.tool_issues.signal import to_tool_issue_trace
from compass.config import RunConfig, TraceConfig
from trace_ingest.catalog import tool_catalog
from trace_ingest.loaders import GymTraceConfig, GymTraceLoader, GymTraceLoadError
from trace_ingest.models import UNSET, SpanKind, TokenCounts


def rollout():
    return {
        "reward": 0.0,
        "reward_components": {"correctness": 0.0},
        "ng_trajectory": {
            "schema_version": "1.0",
            "task_id": "task-1",
            "rollout_id": "rollout-1",
            "invocations": [
                {
                    "invocation_id": "root",
                    "status": "completed",
                    "model_calls": [{"model_call_id": "m1"}],
                    "conversation": [
                        {"role": "user", "content": "Find the fixture."},
                        {
                            "type": "function_call",
                            "call_id": "c1",
                            "name": "lookup",
                            "arguments": '{"id": 42}',
                        },
                        {"type": "function_call_output", "call_id": "c1", "output": None},
                        {"role": "assistant", "content": [{"type": "output_text", "text": "Done"}]},
                    ],
                },
                {
                    "invocation_id": "child",
                    "parent_invocation_id": "root",
                    "conversation": [],
                    "model_calls": [],
                    "status": "incomplete",
                },
            ],
            "model_calls": [
                {
                    "model_call_id": "m1",
                    "started_at": 100.0,
                    "completed_at": 101.5,
                    "request": {"input": [{"role": "user", "content": "Find the fixture."}]},
                    "response": {"id": "r1", "output": []},
                    "response_metadata": {
                        "model": "test-model",
                        "response_id": "r1",
                        "model_ref": {"name": "model"},
                        "response_status": "completed",
                    },
                    "token_stats": {
                        "prompt_tokens": 20,
                        "cached_tokens": 5,
                        "completion_tokens": 3,
                    },
                }
            ],
            "tool_calls": [
                {
                    "invocation_id": "root",
                    "tool_call_id": "c1",
                    "tool_name": "lookup",
                    "started_at": 102.0,
                    "completed_at": 103.0,
                    "status": "failed",
                    "error_type": "LookupError",
                    "output": None,
                }
            ],
            "turns": [
                {
                    "invocation_id": "root",
                    "task_id": "task-1",
                    "rollout_id": "rollout-1",
                    "turn_no": 1,
                    "timestamp": 101.5,
                    "step_count": 0,
                    "resolved": None,
                    "question": "Find the fixture.",
                    "answer": "Done",
                    "model_calls": [{"model_call_id": "m1"}],
                }
            ],
            "gaps": [{"code": "transport_retry_visibility_unavailable"}],
        },
    }


def loader_for(tmp_path, *records, format="jsonl"):
    path = tmp_path / "gym.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, indent=2 if format == "json" else None) for row in records)
    )
    return GymTraceLoader(GymTraceConfig(path=path, format=format))


def test_native_gym_mapping_and_cached_description(tmp_path):
    source = rollout()
    loader = loader_for(tmp_path, source)
    snapshot = loader.load()
    trace = next(iter(snapshot))
    assert trace.id == "rollout-1"
    assert trace.attributes["logical_case_id"] == "task-1"
    assert trace.attributes["gym"] == source
    assert trace.evaluator_results == {"gym.reward": 0, "gym.reward_components": {"correctness": 0}}
    assert trace.aggregate.token_counts is None  # no invented rollout accounting
    agent = trace.root_spans[0]
    assert agent.kind is SpanKind.AGENT
    assert [s.kind for s in agent.children] == [
        SpanKind.LLM,
        SpanKind.CHAIN,
        SpanKind.TOOL,
        SpanKind.AGENT,
    ]
    model, turn, tool, child = agent.children
    assert model.start_time == datetime.fromtimestamp(100, tz=timezone.utc)
    assert model.token_counts == TokenCounts(
        input_tokens=15, cached_input_tokens=5, output_tokens=3
    )
    assert model.model == "test-model"
    assert turn.output == "Done" and turn.token_counts is None
    assert tool.input == {"id": 42} and tool.output is None and tool.error == "LookupError"
    assert tool.tool_call.result_id == "c1" and tool.tool_call.result_count == 1
    assert tool.tool_call.prior_user_text == "Find the fixture."
    assert child.error is None  # incomplete is not an asserted failure
    calls = to_tool_issue_trace(trace).calls
    assert len(calls) == 1 and calls[0].tool_name == "lookup"
    assert loader.describe() == {
        "source": f"gym:{loader.config.path.resolve()}",
        "path": str(loader.config.path.resolve()),
        "format": "jsonl",
        "trace_count": 1,
        "call_count": 1,
        "distinct_logical_cases": 1,
        "gap_count": 1,
    }
    loader.config.path.unlink()
    assert loader.load() is snapshot


def test_corpus_records_stay_separate_and_share_logical_task(tmp_path):
    first, second = rollout(), rollout()
    second["ng_trajectory"]["rollout_id"] = "rollout-2"
    second["ng_trajectory"]["turns"][0]["rollout_id"] = "rollout-2"
    loader = loader_for(tmp_path, first, second)
    assert [trace.id for trace in loader.load()] == ["rollout-1", "rollout-2"]
    assert loader.describe()["distinct_logical_cases"] == 1
    assert len(loader_for(tmp_path, first, format="json").load()) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "version",
        "cycle",
        "invocation",
        "model",
        "tool",
        "turn",
        "identity",
        "result",
        "arguments",
        "time",
        "tokens",
        "ownership",
    ],
)
def test_invalid_or_ambiguous_evidence_rejected(tmp_path, mutation):
    source = rollout()
    t = source["ng_trajectory"]
    if mutation == "version":
        t["schema_version"] = "2.0"
    elif mutation == "cycle":
        t["invocations"][0]["parent_invocation_id"] = "child"
    elif mutation == "invocation":
        t["invocations"].append(copy.deepcopy(t["invocations"][0]))
    elif mutation == "model":
        t["model_calls"].append(copy.deepcopy(t["model_calls"][0]))
    elif mutation == "tool":
        t["tool_calls"].append(copy.deepcopy(t["tool_calls"][0]))
    elif mutation == "turn":
        t["turns"].append(copy.deepcopy(t["turns"][0]))
    elif mutation == "identity":
        t["turns"][0]["task_id"] = "other"
    elif mutation == "result":
        t["invocations"][0]["conversation"][2]["call_id"] = "wrong"
    elif mutation == "arguments":
        t["invocations"][0]["conversation"][1]["arguments"] = '{"x":1,"x":2}'
    elif mutation == "time":
        t["tool_calls"][0]["completed_at"] = 0
    elif mutation == "tokens":
        t["model_calls"][0]["token_stats"]["cached_tokens"] = 21
    elif mutation == "ownership":
        t["invocations"][1]["model_calls"] = [{"model_call_id": "m1"}]
    with pytest.raises(GymTraceLoadError):
        loader_for(tmp_path, source).load()


def test_missing_evidence_is_preserved_without_inventing_relationships(tmp_path):
    source = rollout()
    t = source["ng_trajectory"]
    t["invocations"][0]["model_calls"] = []
    t["invocations"][0]["conversation"].pop(2)
    t["invocations"][1]["parent_invocation_id"] = "unobserved"
    t["model_calls"][0].pop("request")
    t["model_calls"][0]["response"] = None
    t["model_calls"][0]["token_stats"]["cached_tokens"] = None
    t["tool_calls"][0].pop("output")
    trace = next(iter(loader_for(tmp_path, source).load()))
    model = next(s for s in trace.root_spans if s.kind is SpanKind.LLM)
    assert model.input is UNSET and model.output is None and model.token_counts is None
    tool = next(
        s
        for s in trace.root_spans
        if s.kind is SpanKind.AGENT and s.attributes["gym"]["invocation_id"] == "root"
    ).children[1]
    assert tool.kind is SpanKind.TOOL and tool.output is UNSET and tool.tool_call.result_count == 0
    assert {g["code"] for g in trace.attributes["gym_loader_gaps"]} == {
        "model_call_owner_unrecorded",
        "missing_parent_invocation",
    }


@pytest.mark.parametrize(
    "raw", ["", "[]", "{", '{"ng_trajectory": {}, "ng_trajectory": {}}', '{"x": NaN}']
)
def test_malformed_empty_and_nonfinite_json(tmp_path, raw):
    path = tmp_path / "gym.jsonl"
    path.write_text(raw)
    with pytest.raises(GymTraceLoadError):
        GymTraceLoader(GymTraceConfig(path=path)).load()


def test_legacy_records_missing_files_and_duplicate_rollouts(tmp_path):
    with pytest.raises(GymTraceLoadError, match="ng_trajectory"):
        loader_for(tmp_path, {"response": {"output": []}}).load()
    with pytest.raises(GymTraceLoadError):
        GymTraceLoader(GymTraceConfig(path=tmp_path / "missing")).load()
    with pytest.raises(GymTraceLoadError):
        loader_for(tmp_path, rollout(), rollout()).load()


def test_explicit_cli_source_selection(tmp_path):
    path = loader_for(tmp_path, rollout()).config.path
    config = RunConfig(_cli_parse_args=["--trace.gym.path", str(path)])
    assert isinstance(cli._configured_trace_loader(config.trace), GymTraceLoader)
    with pytest.raises(ValidationError, match="exactly one"):
        TraceConfig(gym={"path": path}, atif={"path": path})
    with pytest.raises(ValidationError, match="max_traces"):
        TraceConfig(gym={"path": path}, max_traces=2)


def test_complete_cli_runs_real_tool_evidence(tmp_path, monkeypatch, capsys, select_signals):
    path = loader_for(tmp_path, rollout()).config.path
    compilation = SimpleNamespace(compile_insights=AsyncMock(return_value=[]))
    monkeypatch.setenv("COMPASS_API_KEY", "not-real")
    monkeypatch.setattr(cli, "_build_llm", lambda config, api_key: FakeLLMClient())
    monkeypatch.setattr(cli, "InsightCompilation", lambda llm: compilation)
    output = tmp_path / "insights.yml"
    assert (
        cli.main(
            [
                "--trace.gym.path",
                str(path),
                "--compass-signals",
                json.dumps(select_signals(tool_issues={"include_audit_problems": True})),
                "--output-path",
                str(output),
            ]
        )
        == cli.EXIT_OK
    )
    assert not output.exists()
    assert compilation.compile_insights.await_count == 1
    evidence, snapshot, existing, run_timestamp = compilation.compile_insights.await_args.args
    assert isinstance(run_timestamp, datetime)
    assert len(snapshot) == 1 and existing == []
    assert evidence[0].signal_name == "tool-issues"
    assert evidence[0].artifacts.findings
    assert capsys.readouterr().out == "[]\n"


def test_model_reference_pair_and_ambiguity(tmp_path):
    source = rollout()
    trajectory = source["ng_trajectory"]
    trajectory["invocations"][0]["model_calls"] = [
        {"response_id": "r1", "model_ref": {"name": "model"}}
    ]
    trace = next(iter(loader_for(tmp_path, source).load()))
    assert trace.root_spans[0].children[0].kind is SpanKind.LLM
    other = copy.deepcopy(trajectory["model_calls"][0])
    other["model_call_id"] = "different"
    trajectory["model_calls"].append(other)
    with pytest.raises(GymTraceLoadError):
        loader_for(tmp_path, source).load()


def test_unresolved_references_and_media_are_not_fetched(tmp_path):
    source = rollout()
    source["atif_conversion"] = {"source_trajectory_paths": ["/do/not/read"]}
    root = source["ng_trajectory"]["invocations"][0]
    root["model_calls"].append({"model_call_id": "not-captured"})
    root["conversation"][0]["content"] = [
        {"type": "input_image", "image_url": "https://invalid.invalid/do-not-fetch"}
    ]
    trace = next(iter(loader_for(tmp_path, source).load()))
    assert trace.attributes["gym"] == source
    assert any(
        gap["code"] == "model_reference_unresolved" for gap in trace.attributes["gym_loader_gaps"]
    )


def test_unknown_conversation_tool_is_not_silently_dropped(tmp_path):
    source = rollout()
    source["ng_trajectory"]["invocations"][0]["conversation"].append({"type": "computer_call"})
    with pytest.raises(GymTraceLoadError, match="unsupported Gym conversation item"):
        loader_for(tmp_path, source).load()


def test_enriched_observation_can_supply_result_without_conversation(tmp_path):
    source = rollout()
    source["ng_trajectory"]["invocations"][0]["conversation"] = []
    source["ng_trajectory"]["tool_calls"][0]["output"] = "recorded output"
    trace = next(iter(loader_for(tmp_path, source).load()))
    tool = next(span for span in trace.root_spans[0].children if span.kind is SpanKind.TOOL)
    assert tool.input is UNSET
    assert tool.output == "recorded output"
    assert tool.tool_call.result_id == "c1" and tool.tool_call.result_count == 1


@pytest.mark.parametrize("producer", ["local", "sandboxed"])
@pytest.mark.parametrize("case", ["parallel", "decision", "failed_call"])
def test_upstream_opencode_producer_and_collector_fixtures(producer, case):
    path = Path(__file__).parent / "fixtures/gym_native" / f"opencode-{producer}-{case}.json"
    record = json.loads(path.read_text())
    source = record["ng_trajectory"]
    loader = GymTraceLoader(GymTraceConfig(path=path, format="json"))
    trace = next(iter(loader.load()))
    assert trace.id == source["rollout_id"] == "0-0"
    assert trace.attributes["logical_case_id"] == source["task_id"] == "0"
    assert trace.attributes["gym"] == record
    assert trace.attributes["gym_loader_gaps"] == []
    assert loader.describe()["gap_count"] == len(source["gaps"])
    root = trace.root_spans[0]
    assert isinstance(root.attributes["gym"], dict)
    assert root.attributes["gym"]["invocation_id"] == "root"
    model = next(span for span in root.children if span.kind is SpanKind.LLM)
    assert isinstance(model.attributes["gym"], dict)
    assert model.attributes["gym"]["model_call_id"] == "captured"
    assert model.input is None and model.output is None
    assert model.token_counts is None  # Upstream leaves cached_tokens unknown.
    assert model.error == ("HTTP 503" if case == "failed_call" else None)
    assert trace.aggregate.token_counts is None
    if case == "parallel":
        child = next(span for span in root.children if span.kind is SpanKind.AGENT)
        assert isinstance(child.attributes["gym"], dict)
        assert child.attributes["gym"]["invocation_id"] == "child"
        assert child.attributes["gym"]["spawned_by_tool_call_id"] == "task-1"
        tools = [span for span in root.children if span.kind is SpanKind.TOOL]
        assert [span.tool_name for span in tools] == ["task", "bash"]
        first, second = tools
        assert first.input == {"prompt": "inspect"}
        assert first.output == "done"
        assert second.input == {"command": "pwd"}
        assert second.output == "[Old tool result content cleared]"
        assert first.start_time is not None and first.end_time is not None
        assert second.start_time is not None and second.end_time is not None
        assert first.start_time < second.start_time < second.end_time < first.end_time
        assert (first.end_time - first.start_time).total_seconds() == 2.0
        assert (second.end_time - second.start_time).total_seconds() == 0.4
        assert all(
            tool.tool_call is not None and tool.tool_call.result_count == 1 for tool in tools
        )
        assert len(to_tool_issue_trace(trace).calls) == 2
        assert "turns_unavailable" in {gap["code"] for gap in source["gaps"]}
    else:
        turn = next(span for span in root.children if span.kind is SpanKind.CHAIN)
        assert turn.input is None
        assert turn.output == source["turns"][0]["answer"]
        assert source["turns"][0]["answer"][0]["content"][0]["text"] == "answer"
        assert isinstance(turn.attributes["gym"], dict)
        assert turn.attributes["gym"]["resolved"] is None
        assert not to_tool_issue_trace(trace).calls


def test_missing_attachment_points_to_supported_paths(tmp_path):
    with pytest.raises(GymTraceLoadError, match="trajectory-capabilities"):
        loader_for(tmp_path, {"response": {"output": []}}).load()


def assert_result_counts(trace):
    pending = list(trace.root_spans)
    while pending:
        span = pending.pop()
        pending.extend(span.children)
        if span.kind is SpanKind.TOOL:
            assert span.tool_call is not None
            assert len(span.tool_call.results) == span.tool_call.result_count


@pytest.mark.parametrize(
    "path", sorted((Path(__file__).parent / "fixtures/gym_native").glob("*.json")), ids=str
)
def test_every_tool_span_records_one_result_per_count(path):
    trace = next(iter(GymTraceLoader(GymTraceConfig(path=path, format="json")).load()))
    assert_result_counts(trace)
    for span in trace.root_spans[0].children:
        if span.kind is SpanKind.TOOL and span.tool_call is not None:
            assert span.tool_call.results is not UNSET
            if not span.tool_call.results:
                continue
            assert span.tool_call.results[0].content == span.output
            assert span.tool_call.results[0].is_error is (True if span.error else None)


@pytest.mark.parametrize("empty", [False, True])
def test_responses_create_params_tools_become_root_definitions(tmp_path, empty):
    record = rollout()
    tools = [
        {"type": "function", "name": "search", "parameters": {"type": "object"}},
        {"type": "function", "function": {"name": "open", "parameters": {"type": "string"}}},
        {"type": "function"},
        "not-a-tool",
    ]
    if empty:
        tools = []
    record["responses_create_params"] = {"tools": tools}
    trace = next(iter(loader_for(tmp_path, record).load()))
    root = trace.root_spans[0]
    assert root.kind is SpanKind.AGENT
    assert root.tool_definitions is not UNSET
    assert [definition.raw for definition in root.tool_definitions] == tools
    assert tool_catalog(trace) == ({} if empty else None)


# Responses-envelope records without ng_trajectory.

RESPONSES_FIXTURES = Path(__file__).parent / "fixtures/gym_responses"


def envelope():
    return {
        "_ng_task_index": 3,
        "_ng_rollout_index": 1,
        "reward": 1.0,
        "responses_create_params": {
            "instructions": "Answer from inventory.",
            "input": "How many bolts?",
            "tools": [
                {"type": "function", "name": "lookup", "parameters": {"type": "object"}},
                {"type": "function", "name": "audit", "parameters": {"type": "object"}},
            ],
        },
        "response": {
            "object": "response",
            "id": "resp-1",
            "model": "test-model",
            "status": "completed",
            "usage": {
                "input_tokens": 30,
                "input_tokens_details": {"cached_tokens": 10},
                "output_tokens": 7,
                "total_tokens": 37,
            },
            "output": [
                {"type": "reasoning", "summary": []},
                {
                    "type": "function_call",
                    "call_id": "a",
                    "name": "lookup",
                    "arguments": '{"sku": 1}',
                },
                {"type": "function_call", "call_id": "b", "name": "audit", "arguments": "not json"},
                {"type": "function_call_output", "call_id": "a", "output": "14"},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "14 bolts."}],
                },
            ],
        },
    }


def test_responses_envelope_maps_root_llm_and_paired_tools(tmp_path):
    source = envelope()
    loader = loader_for(tmp_path, source)
    trace = next(iter(loader.load()))
    line = loader.config.path.read_bytes()
    assert trace.id == "gym-" + hashlib.sha256(line.strip()).hexdigest()[:16]
    assert trace.attributes["gym"] == source
    assert trace.attributes["gym_source_shape"] == "rollout_envelope"
    assert trace.attributes["logical_case_id"] == "gym-task-index-3"
    assert trace.evaluator_results == {"gym.reward": 1}
    assert loader.describe()["gap_count"] == 2
    assert trace.attributes["gym_loader_gaps"] == [
        {"code": "ng_trajectory_unavailable"},
        {"code": "responses_output_aggregated"},
    ]
    (root,) = trace.root_spans
    assert root.kind is SpanKind.AGENT
    assert root.input == [
        {"role": "system", "content": "Answer from inventory."},
        {"role": "user", "content": "How many bolts?"},
    ]
    assert root.output == "14 bolts."
    assert tool_catalog(trace) == {"lookup": {"type": "object"}, "audit": {"type": "object"}}
    model, paired, unpaired = root.children
    assert model.kind is SpanKind.LLM and model.model == "test-model"
    assert model.token_counts == TokenCounts(
        input_tokens=20, cached_input_tokens=10, output_tokens=7
    )
    assert model.output == source["response"]["output"]
    assert model.start_time is None and model.end_time is None
    assert paired.input == {"sku": 1} and paired.output == "14"
    assert paired.tool_call.call_id == "a" and paired.tool_call.result_count == 1
    assert paired.tool_call.prior_user_text == "How many bolts?"
    assert [(r.content, r.is_error) for r in paired.tool_call.results] == [("14", None)]
    assert unpaired.input == "not json" and unpaired.output is UNSET
    assert unpaired.tool_call.result_count == 0 and unpaired.tool_call.results == []
    assert_result_counts(trace)
    assert [call.tool_name for call in to_tool_issue_trace(trace).calls] == ["lookup", "audit"]


def test_deeply_nested_arguments_stay_raw(tmp_path):
    source = envelope()
    nested = "[" * 100_000
    source["response"]["output"][1]["arguments"] = nested
    trace = next(iter(loader_for(tmp_path, source).load()))
    paired = trace.root_spans[0].children[1]
    assert paired.input == nested


def test_standalone_responses_object_and_null_attachment(tmp_path):
    response = envelope()["response"]
    response["tools"] = [{"type": "function", "name": "lookup", "parameters": {}}]
    trace = next(iter(loader_for(tmp_path, response, format="json").load()))
    (root,) = trace.root_spans
    assert trace.attributes["gym_source_shape"] == "responses_object"
    assert "logical_case_id" not in trace.attributes
    assert root.input == [] and root.output == "14 bolts."
    assert root.children[0].input is UNSET
    assert tool_catalog(trace) == {"lookup": {}}
    assert_result_counts(trace)
    record = envelope()
    record["ng_trajectory"] = None
    assert next(iter(loader_for(tmp_path, record).load())).root_spans[0].kind is SpanKind.AGENT


def test_results_pair_across_request_input_and_echoed_prefix(tmp_path):
    source = envelope()
    history = [
        {"type": "message", "role": "user", "content": "How many bolts?"},
        {"type": "function_call", "call_id": "a", "name": "lookup", "arguments": "{}"},
    ]
    source["responses_create_params"]["input"] = history
    source["response"]["output"] = [
        {**history[0], "id": "msg-1", "status": "completed"},
        history[1],
        {"type": "function_call_output", "call_id": "a", "output": "14"},
    ]
    trace = next(iter(loader_for(tmp_path, source).load()))
    (tool,) = [span for span in trace.root_spans[0].children if span.kind is SpanKind.TOOL]
    assert tool.attributes["gym"]["item_path"] == "$.responses_create_params.input[1]"
    assert tool.attributes["gym_result"]["item_path"] == "$.response.output[2]"
    assert tool.tool_call.result_count == len(tool.tool_call.results) == 1
    assert trace.root_spans[0].input == [
        {"role": "system", "content": "Answer from inventory."},
        history[0],
    ]
    assert trace.root_spans[0].output is UNSET


def test_response_error_code_marks_model_span(tmp_path):
    source = envelope()
    source["response"]["status"] = "failed"
    source["response"]["error"] = {"code": "server_error", "message": "private"}
    model = next(iter(loader_for(tmp_path, source).load())).root_spans[0].children[0]
    assert model.error == "server_error"
    source["response"]["error"] = None
    model = next(iter(loader_for(tmp_path, source).load())).root_spans[0].children[0]
    assert model.error == "failed"


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_call",
        "duplicate_result",
        "orphan_result",
        "history",
        "prompt_after_echo",
        "prompt_without_input",
        "remote",
        "remote_on_response",
        "item",
        "namespace",
        "output",
        "shape",
    ],
)
def test_ambiguous_responses_records_rejected(tmp_path, mutation):
    source = envelope()
    output = source["response"]["output"]
    if mutation == "duplicate_call":
        output.append(copy.deepcopy(output[1]))
    elif mutation == "duplicate_result":
        output.append(copy.deepcopy(output[3]))
    elif mutation == "orphan_result":
        output[3]["call_id"] = "missing"
    elif mutation == "history":
        output.insert(0, {"role": "user", "content": "Different prompt"})
    elif mutation == "prompt_after_echo":
        request = source["responses_create_params"]
        request["input"] = [{"type": "message", "role": "user", "content": "How many bolts?"}]
        request.pop("instructions", None)
        output[:0] = [
            copy.deepcopy(request["input"][0]),
            {"type": "message", "role": "user", "content": "Another prompt"},
        ]
    elif mutation == "prompt_without_input":
        source["responses_create_params"]["input"] = []
        output.insert(0, {"type": "message", "role": "user", "content": "Hidden prompt"})
    elif mutation == "remote":
        source["responses_create_params"]["previous_response_id"] = "resp-0"
    elif mutation == "remote_on_response":
        source["response"]["previous_response_id"] = "resp-0"
    elif mutation == "item":
        output.append({"type": "web_search_call"})
    elif mutation == "namespace":
        output[1]["namespace"] = "mcp"
    elif mutation == "output":
        source["response"]["output"] = None
    elif mutation == "shape":
        source = {"output": output}
    with pytest.raises(GymTraceLoadError):
        loader_for(tmp_path, source).load()


@pytest.mark.parametrize(
    "field, value",
    [("previous_response_id", "resp-0"), ("conversation", {"id": "conv-0"})],
)
def test_standalone_response_with_remote_state_rejected(tmp_path, field, value):
    response = envelope()["response"]
    response[field] = value
    with pytest.raises(GymTraceLoadError, match="remote conversation state"):
        loader_for(tmp_path, response, format="json").load()


def test_standalone_response_with_prompt_output_rejected(tmp_path):
    response = envelope()["response"]
    response["output"].insert(0, {"type": "message", "role": "system", "content": "Injected"})
    with pytest.raises(GymTraceLoadError, match="prompt-role items"):
        loader_for(tmp_path, response, format="json").load()


def test_eval_author_responses_fixtures_load(tmp_path):
    """Fixtures copied verbatim from eval-author skills/gym-to-atif/evals/files (Apache-2.0)."""

    trace = next(
        iter(
            GymTraceLoader(
                GymTraceConfig(path=RESPONSES_FIXTURES / "source.json", format="json")
            ).load()
        )
    )
    assert trace.root_spans[0].input == [
        {"role": "user", "content": "How many units of BOLT-17 are available?"}
    ]
    assert_result_counts(trace)
    lines = (RESPONSES_FIXTURES / "ambiguous.jsonl").read_bytes().splitlines()
    assert len(lines) == 2
    for index, line in enumerate(lines):
        path = tmp_path / f"row-{index}.json"
        path.write_bytes(line)
        (row,) = GymTraceLoader(GymTraceConfig(path=path, format="json")).load()
        (tool,) = [span for span in row.root_spans[0].children if span.kind is SpanKind.TOOL]
        assert tool.tool_name == "lookup_item" and tool.input == {"sku": "BOLT-17"}
        assert tool.tool_call is not None
        assert tool.tool_call.result_count == len(tool.tool_call.results) == 1
    # Byte-identical rows share a content identity, so one JSONL corpus is a duplicate.
    with pytest.raises(GymTraceLoadError):
        GymTraceLoader(GymTraceConfig(path=RESPONSES_FIXTURES / "ambiguous.jsonl")).load()
