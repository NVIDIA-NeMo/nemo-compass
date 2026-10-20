# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic records from Gym's TrajectoryRecord schema at 399e6783e0e8."""

import copy
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
        if span.kind is SpanKind.TOOL and span.tool_call is not None and span.tool_call.results:
            assert span.tool_call.results[0].content == span.output
            assert span.tool_call.results[0].is_error is (True if span.error else None)


def test_responses_create_params_tools_become_root_definitions(tmp_path):
    record = rollout()
    tools = [
        {"type": "function", "name": "search", "parameters": {"type": "object"}},
        {"type": "function", "function": {"name": "open", "parameters": {"type": "string"}}},
        {"type": "function"},
        "not-a-tool",
    ]
    record["responses_create_params"] = {"tools": tools}
    trace = next(iter(loader_for(tmp_path, record).load()))
    root = trace.root_spans[0]
    assert root.kind is SpanKind.AGENT
    assert root.tool_definitions is not UNSET
    assert [definition.raw for definition in root.tool_definitions] == tools[:2]
    assert tool_catalog(trace) == {"search": {"type": "object"}, "open": {"type": "string"}}
