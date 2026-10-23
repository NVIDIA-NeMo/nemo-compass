# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
from compass.trace_loaders import ATIFTraceConfig, ATIFTraceLoader, ATIFTraceLoadError
from compass.traces import UNSET, Span, SpanKind, TokenCounts, Trace, TraceAggregate
from trace_ingest.catalog import tool_catalog
from trace_ingest.loaders.atif import tool_definitions_from_atif
from trace_ingest.models import ToolDefinition

FIXTURE = Path(__file__).parent / "data" / "atif_trajectories.jsonl"


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


def test_exact_normalization_and_description(tmp_path):
    source = minimal()
    metadata = {key: value for key, value in source.items() if key != "steps"}
    loader = loader_for(tmp_path, source)
    expected = Trace(
        id="run",
        source_url=loader.config.path.resolve().as_uri() + "#L2",
        root_spans=[
            Span(
                id="trajectory",
                kind=SpanKind.AGENT,
                attributes={"atif": metadata},
                children=[
                    Span(
                        id="trajectory/step/1",
                        kind=SpanKind.CHAIN,
                        input={"messages": [{"role": "user", "content": "Help."}]},
                        attributes={"atif": {"step_id": 1, "source": "user"}},
                    )
                ],
            )
        ],
        aggregate=TraceAggregate(),
        attributes={"atif": metadata, "logical_case_id": "run"},
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


def test_realistic_nested_trajectory_and_tool_evidence():
    loader = ATIFTraceLoader(ATIFTraceConfig(path=FIXTURE))
    trace = next(iter(loader.load()))
    root = trace.root_spans[0]
    assert trace.id == "trace-1"
    assert [span.id for span in root.children] == [
        "trajectory/step/1",
        "trajectory/step/2",
        "trajectory/step/3",
        "trajectory/step/4",
        "trajectory/subagent/1",
    ]
    assert root.children[1].start_time == datetime(2026, 9, 1, 10, tzinfo=timezone.utc)
    agent_step = root.children[2]
    assert agent_step.kind is SpanKind.LLM
    assert agent_step.model == "test-model"
    assert agent_step.cost_usd == 0.01
    assert agent_step.token_counts == TokenCounts(
        input_tokens=80, cached_input_tokens=20, output_tokens=10
    )
    metadata = agent_step.attributes["atif"]
    assert isinstance(metadata, dict)
    assert metadata["reasoning_content"] == "Use lookup and delegate research."
    assert root.children[3].kind is SpanKind.CHAIN
    assert root.children[3].model is None
    search, delegate, missing, observation = agent_step.children
    assert search.output is None
    assert missing.output is UNSET
    assert missing.tool_call is not None
    assert search.tool_call is not None
    assert missing.tool_call.result_count == 0
    assert missing.tool_call.result_id is None
    assert search.tool_call.prior_user_text == "Find the report."
    assert delegate.children[0].id == "trajectory/subagent/0"
    assert observation.output == "Background checkpoint."
    assert trace.aggregate.cost_usd == 0.02
    assert trace.aggregate.token_counts == TokenCounts(
        input_tokens=100, cached_input_tokens=20, output_tokens=15
    )
    calls = to_tool_issue_trace(trace).calls
    assert [call.tool_name for call in calls] == ["search", "delegate", "lookup", "save"]
    assert [call.call_index for call in calls] == [0, 1, 2, 3]
    assert calls[2].prior_user_text == "Find a source."
    assert calls[3].prior_user_text == "Find the report."
    assert loader.describe()["call_count"] == 4


@pytest.mark.parametrize(
    "field, value",
    [("arguments", None), ("arguments", UNSET), ("content", None), ("content", UNSET)],
)
def test_missing_and_null_are_not_conflated(tmp_path, field, value):
    source = minimal()
    call = {"tool_call_id": "c", "function_name": "search", "arguments": {}}
    result = {"source_call_id": "c", "content": "ok"}
    target = call if field == "arguments" else result
    if value is UNSET:
        target.pop(field)
    else:
        target[field] = value
    source["steps"].append(
        {
            "step_id": 2,
            "source": "agent",
            "message": "",
            "tool_calls": [call],
            "observation": {"results": [result]},
        }
    )
    trace = next(iter(loader_for(tmp_path, source).load()))
    tool = trace.root_spans[0].children[1].children[0]
    assert (tool.input if field == "arguments" else tool.output) is value
    assert tool.tool_call.result_count == 1


def test_multiple_results_preserve_source_order_and_presence(tmp_path):
    source = json.loads(FIXTURE.read_text())
    results = [
        {"source_call_id": "search", "content": None},
        {"source_call_id": "search"},
        {"source_call_id": "search", "content": "last"},
    ]
    source["steps"][2]["observation"]["results"] = results
    trace = next(iter(loader_for(tmp_path, source).load()))
    tool = trace.root_spans[0].children[2].children[0]
    assert tool.output == results
    assert tool.tool_call.result_count == 3


def test_naive_timestamps_partial_metrics_and_multimodal_content(tmp_path):
    source = minimal()
    source["schema_version"] = "ATIF-v1.8"
    message = [
        {"type": "text", "text": "Read this."},
        {"type": "image", "source": {"media_type": "image/png", "path": "image.png"}},
    ]
    source["steps"][0].update(message=message, timestamp="2026-09-01T10:00:00")
    source["steps"].append(
        {
            "step_id": 2,
            "source": "agent",
            "message": "",
            "metrics": {"prompt_tokens": 100},
            "tool_calls": [{"tool_call_id": "c", "function_name": "read", "arguments": {}}],
        }
    )
    trace = next(iter(loader_for(tmp_path, source).load()))
    user, agent = trace.root_spans[0].children
    assert user.input["messages"][0]["content"] == message
    assert user.start_time is None
    assert user.attributes["atif"]["timestamp"] == "2026-09-01T10:00:00"
    assert agent.token_counts is None
    assert agent.attributes["atif"]["metrics"] == {"prompt_tokens": 100}
    assert agent.children[0].tool_call.prior_user_text == "Read this."


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda t: t.update(schema_version="ATIF-v99"), "schema_version"),
        (lambda t: t.pop("agent"), "agent"),
        (lambda t: t["agent"].update(name=123), "name"),
        (lambda t: t["steps"][0].update(step_id=True), "step_id"),
        (lambda t: t["steps"][0].update(message={}), "message"),
        (lambda t: t["steps"][0].update(observation={"results": {}}), "results"),
        (lambda t: t.pop("session_id"), "stable trace identity"),
        (lambda t: t["steps"][0].update(step_id=2), "sequential"),
        (lambda t: t["steps"].append(t["steps"][0]), "sequential"),
        (lambda t: t["steps"][0].update(source="tool"), "source"),
        (lambda t: t["steps"][0].update(timestamp="not-a-date"), "isoformat"),
        (lambda t: t["steps"][0].update(tool_calls=[]), "require source"),
        (lambda t: t.update(continued_trajectory_ref="next.json"), "continuation"),
        (
            lambda t: t.update(final_metrics={"total_prompt_tokens": 2, "total_cached_tokens": 3}),
            "exceeds",
        ),
        (lambda t: t.update(final_metrics={"total_prompt_tokens": -1}), "non-negative"),
        (lambda t: t.update(final_metrics={"total_cost_usd": -1}), "greater than or equal"),
        (lambda t: t.update(final_metrics={"total_cost_usd": "1"}), "must be a number"),
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
        ("duplicate_call", "duplicate tool_call_id"),
        ("unknown_call", "unknown source_call_id"),
        ("unknown_subagent", "unresolved subagent reference"),
        ("external_subagent", "embed it"),
        ("duplicate_subagent", "unique trajectory_id"),
        ("multiple_parents", "multiple parents"),
    ],
)
def test_invalid_relationships(tmp_path, case, match):
    source = json.loads(FIXTURE.read_text())
    step = source["steps"][2]
    result = step["observation"]["results"][1]
    if case == "duplicate_call":
        step["tool_calls"].append(step["tool_calls"][0])
    elif case == "unknown_call":
        result["source_call_id"] = "unknown"
    elif case == "unknown_subagent":
        result["subagent_trajectory_ref"] = [{"trajectory_id": "unknown"}]
    elif case == "external_subagent":
        result["subagent_trajectory_ref"] = [{"trajectory_path": "subagent.json"}]
    elif case == "duplicate_subagent":
        source["subagent_trajectories"].append(source["subagent_trajectories"][0])
    else:
        step["observation"]["results"].append(result)
    with pytest.raises(ATIFTraceLoadError, match=match):
        loader_for(tmp_path, source).load()


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


def tool_spans(trace):
    pending = list(trace.root_spans)
    while pending:
        span = pending.pop()
        pending.extend(span.children)
        if span.kind is SpanKind.TOOL:
            yield span


def mixed_results_record():
    return {
        "schema_version": "ATIF-v1.7",
        "trajectory_id": "mixed-results",
        "agent": {
            "name": "agent",
            "version": "1",
            "tool_definitions": [{"name": "read", "parameters": {"type": "object"}}],
        },
        "steps": [
            {"step_id": 1, "source": "user", "message": "Go.", "observation": None},
            {
                "step_id": 2,
                "source": "agent",
                "message": "Working in parallel.",
                "tool_calls": [
                    {"tool_call_id": "a", "function_name": "read", "arguments": {"path": "x"}},
                    {"tool_call_id": "b", "function_name": "spawn", "arguments": {}},
                    {"tool_call_id": "c", "function_name": "read"},
                ],
                "observation": {
                    "note": "kept",
                    "results": [
                        {
                            "source_call_id": "b",
                            "content": "spawned",
                            "subagent_trajectory_ref": [{"trajectory_id": "child"}],
                        },
                        {"content": "unlinked"},
                        {"source_call_id": "a", "content": "x", "is_error": True},
                        {"source_call_id": "a", "content": "retry", "extra": {"isError": False}},
                    ],
                },
            },
            {
                "step_id": 3,
                "source": "agent",
                "llm_call_count": 0,
                "message": "Done.",
                "tool_calls": [],
            },
            {"step_id": 4, "source": "agent", "message": "Bye.", "tool_calls": None},
        ],
        "subagent_trajectories": [
            {
                "schema_version": "ATIF-v1.7",
                "trajectory_id": "child",
                "agent": {"name": "child", "version": "1"},
                "steps": [{"step_id": 1, "source": "user", "message": "Sub."}],
                "subagent_trajectories": [],
            }
        ],
    }


def test_mixed_results_record_tool_results_and_errors(tmp_path):
    trace = next(iter(loader_for(tmp_path, mixed_results_record()).load()))
    step = trace.root_spans[0].children[1]
    read, spawn, unanswered, unlinked = step.children
    assert [(r.content, r.is_error) for r in read.tool_call.results] == [
        ("x", True),
        ("retry", False),
    ]
    assert read.error == "ATIF tool result reported is_error"
    assert read.attributes["explicit_error"] is True
    assert spawn.tool_call.results[0].is_error is None
    assert "explicit_error" not in spawn.attributes
    assert spawn.children[0].id == "trajectory/subagent/0"
    assert unanswered.tool_call.results == []
    assert unlinked.attributes["atif"] == {"content": "unlinked"}


def test_explicit_error_result_maps_to_tool_span_error(tmp_path):
    source = minimal()
    source["steps"].append(
        {
            "step_id": 2,
            "source": "agent",
            "message": "",
            "tool_calls": [{"tool_call_id": "a", "function_name": "read", "arguments": {}}],
            "observation": {"results": [{"source_call_id": "a", "content": "x", "is_error": True}]},
        }
    )
    trace = next(iter(loader_for(tmp_path, source).load()))
    tool = trace.root_spans[0].children[1].children[0]
    assert tool.error == "ATIF tool result reported is_error"
    assert tool.attributes["explicit_error"] is True
    assert tool.tool_call.results[0].is_error is True


@pytest.mark.parametrize(
    "flags, is_error, status, explicit_error",
    [
        ({}, None, None, None),
        ({"is_error": False}, False, None, False),
        ({"extra": {"isError": True}}, True, None, True),
        ({"isError": True, "is_error": True}, True, None, True),
        ({"is_error": "yes"}, None, "invalid", None),
        ({"is_error": None}, None, "invalid", None),
        ({"is_error": True, "extra": {"is_error": False}}, None, "conflicting", None),
    ],
)
def test_tool_result_error_flags(tmp_path, flags, is_error, status, explicit_error):
    source = minimal()
    source["steps"].append(
        {
            "step_id": 2,
            "source": "agent",
            "message": "",
            "tool_calls": [{"tool_call_id": "a", "function_name": "read", "arguments": {}}],
            "observation": {"results": [{"source_call_id": "a", "content": "x", **flags}]},
        }
    )
    trace = next(iter(loader_for(tmp_path, source).load()))
    tool = trace.root_spans[0].children[1].children[0]
    result = tool.tool_call.results[0]
    assert result.content == "x"
    assert result.is_error is is_error
    assert result.attributes == ({"is_error_status": status} if status else {})
    assert tool.attributes.get("explicit_error") is explicit_error
    assert (tool.error is not None) is (explicit_error is True)


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
        "not a definition",
        {"description": "nameless"},
    ]
    assert tool_definitions_from_atif(items) == [
        ToolDefinition(
            name="read",
            server="fs",
            description="Read a file.",
            input_schema=read_schema,
            raw=items[0],
        ),
        ToolDefinition(name="write", description="Write.", input_schema=write_schema, raw=items[1]),
        ToolDefinition(raw=items[2]),
        ToolDefinition(description="nameless", raw=items[3]),
    ]
    assert tool_definitions_from_atif(None) == []
    source = minimal()
    source["agent"]["tool_definitions"] = items
    trace = next(iter(loader_for(tmp_path, source).load()))
    assert trace.root_spans[0].tool_definitions == tool_definitions_from_atif(items)
    assert tool_catalog(trace) is None  # Unknown entries cannot be treated as absent tools.

    source["agent"]["tool_definitions"] = [
        {"name": "read", "parameters": read_schema},
        {"name": "read", "parameters": write_schema},
    ]
    assert tool_catalog(next(iter(loader_for(tmp_path, source).load()))) is None

    trace = next(iter(loader_for(tmp_path, minimal()).load()))
    assert trace.root_spans[0].tool_definitions is UNSET
    assert tool_catalog(trace) is None


def test_subagent_tool_definitions_stay_on_subagent_span(tmp_path):
    source = json.loads(FIXTURE.read_text())
    definitions = [{"name": "lookup", "parameters": {"type": "object"}}]
    source["subagent_trajectories"][0]["agent"]["tool_definitions"] = definitions
    trace = next(iter(loader_for(tmp_path, source).load()))
    root = trace.root_spans[0]
    subagent = root.children[2].children[1].children[0]
    assert subagent.id == "trajectory/subagent/0"
    assert root.tool_definitions is UNSET
    assert subagent.tool_definitions == tool_definitions_from_atif(definitions)
    assert tool_catalog(trace) == {"lookup": {"type": "object"}}


def test_every_tool_span_has_one_result_per_matched_source_result(tmp_path):
    records = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]
    records.append(mixed_results_record())
    spans = [span for trace in loader_for(tmp_path, *records).load() for span in tool_spans(trace)]
    assert spans
    for span in spans:
        assert len(span.tool_call.results) == span.tool_call.result_count
        assert len(span.tool_call.results) == len(span.attributes["atif"]["results"])
