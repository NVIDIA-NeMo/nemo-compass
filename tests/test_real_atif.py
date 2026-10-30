# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ATIF trajectories published or produced by Harbor v0.24.0, see fixtures/real/atif/README.md."""

import hashlib
import json
from pathlib import Path

from trace_ingest.loaders.atif import ATIFTraceConfig, ATIFTraceLoader
from trace_ingest.models import Authorship, SpanKind, TokenCounts, TurnRole

FIXTURES = Path(__file__).parent / "fixtures" / "real" / "atif"


def raw(name):
    return json.loads((FIXTURES / name).read_text())


def load(tmp_path, name):
    """Load one pretty-printed Harbor trajectory as a one-line JSONL corpus."""
    corpus = tmp_path / f"{Path(name).stem}.jsonl"
    corpus.write_text(json.dumps(raw(name)) + "\n")
    (trace,) = ATIFTraceLoader(ATIFTraceConfig(path=corpus)).load()
    return trace


def spans_by_id(spans):
    found = {}
    for span in spans:
        found[span.id] = span
        found.update(spans_by_id(span.children))
    return found


def test_claude_code_single_tool_call(tmp_path):
    trace = load(tmp_path, "claude-code-hello-load-atif.json")
    (root,) = trace.root_spans
    spans = spans_by_id(trace.root_spans)

    assert (trace.session_id, trace.session_basis) == (
        "d7d4e19e-608d-44ef-b166-cd050ef274ba",
        "atif.session_id",
    )
    assert {span_id: span.kind for span_id, span in spans.items()} == {
        "trajectory": SpanKind.AGENT,
        "trajectory/step/2": SpanKind.LLM,
        "trajectory/step/2/tool/0": SpanKind.TOOL,
        "trajectory/step/3": SpanKind.LLM,
    }
    assert (root.agent.name, root.agent.version, root.agent.model_name) == (
        "claude-code",
        "2.1.222",
        "claude-opus-5",
    )
    assert (root.run.schema_version, root.run.total_steps) == ("ATIF-v1.7", 3)

    write = spans["trajectory/step/2/tool/0"]
    assert write.tool_name == "Write"
    assert write.input == {"file_path": "/app/hello.txt", "content": "Hello, world!\n"}
    assert write.tool_call.call_id == write.tool_call.result_id == "toolu_01Rmm2GPxpGq9Er31axjfYRv"
    assert write.tool_call.result_count == 1
    assert write.output.startswith("File created successfully at: /app/hello.txt")
    assert write.tool_call.prior_user_text == (
        'Create a file called hello.txt with "Hello, world!" as the content.\n'
    )

    user, call, answer = root.turns
    assert (user.role, user.content) == (
        TurnRole.USER,
        'Create a file called hello.txt with "Hello, world!" as the content.\n',
    )
    assert (call.role, call.authorship, call.content) == (TurnRole.AGENT, Authorship.MODEL, "")
    assert call.llm_span_ids == ["trajectory/step/2"]
    assert call.tool_call_span_ids == ["trajectory/step/2/tool/0"]
    assert (answer.role, answer.authorship) == (TurnRole.AGENT, Authorship.MODEL)
    assert answer.content == "Created `/app/hello.txt` containing `Hello, world!`."
    assert answer.llm_span_ids == ["trajectory/step/3"]
    assert answer.tool_call_span_ids == []

    step = spans["trajectory/step/2"]
    assert step.model == "claude-opus-5"
    assert step.llm_call.call_count == 1
    assert step.token_counts == TokenCounts(
        input_total=18907, cached_input_tokens=13788, output_tokens=82
    )
    # final_metrics equals the sum of both steps' metrics in the source.
    assert trace.aggregate.token_counts == TokenCounts(
        input_total=18907 + 19026, cached_input_tokens=13788 + 18905, output_tokens=82 + 25
    )
    assert trace.aggregate.cost_usd == 0.05176650000000001
    assert trace.losses == []


def test_claude_code_multi_step_tool_calls(tmp_path):
    name = "claude-code-trajectory_pass.json"
    source = raw(name)
    trace = load(tmp_path, name)
    (root,) = trace.root_spans
    spans = spans_by_id(trace.root_spans)

    assert trace.session_id == "a3b10bae-3ebd-4b8e-8779-3d21ce098521"
    assert (root.agent.name, root.agent.model_name) == ("claude-code", "claude-opus-4-6")
    kinds = [span.kind for span in spans.values()]
    assert (kinds.count(SpanKind.AGENT), kinds.count(SpanKind.LLM)) == (1, 11)

    tools = [span for span in spans.values() if span.kind is SpanKind.TOOL]
    assert [tool.tool_name for tool in tools] == [
        "Read", "Bash", "Read", "Bash", "Edit", "Edit", "Edit", "Bash", "Bash", "Read", "Read", "Bash",
    ]  # fmt: skip
    calls = [call for step in source["steps"] for call in step.get("tool_calls", [])]
    results = {
        result["source_call_id"]: result["content"]
        for step in source["steps"]
        for result in step.get("observation", {}).get("results", [])
    }
    assert [tool.tool_call.call_id for tool in tools] == [call["tool_call_id"] for call in calls]
    for tool, call in zip(tools, calls, strict=True):
        assert tool.input == call["arguments"]
        assert tool.tool_call.result_id == call["tool_call_id"]
        assert tool.output == results[call["tool_call_id"]]

    assert [turn.role for turn in root.turns] == [TurnRole.USER] + [TurnRole.AGENT] * 11
    assert [turn.content for turn in root.turns] == [step["message"] for step in source["steps"]]
    # Steps 2 and 3 each issue two tool calls; step 12 issues none.
    assert [len(turn.tool_call_span_ids) for turn in root.turns] == [0, 2, 2] + [1] * 8 + [0]
    for turn in root.turns[1:]:
        for span_id in turn.tool_call_span_ids:
            assert span_id.startswith(f"{turn.llm_span_ids[0]}/tool/")

    assert spans["trajectory/step/12"].output == {
        "role": "assistant",
        "content": source["steps"][11]["message"],
    }
    assert trace.aggregate.token_counts == TokenCounts(
        input_total=485630, cached_input_tokens=461807, output_tokens=3304
    )
    assert trace.aggregate.cost_usd == 0.46238100000000004
    assert trace.losses == []


def test_qwen_code_nested_embedded_subagents(tmp_path):
    trace = load(tmp_path, "qwen-code-nested-subagents.trajectory.json")
    (root,) = trace.root_spans
    spans = spans_by_id(trace.root_spans)

    assert (trace.session_id, trace.session_basis) == ("session-1", "atif.session_id")
    assert {span_id: span.kind for span_id, span in spans.items()} == {
        "trajectory": SpanKind.AGENT,
        "trajectory/step/2": SpanKind.LLM,
        "trajectory/step/2/tool/0": SpanKind.TOOL,
        "trajectory/subagent/0": SpanKind.AGENT,
        "trajectory/subagent/0/step/2": SpanKind.LLM,
        "trajectory/subagent/0/step/2/tool/0": SpanKind.TOOL,
        "trajectory/subagent/0/subagent/0": SpanKind.AGENT,
        "trajectory/subagent/0/subagent/0/step/2": SpanKind.LLM,
    }

    delegate = spans["trajectory/step/2/tool/0"]
    explore = spans["trajectory/subagent/0"]
    nested_call = spans["trajectory/subagent/0/step/2/tool/0"]
    nested = spans["trajectory/subagent/0/subagent/0"]
    assert delegate.children == [explore]
    assert nested_call.children == [nested]

    assert (delegate.tool_name, delegate.input, delegate.output) == (
        "agent",
        {"subagent_type": "Explore"},
        "Done.",
    )
    (ref,) = delegate.tool_call.results[0].subagent_refs
    assert (ref.span_id, ref.trajectory_id, ref.session_id) == (
        "trajectory/subagent/0",
        "Explore-call_parent",
        "session-1",
    )
    assert (nested_call.input, nested_call.output) == (
        {"subagent_type": "general-purpose"},
        "Nested work done.",
    )
    (nested_ref,) = nested_call.tool_call.results[0].subagent_refs
    assert (nested_ref.span_id, nested_ref.trajectory_id) == (
        "trajectory/subagent/0/subagent/0",
        "general-purpose-call_child",
    )

    assert explore.run.trajectory_id == "Explore-call_parent"
    assert explore.agent.extra["subagent_name"] == "Explore"
    assert nested.run.trajectory_id == "general-purpose-call_child"
    assert nested.agent.extra["parent_agent_id"] == "Explore-call_parent"

    assert [(turn.role, turn.content) for turn in root.turns] == [
        (TurnRole.USER, "Delegate."),
        (TurnRole.AGENT, ""),
    ]
    assert root.turns[1].tool_call_span_ids == ["trajectory/step/2/tool/0"]
    assert [(turn.role, turn.content) for turn in explore.turns] == [
        (TurnRole.USER, "Explore."),
        (TurnRole.AGENT, ""),
    ]
    assert explore.turns[1].tool_call_span_ids == ["trajectory/subagent/0/step/2/tool/0"]
    assert [(turn.role, turn.content) for turn in nested.turns] == [
        (TurnRole.USER, "Do nested work."),
        (TurnRole.AGENT, "Nested result."),
    ]
    assert nested.turns[1].llm_span_ids == ["trajectory/subagent/0/subagent/0/step/2"]
    assert trace.losses == []


def test_trajectories_sharing_a_session_id_are_distinct_traces(tmp_path):
    # ATIF-v1.7+ session_id is run-scoped: continuations and siblings may share it, and
    # Harbor root trajectories set no trajectory_id. Both Qwen Code outputs use session-1.
    names = [
        "qwen-code-embedded-subagent.trajectory.json",
        "qwen-code-nested-subagents.trajectory.json",
    ]
    sources = [raw(name) for name in names]
    assert [(source["session_id"], source.get("trajectory_id")) for source in sources] == [
        ("session-1", None),
        ("session-1", None),
    ]
    lines = [json.dumps(source) for source in sources]
    corpus = tmp_path / "qwen-code.jsonl"
    corpus.write_text("".join(f"{line}\n" for line in lines))
    loader = ATIFTraceLoader(ATIFTraceConfig(path=corpus))
    traces = list(loader.load())

    assert [trace.id for trace in traces] == [
        "atif-" + hashlib.sha256(line.encode()).hexdigest()[:16] for line in lines
    ]
    assert [trace.session_id for trace in traces] == ["session-1", "session-1"]
    assert [len(spans_by_id(trace.root_spans)) for trace in traces] == [5, 8]
    description = loader.describe()
    assert (description["trace_count"], description["distinct_sessions"]) == (2, 1)
