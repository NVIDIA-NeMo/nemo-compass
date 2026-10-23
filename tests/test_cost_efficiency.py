# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest
from nooa.unifiedllm import FakeLLMClient

from compass.cli.artifacts import dump_json
from compass.compass_signals.builtins import registered_builtin_signals
from compass.compass_signals.compass_signals import CompassSignalResult, Problem
from compass.compass_signals.cost_efficiency import review as cost_review
from compass.compass_signals.cost_efficiency.prompt_caching_miss.sequences import (
    build_sequences,
    sequence_keys,
)
from compass.compass_signals.cost_efficiency.review import COST_EFFICIENCY_REVIEW
from compass.compass_signals.cost_efficiency.signal import (
    CostEfficiencyArtifacts,
    CostEfficiencyCompassSignal,
    CostEfficiencyConfig,
)
from compass.traces import (
    Span,
    SpanKind,
    TokenCounts,
    Trace,
    TraceAggregate,
    TraceSnapshot,
)

T0 = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
MODEL = "claude-sonnet-5-5"
PLANNER = "planner"


def tool(name: str, description: str = "Look things up.") -> dict:
    return {"name": name, "description": description, "input_schema": {"type": "object"}}


def call(
    span_id: str,
    start: float,
    end: float,
    *,
    uncached: int,
    read: int | None,
    write: int | None,
    output: int = 10,
    tools: tuple[str, ...] = ("search", "read_file"),
    tool_defs: list | None = None,
    params: dict | None = None,
    model: str = MODEL,
    ttl: int | None = 300,
) -> Span:
    return Span(
        id=span_id,
        kind=SpanKind.LLM,
        start_time=T0 + timedelta(seconds=start),
        end_time=T0 + timedelta(seconds=end),
        model=model,
        token_counts=TokenCounts(
            input_tokens=uncached,
            cached_input_tokens=read,
            output_tokens=output,
            cache_write_tokens=write,
            cache_writes_by_ttl={ttl: write} if ttl is not None and write else None,
        ),
        llm_tools=tool_defs if tool_defs is not None else [tool(name) for name in tools],
        llm_params=params if params is not None else {"max_tokens": 1000},
    )


def trace(
    trace_id: str,
    session: str | None,
    calls: list[Span],
    *,
    agent: str | None = PLANNER,
) -> Trace:
    roots = (
        [Span(id=f"{trace_id}-agent", kind=SpanKind.AGENT, agent_id=agent, children=calls)]
        if agent is not None
        else calls
    )
    return Trace(
        id=trace_id,
        root_spans=roots,
        aggregate=TraceAggregate(),
        attributes={"logical_case_id": session} if session is not None else {},
    )


def tool_change_session(index: int) -> Trace:
    """The worked example (SDD §2): an MCP server adds jira_lookup before call 3."""

    return trace(
        f"t{index}",
        f"s{index}",
        [
            call("c1", 0, 8, uncached=1200, read=0, write=40000, output=600),
            call("c2", 10, 16, uncached=2000, read=40000, write=1500, output=450),
            call(
                "c3",
                20,
                30,
                uncached=900,
                read=0,
                write=45000,
                output=700,
                tools=("search", "read_file", "jira_lookup"),
            ),
        ],
    )


def quiet_session(index: int, calls: int) -> Trace:
    return trace(
        f"q{index:02d}",
        f"quiet-{index:02d}",
        [call(f"c{n}", 10 * n, 10 * n + 5, uncached=900, read=0, write=0) for n in range(calls)],
    )


def run(traces: list[Trace], **config: object) -> CompassSignalResult:
    signal = CostEfficiencyCompassSignal(CostEfficiencyConfig.model_validate(config))
    snapshot = TraceSnapshot(traces)
    assert signal.check_prerequisites(snapshot) is None
    result = asyncio.run(signal.analyze(snapshot))
    assert isinstance(result.artifacts, CostEfficiencyArtifacts)
    return result


def kinds(result: CompassSignalResult) -> list[tuple[str, str]]:
    return [(f.trace_id, f.kind) for f in result.artifacts.findings]


def test_worked_example_end_to_end():
    traces = [tool_change_session(i) for i in range(1, 5)]
    traces += [quiet_session(i, 11 if i < 8 else 10) for i in range(16)]
    result = run(traces)

    assert sum(len(list(t.root_spans[0].children)) for t in traces) == 180
    assert kinds(result) == [(f"t{i}", "set_changed") for i in range(1, 5)]
    finding = result.artifacts.findings[0]
    assert finding.tokens == 41500
    assert finding.details == {
        "sequence": f"agent_id:{PLANNER}",
        "cache_age_s": 10.0,
        "idle_s": 4.0,
        "ttl_s": 300,
    }
    assert finding.change_summary == {"added": ["jira_lookup"], "removed": []}
    (card,) = result.artifacts.cards
    assert card.card_id == "ce:cache_miss_after_tool_change"
    assert card.eligible
    (problem,) = result.problems
    assert problem.description == (
        "Cache miss after a tool change in 4 traces (20% of all traces, "
        "20% of traces with token counts; 4 sessions, 20% of sessions with token counts), 4 calls (2.2% of LLM calls with token counts). The "
        "previous call left 41,500 cached tokens; this call read 0 after the tool list "
        "changed. (166,000 input tokens not read from cache.) By kind: tools added or "
        "removed: 4 traces. Tools added: jira_lookup. "
        "Suggested fix: Register every tool before the first call, for example by waiting "
        "for MCP servers to connect; load tools that can only arrive later through the "
        "provider's deferred tool loading, which leaves the tool list unchanged. Figures "
        "cover only the observed calls."
    )
    assert problem.supporting_trace_ids == ("t1", "t2", "t3", "t4")
    assert problem.candidate_trace_ids == ("t1", "t2", "t3", "t4")


def test_tool_change_kinds_share_one_problem():
    edited = [tool("search", "Look things up today."), tool("read_file")]
    traces = [tool_change_session(i) for i in range(1, 3)]
    traces += [
        trace(
            f"e{index}",
            f"e{index}",
            [
                call("c1", 0, 2, uncached=10, read=0, write=5000),
                call("c2", 3, 5, uncached=10, read=0, write=5100, tool_defs=edited),
            ],
        )
        for index in range(2)
    ]
    result = run(traces)
    (card,) = result.artifacts.cards
    assert (card.card_id, card.traces, card.kinds) == (
        "ce:cache_miss_after_tool_change",
        4,
        {"definition_changed": 2, "set_changed": 2},
    )
    (problem,) = result.problems
    assert (
        "By kind: tools added or removed: 2 traces; tool definitions changed: 2 traces."
        in problem.description
    )
    assert "Suggested fixes: Register every tool" in problem.description
    assert "Make tool descriptions and schemas identical" in problem.description


def test_below_minimum_traces_stays_audit_only():
    result = run([tool_change_session(i) for i in range(1, 3)])
    assert result.finding_count == 2
    assert result.problems == ()
    assert result.artifacts.cards[0].audit_reason == "below_minimum_traces"


def test_cross_trace_reference():
    traces = []
    for index in range(3):
        session = f"x{index}"
        traces.append(
            trace(f"x{index}a", session, [call("c1", 0, 10, uncached=10, read=0, write=20000)])
        )
        traces.append(
            trace(
                f"x{index}b",
                session,
                [call("c2", 240, 245, uncached=10, read=0, write=20100, tools=("search",))],
            )
        )
    result = run(traces)
    assert [(f.trace_id, f.reference_trace_id, f.kind) for f in result.artifacts.findings] == [
        (f"x{i}b", f"x{i}a", "set_changed") for i in range(3)
    ]
    assert result.problems


@pytest.mark.parametrize(
    ("second", "kind"),
    [
        ([tool("search", "Changed at 12:01."), tool("read_file")], "definition_changed"),
        ([tool("read_file"), tool("search")], "other"),  # order
        ([tool("search"), {**tool("read_file"), "cache_control": {"type": "ephemeral"}}], "other"),
        (
            [
                {"input_schema": {"type": "object"}, "description": "Look things up.", "name": n}
                for n in ("search", "read_file")
            ],
            "other",  # JSON key order
        ),
    ],
)
def test_tool_change_kinds(second, kind):
    first = [tool("search"), tool("read_file")]
    result = run(
        [
            trace(
                "k",
                "k",
                [
                    call("c1", 0, 2, uncached=10, read=0, write=5000, tool_defs=first),
                    call("c2", 3, 5, uncached=10, read=0, write=5100, tool_defs=second),
                ],
            )
        ]
    )
    assert [f.kind for f in result.artifacts.findings] == [kind]


def test_definition_change_records_names_without_text():
    first = [tool("search", "Look things up.\nUpdated daily."), tool("read_file")]
    second = [
        tool("search", "Look things up.\nUpdated at 12:01.\nUse sparingly."),
        tool("read_file"),
    ]
    traces = [
        trace(
            f"d{index}",
            f"d{index}",
            [
                call("c1", 0, 2, uncached=10, read=0, write=5000, tool_defs=first),
                call("c2", 3, 5, uncached=10, read=0, write=5100, tool_defs=second),
            ],
        )
        for index in range(3)
    ]
    result = run(traces)
    finding = result.artifacts.findings[0]
    assert finding.change_summary == {"changed": ["search"]}
    (problem,) = result.problems
    assert "Changed tools: search." in problem.description
    rendered = dump_json(result.artifacts) + dump_json(result.problems)
    assert "Updated" not in rendered


def test_agent_spans_without_ids_are_separate_sequences():
    """Two agents run one after the other under AGENT spans without IDs: never compared."""

    planner = call("p1", 0, 2, uncached=10, read=0, write=5000)
    coder = call("k1", 3, 5, uncached=10, read=0, write=6000, tools=("edit_file",))
    spans = [
        Span(id="planner", kind=SpanKind.AGENT, children=[planner]),
        Span(id="coder", kind=SpanKind.AGENT, children=[coder]),
    ]
    one = Trace(
        id="t",
        root_spans=spans,
        aggregate=TraceAggregate(),
        attributes={"logical_case_id": "s"},
    )
    assert run([one]).finding_count == 0
    keys = {key for key, _ in sequence_keys(one)}
    assert keys == {"agent_span:t/planner", "agent_span:t/coder"}


def test_a_call_with_its_own_agent_id_joins_that_agent():
    """Relay places a parallel subagent's call under the wrong AGENT span but records its agent."""

    own = call("s1", 0, 2, uncached=10, read=0, write=5000).model_copy(update={"agent_id": "sub"})
    placed = call("m1", 3, 5, uncached=10, read=0, write=6000)
    one = trace("t", "s", [own, placed])
    keys = {visit.span.id: key for key, visit in sequence_keys(one)}
    assert keys["s1"] == "agent_id:sub"
    assert keys["m1"] == f"agent_id:{PLANNER}"


def tool_change_pair(case: str) -> list[Span]:
    """Two calls with a tool change; every case but ``miss`` has one reason it is no miss."""

    first = call("c1", 0, 2, uncached=10, read=0, write=20000)
    second = call("c2", 3, 5, uncached=10, read=0, write=20100, tools=("search",))
    return {
        "miss": [first, second],
        "compaction": [call("c1", 0, 2, uncached=10, read=30000, write=100), second],
        "nothing_cached": [call("c1", 0, 2, uncached=20000, read=0, write=0), second],
        "partial_read": [
            first,
            call("c2", 3, 5, uncached=10, read=5000, write=15100, tools=("search",)),
        ],
        "call_failed": [first, second.model_copy(update={"error": "overloaded"})],
        "cache_read_unknown": [
            first,
            call("c2", 3, 5, uncached=20110, read=None, write=None, tools=("search",)),
        ],
        "cache_write_unknown": [
            call("c1", 0, 2, uncached=10, read=3000, write=None),
            call("c2", 3, 5, uncached=3100, read=0, write=None, tools=("search",)),
        ],
    }[case]


@pytest.mark.parametrize(
    ("case", "findings"),
    [
        ("miss", 1),
        ("compaction", 0),
        ("nothing_cached", 0),
        ("partial_read", 0),
        ("call_failed", 0),
        ("cache_read_unknown", 0),
        ("cache_write_unknown", 0),
    ],
)
def test_only_a_call_reading_none_of_the_cache_is_a_miss(case, findings):
    """The tool list always changes, so only the miss rules decide."""

    assert run([trace("p", "p", tool_change_pair(case))]).finding_count == findings


def references(calls: list[Span]) -> list[tuple[str, str | None]]:
    """(missed call, its reference call) of each finding in one trace."""

    return [
        (finding.span_ids[0], finding.reference_span_id)
        for finding in run([trace("r", "r", calls)]).artifacts.findings
    ]


REFERENCE = call("c1", 0, 2, uncached=10, read=0, write=20000)
MISSED = call("n", 6, 8, uncached=10, read=0, write=20100, tools=("search",))


@pytest.mark.parametrize(
    ("between", "expected"),
    [
        # Still running when n starts, so it cannot have left n a cache.
        (call("p", 3, 20, uncached=10, read=20000, write=0), [("n", "c1")]),
        (
            call("e", 3, 5, uncached=10, read=0, write=20000).model_copy(
                update={"error": "timeout"}
            ),
            [("n", "c1")],
        ),
        (
            call("e", 3, 5, uncached=10, read=0, write=20000).model_copy(update={"end_time": None}),
            [("n", "c1")],
        ),
        (call("u", 3, 5, uncached=20010, read=None, write=None), [("n", "c1")]),
    ],
    ids=["overlapping", "failed", "no_end_time", "cache_read_unknown"],
)
def test_reference_skips_calls_that_cannot_have_left_a_cache(between, expected):
    assert references([REFERENCE, between, MISSED]) == expected


def test_a_call_without_a_finished_earlier_call_has_no_reference():
    overlapping = call("p", 0, 20, uncached=10, read=0, write=20000)
    assert references([overlapping, MISSED]) == []


def test_calls_outside_any_agent_join_the_session_root_sequence():
    first = trace("ra", "s", [REFERENCE], agent=None)
    second = trace("rb", "s", [MISSED], agent=None)
    assert {key for one in (first, second) for key, _ in sequence_keys(one)} == {"root"}
    (finding,) = run([first, second]).artifacts.findings
    assert (finding.trace_id, finding.reference_trace_id, finding.details["sequence"]) == (
        "rb",
        "ra",
        "root",
    )


def test_traces_without_a_session_are_never_compared():
    """Without logical_case_id each trace is its own session; with one, the same calls miss."""

    alone = [trace("na", None, [REFERENCE]), trace("nb", None, [MISSED])]
    sequences = build_sequences(alone)
    assert [(s.session, s.session_id) for s in sequences] == [("na", None), ("nb", None)]
    assert run(alone).finding_count == 0
    assert run([trace("na", "s", [REFERENCE]), trace("nb", "s", [MISSED])]).finding_count == 1


def test_calls_without_a_start_time_are_left_out_of_sequences():
    untimed = MISSED.model_copy(update={"id": "x", "start_time": None})
    (sequence,) = build_sequences([trace("t", "s", [REFERENCE, untimed, MISSED])])
    assert [visit.span.id for visit in sequence.calls] == ["c1", "n"]


def test_outputs_never_contain_request_content():
    canary = "CANARY-7f3a-prompt-content-do-not-leak"
    traces = []
    for index in range(3):
        defs_a = [tool("search", f"{canary} v1"), tool("read_file")]
        defs_b = [tool("search", f"{canary} v2"), tool("read_file")]
        traces.append(
            trace(
                f"pv{index}",
                f"pv{index}",
                [
                    call(
                        "c1",
                        0,
                        2,
                        uncached=10,
                        read=0,
                        write=5000,
                        tool_defs=defs_a,
                        params={"thinking": {"note": canary}},
                    ),
                    call(
                        "c2",
                        3,
                        5,
                        uncached=10,
                        read=0,
                        write=5100,
                        tool_defs=defs_b,
                        params={"thinking": {"note": canary}},
                    ),
                ],
            )
        )
    result = run(traces)
    assert result.problems
    rendered = dump_json(result.artifacts) + dump_json(result.problems)
    assert canary not in rendered
    assert "CANARY" not in rendered
    assert json.loads(dump_json(result.artifacts))["findings"][0]["kind"] == "definition_changed"


def test_no_token_counts_skips_the_signal():
    snapshot = TraceSnapshot(
        [Trace(id="none", root_spans=[Span(id="s", kind=SpanKind.LLM)], aggregate=TraceAggregate())]
    )
    assert (
        CostEfficiencyCompassSignal().check_prerequisites(snapshot)
        == "No LLM spans with token counts"
    )


def skipped_miss(reason: str) -> list[Span]:
    """Two calls whose miss has no tool-change cause: expiry not ruled out, or tools unchanged."""

    first = call(
        "c1", 0, 2, uncached=10, read=0, write=20000, ttl=None if reason == "ttl_unknown" else 300
    )
    second = {
        "ttl_unknown": call("c2", 3, 5, uncached=10, read=0, write=20100, tools=("search",)),
        "expired": call("c2", 400, 405, uncached=10, read=0, write=20100, tools=("search",)),
        "tools_unrecorded": call("c2", 3, 5, uncached=10, read=0, write=20100).model_copy(
            update={"llm_tools": None}
        ),
        "no_tool_change": call(
            "c2", 3, 5, uncached=10, read=0, write=20100, params={"thinking": {"type": "enabled"}}
        ),
    }[reason]
    return [first, second]


@pytest.mark.parametrize("reason", ["ttl_unknown", "expired", "tools_unrecorded", "no_tool_change"])
def test_misses_without_a_tool_change_cause_are_not_findings(reason):
    traces = [trace(f"k{index}", f"k{index}", skipped_miss(reason)) for index in range(3)]
    result = run(traces)
    assert result.finding_count == 0
    assert result.problems == ()


def test_a_tool_change_is_found_even_when_the_model_also_changed():
    """Cause rules do not exclude each other: the tool change is reported regardless."""

    calls = [
        call("c1", 0, 2, uncached=10, read=0, write=20000),
        call("c2", 3, 5, uncached=10, read=0, write=20100, tools=("search",), model="other"),
    ]
    assert kinds(run([trace("m", "m", calls)])) == [("m", "set_changed")]


def review_with(monkeypatch, cited: tuple[str, ...]) -> list[tuple[tuple[str, ...], str]]:
    """Fake IssueDetector citing ``cited``; returns the (trace IDs, problem) of each call."""

    calls = []

    class FakeDetector:
        def __init__(self, *, llm):
            self.llm = llm

        async def detect_issues(self, traces, description, **extra):
            assert description == COST_EFFICIENCY_REVIEW
            assert [c["trace_id"] for c in extra["calls"]] == [t.id for t in traces]
            calls.append((tuple(t.id for t in traces), extra["problem"]))
            return [Problem(description="reviewed", supporting_trace_ids=cited)] if cited else []

    monkeypatch.setattr(cost_review, "IssueDetector", FakeDetector)
    return calls


def reviewed_run(traces: list[Trace]) -> CompassSignalResult:
    signal = CostEfficiencyCompassSignal(CostEfficiencyConfig(), llm=FakeLLMClient())
    return asyncio.run(signal.analyze(TraceSnapshot(traces)))


@pytest.mark.parametrize(
    ("cited", "confirmed", "kept"),
    [
        (("t1", "t3"), ("t1", "t3"), True),
        (("t2",), ("t2",), False),
        ((), (), False),
        (("t1", "elsewhere"), ("t1",), False),  # traces outside the Problem do not count
    ],
)
def test_llm_review_keeps_problems_two_traces_confirm(monkeypatch, cited, confirmed, kept):
    calls = review_with(monkeypatch, cited)
    deterministic = run([tool_change_session(i) for i in range(1, 5)])
    result = reviewed_run([tool_change_session(i) for i in range(1, 5)])

    (problem,) = deterministic.problems
    assert calls == [(("t1", "t2", "t3", "t4"), problem.description)]
    assert result.problems == ((problem,) if kept else ())
    assert result.finding_count == deterministic.finding_count
    assert result.artifacts.review == {"ce:cache_miss_after_tool_change": confirmed}


def test_without_an_llm_the_signal_stays_deterministic(monkeypatch):
    """Tools and tests construct the signal directly; registration always adds an LLM."""

    calls = review_with(monkeypatch, ())
    result = run([tool_change_session(i) for i in range(1, 5)])
    assert len(result.problems) == 1
    assert result.artifacts.review is None
    assert calls == []


def test_registration_requires_an_llm_factory():
    with pytest.raises(ValueError, match="cost-efficiency requires an LLM client factory"):
        registered_builtin_signals(cost_efficiency=CostEfficiencyConfig())
    llm = FakeLLMClient()
    registry = registered_builtin_signals(
        cost_efficiency=CostEfficiencyConfig(), llm_factory=lambda: llm
    )
    signal = registry._signals["cost-efficiency"]
    assert isinstance(signal, CostEfficiencyCompassSignal)
    assert signal.llm is llm


def test_llm_review_includes_reference_traces(monkeypatch):
    """A reference call in an earlier trace of the session is sent with the miss."""

    seen = []

    class FakeDetector:
        def __init__(self, *, llm):
            pass

        async def detect_issues(self, traces, description, **extra):
            seen.append(([t.id for t in traces], extra["calls"]))
            return []

    monkeypatch.setattr(cost_review, "IssueDetector", FakeDetector)
    traces = []
    for i in range(1, 4):
        session = f"s{i}"
        traces.append(
            trace(f"a{i}", session, [call("c1", 0, 8, uncached=900, read=0, write=40000)])
        )
        traces.append(
            trace(
                f"b{i}",
                session,
                [call("c2", 10, 16, uncached=900, read=0, write=40000, tools=("search",))],
            )
        )
    reviewed_run(traces)

    ((trace_ids, calls),) = seen
    assert sorted(trace_ids) == ["a1", "a2", "a3", "b1", "b2", "b3"]
    assert {(c["trace_id"], c["reference_trace_id"]) for c in calls} == {
        ("b1", "a1"),
        ("b2", "a2"),
        ("b3", "a3"),
    }
