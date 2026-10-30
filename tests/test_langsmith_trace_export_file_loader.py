# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from compass.trace_loaders.langsmith import (
    LangSmithTraceExportFileConfig,
    LangSmithTraceExportFileLoader,
    LangSmithTraceLoadError,
)
from compass.traces import SpanKind, Trace
from trace_ingest.models import UNSET, Authorship, ToolResult, TurnRole


def _records(trace: Trace) -> list[dict]:
    """The merged session's per-record provenance entries."""
    records = trace.attributes["records"]
    assert isinstance(records, list)
    entries = [entry for entry in records if isinstance(entry, dict)]
    assert len(entries) == len(records)
    return entries


def trace_export_run_record(
    run_id: str,
    *,
    trace_id: str = "trace-1",
    parent_run_id: str | None = None,
    name: str = "agent",
    run_type: str = "chain",
    start_time: str = "2026-09-01T12:00:00Z",
    end_time: str | None = "2026-09-01T12:00:03Z",
    inputs=None,
    outputs=None,
    error: str | None = None,
    status: str = "success",
    custom_metadata=None,
    tags=None,
    token_usage=None,
    costs=None,
    events=None,
    feedback_stats=None,
    **additional_fields,
):
    """Match the map emitted by langsmith-cli's ExtractRun with --full."""

    return {
        "run_id": run_id,
        "trace_id": trace_id,
        "parent_run_id": parent_run_id,
        "name": name,
        "run_type": run_type,
        "start_time": start_time,
        "end_time": end_time,
        "status": status,
        "duration_ms": 3000 if end_time is not None else None,
        "first_token_time": None,
        "token_usage": token_usage,
        "costs": costs,
        "tags": tags,
        "custom_metadata": custom_metadata,
        "inputs": inputs,
        "outputs": outputs,
        "error": error,
        "events": events,
        "feedback_stats": feedback_stats,
        **additional_fields,
    }


def write_trace_export_file(path, *runs):
    path.write_text("".join(f"{json.dumps(run)}\n" for run in runs), encoding="utf-8")


def test_loader_reads_native_cli_export_and_preserves_hierarchy_and_provenance(tmp_path):
    root = trace_export_run_record(
        "trace-1",
        inputs={"messages": [{"role": "user", "content": "find it"}]},
        outputs={"answer": "done"},
        custom_metadata={"session_id": "session-7", "revision_id": "rev-3"},
        tags=["production"],
        costs={"total_cost": 0.02, "prompt_cost": 0.01},
        token_usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        events=[{"name": "start"}],
        feedback_stats={"quality": {"avg": 1.0}},
        future_cli_field={"kept": True},
    )
    llm = trace_export_run_record(
        "llm-1",
        parent_run_id="trace-1",
        name="acompletion",
        run_type="llm",
        start_time="2026-09-01T12:00:01Z",
        end_time="2026-09-01T12:00:02Z",
        outputs={"content": "use the search tool"},
        costs={"total_cost": 0.005},
    )
    tool = trace_export_run_record(
        "tool-1",
        parent_run_id="trace-1",
        name="search",
        run_type="tool",
        start_time="2026-09-01T12:00:02Z",
        end_time="2026-09-01T12:00:03Z",
        inputs={"query": "x"},
        error="TimeoutError",
        status="error",
    )
    path = tmp_path / "trace-1.jsonl"
    # The CLI returns root-to-leaf order today, but normalization must not rely on it.
    write_trace_export_file(path, tool, root, llm)

    loader = LangSmithTraceExportFileLoader(LangSmithTraceExportFileConfig(path=tmp_path))
    snapshot = loader.load()

    assert snapshot.trace_count == 1
    trace = next(iter(snapshot))
    assert trace.id == "session:session-7"
    # The root's total_cost already includes the LLM's 0.005.
    assert trace.aggregate.cost_usd == 0.02
    assert trace.aggregate.latency_ms == 3000
    assert (trace.session_id, trace.session_basis) == (
        "session-7",
        "langsmith.metadata.session_id",
    )
    assert trace.evaluator_results == {
        "langsmith.feedback_stats.quality": [
            {"trace_id": "trace-1", "value": {"trace-1": {"avg": 1.0}}}
        ]
    }
    (record,) = _records(trace)
    assert record["trace_id"] == "trace-1"
    assert record["source_url"] == f"{path.resolve().as_uri()}#L2"
    record_attributes = record["attributes"]
    assert isinstance(record_attributes, dict)
    assert record_attributes["source_pointer"] == {
        "provider": "langsmith",
        "export_path": str(tmp_path.resolve()),
        "export_file": "trace-1.jsonl",
        "trace_id": "trace-1",
    }
    (session_root,) = trace.root_spans
    (root_span,) = session_root.children
    assert root_span.id == "trace-1/trace-1"
    assert root_span.input == {"messages": [{"role": "user", "content": "find it"}]}
    assert [span.id for span in root_span.children] == ["trace-1/llm-1", "trace-1/tool-1"]
    assert [span.kind for span in root_span.children] == [SpanKind.LLM, SpanKind.TOOL]
    assert [span.cost_usd for span in [root_span, *root_span.children]] == [None, 0.005, None]
    assert session_root.turns is not UNSET
    assert [(turn.content, turn.tool_call_span_ids) for turn in session_root.turns] == [
        ("find it", []),
        ("use the search tool", ["trace-1/tool-1"]),
        ("done", []),
    ]
    langsmith = root_span.attributes["langsmith"]
    assert isinstance(langsmith, dict)
    assert langsmith["total_cost"] == 0.02
    assert langsmith["tags"] == ["production"]
    extra = langsmith["extra"]
    assert isinstance(extra, dict)
    export_metadata = extra["langsmith_cli_export"]
    assert isinstance(export_metadata, dict)
    token_usage = export_metadata["token_usage"]
    assert isinstance(token_usage, dict)
    assert token_usage["total_tokens"] == 12
    assert export_metadata["events"] == [{"name": "start"}]
    assert export_metadata["feedback_stats"] == {"quality": {"avg": 1.0}}
    assert export_metadata["additional_fields"] == {"future_cli_field": {"kept": True}}
    tool_span = root_span.children[1]
    assert tool_span.error == "TimeoutError"
    assert tool_span.tool_name == "search"
    tool_pointer = tool_span.attributes["source_pointer"]
    assert isinstance(tool_pointer, dict)
    assert tool_pointer["line_number"] == 1

    assert loader.report.trace_count == 1
    assert loader.report.run_count == 3
    assert loader.report.unresolved_parent_count == 0
    assert loader.describe() == {
        "source": f"langsmith-trace-export-file:{tmp_path.resolve()}",
        "trace_count": 1,
        "call_count": 1,
        "distinct_cases": 1,
        "distinct_sessions": 1,
        "run_count": 3,
        "unresolved_parent_count": 0,
        "export_path": str(tmp_path.resolve()),
        "export_trace_count": 1,
        "truncated": False,
        "max_traces": 100,
    }


def test_loader_accepts_one_jsonl_file(tmp_path):
    path = tmp_path / "one.jsonl"
    write_trace_export_file(path, trace_export_run_record("trace-1"))

    loader = LangSmithTraceExportFileLoader(LangSmithTraceExportFileConfig(path=path))

    assert loader.load().get_trace_by_id("trace-1").evaluator_results == {}
    assert [trace.id for trace in loader.load()] == ["trace-1"]
    assert loader.describe()["export_path"] == str(path.resolve())


def test_loader_accepts_stitched_jsonl_and_groups_each_complete_trace(tmp_path):
    path = tmp_path / "all.jsonl"
    write_trace_export_file(
        path,
        trace_export_run_record(
            "trace-2",
            trace_id="trace-2",
            start_time="2026-09-02T12:00:00Z",
            end_time="2026-09-02T12:00:03Z",
        ),
        trace_export_run_record("trace-1"),
        trace_export_run_record(
            "child-2",
            trace_id="trace-2",
            parent_run_id="trace-2",
            run_type="tool",
            start_time="2026-09-02T12:00:01Z",
            end_time="2026-09-02T12:00:02Z",
        ),
    )

    loader = LangSmithTraceExportFileLoader(LangSmithTraceExportFileConfig(path=path))

    assert [trace.id for trace in loader.load()] == ["trace-1", "trace-2"]
    assert loader.describe()["export_trace_count"] == 2
    trace_2 = loader.load().get_trace_by_id("trace-2")
    pointer = trace_2.root_spans[0].children[0].attributes["source_pointer"]
    assert isinstance(pointer, dict)
    assert pointer["line_number"] == 3


def test_loader_selects_newest_whole_traces_before_applying_bound(tmp_path):
    write_trace_export_file(
        tmp_path / "b.jsonl",
        trace_export_run_record(
            "trace-b",
            trace_id="trace-b",
            start_time="2026-09-02T10:00:00Z",
            end_time="2026-09-02T10:00:03Z",
        ),
    )
    write_trace_export_file(
        tmp_path / "a.jsonl",
        trace_export_run_record("trace-a", trace_id="trace-a", start_time="2026-09-01T11:00:00Z"),
    )
    loader = LangSmithTraceExportFileLoader(
        LangSmithTraceExportFileConfig(path=tmp_path, max_traces=1)
    )

    assert [trace.id for trace in loader.load()] == ["trace-b"]
    assert loader.describe()["export_trace_count"] == 2
    assert loader.describe()["truncated"] is True


def test_loader_bounds_merged_conversations_and_never_splits_a_thread(tmp_path):
    path = tmp_path / "all.jsonl"
    write_trace_export_file(
        path,
        *(
            trace_export_run_record(
                trace_id,
                trace_id=trace_id,
                start_time=start,
                end_time=None,
                custom_metadata={"thread_id": "x"} if thread else None,
            )
            for trace_id, start, thread in [
                ("oldest", "2026-08-31T12:00:00Z", False),
                ("thread-old", "2026-09-01T12:00:00Z", True),
                ("single", "2026-09-02T12:00:00Z", False),
                ("thread-new", "2026-09-03T12:00:00Z", True),
            ]
        ),
    )
    loader = LangSmithTraceExportFileLoader(LangSmithTraceExportFileConfig(path, max_traces=2))

    traces = list(loader.load())

    assert [trace.id for trace in traces] == ["session:x", "single"]
    assert [record["trace_id"] for record in _records(traces[0])] == ["thread-old", "thread-new"]
    assert all(loss.code != "merge_incomplete" for loss in traces[0].losses)
    description = loader.describe()
    assert description["export_trace_count"] == 4
    assert description["run_count"] == 3
    assert description["truncated"] is True
    assert (description["distinct_cases"], description["distinct_sessions"]) == (2, 1)


def test_loader_ranks_conversations_by_their_newest_root_like_a_live_load(tmp_path):
    path = tmp_path / "all.jsonl"
    write_trace_export_file(
        path,
        *(
            trace_export_run_record(
                trace_id,
                trace_id=trace_id,
                start_time=start,
                end_time=None,
                custom_metadata={"thread_id": "x"} if thread else None,
            )
            for trace_id, start, thread in [
                ("thread-old", "2026-09-01T12:00:00Z", True),
                ("single", "2026-09-02T12:00:00Z", False),
                ("thread-new", "2026-09-03T12:00:00Z", True),
            ]
        ),
    )
    loader = LangSmithTraceExportFileLoader(LangSmithTraceExportFileConfig(path, max_traces=1))

    # The live loader lists roots newest first, so thread x is selected first.
    (trace,) = loader.load()

    assert trace.id == "session:x"
    assert [record["trace_id"] for record in _records(trace)] == ["thread-old", "thread-new"]


def test_loader_merges_traces_that_share_a_thread(tmp_path):
    write_trace_export_file(
        tmp_path / "all.jsonl",
        trace_export_run_record("trace-1", custom_metadata={"thread_id": "t"}),
        trace_export_run_record(
            "trace-2",
            trace_id="trace-2",
            start_time="2026-09-01T12:01:00Z",
            end_time="2026-09-01T12:01:03Z",
            custom_metadata={"thread_id": "t"},
        ),
    )
    loader = LangSmithTraceExportFileLoader(LangSmithTraceExportFileConfig(tmp_path / "all.jsonl"))

    (trace,) = loader.load()

    assert trace.id == "session:t"
    assert (trace.session_id, trace.session_basis) == ("t", "langsmith.metadata.thread_id")
    (session_root,) = trace.root_spans
    assert session_root.kind is SpanKind.AGENT
    assert [span.id for span in session_root.children] == ["trace-1/trace-1", "trace-2/trace-2"]
    assert [record["trace_id"] for record in _records(trace)] == ["trace-1", "trace-2"]
    description = loader.describe()
    assert (description["trace_count"], description["export_trace_count"]) == (1, 2)
    assert description["truncated"] is False


def test_langchain_llm_run_under_a_chain_root_becomes_turns(tmp_path):
    messages = {
        "messages": [
            [
                {
                    "lc": 1,
                    "type": "constructor",
                    "id": ["langchain", "schema", "messages", "HumanMessage"],
                    "kwargs": {"content": "hi"},
                }
            ]
        ]
    }
    path = tmp_path / "langchain.jsonl"
    write_trace_export_file(
        path,
        trace_export_run_record("trace-1", inputs=messages),
        trace_export_run_record(
            "llm-1",
            parent_run_id="trace-1",
            run_type="llm",
            inputs=messages,
            outputs={"generations": [[{"text": "hello"}]]},
        ),
    )

    (trace,) = LangSmithTraceExportFileLoader(LangSmithTraceExportFileConfig(path)).load()

    (root,) = trace.root_spans
    assert root.kind is SpanKind.AGENT
    assert root.turns is not UNSET
    assert [
        (turn.role, turn.authorship, turn.content, turn.llm_span_ids) for turn in root.turns
    ] == [
        (TurnRole.USER, Authorship.UNKNOWN, "hi", []),
        (TurnRole.AGENT, Authorship.MODEL, "hello", ["llm-1"]),
    ]


def test_chain_root_without_a_session_is_promoted_to_the_run_root(tmp_path):
    path = tmp_path / "chain.jsonl"
    write_trace_export_file(path, trace_export_run_record("trace-1", run_type="chain"))

    (trace,) = LangSmithTraceExportFileLoader(LangSmithTraceExportFileConfig(path)).load()

    assert trace.id == "trace-1"
    assert trace.session_id is None
    (root,) = trace.root_spans
    assert root.kind is SpanKind.AGENT
    assert root.attributes["subtype"] == "CHAIN"
    # The export recorded no conversation input, so the promoted root has no turns.
    assert root.turns is UNSET


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("not json\n", "malformed JSON"),
        ("[]\n", "Run must be a JSON object"),
        ("\n", "contains no Runs"),
    ],
)
def test_loader_rejects_malformed_or_empty_jsonl(tmp_path, contents, message):
    path = tmp_path / "bad.jsonl"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(LangSmithTraceLoadError, match=message):
        LangSmithTraceExportFileLoader(LangSmithTraceExportFileConfig(path=path)).load()


def test_loader_requires_full_cli_export(tmp_path):
    record = trace_export_run_record("trace-1")
    del record["inputs"]
    path = tmp_path / "partial.jsonl"
    write_trace_export_file(path, record)

    with pytest.raises(LangSmithTraceLoadError, match=r"missing inputs.*--full"):
        LangSmithTraceExportFileLoader(LangSmithTraceExportFileConfig(path=path)).load()


def test_loader_rejects_multiple_trace_ids_in_a_directory_trace_file(tmp_path):
    path = tmp_path / "mixed.jsonl"
    write_trace_export_file(
        path,
        trace_export_run_record("trace-1"),
        trace_export_run_record("trace-2", trace_id="trace-2"),
    )

    with pytest.raises(LangSmithTraceLoadError, match="mixes trace ids"):
        LangSmithTraceExportFileLoader(LangSmithTraceExportFileConfig(path=tmp_path)).load()


def test_loader_rejects_duplicate_run_ids(tmp_path):
    record = trace_export_run_record("trace-1")
    path = tmp_path / "duplicate-runs.jsonl"
    write_trace_export_file(path, record, record)

    with pytest.raises(LangSmithTraceLoadError, match="duplicate Run id"):
        LangSmithTraceExportFileLoader(LangSmithTraceExportFileConfig(path=path)).load()


def test_loader_rejects_duplicate_trace_ids_across_files(tmp_path):
    write_trace_export_file(tmp_path / "a.jsonl", trace_export_run_record("trace-1"))
    write_trace_export_file(tmp_path / "b.jsonl", trace_export_run_record("trace-1"))

    with pytest.raises(LangSmithTraceLoadError, match="duplicate trace id"):
        LangSmithTraceExportFileLoader(LangSmithTraceExportFileConfig(path=tmp_path)).load()


def test_loader_rejects_missing_parent_as_an_incomplete_trace(tmp_path):
    path = tmp_path / "incomplete.jsonl"
    write_trace_export_file(
        path,
        trace_export_run_record("trace-1"),
        trace_export_run_record("tool-1", parent_run_id="missing", run_type="tool"),
    )

    with pytest.raises(LangSmithTraceLoadError, match=r"incomplete.*missing parent"):
        LangSmithTraceExportFileLoader(LangSmithTraceExportFileConfig(path=path)).load()


def test_loader_rejects_multiple_roots(tmp_path):
    path = tmp_path / "roots.jsonl"
    write_trace_export_file(
        path, trace_export_run_record("trace-1"), trace_export_run_record("other-root")
    )

    with pytest.raises(LangSmithTraceLoadError, match="exactly one root Run"):
        LangSmithTraceExportFileLoader(LangSmithTraceExportFileConfig(path=path)).load()


def test_loader_rejects_parent_cycles(tmp_path):
    path = tmp_path / "cycle.jsonl"
    write_trace_export_file(
        path,
        trace_export_run_record("trace-1"),
        trace_export_run_record("child-1", parent_run_id="child-2"),
        trace_export_run_record("child-2", parent_run_id="child-1"),
    )

    with pytest.raises(LangSmithTraceLoadError, match="parent cycle"):
        LangSmithTraceExportFileLoader(LangSmithTraceExportFileConfig(path=path)).load()


def test_loader_rejects_missing_paths_and_directories_without_exports(tmp_path):
    with pytest.raises(LangSmithTraceLoadError, match="does not exist"):
        LangSmithTraceExportFileLoader(
            LangSmithTraceExportFileConfig(path=tmp_path / "missing")
        ).load()
    with pytest.raises(LangSmithTraceLoadError, match="contains no .jsonl"):
        LangSmithTraceExportFileLoader(LangSmithTraceExportFileConfig(path=tmp_path)).load()


@pytest.mark.parametrize("max_traces", [0, -1])
def test_config_rejects_nonpositive_trace_bounds(max_traces):
    with pytest.raises(ValueError, match="max_traces must be at least 1"):
        LangSmithTraceExportFileConfig(path=Path("export"), max_traces=max_traces)


def test_loader_normalizes_timestamps_to_utc(tmp_path):
    path = tmp_path / "timezone.jsonl"
    write_trace_export_file(
        path,
        trace_export_run_record(
            "trace-1",
            start_time="2026-09-01T08:00:00-04:00",
            end_time="2026-09-01T08:00:01-04:00",
        ),
    )

    root = next(
        iter(LangSmithTraceExportFileLoader(LangSmithTraceExportFileConfig(path=path)).load())
    )

    assert root.root_spans[0].start_time == datetime(2026, 9, 1, 12, tzinfo=timezone.utc)


def test_tool_results_mark_errors_and_match_result_counts(tmp_path):
    root = trace_export_run_record("trace-1", inputs={"q": "x"}, outputs={"a": "y"})
    failed = trace_export_run_record(
        "tool-1",
        parent_run_id="trace-1",
        name="search",
        run_type="tool",
        start_time="2026-09-01T12:00:02Z",
        end_time="2026-09-01T12:00:03Z",
        outputs={"output": "boom"},
        error="TimeoutError",
        status="error",
    )
    ok = trace_export_run_record(
        "tool-2",
        parent_run_id="trace-1",
        name="search",
        run_type="tool",
        start_time="2026-09-01T12:00:04Z",
        end_time="2026-09-01T12:00:05Z",
        outputs={"output": "fine"},
    )
    write_trace_export_file(tmp_path / "trace-1.jsonl", root, failed, ok)
    trace = next(
        iter(LangSmithTraceExportFileLoader(LangSmithTraceExportFileConfig(path=tmp_path)).load())
    )
    spans = {span.id: span for span in trace.root_spans[0].children}
    calls = {key: span.tool_call for key, span in spans.items() if span.tool_call is not None}
    assert calls.keys() == spans.keys()
    assert calls["tool-1"].results == [ToolResult(content=spans["tool-1"].output, is_error=True)]
    assert calls["tool-2"].results == [ToolResult(content=spans["tool-2"].output, is_error=None)]
    for call in calls.values():
        assert call.results is not UNSET
        assert len(call.results) == call.result_count
