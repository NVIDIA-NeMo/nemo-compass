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
from compass.traces import SpanKind
from trace_ingest.models import UNSET, ToolResult


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
    assert trace.id == "trace-1"
    assert trace.aggregate.cost_usd == 0.025
    assert trace.aggregate.latency_ms == 3000
    assert (trace.session_id, trace.session_basis) == (
        "session-7",
        "langsmith.metadata.session_id",
    )
    assert trace.attributes["source_pointer"] == {
        "provider": "langsmith",
        "export_path": str(tmp_path.resolve()),
        "export_file": "trace-1.jsonl",
        "trace_id": "trace-1",
    }
    root_span = trace.root_spans[0]
    assert root_span.input == {"messages": [{"role": "user", "content": "find it"}]}
    assert [span.id for span in root_span.children] == ["llm-1", "tool-1"]
    assert [span.kind for span in root_span.children] == [SpanKind.LLM, SpanKind.TOOL]
    langsmith = root_span.attributes["langsmith"]
    assert isinstance(langsmith, dict)
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
