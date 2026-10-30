#!/usr/bin/env -S uv run
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the bundled sample corpus.

The corpus has to do a specific job: make every one of the tool-issue signal's
nineteen finding types fire, give the anomaly-and-pattern signal enough recurrence to populate all of its grouping
sections — all while staying small enough to read. Hand-writing fourteen traces of inline JSON
to hit those targets is where mistakes hide, so it is generated here instead
and the output is committed. ``tests/test_shipped_corpora.py`` regenerates into a
tmpdir and asserts byte equality, so the data can never drift from this file.

Fully deterministic: no RNG, no clock. Run it with::

    ./tools/generate_demo_corpus.py

The synthetic venue is a "DocOps agent" with five catalogued tools. It uses the
literal name ``CodeExecutionTool`` so the built-in traceback handling is exercised.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, TypedDict, Unpack

from pydantic import JsonValue

from compass.traces import UNSET, Span, SpanKind, ToolCall, Trace, TraceAggregate
from trace_ingest.models import ToolDefinition, ToolResult

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "src" / "compass" / "data"

TIMEOUT = "Error: connection timed out after 30s"

TOOL_CATALOG: dict[str, Any] = {
    "FileSearchTool": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "FileReadTool": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    },
    "CodeExecutionTool": {
        "type": "object",
        "properties": {
            "code": {"type": "string"},
            "mode": {"type": "string", "enum": ["python", "shell"]},
        },
        "required": ["code"],
        "additionalProperties": False,
    },
    "DatabaseQueryTool": {
        "type": "object",
        "properties": {"sql": {"type": "string"}},
        "required": ["sql"],
        "additionalProperties": False,
    },
    # A known tool whose argument schema was never captured. Enables
    # unknown_tool detection while abstaining from argument checks.
    "SessionTool": None,
}
TOOL_DEFINITIONS = [
    ToolDefinition(
        name=name,
        input_schema=schema,
        raw={"name": name, "parameters": schema},
    )
    for name, schema in TOOL_CATALOG.items()
]

VERDICT_DEAD_END = "dead_end: no artifact produced"
VERDICT_COMPLETED = "completed"
MISSING_RESULT = object()


class _CallFields(TypedDict, total=False):
    result_missing: bool
    result_count: int
    explicit_error: bool
    outcome_marker: str
    extra: dict[str, JsonValue]
    result_id: str
    instrumentation_alias_of: str
    prior_user_text: str
    returned_data: JsonValue


class TraceBuilder:
    """Build one canonical trace for the bundled synthetic corpus."""

    def __init__(self, trace_id: str, case_id: str, task_text: str) -> None:
        self.trace_id = trace_id
        self.case_id = case_id
        self.task_text = task_text
        self.spans: list[Span] = []
        self.orphans: list[dict[str, Any]] = []
        self.extras: dict[str, Any] = {}
        self.call_count = 0
        self.duration_ms = 0.0
        self._span_ids: set[str] = set()

    def _span_id(self, source_id: str) -> str:
        if source_id not in self._span_ids:
            self._span_ids.add(source_id)
            return source_id
        occurrence = 2
        while f"{source_id}#{occurrence}" in self._span_ids:
            occurrence += 1
        span_id = f"{source_id}#{occurrence}"
        self._span_ids.add(span_id)
        return span_id

    def call(
        self,
        tool_name: str,
        arguments: JsonValue,
        *,
        result: object = MISSING_RESULT,
        call_id: str | None = None,
        duration_ms: float = 120.0,
        **fields: Unpack[_CallFields],
    ) -> TraceBuilder:
        index = self.call_count
        self.call_count += 1
        resolved_call_id = call_id or f"{self.trace_id}#{index}"
        result_is_missing = result is MISSING_RESULT or fields.get("result_missing") is True
        if fields.get("result_count") == 0:
            result_is_missing = True
        result_count = 0 if result_is_missing else int(fields.get("result_count", 1))
        explicit_error = fields.get("explicit_error")
        self.duration_ms += duration_ms

        span_fields: dict[str, Any] = {
            "id": self._span_id(resolved_call_id),
            "kind": SpanKind.TOOL,
            "tool_name": tool_name,
            "input": arguments,
            "error": (
                str(fields.get("outcome_marker") or "explicit_error")
                if explicit_error is True
                else None
            ),
            "attributes": {
                "name": tool_name,
                "subtype": "tool",
                "summary": "",
                "explicit_error": explicit_error,
                "outcome_marker": fields.get("outcome_marker"),
                "duration_ms": duration_ms,
                "source_pointer": {"trace_id": self.trace_id, "call_index": index},
                "extra": dict(fields.get("extra") or {}),
            },
            "tool_call": ToolCall(
                call_id=resolved_call_id,
                index=index,
                result_id=fields.get("result_id"),
                result_count=result_count,
                instrumentation_alias_of=fields.get("instrumentation_alias_of"),
                prior_user_text=fields.get("prior_user_text") or self.task_text,
                returned_data=fields.get("returned_data", UNSET),
                # Each matched result carries the recorded output; a duplicate
                # delivery repeats it. is_error mirrors the recorded flag, if any.
                results=[
                    ToolResult(content=result, is_error=explicit_error) for _ in range(result_count)
                ],
            ),
        }
        if not result_is_missing:
            span_fields["output"] = result
        self.spans.append(Span(**span_fields))
        return self

    def step(
        self,
        step_type: str,
        name: str,
        *,
        content: str = "",
    ) -> TraceBuilder:
        step_index = len(self.spans)
        kind = (
            SpanKind.EVALUATOR
            if step_type == "evaluation"
            else SpanKind.AGENT
            if step_type in {"agent", "agent_step", "planning", "user"}
            else SpanKind.UNKNOWN
        )
        span_fields: dict[str, Any] = {
            "id": self._span_id(f"{self.trace_id}#step-{step_index}"),
            "kind": kind,
            "attributes": {
                "name": name or None,
                "subtype": step_type,
                "summary": content,
                "source_pointer": {},
                "extra": {},
            },
        }
        if content:
            span_fields["output"] = content
        self.spans.append(Span(**span_fields))
        return self

    def build(
        self,
        *,
        verdict: str,
        cost: float,
        metrics: dict[str, float] | None = None,
        complete_provenance_context: bool = False,
    ) -> Trace:
        attributes: dict[str, Any] = {
            "complete_provenance_context": complete_provenance_context,
            "orphan_results": self.orphans,
            "source_pointer": {"dataset": "docops-sample", "run_id": self.trace_id},
            "metrics": metrics or {},
            "extra": self.extras,
            "task_text": self.task_text,
            "observed_verdict": verdict,
        }
        # The agent root declares the catalogued tools; every step runs under it.
        agent = Span(
            id=self._span_id(f"{self.trace_id}#agent"),
            kind=SpanKind.AGENT,
            attributes={"name": "docops-agent", "source_pointer": {}},
            tool_definitions=TOOL_DEFINITIONS,
            children=self.spans,
        )
        return Trace(
            id=self.trace_id,
            case_id=self.case_id,
            root_spans=[agent],
            aggregate=TraceAggregate(cost_usd=cost, latency_ms=self.duration_ms or None),
            evaluator_results={"observed_verdict": verdict},
            attributes=attributes,
        )


def _plan(builder: TraceBuilder, text: str) -> TraceBuilder:
    return builder.step("planning", "plan", content=text)


def _close(builder: TraceBuilder, text: str) -> TraceBuilder:
    return builder.step("evaluation", "boundary", content=text)


# -- the traces ------------------------------------------------------------


def search_trace(suffix: str, case: str) -> Trace:
    """Search-shaped work. Recurs three times to qualify two card types.

    Contributes ``explicit_tool_failure`` (error_prefix) and ``unknown_tool``
    across three independent logical cases, which is what makes those two
    cards eligible for insight compilation.
    """
    task = f"Locate the {suffix} specification and remove the stale copy."
    b = TraceBuilder(f"docops-search-{suffix}", case, task)
    _plan(b, f"Search for the {suffix} spec, then delete the stale duplicate.")
    b.call(
        "FileSearchTool",
        {"query": f"{suffix} specification", "limit": 5},
        result={"content": f"3 matches for {suffix}", "returned_data": True},
        duration_ms=810.0,
    )
    b.call(
        "FileSearchTool",
        {"query": f"{suffix} duplicate"},
        result={"content": TIMEOUT},
        duration_ms=30_000.0,
    )
    # Not in the catalog: fires unknown_tool.
    b.call(
        "FileDeleteTool",
        {"path": f"/docs/{suffix}-old.md"},
        result={"content": "deleted"},
        duration_ms=90.0,
    )
    _close(b, "The stale copy was removed successfully.")
    return b.build(verdict=VERDICT_DEAD_END, cost=0.42, metrics={"turn_count": 5.0})


def contract_trace() -> Trace:
    """One trace covering all six argument-contract findings."""
    task = "Search the archive and run the mesh check."
    b = TraceBuilder("docops-contract", "case-contract", task)
    _plan(b, "Search, then execute the verification script.")
    # missing_required_argument: no 'query'
    b.call("FileSearchTool", {"limit": 3}, result={"content": "rejected: missing input query"})
    # unknown_argument: 'recursive' under additionalProperties:false
    b.call(
        "FileSearchTool",
        {"query": "archive", "recursive": True},
        result={"content": "rejected: unknown argument"},
    )
    # argument_type_mismatch: limit is a string
    b.call(
        "FileSearchTool",
        {"query": "archive", "limit": "five"},
        result={"content": "Error: limit must be an integer"},
    )
    # json_schema_violation: violates `minimum`, which is not one of the four
    # specially named validators
    b.call(
        "FileSearchTool",
        {"query": "archive", "limit": -3},
        result={"content": "Error: limit must be positive"},
    )
    # argument_enum_violation: mode outside the enum
    b.call(
        "CodeExecutionTool",
        {"code": "check()", "mode": "turbo"},
        result={"content": "Error: unsupported mode"},
    )
    # malformed_tool_call: arguments are not an object at all
    b.call("DatabaseQueryTool", "sql=SELECT 1", result={"content": "Error: malformed request"})
    _close(b, "Unable to complete the verification.")
    return b.build(verdict=VERDICT_DEAD_END, cost=0.31, metrics={"turn_count": 8.0})


def instrumentation_trace() -> Trace:
    """The five instrumentation-integrity findings."""
    task = "Read the changelog and summarise it."
    b = TraceBuilder("docops-instrumentation", "case-instrumentation", task)
    _plan(b, "Read the changelog, then summarise.")
    b.call(
        "FileReadTool",
        {"path": "/docs/CHANGELOG.md"},
        result={"content": "v1.2 released"},
        call_id="docops-instrumentation#dup",
    )
    # duplicate_call_id: the harness reused an id
    b.call(
        "FileReadTool",
        {"path": "/docs/CHANGELOG-2.md"},
        result={"content": "v1.1 released"},
        call_id="docops-instrumentation#dup",
    )
    # missing_tool_result: no result key at all
    b.call("FileReadTool", {"path": "/docs/NOTES.md"})
    # duplicate_tool_result
    b.call(
        "FileReadTool", {"path": "/docs/README.md"}, result={"content": "readme"}, result_count=2
    )
    # call_result_id_mismatch: the result claims to answer a different call
    b.call(
        "FileReadTool", {"path": "/docs/LICENSE"}, result={"content": "license"}, result_id="res-88"
    )
    # mapped_instrumentation_alias
    b.call(
        "UnknownTool",
        {"code": "print(1)", "mode": "python"},
        result={"content": "1"},
        instrumentation_alias_of="CodeExecutionTool",
    )
    # orphan_tool_result
    b.orphans.append(
        {
            "result_id": "docops-instrumentation#orphan",
            "tool_name": "FileReadTool",
            "content": "an unmatched result",
            "source_pointer": {"trace_id": "docops-instrumentation", "message_index": 21},
        }
    )
    _close(b, "Summary produced.")
    return b.build(verdict=VERDICT_COMPLETED, cost=0.55, metrics={"turn_count": 9.0})


def repeated_retry_trace() -> Trace:
    """Three byte-identical failing calls: repeated_identical_failed_call."""
    task = "Pull the release table from the docs database."
    b = TraceBuilder("docops-retry-identical", "case-retry-identical", task)
    _plan(b, "Query the release table.")
    for _ in range(3):
        b.call(
            "DatabaseQueryTool",
            {"sql": "SELECT * FROM releases"},
            result={"content": TIMEOUT},
            duration_ms=30_000.0,
        )
    _close(b, "Could not reach the database.")
    return b.build(verdict=VERDICT_DEAD_END, cost=0.90, metrics={"turn_count": 6.0})


def modified_retry_trace() -> Trace:
    """Three different argument sets, same failure class: modified_retry_same_failure."""
    task = "Open the design document."
    b = TraceBuilder("docops-retry-modified", "case-retry-modified", task)
    _plan(b, "Try the likely paths for the design document.")
    for path in ("/docs/design.md", "/docs/design/index.md", "/design.md"):
        b.call(
            "FileReadTool", {"path": path}, result={"content": "Error: no such file or directory"}
        )
    _close(b, "The design document could not be located.")
    return b.build(verdict=VERDICT_DEAD_END, cost=0.22, metrics={"turn_count": 5.0})


def placeholder_trace() -> Trace:
    """A failed call whose argument was never substituted."""
    task = "Export the report to the configured output directory."
    b = TraceBuilder("docops-placeholder", "case-placeholder", task)
    _plan(b, "Write the report to the output directory.")
    # PLACEHOLDER matches the *whole* stripped value, so the argument must be
    # exactly the unsubstituted token.
    b.call(
        "FileReadTool",
        {"path": "<output_dir>"},
        result={"content": "Error: no such file or directory"},
    )
    b.call("FileReadTool", {"path": "{{REPORT_PATH}}"}, result={"content": "Error: invalid path"})
    _close(b, "The export failed.")
    return b.build(verdict=VERDICT_DEAD_END, cost=0.18, metrics={"turn_count": 4.0})


def provenance_trace() -> Trace:
    """An identifier the agent invented, explicitly rejected by the tool.

    Requires complete_provenance_context: the conclusion "ungrounded" is only
    sound when every user message and prior result was captured. The identifier
    must appear in the rejection text and nowhere in the prior context.
    """
    task = "Open the current session document and append the release note."
    b = TraceBuilder("docops-provenance", "case-provenance", task)
    _plan(b, "Open the session document, then append.")
    b.call(
        "FileSearchTool",
        {"query": "session document"},
        result={"content": "1 match: /docs/session.md", "returned_data": True},
    )
    b.call(
        "SessionTool",
        {"document_id": "DOC-99213"},
        result={"content": "Error: invalid document DOC-99213: not found"},
    )
    _close(b, "Could not open the document.")
    return b.build(
        verdict=VERDICT_DEAD_END,
        cost=0.27,
        metrics={"turn_count": 5.0},
        complete_provenance_context=True,
    )


def state_trace(suffix: str, case: str) -> Trace:
    """A tool refusing to act because a precondition is unmet."""
    task = f"Append the {suffix} note to the active document."
    b = TraceBuilder(f"docops-state-{suffix}", case, task)
    _plan(b, "Append the note to the active document.")
    b.call(
        "SessionTool",
        {"action": "append", "text": f"{suffix} note"},
        result={"content": "[NO_ACTIVE_SESSION] open a session before calling append"},
    )
    b.call(
        "SessionTool",
        {"action": "open"},
        result={"content": "Error: you must open a workspace first"},
    )
    _close(b, "The note was appended successfully.")
    return b.build(verdict=VERDICT_DEAD_END, cost=0.14, metrics={"turn_count": 4.0})


def code_trace() -> Trace:
    """A Python traceback, decoded only because the tool is a code runner."""
    task = "Run the mesh verification script."
    b = TraceBuilder("docops-code", "case-code", task)
    _plan(b, "Run the verification script.")
    b.call(
        "CodeExecutionTool",
        {"code": "import mesh; mesh.check()", "mode": "python"},
        result={
            "content": "Traceback (most recent call last):\n"
            '  File "<stdin>", line 1, in <module>\n'
            "ModuleNotFoundError: No module named 'mesh'"
        },
        duration_ms=2210.0,
    )
    b.call(
        "CodeExecutionTool",
        {"code": "print('ok')", "mode": "python"},
        result={"content": "ok", "returned_data": True},
    )
    b.step("agent", "reflect", content="The mesh module is not installed in this environment.")
    _close(b, "Verification completed successfully.")
    return b.build(verdict=VERDICT_COMPLETED, cost=1.10, metrics={"turn_count": 6.0})


def stagnation_trace(suffix: str, case: str) -> Trace:
    """The same failing search repeated with no progress between attempts.

    Fires ``repeated_identical_failed_call``, and gives the anomaly-and-pattern signal a trajectory whose
    terminal evaluation claims success the calls do not support.
    """
    task = f"Find the {suffix} owner in the directory."
    b = TraceBuilder(f"docops-stagnation-{suffix}", case, task)
    b.step("planning", "plan", content="Search the directory repeatedly.")
    b.call(
        "FileSearchTool",
        {"query": f"{suffix} owner"},
        result={"content": "no matches", "returned_data": False},
    )
    for _ in range(4):
        b.call(
            "FileSearchTool",
            {"query": f"{suffix} owner"},
            result={"content": "no matches", "returned_data": False},
        )
    b.step("evaluation", "boundary", content="The owner was identified successfully.")
    return b.build(verdict=VERDICT_DEAD_END, cost=0.35, metrics={"turn_count": 7.0})


def outlier_trace() -> Trace:
    """One deliberately extreme trace so the 2% contamination default flags something.

    Long, repetitive, and one very large output — the shape the anomaly-and-pattern feature
    vector is built to separate.
    """
    task = "Reindex the entire documentation tree."
    b = TraceBuilder("docops-outlier", "case-outlier", task)
    _plan(b, "Walk the whole tree and reindex.")
    for _ in range(28):
        b.call(
            "FileSearchTool",
            {"query": "*"},
            result={"content": "x" * 40_000, "returned_data": False},
            duration_ms=9_000.0,
        )
    _close(b, "Reindex incomplete.")
    return b.build(verdict=VERDICT_DEAD_END, cost=9.90, metrics={"turn_count": 30.0})


def clean_trace(suffix: str, case: str) -> Trace:
    """A trace with nothing wrong, so 'no findings' is represented too."""
    task = f"Summarise the {suffix} guide."
    b = TraceBuilder(f"docops-clean-{suffix}", case, task)
    _plan(b, "Find the guide, read it, summarise.")
    b.call(
        "FileSearchTool",
        {"query": f"{suffix} guide", "limit": 3},
        result={"content": f"1 match: /docs/{suffix}.md", "returned_data": True},
        duration_ms=340.0,
    )
    b.call(
        "FileReadTool",
        {"path": f"/docs/{suffix}.md"},
        result={"content": f"The {suffix} guide explains the workflow.", "returned_data": True},
        duration_ms=210.0,
    )
    b.step("agent", "reflect", content="The guide covers the workflow end to end.")
    _close(b, "Summary produced successfully.")
    return b.build(verdict=VERDICT_COMPLETED, cost=0.20, metrics={"turn_count": 4.0})


def build_corpus() -> list[Trace]:
    """Fourteen traces over twelve logical cases.

    Two cases (``case-search-a`` and ``case-state-a``) cover two traces each,
    so ``case_id`` is demonstrably not a synonym for ``trace_id`` —
    which is what the card-eligibility gate depends on.
    """
    return [
        search_trace("alpha", "case-search-a"),
        search_trace("alpha-retry", "case-search-a"),  # same logical case
        search_trace("beta", "case-search-b"),
        search_trace("gamma", "case-search-c"),
        contract_trace(),
        instrumentation_trace(),
        repeated_retry_trace(),
        modified_retry_trace(),
        placeholder_trace(),
        provenance_trace(),
        state_trace("a", "case-state-a"),
        state_trace("a-retry", "case-state-a"),  # same logical case
        state_trace("b", "case-state-b"),
        code_trace(),
        stagnation_trace("ops", "case-stagnation"),
        outlier_trace(),
        clean_trace("onboarding", "case-clean-a"),
        clean_trace("release", "case-clean-b"),
    ]


def write_outputs(data_dir: Path) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = data_dir / "sample_corpus.jsonl"
    lines = [
        json.dumps(
            trace.model_dump(mode="json", exclude_unset=True),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for trace in build_corpus()
    ]
    corpus_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"corpus": corpus_path}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=DATA_DIR, help="output directory (default: bundled data dir)"
    )
    args = parser.parse_args()

    written = write_outputs(args.out)
    corpus = build_corpus()
    print(f"traces           : {len(corpus)}")
    print(
        "calls            : "
        f"{sum(span.kind is SpanKind.TOOL for trace in corpus for span in trace.root_spans[0].children)}"
    )
    print(f"cases            : {len({t.case_id for t in corpus})}")
    for name, path in written.items():
        print(
            f"{name:17}: {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
