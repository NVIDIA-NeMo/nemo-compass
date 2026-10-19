# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Run normalization for the LangSmith API and trace export files."""

from __future__ import annotations

import heapq
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from pydantic import JsonValue

from trace_ingest.models import (
    UNSET,
    Span,
    SpanKind,
    ToolCall,
    Trace,
    TraceAggregate,
)

LANGSMITH_DEFAULT_MAX_TRACES = 100


class _LangSmithRun(Protocol):
    """Run-shaped data shared by SDK responses and parsed trace exports."""

    @property
    def id(self) -> object: ...

    @property
    def parent_run_id(self) -> object: ...

    @property
    def start_time(self) -> object: ...


class LangSmithTraceLoadError(RuntimeError):
    """A LangSmith source could not be resolved, fetched, or normalized."""


@dataclass(frozen=True)
class LangSmithTraceLoadReport:
    """Diagnostics produced by the most recent load."""

    trace_count: int = 0
    run_count: int = 0
    unresolved_parent_count: int = 0


def normalize_trace(
    trace_id: str,
    selected_root: _LangSmithRun,
    runs: Sequence[_LangSmithRun],
    *,
    source_pointer: Mapping[str, JsonValue],
    langsmith_attributes: Mapping[str, JsonValue],
    run_source_pointers: Mapping[str, Mapping[str, JsonValue]] | None = None,
) -> tuple[Trace, int]:
    """Normalize one complete tree from the LangSmith API or a trace export file."""

    by_id: dict[str, _LangSmithRun] = {}
    for run in runs:
        run_id = required_id(run, "id", f"LangSmith trace {trace_id!r} Run")
        if run_id in by_id:
            raise LangSmithTraceLoadError(
                f"LangSmith trace {trace_id!r} contains duplicate Run id {run_id!r}"
            )
        by_id[run_id] = run

    root_id = required_id(selected_root, "id", "LangSmith selected root Run")
    root = by_id.get(root_id)
    if root is None:
        raise LangSmithTraceLoadError(
            f"LangSmith trace {trace_id!r} did not include its selected root Run"
        )

    effective_parents: dict[str, str | None] = {}
    unresolved: dict[str, str] = {}
    for run_id, run in by_id.items():
        parent = getattr(run, "parent_run_id", None)
        parent_id = str(parent) if parent not in (None, "") else None
        if parent_id is not None and parent_id not in by_id:
            unresolved[run_id] = parent_id
            parent_id = None
        effective_parents[run_id] = parent_id

    if effective_parents[root_id] is not None:
        raise LangSmithTraceLoadError(
            f"LangSmith trace {trace_id!r} selected root Run has a parent"
        )

    ordered = _canonical_run_order(trace_id, by_id, effective_parents)
    base_pointer: dict[str, JsonValue] = {**source_pointer, "trace_id": trace_id}
    normalized_by_id: dict[str, Span] = {}
    root_spans: list[Span] = []
    feedback_by_name: dict[str, dict[str, JsonValue]] = {}
    tool_call_index = 0
    for run in ordered:
        run_id = required_id(run, "id", "LangSmith Run")
        normalized = _normalize_run(
            run,
            unresolved_parent_id=unresolved.get(run_id),
            base_pointer=base_pointer,
            source_pointer=(run_source_pointers or {}).get(run_id),
        )
        feedback = getattr(run, "feedback_stats", None)
        if feedback is not None:
            if not isinstance(feedback, Mapping):
                raise LangSmithTraceLoadError(
                    f"LangSmith Run {run_id!r} feedback_stats must be an object or null"
                )
            for name, stats in feedback.items():
                if stats is None:
                    continue
                if not isinstance(name, str) or not name or not isinstance(stats, Mapping):
                    raise LangSmithTraceLoadError(
                        f"LangSmith Run {run_id!r} has invalid feedback statistics {name!r}"
                    )
                if stats:
                    # Keep per-run aggregates intact, including repeated feedback names.
                    key = f"langsmith.feedback_stats.{name}"
                    feedback_by_name.setdefault(key, {})[run_id] = _json_value(stats)
        if normalized.kind is SpanKind.TOOL:
            normalized.tool_call = ToolCall(
                call_id=run_id,
                index=tool_call_index,
                result_count=0 if normalized.output is UNSET else 1,
            )
            tool_call_index += 1
        normalized_by_id[run_id] = normalized
        parent_id = effective_parents[run_id]
        if parent_id is None:
            root_spans.append(normalized)
        else:
            normalized_by_id[parent_id].children.append(normalized)

    cost = _trace_cost_usd(runs, trace_id)
    logical_case_id = _logical_case_id(root)
    attributes: dict[str, JsonValue] = {
        "source_pointer": base_pointer,
        "langsmith": dict(langsmith_attributes),
    }
    if logical_case_id is not None:
        attributes["logical_case_id"] = logical_case_id
    return (
        Trace(
            id=trace_id,
            root_spans=root_spans,
            aggregate=TraceAggregate(
                cost_usd=cost,
                latency_ms=_trace_latency_ms(ordered),
            ),
            attributes=attributes,
            evaluator_results=dict(feedback_by_name),
        ),
        len(unresolved),
    )


def _canonical_run_order(
    trace_id: str,
    by_id: Mapping[str, _LangSmithRun],
    effective_parents: Mapping[str, str | None],
) -> list[_LangSmithRun]:
    children: dict[str, list[str]] = {run_id: [] for run_id in by_id}
    indegree = {run_id: 0 for run_id in by_id}
    for run_id, parent_id in effective_parents.items():
        if parent_id is not None:
            children[parent_id].append(run_id)
            indegree[run_id] += 1

    ready: list[tuple[datetime, str, str]] = []
    for run_id, degree in indegree.items():
        if degree == 0:
            heapq.heappush(ready, _run_sort_key(by_id[run_id], run_id))

    ordered: list[_LangSmithRun] = []
    while ready:
        _, _, run_id = heapq.heappop(ready)
        ordered.append(by_id[run_id])
        for child_id in children[run_id]:
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                heapq.heappush(ready, _run_sort_key(by_id[child_id], child_id))

    if len(ordered) != len(by_id):
        raise LangSmithTraceLoadError(f"LangSmith trace {trace_id!r} contains a parent cycle")
    return ordered


def trace_sort_key(root: _LangSmithRun, trace_id: str) -> tuple[datetime, str]:
    """Return the stable ordering key shared by both LangSmith loaders."""

    return (_run_started_at(root), trace_id)


def _run_sort_key(run: _LangSmithRun, run_id: str) -> tuple[datetime, str, str]:
    dotted_order = str(getattr(run, "dotted_order", None) or "")
    return (_run_started_at(run), dotted_order, run_id)


def _run_started_at(run: _LangSmithRun) -> datetime:
    value = getattr(run, "start_time", None)
    if not isinstance(value, datetime):
        run_id = str(getattr(run, "id", "<unknown>"))
        raise LangSmithTraceLoadError(f"LangSmith Run {run_id!r} has no start time")
    if value.tzinfo is None or value.utcoffset() is None:
        run_id = str(getattr(run, "id", "<unknown>"))
        raise LangSmithTraceLoadError(
            f"LangSmith Run {run_id!r} start time must include a timezone"
        )
    return value.astimezone(timezone.utc)


def _normalize_run(
    run: _LangSmithRun,
    *,
    unresolved_parent_id: str | None,
    base_pointer: Mapping[str, JsonValue],
    source_pointer: Mapping[str, JsonValue] | None,
) -> Span:
    run_id = required_id(run, "id", "LangSmith Run")
    kind, subtype = _map_run_kind(getattr(run, "run_type", None))
    started_at = _run_started_at(run)
    ended_at = _optional_datetime(getattr(run, "end_time", None), run_id, "end")
    duration_ms = None
    if ended_at is not None:
        duration_ms = (ended_at - started_at).total_seconds() * 1000
        if duration_ms < 0:
            raise LangSmithTraceLoadError(f"LangSmith Run {run_id!r} ends before it starts")

    pointer: dict[str, JsonValue] = {
        **base_pointer,
        "run_id": run_id,
        **(source_pointer or {}),
    }
    app_path = getattr(run, "app_path", None)
    if app_path not in (None, ""):
        pointer["app_path"] = str(app_path)
    if unresolved_parent_id is not None:
        pointer["unresolved_parent_run_id"] = unresolved_parent_id

    status, error = _map_run_status(run)
    name = getattr(run, "name", None)
    name = str(name) if name not in (None, "") else None
    return Span(
        id=run_id,
        kind=kind,
        start_time=started_at,
        end_time=ended_at,
        input=_optional_json_field(getattr(run, "inputs", None)),
        output=_optional_json_field(getattr(run, "outputs", None)),
        tool_name=name if kind is SpanKind.TOOL else None,
        cost_usd=_run_cost_usd(run, run_id),
        error=error,
        attributes={
            "name": name,
            "subtype": subtype,
            "duration_ms": duration_ms,
            "source_pointer": pointer,
            "status": status,
            "langsmith": {
                "run_type": str(getattr(run, "run_type", None) or ""),
                "tags": _json_value(getattr(run, "tags", None)),
                "extra": _json_value(getattr(run, "extra", None)),
                "feedback_stats": _json_value(getattr(run, "feedback_stats", None)),
                "dotted_order": str(getattr(run, "dotted_order", None) or "") or None,
            },
        },
    )


def _map_run_kind(run_type: object) -> tuple[SpanKind, str | None]:
    provider_type = str(run_type or "UNKNOWN").upper()
    aliases = {
        "LLM": SpanKind.LLM,
        "TOOL": SpanKind.TOOL,
        "AGENT": SpanKind.AGENT,
        "CHAIN": SpanKind.CHAIN,
        "RETRIEVER": SpanKind.RETRIEVER,
        "EMBEDDING": SpanKind.EMBEDDING,
        "RERANKER": SpanKind.RERANKER,
        "EVALUATOR": SpanKind.EVALUATOR,
        "GUARDRAIL": SpanKind.GUARDRAIL,
    }
    kind = aliases.get(provider_type)
    if kind is None:
        return SpanKind.UNKNOWN, provider_type
    return kind, None


def _map_run_status(run: _LangSmithRun) -> tuple[str, str | None]:
    error = getattr(run, "error", None)
    if error not in (None, ""):
        return "ERROR", str(error)
    status = str(getattr(run, "status", None) or "").lower()
    if status == "success":
        return "SUCCESS", None
    if status == "error":
        return "ERROR", "LangSmith Run status ERROR"
    if status in {"cancelled", "canceled"}:
        return "CANCELLED", None
    return "UNKNOWN", None


def _optional_datetime(value: object, run_id: str, field_name: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, datetime):
        raise LangSmithTraceLoadError(
            f"LangSmith Run {run_id!r} {field_name} time is not a datetime"
        )
    if value.tzinfo is None or value.utcoffset() is None:
        raise LangSmithTraceLoadError(
            f"LangSmith Run {run_id!r} {field_name} time must include a timezone"
        )
    return value.astimezone(timezone.utc)


def _optional_json_field(value: object) -> JsonValue | UNSET:
    if value is None:
        return UNSET
    return _json_value(value)


def walk_spans(spans: Sequence[Span]) -> Iterator[Span]:
    """Yield every normalized span in stable depth-first order."""

    pending = list(reversed(spans))
    while pending:
        span = pending.pop()
        yield span
        pending.extend(reversed(span.children))


def _json_value(value: object) -> JsonValue:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise LangSmithTraceLoadError(
            f"LangSmith Run input/output is not JSON-serializable: {error}"
        ) from error


def _trace_cost_usd(runs: Sequence[_LangSmithRun], trace_id: str) -> float | None:
    costs: list[float] = []
    for run in runs:
        value = getattr(run, "total_cost", None)
        if value is None:
            continue
        try:
            cost = float(value)
        except (TypeError, ValueError) as error:
            raise LangSmithTraceLoadError(
                f"LangSmith trace {trace_id!r} has invalid Run cost {value!r}"
            ) from error
        if not math.isfinite(cost) or cost < 0:
            raise LangSmithTraceLoadError(
                f"LangSmith trace {trace_id!r} has invalid Run cost {value!r}"
            )
        costs.append(cost)
    return sum(costs) if costs else None


def _run_cost_usd(run: _LangSmithRun, run_id: str) -> float | None:
    value = getattr(run, "total_cost", None)
    if value is None:
        return None
    try:
        cost = float(value)
    except (TypeError, ValueError) as error:
        raise LangSmithTraceLoadError(
            f"LangSmith Run {run_id!r} has invalid cost {value!r}"
        ) from error
    if not math.isfinite(cost) or cost < 0:
        raise LangSmithTraceLoadError(f"LangSmith Run {run_id!r} has invalid cost {value!r}")
    return cost


def _trace_latency_ms(runs: Sequence[_LangSmithRun]) -> float | None:
    starts: list[datetime] = []
    ends: list[datetime] = []
    for run in runs:
        run_id = required_id(run, "id", "LangSmith Run")
        starts.append(_run_started_at(run))
        end = _optional_datetime(getattr(run, "end_time", None), run_id, "end")
        if end is not None:
            ends.append(end)
    if not starts or not ends:
        return None
    return (max(ends) - min(starts)).total_seconds() * 1_000


def _logical_case_id(root: _LangSmithRun) -> str | None:
    extra = getattr(root, "extra", None)
    if not isinstance(extra, Mapping):
        return None
    metadata = extra.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    for key in ("thread_id", "session_id", "conversation_id"):
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def required_id(value: object, field_name: str, context: str) -> str:
    """Read one required provider identifier as a string."""

    identifier = getattr(value, field_name, None)
    if identifier in (None, ""):
        raise LangSmithTraceLoadError(f"{context} has no {field_name.replace('_', ' ')}")
    return str(identifier)
