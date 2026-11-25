# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read complete Arize AX traces through the public v2 REST API."""

from __future__ import annotations

import heapq
import json
import math
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue

from trace_ingest.loaders.helper.merge import Record, assemble, select_sessions
from trace_ingest.loaders.helper.token_counts import token_counts
from trace_ingest.loaders.helper.tool_definitions import tool_definitions
from trace_ingest.loaders.helper.turns import parse_json, run_root
from trace_ingest.loaders.trace_loaders import TraceDescription
from trace_ingest.models import (
    UNSET,
    Loss,
    Span,
    SpanKind,
    ToolCall,
    ToolResult,
    Trace,
    TraceAggregate,
    TraceSnapshot,
)
from trace_ingest.source_links import validate_source_url

ARIZE_AX_DEFAULT_MAX_TRACES = 100
_PAGE_SIZE = 100


class ArizeAXTraceLoadError(RuntimeError):
    """The AX selection could not be fetched or normalized completely."""


def validate_arize_ax_selection(project_id: str, start: datetime, end: datetime) -> None:
    """Require a project and an explicit timezone-aware selection window."""
    if not project_id.strip():
        raise ValueError("Arize AX project_id must not be blank")
    for value in (start, end):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Arize AX timestamps must include a timezone")
    if end <= start:
        raise ValueError("Arize AX to_timestamp must be after from_timestamp")


@dataclass(frozen=True)
class ArizeAXTraceConfig:
    """Select newest AX conversations in a bounded time window."""

    project_id: str
    from_timestamp: datetime
    to_timestamp: datetime
    api_url: str | None = None
    filter: str | None = None
    max_traces: int = ARIZE_AX_DEFAULT_MAX_TRACES

    def __post_init__(self) -> None:
        validate_arize_ax_selection(self.project_id, self.from_timestamp, self.to_timestamp)
        if self.max_traces < 1:
            raise ValueError("max_traces must be at least 1")


class ArizeAXTraceDescription(TraceDescription):
    """Selection and corpus facts without credentials."""

    api_url: str
    project_id: str
    from_timestamp: str
    to_timestamp: str
    filter: str | None
    max_traces: int
    span_count: int


class _Context(BaseModel):
    model_config = ConfigDict(strict=True)
    trace_id: str = Field(min_length=1)
    span_id: str = Field(min_length=1)


class _AXSpan(BaseModel):
    """Documented AX span envelope; retain extra provider JSON fields."""

    model_config = ConfigDict(extra="allow")
    context: _Context
    name: str
    kind: str | None = None
    parent_id: str | None = None
    start_time: AwareDatetime
    end_time: AwareDatetime
    status_code: str | None = None
    status_message: str | None = None
    attributes: dict[str, JsonValue] | None = None
    evaluations: list[dict[str, JsonValue]] | None = None
    annotations: list[dict[str, JsonValue]] | None = None
    trace_annotations: list[dict[str, JsonValue]] | None = None


class _TraceSummary(BaseModel):
    trace_id: str = Field(min_length=1)
    root_span_id: str = Field(min_length=1)
    start_time: AwareDatetime
    end_time: AwareDatetime


@dataclass
class ArizeAXTraceLoader:
    """Discover traces, then page every span of each selected trace.

    AX can cap the spans embedded in trace-list responses, even when
    ``spans_truncated`` is false. Never use that embedded list as the corpus.
    """

    config: ArizeAXTraceConfig
    client: httpx.Client | None = field(default=None, repr=False)
    _snapshot: TraceSnapshot | None = field(default=None, init=False, repr=False)
    _api_url: str = field(default="", init=False)

    def load(self) -> TraceSnapshot:
        self._snapshot = None
        key = os.environ.get("ARIZE_AX_API_KEY", "").strip()
        if not key:
            raise ArizeAXTraceLoadError("ARIZE_AX_API_KEY is required")
        base = (
            self.config.api_url or os.environ.get("ARIZE_AX_API_URL") or "https://api.arize.com"
        ).rstrip("/")
        try:
            validate_source_url(base)
            parsed = urlsplit(base)
            if parsed.scheme not in {"http", "https"} or parsed.query or parsed.fragment:
                raise ValueError(
                    "Arize AX api_url must be an HTTP(S) base URL without query or fragment"
                )
            self._api_url = base
            if self.client is not None:
                snapshot = self._load(self.client, key)
            else:
                with httpx.Client(timeout=60) as client:
                    snapshot = self._load(client, key)
        except ArizeAXTraceLoadError:
            raise
        except httpx.HTTPStatusError as error:
            raise ArizeAXTraceLoadError(
                f"Arize AX request failed (HTTP {error.response.status_code})"
            ) from error
        except httpx.RequestError as error:
            raise ArizeAXTraceLoadError(
                "Arize AX request failed; check API URL and connectivity"
            ) from error
        except (ValueError, TypeError, OverflowError) as error:
            raise ArizeAXTraceLoadError(
                f"Invalid Arize AX response or configuration: {error}"
            ) from error
        self._snapshot = snapshot
        return snapshot

    def _pages(
        self, client: httpx.Client, key: str, resource: str, body: dict[str, Any], *, page_size: int
    ) -> Iterator[dict[str, Any]]:
        cursor = None
        seen: set[str] = set()
        while True:
            params: dict[str, str | int] = {"limit": page_size}
            if cursor is not None:
                params["cursor"] = cursor
            response = client.post(
                f"{self._api_url}/v2/{resource}",
                headers={"Authorization": f"Bearer {key}"},
                params=params,
                json=body,
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict) or not isinstance(data.get(resource), list):
                raise ArizeAXTraceLoadError(f"Arize AX {resource} response must contain a list")
            pagination = data.get("pagination")
            if not isinstance(pagination, dict) or not isinstance(pagination.get("has_more"), bool):
                raise ArizeAXTraceLoadError("Arize AX response is missing pagination metadata")
            yield from data[resource]
            if not pagination["has_more"]:
                break
            cursor = pagination.get("next_cursor")
            if not isinstance(cursor, str) or not cursor or cursor in seen:
                raise ArizeAXTraceLoadError("Arize AX pagination cursor is missing or repeated")
            seen.add(cursor)

    def _load(self, client: httpx.Client, key: str) -> TraceSnapshot:
        cache: dict[str, Record] = {}
        records, partial = select_sessions(
            self._records(client, key, cache),
            key=lambda record: (
                ("session", record.trace.session_id)
                if record.trace.session_id
                else ("trace", record.trace.id)
            ),
            item_id=lambda record: record.trace.id,
            limit=self.config.max_traces,
            reread=lambda session_id: self._records(client, key, cache, session_id),
        )
        return TraceSnapshot(assemble(records, partial=partial))

    def _records(
        self,
        client: httpx.Client,
        key: str,
        cache: dict[str, Record],
        session_id: str | None = None,
    ) -> Iterator[Record]:
        """Normalize each distinct provider trace before choosing its conversation key."""
        config = self.config
        selection: dict[str, Any] = {
            "project_id": config.project_id,
            "start_time": config.from_timestamp.isoformat(),
            "end_time": config.to_timestamp.isoformat(),
        }
        restriction = config.filter
        if session_id is not None:
            session_filter = f"attributes.session.id = {_literal(session_id)}"
            restriction = (
                session_filter if restriction is None else f"({restriction}) AND ({session_filter})"
            )
        if restriction is not None:
            selection["filter"] = restriction
        seen: set[str] = set()
        for raw in self._pages(
            client, key, "traces", selection, page_size=min(_PAGE_SIZE, config.max_traces + 1)
        ):
            summary = _TraceSummary.model_validate(raw)
            if summary.end_time < summary.start_time:
                raise ArizeAXTraceLoadError("Arize AX trace ends before it starts")
            if summary.trace_id in seen:
                # AX lists a multi-root trace once per root; its full span query
                # already includes all roots, so the trace is selected only once.
                continue
            seen.add(summary.trace_id)
            if summary.trace_id not in cache:
                trace = self._fetch_trace(client, key, summary)
                cache[summary.trace_id] = _record(trace)
            yield cache[summary.trace_id]

    def _fetch_trace(self, client: httpx.Client, key: str, summary: _TraceSummary) -> Trace:
        config = self.config
        body = {
            "project_id": config.project_id,
            "start_time": min(config.from_timestamp, summary.start_time).isoformat(),
            # Datetime rounds nanosecond summaries down to microseconds; the API
            # upper bound is exclusive, so leave a margin for the final span.
            "end_time": max(
                config.to_timestamp, summary.end_time + timedelta(seconds=1)
            ).isoformat(),
            "filter": f"context.trace_id = {_literal(summary.trace_id)}",
        }
        spans = [
            _AXSpan.model_validate(row)
            for row in self._pages(client, key, "spans", body, page_size=_PAGE_SIZE)
        ]
        if not any(span.context.span_id == summary.root_span_id for span in spans):
            raise ArizeAXTraceLoadError("Arize AX span query did not return the selected root")
        return _normalize_trace(summary.trace_id, spans)

    def describe(self) -> ArizeAXTraceDescription:
        traces = list(self._snapshot) if self._snapshot is not None else []
        spans = [span for trace in traces for span in _walk(trace.root_spans)]
        return ArizeAXTraceDescription(
            source="arize_ax",
            trace_count=len(traces),
            span_count=sum("arize_ax" in span.attributes for span in spans),
            call_count=sum(span.kind is SpanKind.TOOL for span in spans),
            distinct_cases=len({trace.case_id or trace.id for trace in traces}),
            distinct_sessions=len({trace.session_id for trace in traces if trace.session_id}),
            api_url=self._api_url,
            project_id=self.config.project_id,
            from_timestamp=self.config.from_timestamp.isoformat(),
            to_timestamp=self.config.to_timestamp.isoformat(),
            filter=self.config.filter,
            max_traces=self.config.max_traces,
        )


def _literal(value: str) -> str:
    """AX uses SQL string literals, including doubled apostrophes."""
    return "'" + value.replace("'", "''") + "'"


def _record(trace: Trace) -> Record:
    root = run_root(trace)
    return Record(
        trace,
        conversation_input=root.input if root is not None else UNSET,
        conversation_output=root.output if root is not None else UNSET,
    )


def _walk(roots: list[Span]) -> Iterator[Span]:
    pending = list(reversed(roots))
    while pending:
        span = pending.pop()
        yield span
        pending.extend(reversed(span.children))


def _payload(attrs: dict[str, JsonValue], field: str) -> JsonValue | UNSET:
    key = f"{field}.value"
    if key not in attrs:
        return UNSET
    value = attrs[key]
    if attrs.get(f"{field}.mime_type") == "application/json" and isinstance(value, str):
        return json.loads(value)
    return value


def _count(value: JsonValue | None) -> int | None:
    """AX can return OpenInference token counts as integers or decimal strings."""
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        return int(value)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _cost(value: JsonValue | None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        cost = float(value)
    except ValueError:
        return None
    return cost if math.isfinite(cost) and cost >= 0 else None


def _normalize_span(provider: _AXSpan) -> Span:
    attrs = provider.attributes or {}
    kind = (
        SpanKind(provider.kind)
        if provider.kind in SpanKind._value2member_map_
        else SpanKind.UNKNOWN
    )
    output = _payload(attrs, "output")
    error = provider.status_message or "ERROR" if provider.status_code == "ERROR" else None
    span = Span(
        id=provider.context.span_id,
        kind=kind,
        start_time=provider.start_time,
        end_time=provider.end_time,
        input=_payload(attrs, "input"),
        output=output,
        error=error,
        model=attrs.get("llm.model_name") if isinstance(attrs.get("llm.model_name"), str) else None,
        cost_usd=_cost(attrs.get("llm.cost.total")),
        token_counts=token_counts(
            input_total=_count(attrs.get("llm.token_count.prompt")),
            output_tokens=_count(attrs.get("llm.token_count.completion")),
            source="arize_ax.openinference",
        ),
        attributes={"arize_ax": provider.model_dump(mode="json", exclude_unset=True)},
    )
    if kind is SpanKind.TOOL:
        name = attrs.get("tool.name")
        span.tool_name = name if isinstance(name, str) else provider.name
        results = (
            []
            if output is UNSET
            else [ToolResult(content=output, is_error=True if error else None)]
        )
        span.tool_call = ToolCall(call_id=span.id, result_count=len(results), results=results)
    if kind in (SpanKind.AGENT, SpanKind.LLM):
        tools = attrs.get("llm.tools")
        if isinstance(tools, list):
            definitions = tool_definitions(
                [
                    parse_json(
                        item.get("tool.json_schema")
                        if isinstance(item, dict) and "tool.json_schema" in item
                        else item
                    )
                    for item in tools
                ]
            )
            assert definitions is not None  # The source is a recorded list, even when empty.
            for definition, item in zip(definitions, tools, strict=True):
                definition.raw = item
            span.tool_definitions = definitions
    if kind is SpanKind.LLM:
        params = parse_json(attrs.get("llm.invocation_parameters"))
        if isinstance(params, dict):
            span.llm_params = {
                k: v for k, v in params.items() if k not in {"model", "tools", "messages", "prompt"}
            }
    return span


def _normalize_trace(trace_id: str, providers: list[_AXSpan]) -> Trace:
    by_id = {span.context.span_id: span for span in providers}
    if len(by_id) != len(providers):
        raise ArizeAXTraceLoadError(f"Duplicate Arize AX span ID in trace {trace_id}")
    if not by_id:
        raise ArizeAXTraceLoadError(f"Arize AX trace {trace_id} has no spans")
    children: dict[str, list[str]] = {key: [] for key in by_id}
    roots = []
    for key, provider in by_id.items():
        if provider.context.trace_id != trace_id:
            raise ArizeAXTraceLoadError("Arize AX span belongs to a different trace")
        if not provider.parent_id:
            roots.append(key)
        elif provider.parent_id not in by_id:
            raise ArizeAXTraceLoadError(f"Arize AX trace {trace_id} is incomplete: missing parent")
        else:
            children[provider.parent_id].append(key)

    def order(key: str) -> tuple[datetime, str]:
        return by_id[key].start_time, key

    ready = [(order(key), key) for key in roots]
    heapq.heapify(ready)
    normalized: dict[str, Span] = {}
    while ready:
        _, key = heapq.heappop(ready)
        provider = by_id[key]
        span = _normalize_span(provider)
        normalized[key] = span
        if provider.parent_id:
            normalized[provider.parent_id].children.append(span)
        for child in children[key]:
            heapq.heappush(ready, (order(child), child))
    if len(normalized) != len(by_id):
        raise ArizeAXTraceLoadError(f"Arize AX trace {trace_id} contains a parent cycle")
    evaluations: dict[str, JsonValue] = {}
    for provider in providers:
        for category in ("evaluations", "annotations", "trace_annotations"):
            for index, result in enumerate(getattr(provider, category) or []):
                name = result.get("name") or str(index)
                evaluations[f"arize_ax/{category}/{provider.context.span_id}/{index}/{name}"] = (
                    result
                )
    sessions = {
        value
        for span in providers
        if isinstance(value := (span.attributes or {}).get("session.id"), str) and value
    }
    session_id = next(iter(sessions)) if len(sessions) == 1 else None
    return Trace(
        id=trace_id,
        session_id=session_id,
        session_basis="arize_ax.session.id" if session_id is not None else None,
        root_spans=[normalized[key] for key in sorted(roots, key=order)],
        aggregate=TraceAggregate(
            latency_ms=(
                max(s.end_time for s in providers) - min(s.start_time for s in providers)
            ).total_seconds()
            * 1000
        ),
        evaluator_results=evaluations,
        attributes={"source_pointer": {"provider": "arize_ax", "trace_id": trace_id}},
        losses=[Loss(code="conflicting_session_ids", detail={"session_ids": sorted(sessions)})]
        if len(sessions) > 1
        else [],
    )
