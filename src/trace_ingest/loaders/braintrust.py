# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load complete Braintrust traces via the documented SQL query API.

Schema and pagination: https://www.braintrust.dev/docs/api-reference/query
Span fields: https://www.braintrust.dev/docs/instrument/advanced-tracing
"""

from __future__ import annotations

import heapq
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import quote, urlencode

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from trace_ingest.loaders.trace_loaders import TraceDescription
from trace_ingest.models import (
    UNSET,
    Span,
    SpanKind,
    ToolCall,
    Trace,
    TraceAggregate,
    TraceSnapshot,
)
from trace_ingest.source_links import http_source_url

BRAINTRUST_DEFAULT_MAX_TRACES = 100
_DEFAULT_API_URL = "https://api.braintrust.dev"
_PAGE_SIZE = 100


class BraintrustTraceLoadError(RuntimeError):
    """A Braintrust query or trace cannot be loaded safely."""


def validate_braintrust_selection(
    project_id: str | None,
    experiment_id: str | None,
    from_timestamp: datetime,
    to_timestamp: datetime,
) -> None:
    """Require one source and a bounded, timezone-aware root creation window."""
    if (project_id is None) == (experiment_id is None):
        raise ValueError("Braintrust requires exactly one of project_id or experiment_id")
    if not (project_id or experiment_id or "").strip():
        raise ValueError("Braintrust source ID must not be blank")
    for name, value in (("from_timestamp", from_timestamp), ("to_timestamp", to_timestamp)):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{name} must include a timezone")
    if to_timestamp <= from_timestamp:
        raise ValueError("to_timestamp must be after from_timestamp")


@dataclass(frozen=True)
class BraintrustTraceConfig:
    """Select root spans created in [from_timestamp, to_timestamp)."""

    from_timestamp: datetime
    to_timestamp: datetime
    project_id: str | None = None
    experiment_id: str | None = None
    api_url: str | None = None
    max_traces: int = BRAINTRUST_DEFAULT_MAX_TRACES
    org_name: str | None = None
    app_url: str | None = None

    def __post_init__(self) -> None:
        validate_braintrust_selection(
            self.project_id, self.experiment_id, self.from_timestamp, self.to_timestamp
        )
        if self.max_traces < 1:
            raise ValueError("max_traces must be at least 1")


class BraintrustTraceDescription(TraceDescription):
    """Source selection and corpus diagnostics; never includes credentials."""

    api_url: str
    project_id: str | None
    experiment_id: str | None
    from_timestamp: str
    to_timestamp: str
    max_traces: int
    span_count: int


class _ProviderSpan(BaseModel):
    """Documented Braintrust span fields, preserving unmodeled JSON metadata."""

    model_config = ConfigDict(extra="allow", strict=True)
    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)
    id: str = Field(min_length=1)
    span_id: str = Field(min_length=1)
    root_span_id: str = Field(min_length=1)
    is_root: bool | None = None
    span_parents: list[str] | None = None
    span_attributes: dict[str, JsonValue] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    metrics: dict[str, JsonValue] = Field(default_factory=dict)
    scores: dict[str, float | None] | None = None
    input: JsonValue | UNSET = UNSET
    output: JsonValue | UNSET = UNSET
    error: str | None = None


@dataclass
class BraintrustTraceLoader:
    """Fetch bounded root selections and all their spans, with an injectable HTTP client."""

    config: BraintrustTraceConfig
    client: httpx.Client | None = field(default=None, repr=False)
    _snapshot: TraceSnapshot = field(
        default_factory=lambda: TraceSnapshot([]), init=False, repr=False
    )
    _api_url: str | None = field(default=None, init=False, repr=False)

    def load(self) -> TraceSnapshot:
        """Normalize complete trees and publish the snapshot only after successful loading."""
        api_url = _resolved_api_url(self.config.api_url)
        try:
            if self.client is not None:
                snapshot = self._load(self.client, api_url)
            else:
                api_key = os.environ.get("BRAINTRUST_API_KEY", "").strip()
                if not api_key:
                    raise BraintrustTraceLoadError("Braintrust loading requires BRAINTRUST_API_KEY")
                with httpx.Client(
                    headers={"Authorization": f"Bearer {api_key}"}, timeout=60.0
                ) as client:
                    snapshot = self._load(client, api_url)
        except httpx.HTTPStatusError as error:
            raise BraintrustTraceLoadError(
                f"Braintrust query failed with HTTP {error.response.status_code}"
            ) from error
        except httpx.RequestError as error:
            raise BraintrustTraceLoadError("Braintrust query request failed") from error
        except (ValueError, OverflowError) as error:
            raise BraintrustTraceLoadError(f"Invalid Braintrust response: {error}") from error
        self._api_url = api_url
        self._snapshot = snapshot
        return snapshot

    def _load(self, client: httpx.Client, api_url: str) -> TraceSnapshot:
        source_id = self.config.project_id or self.config.experiment_id
        table = "project_logs" if self.config.project_id is not None else "experiment"
        source = f"{table}({_literal(str(source_id))})"
        selection = (
            f"SELECT root_span_id, created FROM {source} WHERE is_root "
            f"AND created >= {_literal(self.config.from_timestamp.isoformat())} "
            f"AND created < {_literal(self.config.to_timestamp.isoformat())} "
            f"ORDER BY _pagination_key DESC LIMIT {min(_PAGE_SIZE, self.config.max_traces)}"
        )
        selected: dict[str, datetime] = {}
        for row in _query_rows(client, api_url, selection):
            trace_id = row.get("root_span_id")
            created = row.get("created")
            if not isinstance(trace_id, str) or not trace_id or not isinstance(created, str):
                raise BraintrustTraceLoadError("Braintrust root requires root_span_id and created")
            timestamp = datetime.fromisoformat(created.replace("Z", "+00:00"))
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise BraintrustTraceLoadError("Braintrust root created must include a timezone")
            if trace_id in selected:
                raise BraintrustTraceLoadError(f"duplicate Braintrust root {trace_id!r}")
            selected[trace_id] = timestamp
            if len(selected) == self.config.max_traces:
                break

        def traces() -> Iterator[Trace]:
            for trace_id in sorted(selected, key=lambda key: (selected[key], key)):
                query = (
                    f"SELECT * FROM {source} WHERE root_span_id = {_literal(trace_id)} "
                    f"ORDER BY _pagination_key ASC LIMIT {_PAGE_SIZE}"
                )
                rows = list(_query_rows(client, api_url, query))
                pointer: dict[str, JsonValue] = {
                    "provider": "braintrust",
                    "api_url": api_url,
                    "project_id": self.config.project_id,
                    "experiment_id": self.config.experiment_id,
                    "trace_id": trace_id,
                }
                trace = _normalize_trace(trace_id, rows, pointer)
                org_name = self.config.org_name or os.environ.get("BRAINTRUST_ORG_NAME")
                app_url = http_source_url(
                    self.config.app_url
                    or os.environ.get("BRAINTRUST_APP_URL")
                    or "https://www.braintrust.dev"
                )
                if org_name and app_url:
                    pending = list(trace.root_spans)
                    while pending:
                        span = pending.pop()
                        pending.extend(span.children)
                        span_pointer = span.attributes["source_pointer"]
                        assert isinstance(span_pointer, dict)
                        query = urlencode(
                            {
                                "object_type": "project_logs"
                                if self.config.project_id
                                else "experiment",
                                "object_id": self.config.project_id or self.config.experiment_id,
                                "id": span_pointer["row_id"],
                            }
                        )
                        span.source_url = (
                            f"{app_url.rstrip('/')}/app/{quote(org_name, safe='')}/object?{query}"
                        )
                    trace.source_url = trace.root_spans[0].source_url
                yield trace

        return TraceSnapshot(traces())

    def describe(self) -> BraintrustTraceDescription:
        """Describe the last successfully loaded corpus and its selection."""
        api_url = self._api_url or _resolved_api_url(self.config.api_url)
        span_count = call_count = 0
        for trace in self._snapshot:
            pending = list(trace.root_spans)
            while pending:
                span = pending.pop()
                span_count += 1
                call_count += span.kind is SpanKind.TOOL
                pending.extend(span.children)
        table = "project_logs" if self.config.project_id is not None else "experiment"
        source_id = self.config.project_id or self.config.experiment_id
        return {
            "source": f"braintrust:{api_url}#{table}/{source_id}",
            "api_url": api_url,
            "project_id": self.config.project_id,
            "experiment_id": self.config.experiment_id,
            "from_timestamp": self.config.from_timestamp.isoformat(),
            "to_timestamp": self.config.to_timestamp.isoformat(),
            "max_traces": self.config.max_traces,
            "span_count": span_count,
            "trace_count": self._snapshot.trace_count,
            "call_count": call_count,
            "distinct_logical_cases": self._snapshot.trace_count,
        }


def _resolved_api_url(configured: str | None) -> str:
    return (configured or os.environ.get("BRAINTRUST_API_URL") or _DEFAULT_API_URL).rstrip("/")


def _literal(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "''") + "'"


def _query_rows(client: httpx.Client, api_url: str, query: str) -> Iterator[dict[str, JsonValue]]:
    cursor: str | None = None
    seen: set[str] = set()
    while True:
        paged_query = query if cursor is None else f"{query} OFFSET {_literal(cursor)}"
        response = client.post(f"{api_url}/btql", json={"query": paged_query, "fmt": "json"})
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise BraintrustTraceLoadError("Braintrust query response requires a data array")
        rows = payload["data"]
        cursor = response.headers.get("x-bt-cursor")
        if cursor:
            if cursor in seen or not rows:
                raise BraintrustTraceLoadError("Braintrust pagination made no progress")
            seen.add(cursor)
        for row in rows:
            if not isinstance(row, dict):
                raise BraintrustTraceLoadError("Braintrust query row must be an object")
            yield row
        if not cursor:
            return


def _normalize_trace(
    trace_id: str, rows: list[dict[str, JsonValue]], pointer: dict[str, JsonValue]
) -> Trace:
    by_id: dict[str, _ProviderSpan] = {}
    row_ids: set[str] = set()
    try:
        for row in rows:
            span = _ProviderSpan.model_validate(row)
            if span.root_span_id != trace_id:
                raise BraintrustTraceLoadError("Braintrust span belongs to a different trace")
            if span.span_id in by_id or span.id in row_ids:
                raise BraintrustTraceLoadError(
                    f"duplicate Braintrust span or row id in {trace_id!r}"
                )
            by_id[span.span_id] = span
            row_ids.add(span.id)
        # root_span_id identifies the trace; with current SDKs it need not equal
        # the root's span_id. Prefer the API's is_root marker when supplied.
        roots = [
            key
            for key, span in by_id.items()
            if span.is_root is True or (span.is_root is None and not span.span_parents)
        ]
        if not roots:
            raise BraintrustTraceLoadError(
                f"Braintrust trace {trace_id!r} is missing its root span"
            )
        if len(roots) != 1:
            raise BraintrustTraceLoadError(
                "Braintrust non-root span requires exactly one parent; multiple roots found"
            )
        root_id = roots[0]
        children: dict[str, list[str]] = {key: [] for key in by_id}
        for key, span in by_id.items():
            if key == root_id:
                if span.span_parents:
                    raise BraintrustTraceLoadError("Braintrust root span has parents")
            else:
                if span.span_parents is None or len(span.span_parents) != 1:
                    raise BraintrustTraceLoadError(
                        "Braintrust non-root span requires exactly one parent"
                    )
                parent = span.span_parents[0]
                if parent not in by_id:
                    raise BraintrustTraceLoadError("Braintrust trace is incomplete: missing parent")
                children[parent].append(key)

        ready = [(_start_key(by_id[root_id]), root_id)]
        normalized: dict[str, Span] = {}
        tool_index = 0
        while ready:
            _, key = heapq.heappop(ready)
            provider = by_id[key]
            span = _normalize_span(provider, pointer)
            if span.kind is SpanKind.TOOL:
                span.tool_call = ToolCall(
                    call_id=key, index=tool_index, result_count=0 if span.output is UNSET else 1
                )
                tool_index += 1
            normalized[key] = span
            if provider.span_parents:
                normalized[provider.span_parents[0]].children.append(span)
            for child in children[key]:
                heapq.heappush(ready, (_start_key(by_id[child]), child))
        if len(normalized) != len(by_id):
            raise BraintrustTraceLoadError(f"Braintrust trace {trace_id!r} contains a parent cycle")
        starts = [s.start_time for s in normalized.values() if s.start_time is not None]
        ends = [s.end_time for s in normalized.values() if s.end_time is not None]
        root = by_id[root_id]
        return Trace(
            id=trace_id,
            root_spans=[normalized[root_id]],
            aggregate=TraceAggregate(
                latency_ms=(max(ends) - min(starts)).total_seconds() * 1000
                if starts and ends
                else None
            ),
            evaluator_results=dict(root.scores or {}),
            attributes={"source_pointer": pointer},
        )
    except (ValidationError, ValueError, OverflowError) as error:
        raise BraintrustTraceLoadError(f"Invalid Braintrust trace {trace_id!r}: {error}") from error


def _metric(span: _ProviderSpan, name: str) -> float | None:
    value = span.metrics.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BraintrustTraceLoadError(f"Braintrust metric {name!r} must be numeric")
    return float(value)


def _start_key(span: _ProviderSpan) -> float:
    value = _metric(span, "start")
    return value if value is not None else float("inf")


def _normalize_span(provider: _ProviderSpan, pointer: dict[str, JsonValue]) -> Span:
    provider_type = provider.span_attributes.get("type")
    kind = {
        "llm": SpanKind.LLM,
        "tool": SpanKind.TOOL,
        "score": SpanKind.EVALUATOR,
        "function": SpanKind.CHAIN,
        "task": SpanKind.CHAIN,
        "eval": SpanKind.CHAIN,
    }.get(str(provider_type), SpanKind.UNKNOWN)
    start = _metric(provider, "start")
    end = _metric(provider, "end")
    name = provider.span_attributes.get("name")
    model = provider.metadata.get("model")
    return Span(
        id=provider.span_id,
        kind=kind,
        start_time=datetime.fromtimestamp(start, timezone.utc) if start is not None else None,
        end_time=datetime.fromtimestamp(end, timezone.utc) if end is not None else None,
        input=provider.input,
        output=provider.output,
        error=provider.error,
        tool_name=name if kind is SpanKind.TOOL and isinstance(name, str) else None,
        model=model if isinstance(model, str) else None,
        cost_usd=_metric(provider, "estimated_cost"),
        attributes={
            "name": name,
            "subtype": provider_type,
            "source_pointer": {**pointer, "span_id": provider.span_id, "row_id": provider.id},
            "braintrust": provider.model_dump(mode="json", exclude_unset=True),
        },
    )
