# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load a bounded NeMo Platform Intake query as canonical traces.
Importantly, this loader cannot introduce a dependency on NeMo Platform packages or models.
This is required so that the loader can be used /by/ NeMo Platform without creating a circular dependency.
"""

from __future__ import annotations

import heapq
import json
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Literal, TypeVar
from urllib.parse import quote, urlsplit

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationInfo,
    field_validator,
    model_validator,
)

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

_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])


class IntakeLoadError(RuntimeError):
    """An Intake request or normalization failed with contextual diagnostics."""


class IntakeStatus(str, Enum):
    """Status values exposed by Intake trace and span records."""

    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class _Pagination(_WireModel):
    page: int = Field(ge=1)
    page_size: int = Field(ge=1)
    current_page_size: int = Field(ge=0)
    total_pages: int = Field(ge=0)
    total_results: int = Field(ge=0)


class _EvaluationContext(_WireModel):
    evaluation_name: str | None = None
    test_case_name: str | None = None


class _IntakeTrace(_WireModel):
    id: str
    root_span_id: str | None = None
    session_id: str
    workspace: str
    name: str | None = None
    input: str | None = None
    output: str | None = None
    evaluation_context: _EvaluationContext | None = None
    agent_name: str | None = None
    started_at: datetime
    ended_at: datetime | None = None
    duration_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    status: IntakeStatus
    cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class _IntakeSpan(_WireModel):
    span_id: str
    session_id: str
    workspace: str
    parent_span_id: str | None = None
    kind: str
    name: str | None = None
    source: str
    trace_id: str | None = None
    started_at: datetime
    ended_at: datetime | None = None
    status: IntakeStatus
    error_type: str | None = None
    error_message: str | None = None
    tool_name: str | None = None
    model: str | None = None
    cost_total_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    input: str | None = None
    output: str | None = None
    raw_attributes: str | None = None


class _EvaluatorResult(_WireModel):
    evaluator_result_id: str
    span_id: str
    session_id: str
    workspace: str
    name: str = Field(min_length=1)
    value: float | None = Field(default=None, allow_inf_nan=False)
    string_value: str | None = None
    data_type: Literal["NUMERIC", "BOOLEAN", "CATEGORICAL", "TEXT"]
    comment: str | None = None
    created_by: str | None = None
    created_at: datetime
    ingested_at: datetime

    @model_validator(mode="after")
    def value_matches_type(self) -> _EvaluatorResult:
        if self.data_type in {"NUMERIC", "BOOLEAN"} and self.value is None:
            raise ValueError(f"{self.data_type} evaluator result requires value")
        if self.data_type in {"CATEGORICAL", "TEXT"} and self.string_value is None:
            raise ValueError(f"{self.data_type} evaluator result requires string_value")
        if self.data_type == "BOOLEAN" and self.value not in (0, 1):
            raise ValueError("BOOLEAN evaluator result requires value 0 or 1")
        return self


class _EvaluatorResultPage(_WireModel):
    data: list[_EvaluatorResult]
    pagination: _Pagination


class _Evaluation(_WireModel):
    name: str = Field(min_length=1)


class _EvaluationPage(_WireModel):
    data: list[_Evaluation]
    pagination: _Pagination


class _TracePage(_WireModel):
    data: list[_IntakeTrace]
    pagination: _Pagination


class _SpanPage(_WireModel):
    data: list[_IntakeSpan]
    pagination: _Pagination


class IntakeTraceQuery(_StrictModel):
    """A mandatory time-bounded root-trace selection."""

    started_at_gte: datetime
    started_at_lte: datetime
    agent_name: str | None = None
    experiment_id: str | None = Field(
        default=None, min_length=1, description="Restrict traces to evaluations in this experiment."
    )
    evaluation_name: str | None = None
    test_case_name: str | None = None
    session_id: str | None = None
    status: IntakeStatus | None = None
    max_traces: int | None = Field(default=None, ge=1)
    sort: Literal["started_at", "-started_at"] = "started_at"

    @field_validator("started_at_gte", "started_at_lte")
    @classmethod
    def timestamps_include_timezone(cls, value: datetime, info: ValidationInfo) -> datetime:
        if value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must include a timezone")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def bounds_are_ordered(self) -> IntakeTraceQuery:
        if self.started_at_gte > self.started_at_lte:
            raise ValueError("started_at_gte must not follow started_at_lte")
        return self

    def params(self) -> dict[str, str]:
        params = {
            "filter[started_at][gte]": self.started_at_gte.isoformat(),
            "filter[started_at][lte]": self.started_at_lte.isoformat(),
        }
        for name in ("agent_name", "evaluation_name", "test_case_name", "session_id"):
            value = getattr(self, name)
            if value is not None:
                params[f"filter[{name}]"] = value
        if self.status is not None:
            params["filter[status]"] = self.status.value
        return params

    def describe(self) -> dict[str, JsonValue]:
        return self.model_dump(mode="json", exclude_none=True)


class IntakeTraceLoaderConfig(_StrictModel):
    """Validated configuration for one bounded Intake snapshot."""

    base_url: str = Field(min_length=1)
    workspace: str = Field(min_length=1)
    query: IntakeTraceQuery
    page_size: int = Field(default=100, ge=1, le=1000)
    timeout_seconds: float = Field(default=30.0, gt=0, allow_inf_nan=False)

    @field_validator("base_url")
    @classmethod
    def valid_http_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        parsed = urlsplit(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("base_url must not contain credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("base_url must not include a query or fragment")
        return normalized

    @field_validator("workspace")
    @classmethod
    def non_empty_workspace(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("workspace must not be blank")
        return normalized


class IntakeTraceDescription(TraceDescription):
    """Run metadata for a live Intake query."""

    span_count: int
    workspace: str
    query: dict[str, JsonValue]


_Page = TypeVar("_Page", _TracePage, _SpanPage, _EvaluatorResultPage, _EvaluationPage)


class _IntakeClient:
    def __init__(
        self,
        *,
        base_url: str,
        workspace: str,
        page_size: int,
        timeout_seconds: float,
        transport: httpx.BaseTransport | None,
    ) -> None:
        workspace_path = quote(workspace, safe="")
        self._prefix = f"{base_url.rstrip('/')}/apis/intake/v2/workspaces/{workspace_path}"
        self._page_size = page_size
        access_token = os.environ.get("NMP_ACCESS_TOKEN")
        headers = {"Authorization": f"Bearer {access_token}"} if access_token else {}
        self._client = httpx.Client(timeout=timeout_seconds, transport=transport, headers=headers)

    def __enter__(self) -> _IntakeClient:
        return self

    def __exit__(self, *_args: object) -> None:
        self._client.close()

    def trace_pages(self, query: IntakeTraceQuery) -> Iterator[_TracePage]:
        yield from self._pages(
            endpoint="traces",
            model=_TracePage,
            params={**query.params(), "sort": query.sort, "mode": "detailed"},
        )

    def traces(self, query: IntakeTraceQuery) -> Iterator[_IntakeTrace]:
        def records(selection: IntakeTraceQuery) -> Iterator[_IntakeTrace]:
            for page in self.trace_pages(selection):
                yield from page.data

        if query.experiment_id is None:
            yield from records(query)
            return

        names = {
            evaluation.name
            for page in self._pages(
                endpoint="evaluations",
                model=_EvaluationPage,
                params={"filter[experiment_id]": query.experiment_id, "sort": "name"},
            )
            for evaluation in page.data
        }
        if query.evaluation_name is not None:
            names.intersection_update({query.evaluation_name})
        # Merge the ordered selections before applying the shared trace limit.
        yield from heapq.merge(
            *(
                records(query.model_copy(update={"evaluation_name": name}))
                for name in sorted(names)
            ),
            key=lambda trace: _aware_utc(trace.started_at),
            reverse=query.sort == "-started_at",
        )

    def span_pages(self, trace_id: str) -> Iterator[_SpanPage]:
        yield from self._pages(
            endpoint="spans",
            model=_SpanPage,
            params={
                "filter[trace_id]": trace_id,
                "sort": "started_at",
                "mode": "detailed",
            },
        )

    def evaluator_result_pages(self, session_id: str) -> Iterator[_EvaluatorResultPage]:
        yield from self._pages(
            endpoint="evaluator-results",
            model=_EvaluatorResultPage,
            params={"filter[session_id]": session_id, "sort": "created_at"},
        )

    def _pages(
        self,
        *,
        endpoint: Literal["traces", "spans", "evaluator-results", "evaluations"],
        model: type[_Page],
        params: Mapping[str, str],
    ) -> Iterator[_Page]:
        page_number = 1
        received = 0
        expected_total: int | None = None
        expected_pages: int | None = None
        while True:
            request_params: dict[str, str | int] = {
                **params,
                "page": page_number,
                "page_size": self._page_size,
            }
            url = f"{self._prefix}/{endpoint}"
            try:
                response = self._client.get(url, params=request_params)
                response.raise_for_status()
                page = model.model_validate(response.json())
            except (httpx.HTTPError, ValueError) as error:
                raise IntakeLoadError(
                    f"Intake {endpoint} request failed at page {page_number}: {error}"
                ) from error

            pagination = page.pagination
            if pagination.page != page_number:
                raise IntakeLoadError(
                    f"Intake {endpoint} returned page {pagination.page} while page "
                    f"{page_number} was requested"
                )
            if pagination.current_page_size != len(page.data):
                raise IntakeLoadError(
                    f"Intake {endpoint} page {page_number} reports "
                    f"current_page_size={pagination.current_page_size} but returned "
                    f"{len(page.data)} records"
                )
            if expected_total is None:
                expected_total = pagination.total_results
                expected_pages = pagination.total_pages
            elif (
                pagination.total_results != expected_total
                or pagination.total_pages != expected_pages
            ):
                raise IntakeLoadError(
                    f"Intake {endpoint} pagination changed during the bounded scan"
                )

            received += len(page.data)
            last_page = page_number >= pagination.total_pages
            if received > expected_total or (last_page and received != expected_total):
                raise IntakeLoadError(
                    f"Intake {endpoint} expected {expected_total} records but received {received}"
                )
            if not page.data and not last_page:
                raise IntakeLoadError(f"Intake {endpoint} returned an empty non-final page")

            yield page
            if page_number >= pagination.total_pages:
                break
            page_number += 1


@dataclass
class IntakeTraceLoader:
    """Fetch and normalize one bounded Intake selection in memory."""

    config: IntakeTraceLoaderConfig
    transport: httpx.BaseTransport | None = field(default=None, repr=False)
    _snapshot: TraceSnapshot | None = field(default=None, init=False, repr=False)

    def load(self) -> TraceSnapshot:
        if self._snapshot is not None:
            return self._snapshot

        with _IntakeClient(
            base_url=self.config.base_url,
            workspace=self.config.workspace,
            page_size=self.config.page_size,
            timeout_seconds=self.config.timeout_seconds,
            transport=self.transport,
        ) as client:
            try:
                snapshot = TraceSnapshot(self._traces(client))
            except (ValueError, TypeError) as error:
                raise IntakeLoadError(f"Failed to load Intake traces: {error}") from error
        if not snapshot.trace_count:
            raise IntakeLoadError("bounded Intake scan returned no traces")
        self._snapshot = snapshot
        return snapshot

    def _traces(self, client: _IntakeClient) -> Iterator[Trace]:
        count = 0
        for intake_trace in client.traces(self.config.query):
            raw_spans = [
                span for span_page in client.span_pages(intake_trace.id) for span in span_page.data
            ]
            raw_results = [
                result
                for result_page in client.evaluator_result_pages(intake_trace.session_id)
                for result in result_page.data
            ]
            try:
                yield _normalize_trace(
                    intake_trace,
                    raw_spans,
                    raw_results,
                    base_url=self.config.base_url,
                    workspace=self.config.workspace,
                )
            except (ValueError, TypeError) as error:
                raise IntakeLoadError(
                    f"failed to normalize Intake trace {intake_trace.id!r}: {error}"
                ) from error
            count += 1
            if count == self.config.query.max_traces:
                return

    def describe(self) -> IntakeTraceDescription:
        """Describe the configured source and most recently loaded corpus."""
        traces = self._snapshot or ()
        span_count = call_count = 0
        for trace in traces:
            pending = list(trace.root_spans)
            while pending:
                span = pending.pop()
                span_count += 1
                call_count += span.kind is SpanKind.TOOL
                pending.extend(span.children)
        return {
            "source": (
                f"intake:{self.config.base_url}/apis/intake/v2/workspaces/"
                f"{quote(self.config.workspace, safe='')}"
            ),
            "trace_count": len(traces),
            "call_count": call_count,
            "distinct_logical_cases": len(
                {str(trace.attributes.get("logical_case_id") or trace.id) for trace in traces}
            ),
            "span_count": span_count,
            "workspace": self.config.workspace,
            "query": self.config.query.describe(),
        }


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _payload(
    record: _IntakeTrace | _IntakeSpan,
    field_name: Literal["input", "output"],
) -> JsonValue | UNSET:
    value = record.input if field_name == "input" else record.output
    # Intake uses SQL null for absent text; the JSON string "null" is explicit null.
    if value is None:
        return UNSET
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _raw_attributes(span: _IntakeSpan) -> dict[str, JsonValue]:
    if not span.raw_attributes:
        return {}
    try:
        value = json.loads(span.raw_attributes)
    except json.JSONDecodeError as error:
        raise ValueError(f"span {span.span_id!r} has invalid raw_attributes JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"span {span.span_id!r} raw_attributes must be a JSON object")
    return _JSON_OBJECT.validate_python(value)


def _normalize_trace(
    raw: _IntakeTrace,
    raw_spans: list[_IntakeSpan],
    raw_results: list[_EvaluatorResult],
    *,
    base_url: str,
    workspace: str,
) -> Trace:
    for span in raw_spans:
        if span.trace_id is not None and span.trace_id != raw.id:
            raise ValueError(
                f"span {span.span_id!r} belongs to trace {span.trace_id!r}, not {raw.id!r}"
            )

    ordered = _canonical_spans(raw.id, raw_spans)
    base_pointer: dict[str, JsonValue] = {
        "provider": "nemo-platform-intake",
        "base_url": base_url.rstrip("/"),
        "workspace": workspace,
        "trace_id": raw.id,
    }
    normalized_by_id: dict[str, Span] = {}
    root_spans: list[Span] = []
    tool_call_index = 0
    for raw_span in ordered:
        normalized = _normalize_span(
            raw_span,
            base_pointer=base_pointer,
            tool_call_index=tool_call_index,
        )
        if normalized.kind is SpanKind.TOOL:
            tool_call_index += 1
        normalized_by_id[raw_span.span_id] = normalized
        if raw_span.parent_span_id is None:
            root_spans.append(normalized)
        else:
            normalized_by_id[raw_span.parent_span_id].children.append(normalized)

    trace_input = _payload(raw, "input")
    trace_output = _payload(raw, "output")
    intake_attributes: dict[str, JsonValue] = {
        "root_span_id": raw.root_span_id,
        "session_id": raw.session_id,
        "workspace": raw.workspace,
        "name": raw.name,
        "agent_name": raw.agent_name,
        "started_at": _aware_utc(raw.started_at).isoformat(),
        "ended_at": _aware_utc(raw.ended_at).isoformat() if raw.ended_at is not None else None,
        "status": raw.status.value,
    }
    if trace_output is not UNSET:
        intake_attributes["output"] = trace_output
    if raw.evaluation_context is not None:
        intake_attributes["evaluation_context"] = raw.evaluation_context.model_dump(mode="json")

    attributes: dict[str, JsonValue] = {
        "source_pointer": base_pointer,
        "intake": intake_attributes,
    }
    if trace_input is not UNSET:
        attributes["task_text"] = trace_input
    logical_case_id = (
        raw.evaluation_context.test_case_name if raw.evaluation_context is not None else None
    )
    if logical_case_id is not None:
        attributes["logical_case_id"] = logical_case_id
    catalog = _atif_tool_catalog(ordered)
    if catalog is not None:
        attributes["tool_catalog"] = catalog

    return Trace(
        id=raw.id,
        evaluator_results=_normalize_evaluator_results(raw, raw_spans, raw_results),
        root_spans=root_spans,
        aggregate=TraceAggregate(
            cost_usd=raw.cost_usd,
            latency_ms=_trace_duration_ms(raw),
        ),
        attributes=attributes,
    )


def _normalize_evaluator_results(
    trace: _IntakeTrace,
    spans: list[_IntakeSpan],
    results: list[_EvaluatorResult],
) -> dict[str, JsonValue]:
    span_ids = {span.span_id for span in spans}
    seen_ids: set[str] = set()
    seen_targets: set[tuple[str, str]] = set()
    by_name: dict[str, list[JsonValue]] = {}
    for result in sorted(results, key=lambda row: (row.name, row.span_id, row.evaluator_result_id)):
        if result.session_id != trace.session_id or result.workspace != trace.workspace:
            raise ValueError(
                f"evaluator result {result.evaluator_result_id!r} belongs to another session or workspace"
            )
        if result.span_id not in span_ids:
            # Sessions can contain multiple traces; results belong to their target span.
            continue
        target = (result.span_id, result.name)
        if result.evaluator_result_id in seen_ids or target in seen_targets:
            raise ValueError(
                f"duplicate evaluator result for span {result.span_id!r}, name {result.name!r}"
            )
        seen_ids.add(result.evaluator_result_id)
        seen_targets.add(target)
        by_name.setdefault(result.name, []).append(
            result.model_dump(mode="json", exclude_none=True)
        )
    # Preserve repeated metric names on different spans without overwriting either result.
    return {name: values[0] if len(values) == 1 else values for name, values in by_name.items()}


def _normalize_span(
    raw: _IntakeSpan,
    *,
    base_pointer: dict[str, JsonValue],
    tool_call_index: int,
) -> Span:
    input_value = _payload(raw, "input")
    output_value = _payload(raw, "output")
    kind, subtype = _span_kind(raw.kind)
    tool_call: ToolCall | None = None
    tool_name = raw.tool_name
    if kind is SpanKind.TOOL and raw.source == "atif":
        input_value, output_value, tool_name, tool_call = _normalize_atif_tool_payload(
            raw,
            input_value=input_value,
            output_value=output_value,
        )
    if kind is SpanKind.TOOL:
        details = tool_call or ToolCall()
        tool_call = details.model_copy(
            update={"index": tool_call_index, "result_count": 0 if output_value is UNSET else 1}
        )

    started_at = _aware_utc(raw.started_at)
    ended_at = _aware_utc(raw.ended_at) if raw.ended_at is not None else None
    duration_ms = (ended_at - started_at).total_seconds() * 1000.0 if ended_at is not None else None
    explicit_error = (
        True
        if raw.status is IntakeStatus.ERROR
        else False
        if raw.status is IntakeStatus.SUCCESS
        else None
    )
    error = None
    if raw.status is IntakeStatus.ERROR:
        error = raw.error_message or raw.error_type or "Intake span status error"
    attributes: dict[str, JsonValue] = {
        "name": raw.name,
        "subtype": subtype,
        "summary": raw.error_message if output_value is UNSET else None,
        "duration_ms": duration_ms,
        "source_pointer": {**base_pointer, "span_id": raw.span_id},
        "status": raw.status.value,
        "explicit_error": explicit_error,
        "outcome_marker": raw.error_type,
        "intake": {
            "session_id": raw.session_id,
            "workspace": raw.workspace,
            "source": raw.source,
            "kind": raw.kind,
            "parent_span_id": raw.parent_span_id,
            "raw_attributes": _raw_attributes(raw),
        },
    }
    return Span(
        id=raw.span_id,
        kind=kind,
        # Children are attached later; keep them in exclude_unset serialization.
        children=[],
        start_time=started_at,
        end_time=ended_at,
        input=input_value,
        output=output_value,
        model=raw.model,
        cost_usd=raw.cost_total_usd,
        tool_name=tool_name if kind is SpanKind.TOOL else None,
        tool_call=tool_call,
        error=error,
        attributes=attributes,
    )


def _span_kind(value: str) -> tuple[SpanKind, str | None]:
    normalized = value.upper()
    try:
        return SpanKind(normalized), None
    except ValueError:
        return SpanKind.UNKNOWN, normalized


def _trace_duration_ms(raw: _IntakeTrace) -> float | None:
    if raw.duration_ms is not None:
        return raw.duration_ms
    if raw.ended_at is None:
        return None
    return (_aware_utc(raw.ended_at) - _aware_utc(raw.started_at)).total_seconds() * 1000.0


def _canonical_spans(trace_id: str, spans: list[_IntakeSpan]) -> list[_IntakeSpan]:
    by_id: dict[str, _IntakeSpan] = {}
    children: dict[str, list[str]] = {}
    for span in spans:
        if span.span_id in by_id:
            raise ValueError(f"trace {trace_id!r} contains duplicate span id {span.span_id!r}")
        by_id[span.span_id] = span
    for span in spans:
        if span.parent_span_id is None:
            continue
        if span.parent_span_id not in by_id:
            raise ValueError(
                f"trace {trace_id!r} span {span.span_id!r} references missing parent "
                f"{span.parent_span_id!r}"
            )
        children.setdefault(span.parent_span_id, []).append(span.span_id)

    ready = [
        (_aware_utc(span.started_at), span.span_id) for span in spans if span.parent_span_id is None
    ]
    heapq.heapify(ready)
    result: list[_IntakeSpan] = []
    while ready:
        _, span_id = heapq.heappop(ready)
        result.append(by_id[span_id])
        for child_id in children.get(span_id, []):
            heapq.heappush(ready, (_aware_utc(by_id[child_id].started_at), child_id))
    if len(result) != len(spans):
        raise ValueError(f"trace {trace_id!r} contains a parent cycle")
    return result


def _normalize_atif_tool_payload(
    raw: _IntakeSpan,
    *,
    input_value: JsonValue | UNSET,
    output_value: JsonValue | UNSET,
) -> tuple[JsonValue | UNSET, JsonValue | UNSET, str | None, ToolCall]:
    call_id: str | None = None
    result_id: str | None = None
    tool_name = raw.tool_name
    arguments = input_value
    result = output_value
    if isinstance(input_value, Mapping):
        if input_value.get("tool_call_id") is not None:
            call_id = str(input_value["tool_call_id"])
        if input_value.get("function_name") is not None:
            tool_name = str(input_value["function_name"])
        if "arguments" in input_value:
            arguments = input_value["arguments"]
    if isinstance(output_value, Mapping):
        if output_value.get("source_call_id") is not None:
            result_id = str(output_value["source_call_id"])
        if "content" in output_value:
            result = output_value["content"]
    return (
        arguments,
        result,
        tool_name,
        ToolCall(call_id=call_id, result_id=result_id),
    )


def _atif_tool_catalog(spans: list[_IntakeSpan]) -> dict[str, JsonValue] | None:
    catalog: dict[str, JsonValue] = {}
    for span in spans:
        if span.source != "atif":
            continue
        agent = _raw_attributes(span).get("agent")
        definitions = agent.get("tool_definitions") if isinstance(agent, Mapping) else None
        if not isinstance(definitions, list):
            continue
        for definition in definitions:
            if not isinstance(definition, Mapping) or not definition.get("name"):
                continue
            name = str(definition["name"])
            schema = definition.get("parameters") or definition.get("input_schema")
            catalog[name] = dict(schema) if isinstance(schema, Mapping) else None
    return catalog or None
