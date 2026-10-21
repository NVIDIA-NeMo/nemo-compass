# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load MLflow traces into the provider-neutral trace contracts."""

from __future__ import annotations

import hashlib
import heapq
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlencode

from pydantic import JsonValue

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
from trace_ingest.source_links import file_source_url, http_source_url

if TYPE_CHECKING:
    from mlflow import MlflowClient
    from mlflow.entities import Span as MLflowSpan
    from mlflow.entities import SpanStatus as MLflowSpanStatus
    from mlflow.entities import Trace as MLflowTrace

_INPUTS_ATTRIBUTE = "mlflow.spanInputs"
_OUTPUTS_ATTRIBUTE = "mlflow.spanOutputs"
_SPAN_TYPE_ATTRIBUTE = "mlflow.spanType"
_SESSION_METADATA = "mlflow.trace.session"
_SESSION_ATTRIBUTE = "session.id"
_ORDER_BY = ["timestamp_ms DESC", "request_id ASC"]
_SEARCH_PAGE_SIZE = 500

MLFLOW_DEFAULT_MAX_TRACES = 10_000

__all__ = [
    "MLflowFileTraceConfig",
    "MLflowFileTraceDescription",
    "MLflowFileTraceLoader",
    "MLFLOW_DEFAULT_MAX_TRACES",
    "MLflowLoadReport",
    "MLflowTraceConfig",
    "MLflowTraceDescription",
    "MLflowTraceLoadError",
    "MLflowTraceLoader",
]


class MLflowTraceLoadError(RuntimeError):
    """An MLflow source could not be resolved, fetched, or normalized."""


@dataclass(frozen=True)
class MLflowLoadReport:
    """Diagnostics produced by the most recent load."""

    trace_count: int = 0
    span_count: int = 0
    unresolved_parent_count: int = 0
    call_count: int = 0
    distinct_logical_cases: int = 0


class MLflowTraceDescription(TraceDescription):
    """Run metadata for a live MLflow trace query."""

    span_count: int
    unresolved_parent_count: int
    experiment_name: str
    experiment_id: str
    tracking_uri: str
    filter: str | None
    max_traces: int


class MLflowFileTraceDescription(TraceDescription):
    """Run metadata for a native MLflow trace export."""

    span_count: int
    unresolved_parent_count: int
    export_path: str
    sha256: str | None
    continuation_token_present: bool
    export_trace_count: int
    truncated: bool
    max_traces: int


@dataclass(frozen=True)
class MLflowTraceConfig:
    """Configuration for loading traces from an MLflow experiment."""

    experiment_name: str
    tracking_uri: str | None = None
    filter_string: str | None = None
    max_traces: int = MLFLOW_DEFAULT_MAX_TRACES

    def __post_init__(self) -> None:
        if not self.experiment_name.strip():
            raise ValueError("experiment_name must not be empty")
        if self.max_traces < 1:
            raise ValueError("max_traces must be at least 1")


@dataclass
class MLflowTraceLoader:
    """Load a bounded selection of traces from one MLflow experiment.

    ``client`` is an injection seam for tests and custom authentication. When it
    is omitted, MLflow is imported only when :meth:`load` is called, so local
    file users do not need the optional dependency.
    """

    config: MLflowTraceConfig
    client: MlflowClient | None = field(default=None, repr=False)
    report: MLflowLoadReport = field(default_factory=MLflowLoadReport, init=False)
    _experiment_id: str | None = field(default=None, init=False, repr=False)
    _resolved_tracking_uri: str | None = field(default=None, init=False, repr=False)

    def load(self) -> TraceSnapshot:
        """Fetch and normalize the configured MLflow corpus."""

        try:
            client = self.client or _new_mlflow_client(self.config.tracking_uri)
            tracking_uri = self.config.tracking_uri or _client_tracking_uri(client)
            experiment = client.get_experiment_by_name(self.config.experiment_name)
            if experiment is None:
                raise MLflowTraceLoadError(
                    f"MLflow experiment {self.config.experiment_name!r} was not found at "
                    f"{tracking_uri}"
                )
            experiment_id = str(experiment.experiment_id)
            sort_keys: dict[str, tuple[int, str]] = {}
            logical_cases: set[str] = set()
            unresolved_parent_count = 0
            span_count = 0
            call_count = 0

            def traces() -> Iterator[Trace]:
                nonlocal unresolved_parent_count, span_count, call_count
                for page in self._search_trace_pages(client, experiment_id):
                    for provider_trace in page:
                        trace, detached, trace_span_count, trace_call_count = _normalize_trace(
                            provider_trace,
                            source_pointer={
                                "provider": "mlflow",
                                "tracking_uri": tracking_uri,
                                "experiment_name": self.config.experiment_name,
                                "experiment_id": experiment_id,
                            },
                        )
                        trace.source_url = _trace_url(tracking_uri, experiment_id, trace.id)
                        sort_keys[trace.id] = _trace_sort_key(provider_trace)
                        logical_cases.add(_logical_case(trace))
                        unresolved_parent_count += detached
                        span_count += trace_span_count
                        call_count += trace_call_count
                        yield trace

            snapshot = TraceSnapshot(traces())
            snapshot.sort(sort_keys.__getitem__)
        except (MLflowTraceLoadError, ModuleNotFoundError):
            raise
        except Exception as error:
            raise MLflowTraceLoadError(
                f"Failed to load MLflow experiment {self.config.experiment_name!r}: {error}"
            ) from error

        self._experiment_id = experiment_id
        self._resolved_tracking_uri = tracking_uri
        self.report = MLflowLoadReport(
            trace_count=snapshot.trace_count,
            span_count=span_count,
            unresolved_parent_count=unresolved_parent_count,
            call_count=call_count,
            distinct_logical_cases=len(logical_cases),
        )
        return snapshot

    def _search_trace_pages(
        self, client: MlflowClient, experiment_id: str
    ) -> Iterator[list[MLflowTrace]]:
        loaded_count = 0
        seen: set[str] = set()
        seen_tokens: set[str] = set()
        page_token: str | None = None

        while loaded_count < self.config.max_traces:
            remaining = self.config.max_traces - loaded_count
            page_size = min(remaining, _SEARCH_PAGE_SIZE)
            page = client.search_traces(
                locations=[experiment_id],
                filter_string=self.config.filter_string,
                max_results=page_size,
                order_by=list(_ORDER_BY),
                page_token=page_token,
                include_spans=True,
            )
            items = list(page)
            next_token = page.token
            if not items:
                if next_token:
                    raise MLflowTraceLoadError(
                        "MLflow search returned an empty page with a continuation token"
                    )
                break
            if len(items) > remaining:
                raise MLflowTraceLoadError(
                    f"MLflow search returned {len(items)} traces when at most {remaining} "
                    "were requested"
                )

            for item in items:
                trace_id = _trace_id(item)
                if trace_id in seen:
                    raise MLflowTraceLoadError(
                        f"MLflow search returned duplicate trace id {trace_id!r}"
                    )
                seen.add(trace_id)
            loaded_count += len(items)
            yield items

            if not next_token:
                break
            page_token = str(next_token)
            if page_token in seen_tokens:
                raise MLflowTraceLoadError(
                    f"MLflow search repeated continuation token {page_token!r}"
                )
            seen_tokens.add(page_token)

    def describe(self) -> MLflowTraceDescription:
        """Describe the normalized corpus and its MLflow coordinates."""

        tracking_uri = self._resolved_tracking_uri or self.config.tracking_uri or "<configured>"
        experiment_id = self._experiment_id or "<unresolved>"
        return {
            "source": _source_label(tracking_uri, experiment_id),
            "trace_count": self.report.trace_count,
            "call_count": self.report.call_count,
            "distinct_logical_cases": self.report.distinct_logical_cases,
            "span_count": self.report.span_count,
            "unresolved_parent_count": self.report.unresolved_parent_count,
            "experiment_name": self.config.experiment_name,
            "experiment_id": experiment_id,
            "tracking_uri": tracking_uri,
            "filter": self.config.filter_string,
            "max_traces": self.config.max_traces,
        }


@dataclass(frozen=True)
class MLflowFileTraceConfig:
    """Configuration for loading a native MLflow trace export."""

    path: Path
    max_traces: int = MLFLOW_DEFAULT_MAX_TRACES

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if self.max_traces < 1:
            raise ValueError("max_traces must be at least 1")


@dataclass
class MLflowFileTraceLoader:
    """Load traces from a native MLflow JSON export on the local filesystem."""

    config: MLflowFileTraceConfig
    report: MLflowLoadReport = field(default_factory=MLflowLoadReport, init=False)
    _continuation_token_present: bool = field(default=False, init=False, repr=False)
    _digest: str | None = field(default=None, init=False, repr=False)
    _export_trace_count: int = field(default=0, init=False, repr=False)

    def load(self) -> TraceSnapshot:
        """Parse and normalize the configured native MLflow export."""

        resolved_path = self.config.path.resolve()
        try:
            records, continuation_token_present, digest, export_trace_count = _read_mlflow_export(
                self.config.path, self.config.max_traces
            )
            sort_keys: dict[str, tuple[int, str]] = {}
            logical_cases: set[str] = set()
            unresolved_parent_count = 0
            span_count = 0
            call_count = 0
            seen: set[str] = set()

            def traces() -> Iterator[Trace]:
                nonlocal unresolved_parent_count, span_count, call_count
                for provider_trace in map(_mlflow_trace_from_dict, records):
                    trace_id = _trace_id(provider_trace)
                    if trace_id in seen:
                        raise MLflowTraceLoadError(
                            f"MLflow export {str(self.config.path)!r} contains duplicate trace id "
                            f"{trace_id!r}"
                        )
                    seen.add(trace_id)
                    if not provider_trace.data.spans:
                        raise MLflowTraceLoadError(
                            f"MLflow export trace {trace_id!r} has no spans; export complete traces "
                            "without --no-include-spans"
                        )
                    trace, detached, trace_span_count, trace_call_count = _normalize_trace(
                        provider_trace,
                        source_pointer={
                            "provider": "mlflow",
                            "export_path": str(resolved_path),
                        },
                    )
                    trace.source_url = file_source_url(resolved_path)
                    sort_keys[trace.id] = _trace_sort_key(provider_trace)
                    logical_cases.add(_logical_case(trace))
                    unresolved_parent_count += detached
                    span_count += trace_span_count
                    call_count += trace_call_count
                    yield trace

            snapshot = TraceSnapshot(traces())
            snapshot.sort(sort_keys.__getitem__)
        except (MLflowTraceLoadError, FileNotFoundError, ModuleNotFoundError):
            raise
        except Exception as error:
            raise MLflowTraceLoadError(
                f"Failed to load MLflow export {str(self.config.path)!r}: {error}"
            ) from error

        self._continuation_token_present = continuation_token_present
        self._digest = digest
        self._export_trace_count = export_trace_count
        self.report = MLflowLoadReport(
            trace_count=snapshot.trace_count,
            span_count=span_count,
            unresolved_parent_count=unresolved_parent_count,
            call_count=call_count,
            distinct_logical_cases=len(logical_cases),
        )
        return snapshot

    def describe(self) -> MLflowFileTraceDescription:
        """Describe the normalized corpus and its export provenance."""

        resolved_path = self.config.path.resolve()
        return {
            "source": f"mlflow-export:{resolved_path}",
            "trace_count": self.report.trace_count,
            "call_count": self.report.call_count,
            "distinct_logical_cases": self.report.distinct_logical_cases,
            "span_count": self.report.span_count,
            "unresolved_parent_count": self.report.unresolved_parent_count,
            "export_path": str(resolved_path),
            "sha256": self._digest,
            "continuation_token_present": self._continuation_token_present,
            "export_trace_count": self._export_trace_count,
            "truncated": self._export_trace_count > self.report.trace_count,
            "max_traces": self.config.max_traces,
        }


def _new_mlflow_client(tracking_uri: str | None) -> MlflowClient:
    from mlflow import MlflowClient

    return MlflowClient(tracking_uri=tracking_uri)


def _mlflow_trace_from_dict(record: dict[str, Any]) -> MLflowTrace:
    from mlflow.entities import Trace as ProviderTrace

    try:
        return ProviderTrace.from_dict(record)
    except Exception as error:
        info = record.get("info")
        trace_id = info.get("trace_id") if isinstance(info, dict) else None
        context = f" trace {trace_id!r}" if trace_id else ""
        raise MLflowTraceLoadError(f"Failed to parse MLflow export{context}: {error}") from error


def _read_mlflow_export(path: Path, max_traces: int) -> tuple[list[dict[str, Any]], bool, str, int]:
    contents = path.read_bytes()
    records, continuation_token_present = _parse_mlflow_export(contents, path)
    return (
        records[:max_traces],
        continuation_token_present,
        hashlib.sha256(contents).hexdigest(),
        len(records),
    )


def _parse_mlflow_export(contents: bytes, path: Path) -> tuple[list[dict[str, Any]], bool]:
    try:
        text = contents.decode("utf-8")
    except UnicodeDecodeError as error:
        raise MLflowTraceLoadError(f"MLflow export {str(path)!r} is not UTF-8 text") from error

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = _parse_mlflow_jsonl(text, path)

    continuation_token_present = False
    if isinstance(payload, list):
        raw_records = payload
    elif isinstance(payload, dict) and "info" in payload and "data" in payload:
        raw_records = [payload]
    elif isinstance(payload, dict) and "traces" in payload:
        raw_records = payload["traces"]
        if not isinstance(raw_records, list):
            raise MLflowTraceLoadError(
                f"MLflow export {str(path)!r} field 'traces' must be a JSON array"
            )
        continuation_token_present = payload.get("next_page_token") not in (None, "")
    else:
        raise MLflowTraceLoadError(
            f"MLflow export {str(path)!r} must contain a trace, a trace array, "
            "a search result with a 'traces' array, or one trace per JSONL line"
        )

    records: list[dict[str, Any]] = []
    for index, record in enumerate(raw_records, start=1):
        if not isinstance(record, dict) or "info" not in record or "data" not in record:
            raise MLflowTraceLoadError(
                f"MLflow export {str(path)!r} trace {index} must contain 'info' and 'data'"
            )
        records.append(record)
    return records, continuation_token_present


def _parse_mlflow_jsonl(text: str, path: Path) -> list[object]:
    records: list[object] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise MLflowTraceLoadError(
                f"MLflow export {str(path)!r} is neither valid JSON nor JSONL; "
                f"line {line_number}: {error.msg}"
            ) from error
    if not records:
        raise MLflowTraceLoadError(f"MLflow export {str(path)!r} contains no traces")
    return records


def _client_tracking_uri(client: MlflowClient) -> str:
    value = client.tracking_uri
    return str(value) if value else "<configured>"


def _source_label(tracking_uri: str, experiment_id: str) -> str:
    return f"mlflow:{tracking_uri}#experiment/{experiment_id}"


def _trace_id(provider_trace: MLflowTrace) -> str:
    trace_id = provider_trace.info.trace_id
    if not trace_id:
        raise MLflowTraceLoadError("MLflow returned a trace without a trace id")
    return trace_id


def _trace_sort_key(provider_trace: MLflowTrace) -> tuple[int, str]:
    return (provider_trace.info.request_time, _trace_id(provider_trace))


def _normalize_trace(
    provider_trace: MLflowTrace,
    *,
    source_pointer: dict[str, JsonValue],
) -> tuple[Trace, int, int, int]:
    trace_id = _trace_id(provider_trace)
    provider_spans = provider_trace.data.spans
    by_id: dict[str, MLflowSpan] = {}
    for provider_span in provider_spans:
        span_id = str(provider_span.span_id)
        if span_id in by_id:
            raise MLflowTraceLoadError(
                f"MLflow trace {trace_id!r} contains duplicate span id {span_id!r}"
            )
        by_id[span_id] = provider_span

    effective_parents: dict[str, str | None] = {}
    unresolved: dict[str, str] = {}
    for span_id, provider_span in by_id.items():
        parent = provider_span.parent_id
        parent_id = str(parent) if parent else None
        if parent_id is not None and parent_id not in by_id:
            unresolved[span_id] = parent_id
            parent_id = None
        effective_parents[span_id] = parent_id

    ordered = _canonical_span_order(trace_id, by_id, effective_parents)
    base_pointer: dict[str, JsonValue] = {**source_pointer, "trace_id": trace_id}
    normalized_by_id: dict[str, Span] = {}
    root_spans: list[Span] = []
    tool_call_index = 0
    for provider_span in ordered:
        span_id = str(provider_span.span_id)
        normalized = _normalize_span(
            provider_span,
            unresolved_parent_id=unresolved.get(span_id),
            base_pointer=base_pointer,
        )
        if normalized.kind is SpanKind.TOOL:
            normalized.tool_call = ToolCall(
                call_id=span_id,
                index=tool_call_index,
                result_count=0 if normalized.output is UNSET else 1,
            )
            tool_call_index += 1
        normalized_by_id[span_id] = normalized
        parent_id = effective_parents[span_id]
        if parent_id is None:
            root_spans.append(normalized)
        else:
            normalized_by_id[parent_id].children.append(normalized)

    metadata = dict(provider_trace.info.trace_metadata or {})
    logical_case_id = _logical_case_id(metadata, ordered, effective_parents)
    assessments = [assessment.to_dictionary() for assessment in provider_trace.info.assessments]
    evaluator_results: dict[str, JsonValue] = {}
    for row in assessments:
        name = row.get("assessment_name")
        if not isinstance(name, str) or not name or row.get("valid") is False:
            continue
        if name in evaluator_results:
            raise MLflowTraceLoadError(f"duplicate active assessment {name!r}")
        evaluator_results[name] = _json_value(row)
    attributes: dict[str, JsonValue] = {
        "source_pointer": base_pointer,
        "mlflow": {
            "request_time": provider_trace.info.request_time,
            "trace_metadata": _json_value(metadata),
            "assessments": _json_value(assessments),
        },
    }
    if logical_case_id is not None:
        attributes["logical_case_id"] = logical_case_id

    trace = Trace(
        id=trace_id,
        root_spans=root_spans,
        aggregate=TraceAggregate(latency_ms=_trace_latency_ms(ordered)),
        evaluator_results=evaluator_results,
        attributes=attributes,
    )
    return (
        trace,
        len(unresolved),
        len(normalized_by_id),
        tool_call_index,
    )


def _canonical_span_order(
    trace_id: str,
    by_id: dict[str, MLflowSpan],
    effective_parents: dict[str, str | None],
) -> list[MLflowSpan]:
    children: dict[str, list[str]] = {span_id: [] for span_id in by_id}
    indegree = {span_id: 0 for span_id in by_id}
    for span_id, parent_id in effective_parents.items():
        if parent_id is not None:
            children[parent_id].append(span_id)
            indegree[span_id] += 1

    ready: list[tuple[int, str]] = []
    for span_id, degree in indegree.items():
        if degree == 0:
            heapq.heappush(ready, (by_id[span_id].start_time_ns, span_id))

    ordered: list[MLflowSpan] = []
    while ready:
        _, span_id = heapq.heappop(ready)
        ordered.append(by_id[span_id])
        for child_id in children[span_id]:
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                heapq.heappush(ready, (by_id[child_id].start_time_ns, child_id))

    if len(ordered) != len(by_id):
        raise MLflowTraceLoadError(f"MLflow trace {trace_id!r} contains a parent cycle")
    return ordered


def _normalize_span(
    provider_span: MLflowSpan,
    *,
    unresolved_parent_id: str | None,
    base_pointer: dict[str, JsonValue],
) -> Span:
    span_id = str(provider_span.span_id)
    provider_type = _provider_span_type(provider_span)
    kind, subtype = _map_span_kind(provider_type)
    started_at = _datetime_from_ns(provider_span.start_time_ns)
    ended_at = _datetime_from_ns(provider_span.end_time_ns)
    duration_ms = None
    if started_at is not None and ended_at is not None:
        duration_ms = (ended_at - started_at).total_seconds() * 1000

    pointer = {**base_pointer, "span_id": span_id}
    if unresolved_parent_id is not None:
        pointer["unresolved_parent_span_id"] = unresolved_parent_id
    status, status_description = _span_status(provider_span.status)
    error = None
    if status == "ERROR":
        error = status_description or "MLflow span status ERROR"
    name = provider_span.name
    return Span(
        id=span_id,
        kind=kind,
        start_time=started_at,
        end_time=ended_at,
        input=_span_field(provider_span, _INPUTS_ATTRIBUTE, provider_span.inputs),
        output=_span_field(provider_span, _OUTPUTS_ATTRIBUTE, provider_span.outputs),
        tool_name=name if kind is SpanKind.TOOL else None,
        error=error,
        attributes={
            "name": str(name) if name is not None else None,
            "subtype": subtype,
            "duration_ms": duration_ms,
            "source_pointer": pointer,
            "status": status,
            "status_description": status_description,
            "mlflow": {"attributes": _json_value(provider_span.attributes)},
        },
    )


def _provider_span_type(provider_span: MLflowSpan) -> str:
    provider_type = provider_span.span_type
    if provider_type:
        return provider_type.upper()
    attributes = provider_span.attributes
    provider_type = attributes.get(_SPAN_TYPE_ATTRIBUTE) or attributes.get(
        "openinference.span.kind"
    )
    return str(provider_type or "UNKNOWN").upper()


def _map_span_kind(provider_type: str) -> tuple[SpanKind, str | None]:
    if provider_type == "CHAT_MODEL":
        return SpanKind.LLM, provider_type
    try:
        return SpanKind(provider_type), None
    except ValueError:
        return SpanKind.UNKNOWN, provider_type


def _span_status(provider_status: MLflowSpanStatus) -> tuple[str, str | None]:
    code = str(provider_status.status_code.value)
    description = provider_status.description
    return code, str(description) if description else None


def _trace_latency_ms(provider_spans: list[MLflowSpan]) -> float | None:
    starts = [
        started
        for span in provider_spans
        if (started := _datetime_from_ns(span.start_time_ns)) is not None
    ]
    ends = [
        ended
        for span in provider_spans
        if (ended := _datetime_from_ns(span.end_time_ns)) is not None
    ]
    if not starts or not ends:
        return None
    return (max(ends) - min(starts)).total_seconds() * 1_000


def _logical_case(trace: Trace) -> str:
    logical_case_id = trace.attributes.get("logical_case_id")
    return str(logical_case_id) if logical_case_id is not None else trace.id


def _span_field(provider_span: MLflowSpan, attribute_key: str, value: object) -> JsonValue | UNSET:
    attributes = provider_span.attributes
    if attribute_key not in attributes:
        return UNSET
    # OTLP ingestion in MLflow 3.15.2 leaves an explicit JSON null as the
    # literal string ``"null"``. Attribute presence has already established
    # that this is an observed value rather than a missing input/output.
    if value == "null" and attributes.get("output.mime_type") == "application/json":
        return None
    return _json_value(value)


def _json_value(value: object) -> JsonValue:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise MLflowTraceLoadError(f"MLflow value is not JSON-serializable: {error}") from error


def _datetime_from_ns(value: int | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromtimestamp(int(value) / 1_000_000_000, tz=timezone.utc)


def _logical_case_id(
    metadata: dict[str, str],
    ordered_spans: list[MLflowSpan],
    effective_parents: dict[str, str | None],
) -> str | None:
    session = metadata.get(_SESSION_METADATA)
    if session in (None, ""):
        for provider_span in ordered_spans:
            if effective_parents[str(provider_span.span_id)] is not None:
                continue
            attributes = provider_span.attributes
            session = attributes.get(_SESSION_ATTRIBUTE)
            if session not in (None, ""):
                break
    return str(session) if session not in (None, "") else None


def _trace_url(tracking_uri: str, experiment_id: str, trace_id: str) -> str | None:
    # MLflow's UI supports both the legacy table and the current trace selector.
    base = http_source_url(tracking_uri)
    if base is None:
        return None
    query = urlencode({"selectedEvaluationId": trace_id, "traceId": trace_id})
    return f"{base.rstrip('/')}/#/experiments/{quote(experiment_id, safe='')}/traces?{query}"
