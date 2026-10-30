# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize Langfuse observations into provider-neutral traces."""

from __future__ import annotations

import heapq
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from urllib.parse import quote

from pydantic import JsonValue

from trace_ingest.loaders.helper import llm_span
from trace_ingest.loaders.helper.llm_span import SpanData, nemo_relay
from trace_ingest.loaders.helper.tool_definitions import tool_definitions
from trace_ingest.loaders.langfuse.config import LangfuseTraceLoadError
from trace_ingest.models import (
    UNSET,
    Span,
    SpanKind,
    ToolCall,
    ToolDefinition,
    ToolResult,
    Trace,
    TraceAggregate,
)
from trace_ingest.source_links import http_source_url

if TYPE_CHECKING:
    from langfuse.api.commons.types.observation_v2 import ObservationV2
    from langfuse.api.commons.types.observations_view import ObservationsView
    from langfuse.api.commons.types.trace_with_details import TraceWithDetails
    from langfuse.api.commons.types.trace_with_full_details import TraceWithFullDetails
    from langfuse.api.scores_v3.types.score_v3 import ScoreV3

    LangfuseObservation = ObservationsView | ObservationV2


def _trace_id(provider_trace: TraceWithDetails | TraceWithFullDetails) -> str:
    if not provider_trace.id:
        raise LangfuseTraceLoadError("Langfuse returned a trace without an id")
    return str(provider_trace.id)


def _trace_timestamp(provider_trace: TraceWithDetails | TraceWithFullDetails) -> datetime:
    value = provider_trace.timestamp
    if value.tzinfo is None or value.utcoffset() is None:
        raise LangfuseTraceLoadError(
            f"Langfuse trace {provider_trace.id!r} timestamp must include a timezone"
        )
    return value.astimezone(timezone.utc)


def _normalize_trace(
    provider_trace: TraceWithFullDetails,
    *,
    source_pointer: dict[str, JsonValue],
    require_complete_tree: bool = False,
) -> Trace:
    trace_id = _trace_id(provider_trace)
    root_spans, ordered = _normalize_tree(
        trace_id, provider_trace.observations, source_pointer, require_complete_tree
    )
    attributes: dict[str, JsonValue] = {
        "source_pointer": source_pointer,
        "langfuse": {
            "trace_name": provider_trace.name,
            "timestamp": _trace_timestamp(provider_trace).isoformat(),
            "input": _json_value(provider_trace.input),
            "output": _json_value(provider_trace.output),
            "user_id": provider_trace.user_id,
            "session_id": provider_trace.session_id,
            "release": provider_trace.release,
            "version": provider_trace.version,
            "environment": provider_trace.environment,
            "tags": _json_value(provider_trace.tags),
            "public": provider_trace.public,
            "metadata": _json_value(provider_trace.metadata),
            "html_path": provider_trace.html_path,
            "scores": _json_value(provider_trace.scores),
        },
    }

    # Keep repeated evaluations and observation associations rather than choosing a winner.
    scores_by_name: dict[str, list[JsonValue]] = {}
    for score in sorted(provider_trace.scores, key=lambda score: (score.name, score.id)):
        scores_by_name.setdefault(score.name, []).append(_json_value(score))

    return Trace(
        id=trace_id,
        session_id=str(provider_trace.session_id) if provider_trace.session_id else None,
        session_basis="langfuse.sessionId" if provider_trace.session_id else None,
        root_spans=root_spans,
        aggregate=TraceAggregate(
            cost_usd=_trace_cost(provider_trace, ordered),
            latency_ms=_trace_latency_ms(provider_trace, ordered),
        ),
        attributes=attributes,
        evaluator_results={name: scores for name, scores in scores_by_name.items()},
    )


def _normalize_tree(
    trace_id: str,
    observations: Sequence[LangfuseObservation],
    source_pointer: dict[str, JsonValue],
    require_complete_tree: bool = False,
) -> tuple[list[Span], list[LangfuseObservation]]:
    by_id: dict[str, LangfuseObservation] = {}
    for observation in observations:
        observation_id = str(observation.id)
        if observation_id in by_id:
            raise LangfuseTraceLoadError(
                f"Langfuse trace {trace_id!r} contains duplicate observation id {observation_id!r}"
            )
        by_id[observation_id] = observation

    effective_parents: dict[str, str | None] = {}
    unresolved: dict[str, str] = {}
    for observation_id, observation in by_id.items():
        parent_id = (
            str(observation.parent_observation_id) if observation.parent_observation_id else None
        )
        if parent_id is not None and parent_id not in by_id:
            if require_complete_tree:
                raise LangfuseTraceLoadError("incomplete observation tree: missing parent")
            unresolved[observation_id] = parent_id
            parent_id = None
        effective_parents[observation_id] = parent_id

    ordered = _canonical_observation_order(trace_id, by_id, effective_parents)
    normalized_by_id: dict[str, Span] = {}
    root_spans: list[Span] = []
    tool_call_index = 0
    for observation in ordered:
        observation_id = str(observation.id)
        normalized = _normalize_observation(
            observation,
            unresolved_parent_id=unresolved.get(observation_id),
            base_pointer=source_pointer,
        )
        if normalized.kind is SpanKind.TOOL:
            results = (
                []
                if normalized.output is UNSET
                else [
                    ToolResult(
                        content=normalized.output, is_error=True if normalized.error else None
                    )
                ]
            )
            normalized.tool_call = ToolCall(
                call_id=observation_id,
                index=tool_call_index,
                result_count=len(results),
                results=results,
            )
            tool_call_index += 1
        normalized_by_id[observation_id] = normalized
        parent_id = effective_parents[observation_id]
        if parent_id is None:
            root_spans.append(normalized)
        else:
            normalized_by_id[parent_id].children.append(normalized)

    return root_spans, ordered


def _v4_observation_context(observation: LangfuseObservation) -> dict[str, JsonValue]:
    from langfuse.api.commons.types.observation_v2 import ObservationV2

    if not isinstance(observation, ObservationV2):
        return {}
    return {
        "trace_name": observation.trace_name,
        "user_id": observation.user_id,
        "session_id": observation.session_id,
        "tags": _json_value(observation.tags),
        "release": observation.release,
        "is_root_observation": observation.is_root_observation,
    }


def _normalize_v4_trace(
    trace_id: str,
    observations: Sequence[ObservationV2],
    scores: Sequence[ScoreV3],
    base_url: str | None,
) -> Trace:
    pointer: dict[str, JsonValue] = {
        "provider": "langfuse",
        **({"base_url": base_url} if base_url else {}),
        "trace_id": trace_id,
    }
    roots, ordered = _normalize_tree(trace_id, observations, pointer)
    # No single trace-level I/O or session is authoritative when several physical
    # roots exist. Preserve per-observation values rather than choosing arbitrarily.
    physical_roots = [obs for obs in observations if obs.parent_observation_id is None]
    root = physical_roots[0] if len(physical_roots) == 1 else None
    attributes: dict[str, JsonValue] = {
        "source_pointer": pointer,
        "langfuse": {
            "api_version": "v4",
            "trace_name": root.trace_name if root else None,
            "timestamp": min(_started_at(obs) for obs in observations).isoformat(),
            "input": _json_value(_io_value(root.input))
            if root and root.input is not None
            else None,
            "output": _json_value(_io_value(root.output))
            if root and root.output is not None
            else None,
            "scores": _json_value(scores),
        },
    }
    sessions = {obs.session_id for obs in observations if obs.session_id}
    session_id = next(iter(sessions)) if len(sessions) == 1 else None
    results: dict[str, list[JsonValue]] = {}
    for score in sorted(scores, key=lambda item: (item.name, item.id)):
        results.setdefault(score.name, []).append(_json_value(score))
    costs = [cost for obs in observations if (cost := _observation_cost(obs)) is not None]
    starts = [_started_at(obs) for obs in observations]
    ends = [end for obs in observations if (end := _optional_datetime(obs.end_time)) is not None]
    project_ids = {obs.project_id for obs in observations}
    if len(project_ids) != 1:
        raise LangfuseTraceLoadError("Langfuse returned observations from multiple projects")
    project_id = next(iter(project_ids))
    return Trace(
        id=trace_id,
        session_id=session_id,
        session_basis="langfuse.sessionId" if session_id else None,
        root_spans=roots,
        source_url=http_source_url(
            base_url.rstrip("/")
            + f"/project/{quote(project_id, safe='')}/traces/{quote(trace_id, safe='')}"
        )
        if base_url
        else None,
        aggregate=TraceAggregate(
            cost_usd=sum(costs) if costs else None,
            latency_ms=(max(ends) - min(starts)).total_seconds() * 1000 if ends else None,
        ),
        attributes=attributes,
        evaluator_results=results,
    )


def _tool_definitions_from_input(value: object) -> list[ToolDefinition] | None:
    """Read the platform's recorded list, preserving each entry verbatim."""

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    return tool_definitions(value.get("tools")) if isinstance(value, Mapping) else None


def _canonical_observation_order(
    trace_id: str,
    by_id: Mapping[str, LangfuseObservation],
    effective_parents: Mapping[str, str | None],
) -> list[LangfuseObservation]:
    children: dict[str, list[str]] = {observation_id: [] for observation_id in by_id}
    indegree = {observation_id: 0 for observation_id in by_id}
    for observation_id, parent_id in effective_parents.items():
        if parent_id is not None:
            children[parent_id].append(observation_id)
            indegree[observation_id] += 1

    ready: list[tuple[datetime, str]] = []
    for observation_id, degree in indegree.items():
        if degree == 0:
            heapq.heappush(ready, (_started_at(by_id[observation_id]), observation_id))

    ordered: list[LangfuseObservation] = []
    while ready:
        _, observation_id = heapq.heappop(ready)
        ordered.append(by_id[observation_id])
        for child_id in children[observation_id]:
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                heapq.heappush(ready, (_started_at(by_id[child_id]), child_id))
    if len(ordered) != len(by_id):
        raise LangfuseTraceLoadError(f"Langfuse trace {trace_id!r} contains a parent cycle")
    return ordered


def _normalize_observation(
    observation: LangfuseObservation,
    *,
    unresolved_parent_id: str | None,
    base_pointer: dict[str, JsonValue],
) -> Span:
    observation_id = str(observation.id)
    kind, subtype = _map_observation_kind(observation.type)
    data = _span_data(observation)
    llm = llm_span.normalize(data)
    if llm is not None and kind is not SpanKind.LLM:
        # Relay's full export types its LLM calls as plain spans.
        kind, subtype = SpanKind.LLM, data.type
    definitions = None
    if kind in (SpanKind.AGENT, SpanKind.LLM):
        definitions = llm.tool_definitions if llm else None
        if definitions is None:
            definitions = _tool_definitions_from_input(observation.input)
    started_at = _started_at(observation)
    ended_at = _optional_datetime(observation.end_time)
    pointer: dict[str, JsonValue] = {**base_pointer, "observation_id": observation_id}
    if unresolved_parent_id is not None:
        pointer["unresolved_parent_observation_id"] = unresolved_parent_id

    level = _enum_value(observation.level)
    status_message = str(observation.status_message) if observation.status_message else None
    error = status_message or "Langfuse observation level ERROR" if level == "ERROR" else None
    name = str(observation.name) if observation.name else None
    # Langfuse's own model, which it prices by; else the one the span records,
    # such as Relay's for calls Langfuse does not type as generations.
    model = str(observation.model) if observation.model else (llm.model if llm else None)
    return Span(
        id=observation_id,
        kind=kind,
        start_time=started_at,
        end_time=ended_at,
        input=_io_value(observation.input),
        output=_io_value(observation.output),
        cost_usd=_observation_cost(observation),
        model=model,
        token_counts=llm.token_counts if llm else None,
        tool_definitions=definitions if definitions is not None else UNSET,
        llm_params=llm.params if llm else None,
        agent_id=nemo_relay.agent_id(data),
        tool_name=name if kind is SpanKind.TOOL else None,
        error=error,
        attributes={
            "name": name,
            "subtype": subtype,
            "duration_ms": ((ended_at - started_at).total_seconds() * 1_000 if ended_at else None),
            "source_pointer": pointer,
            "status": level,
            "status_message": status_message,
            "langfuse": {
                "metadata": _json_value(observation.metadata),
                "usage_details": _json_value(observation.usage_details),
                "cost_details": _json_value(observation.cost_details),
                "model_parameters": _json_value(observation.model_parameters),
                "version": observation.version,
                "environment": observation.environment,
                "prompt_id": observation.prompt_id,
                "prompt_name": observation.prompt_name,
                "prompt_version": observation.prompt_version,
                **_v4_observation_context(observation),
            },
        },
    )


def _map_observation_kind(value: object) -> tuple[SpanKind, str | None]:
    provider_type = str(value or "UNKNOWN").upper()
    if provider_type == "GENERATION":
        return SpanKind.LLM, provider_type
    try:
        return SpanKind(provider_type), None
    except ValueError:
        return SpanKind.UNKNOWN, provider_type


def _span_data(observation: LangfuseObservation) -> SpanData:
    """What the LLM span readers need, from where Langfuse keeps it: OTel span
    attributes and the instrumentation scope under ``metadata``, its own usage in
    ``usageDetails``."""

    metadata = observation.metadata if isinstance(observation.metadata, Mapping) else {}
    scope = metadata.get("scope")
    attributes = metadata.get("attributes")
    return SpanData(
        name=str(observation.name) if observation.name else None,
        type=str(observation.type) if observation.type else None,
        scope=str(scope.get("name")) if isinstance(scope, Mapping) and scope.get("name") else None,
        attributes=attributes if isinstance(attributes, Mapping) else {},
        usage_details=observation.usage_details or {},
    )


def _started_at(observation: LangfuseObservation) -> datetime:
    value = observation.start_time
    if value.tzinfo is None or value.utcoffset() is None:
        raise LangfuseTraceLoadError(
            f"Langfuse observation {observation.id!r} start time must include a timezone"
        )
    return value.astimezone(timezone.utc)


def _optional_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise LangfuseTraceLoadError("Langfuse observation end time must include a timezone")
    return value.astimezone(timezone.utc)


def _observation_cost(observation: LangfuseObservation) -> float | None:
    details = observation.cost_details or {}
    cost = details.get("total")
    if cost is None:
        from langfuse.api.commons.types.observation_v2 import ObservationV2

        cost = (
            observation.total_cost
            if isinstance(observation, ObservationV2)
            else observation.calculated_total_cost
        )
    return float(cost) if cost is not None and cost >= 0 else None


def _trace_cost(
    provider_trace: TraceWithFullDetails,
    observations: Sequence[LangfuseObservation],
) -> float | None:
    if provider_trace.total_cost is not None and provider_trace.total_cost >= 0:
        return float(provider_trace.total_cost)
    costs = [
        cost for observation in observations if (cost := _observation_cost(observation)) is not None
    ]
    return sum(costs) if costs else None


def _trace_latency_ms(
    provider_trace: TraceWithFullDetails,
    observations: Sequence[LangfuseObservation],
) -> float | None:
    if provider_trace.latency is not None and provider_trace.latency >= 0:
        return float(provider_trace.latency) * 1_000
    starts = [_started_at(observation) for observation in observations]
    ends = [
        end
        for observation in observations
        if (end := _optional_datetime(observation.end_time)) is not None
    ]
    if not starts or not ends:
        return None
    return (max(ends) - min(starts)).total_seconds() * 1_000


def _io_value(value: object) -> JsonValue | UNSET:
    if value is None:
        # The v3 API does not distinguish a missing value from an explicitly
        # recorded JSON null.
        return UNSET
    if isinstance(value, str):
        try:
            return _json_value(json.loads(value))
        except json.JSONDecodeError:
            pass
    return _json_value(value)


def _enum_value(value: object) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    return str(raw).upper()


def _json_default(value: object) -> object:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json", by_alias=True)
    legacy_dict = getattr(value, "dict", None)
    if callable(legacy_dict):
        return legacy_dict(by_alias=True)
    return str(value)


def _json_value(value: object) -> JsonValue:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False, default=_json_default))
