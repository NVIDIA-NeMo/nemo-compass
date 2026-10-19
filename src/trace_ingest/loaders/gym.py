# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read Gym rollout JSON/JSONL with the versioned ``ng_trajectory`` attachment.

Source: NVIDIA-NeMo/Gym@399e6783e0e879424fc20a23f8e2d46ffa4cfdee,
nemo_gym/rollout_observability.py (TrajectoryRecord), rollout_collection.py.
Supported producer paths and required observability configuration:
https://docs.nvidia.com/nemo/gym/reference/trajectory-capabilities/

Only persisted rollouts with ng_trajectory schema 1.0 are accepted. Coverage is
path-dependent; recorded gaps and partial evidence are preserved. Responses-only
exports and ATIF conversion are outside this loader's contract.
No Gym runtime, inference, media fetching, or filesystem reference resolution.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from trace_ingest.loaders.trace_loaders import TraceDescription
from trace_ingest.models import (
    UNSET,
    Span,
    SpanKind,
    TokenCounts,
    ToolCall,
    Trace,
    TraceAggregate,
    TraceSnapshot,
)
from trace_ingest.source_links import file_source_url

__all__ = ["GymTraceConfig", "GymTraceDescription", "GymTraceLoader", "GymTraceLoadError"]


class GymTraceLoadError(ValueError):
    """A Gym export cannot be normalized unambiguously."""


class GymTraceConfig(BaseModel):
    """Select a complete local rollout file, without a trace limit.

    ``path`` resolves from the working directory. ``jsonl`` reads one rollout
    per nonblank line; ``json`` reads a single, optionally pretty-printed object.
    Both formats require the rollout envelope containing ``ng_trajectory``.
    """

    model_config = ConfigDict(extra="forbid")
    path: Path
    format: Literal["jsonl", "json"] = "jsonl"


class GymTraceDescription(TraceDescription):
    """File identity and counts from the cached corpus.

    ``call_count`` counts tool spans; ``distinct_logical_cases`` counts task IDs.
    ``gap_count`` sums source-reported gaps and unresolved ownership/parent gaps
    added by this loader. Zero gaps is not proof of complete producer coverage.
    """

    path: str
    format: str
    gap_count: int


def _object(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("expected an object")
    return value


def _rows(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("expected an array")
    return [_object(row) for row in value]


def _id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("expected a nonempty identifier")
    return value


def _decode(raw: bytes | str) -> object:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def number(value: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("nonfinite JSON number")
        return result

    return json.loads(raw, object_pairs_hook=pairs, parse_float=number, parse_constant=number)


def _timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("invalid Gym timestamp")
    return datetime.fromtimestamp(value, tz=timezone.utc)


def _tokens(raw: object) -> TokenCounts | None:
    """Normalize complete token accounting; leave partial counts in raw metadata."""

    stats = _object(raw)
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens",
        "reasoning_tokens",
        "total_tokens",
    ):
        value = stats.get(key)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("invalid Gym token count")
    prompt, cached, completion = (
        stats.get(key) for key in ("prompt_tokens", "cached_tokens", "completion_tokens")
    )
    if prompt is not None and cached is not None and cached > prompt:
        raise ValueError("cached tokens exceed prompt tokens")
    if prompt is None or cached is None or completion is None:
        return None
    return TokenCounts(
        input_tokens=prompt - cached, cached_input_tokens=cached, output_tokens=completion
    )


def _error(row: dict[str, Any], key: str = "status") -> str | None:
    error = row.get("error_type") or row.get("error_category")
    if error is not None:
        return _id(error)
    code = row.get("status_code")
    if key == "response_status" and type(code) is int and code >= 400:
        return f"HTTP {code}"
    status = row.get(key)
    return status if status in ("failed", "timeout", "cancelled") else None


def _conversation_tools(invocation: dict[str, Any], prefix: str) -> dict[str, Span]:
    """Pair conversation calls/results by invocation-scoped IDs in source order.

    A missing result remains UNSET; an explicit result with null output remains
    a matched result. Observations may enrich these spans later with timing.
    """

    calls: dict[str, Span] = {}
    user: str | None = None
    for index, item in enumerate(_rows(invocation.get("conversation", []))):
        if item.get("type", "message") not in (
            "message",
            "reasoning",
            "function_call",
            "function_call_output",
        ):
            raise GymTraceLoadError("unsupported Gym conversation item; add an explicit mapping")
        if item.get("role") == "user":
            content = item.get("content")
            user = (
                content
                if isinstance(content, str)
                else "\n".join(
                    part["text"]
                    for part in _rows(content)
                    if part.get("type") in ("input_text", "text")
                    and isinstance(part.get("text"), str)
                )
            )
        if item.get("type") == "function_call":
            if item.get("namespace") is not None:
                raise GymTraceLoadError("namespaced Gym functions require an explicit name mapping")
            call_id = _id(item.get("call_id"))
            if call_id in calls:
                raise ValueError("duplicate invocation-scoped tool call ID")
            arguments = item.get("arguments", UNSET)
            if isinstance(arguments, str):
                arguments = _object(_decode(arguments))
            elif arguments is not UNSET and arguments is not None:
                arguments = _object(arguments)
            calls[call_id] = Span(
                id=f"{prefix}/tool/{len(calls)}",
                kind=SpanKind.TOOL,
                tool_name=_id(item.get("name")),
                input=arguments,
                tool_call=ToolCall(
                    call_id=call_id, index=len(calls), result_count=0, prior_user_text=user
                ),
                attributes={"gym": {"call": item, "conversation_index": index}},
            )
        elif item.get("type") == "function_call_output":
            call_id = _id(item.get("call_id"))
            if call_id not in calls:
                raise ValueError("tool result has no preceding invocation-scoped call")
            span = calls[call_id]
            assert span.tool_call is not None
            if span.tool_call.result_count:
                raise ValueError("duplicate tool result")
            span.output = item.get("output", UNSET)
            span.tool_call.result_id = call_id
            span.tool_call.result_count = 1
            span.attributes["gym_result"] = {"item": item, "conversation_index": index}
    return calls


def _model_owners(
    model_calls: list[dict[str, Any]],
    invocations: dict[str, dict[str, Any]],
    gaps: list[dict[str, Any]],
) -> dict[int, str]:
    """Join captured calls using explicit IDs or exact model/response pairs.

    Missing references become gaps; ambiguous matches and multiple owners fail
    instead of inferring ownership from ordering or timestamps.
    """

    owners: dict[int, str] = {}
    for identifier, invocation in invocations.items():
        for ref in _rows(invocation.get("model_calls", [])):
            if not ref.get("model_call_id") and not (
                ref.get("model_ref") is not None and ref.get("response_id")
            ):
                raise ValueError("model reference has no join key")
            matches = []
            for index, call in enumerate(model_calls):
                metadata = _object(call.get("response_metadata", {}))
                if ref.get("model_call_id") is not None:
                    matched = ref["model_call_id"] == call.get("model_call_id")
                else:
                    matched = ref["response_id"] == metadata.get("response_id") and ref[
                        "model_ref"
                    ] == metadata.get("model_ref")
                if matched:
                    matches.append(index)
            if len(matches) > 1:
                raise ValueError("ambiguous model reference")
            if not matches:
                gaps.append(
                    {
                        "code": "model_reference_unresolved",
                        "invocation_id": identifier,
                        "reference": ref,
                    }
                )
                continue
            index = matches[0]
            if index in owners and owners[index] != identifier:
                raise ValueError("model call has multiple invocation owners")
            owners[index] = identifier
    return owners


def _normalize(record: dict[str, Any]) -> Trace:
    """Map one native attachment while retaining the complete source envelope.

    Rollout IDs identify traces; task IDs group logical cases. Invocations,
    captured calls, tools and semantic turns map to separate span kinds.
    Source gaps remain under ``attributes.gym``; additional join gaps are in
    ``gym_loader_gaps``. Rewards stay evaluation signals, not execution errors.
    """

    if "ng_trajectory" not in record:
        raise GymTraceLoadError(
            "Gym rollout requires ng_trajectory schema 1.0 from a supported producer path; "
            "see https://docs.nvidia.com/nemo/gym/reference/trajectory-capabilities/"
        )
    source = _object(record["ng_trajectory"])
    if source.get("schema_version") != "1.0":
        raise GymTraceLoadError("unsupported ng_trajectory schema_version; expected 1.0")
    rollout, task = _id(source.get("rollout_id")), _id(source.get("task_id"))
    gaps = _rows(source.get("gaps", []))
    invocations = _rows(source.get("invocations", []))
    by_id: dict[str, dict[str, Any]] = {}
    spans: dict[str, Span] = {}
    tools: dict[str, dict[str, Span]] = {}
    extra_gaps: list[dict[str, Any]] = []
    for index, invocation in enumerate(invocations):
        identifier = _id(invocation.get("invocation_id"))
        if identifier in by_id:
            raise ValueError("duplicate invocation ID")
        by_id[identifier] = invocation
        span = Span(
            id=f"gym/invocation/{index}",
            kind=SpanKind.AGENT,
            error=_error(invocation),
            input={
                "messages": [
                    item
                    for item in _rows(invocation["conversation"])
                    if item.get("type", "message") == "message"
                ]
            }
            if "conversation" in invocation
            else UNSET,
            attributes={"gym": invocation},
        )
        spans[identifier] = span
        tools[identifier] = _conversation_tools(invocation, span.id)

    roots: list[Span] = []
    for identifier, invocation in by_id.items():
        chain: set[str] = set()
        current: str | None = identifier
        while current in by_id:
            if current in chain:
                raise ValueError("cyclic invocation parents")
            chain.add(current)
            current = by_id[current].get("parent_invocation_id")
        parent = invocation.get("parent_invocation_id")
        if parent is not None and parent not in spans:
            extra_gaps.append({"code": "missing_parent_invocation", "invocation_id": identifier})
        (spans[parent].children if parent in spans else roots).append(spans[identifier])

    model_calls = _rows(source.get("model_calls", []))
    model_ids = [
        _id(row["model_call_id"]) for row in model_calls if row.get("model_call_id") is not None
    ]
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("duplicate model call ID")
    owners = _model_owners(model_calls, by_id, extra_gaps)
    for index, call in enumerate(model_calls):
        metadata = _object(call.get("response_metadata", {}))
        span = Span(
            id=f"gym/model/{index}",
            kind=SpanKind.LLM,
            input=call.get("request", UNSET),
            output=call.get("response", UNSET),
            start_time=_timestamp(call.get("started_at")),
            end_time=_timestamp(call.get("completed_at")),
            token_counts=_tokens(call.get("token_stats", {})),
            model=metadata.get("model"),
            error=_error(metadata, "response_status"),
            attributes={"gym": call},
        )
        (spans[owners[index]].children if index in owners else roots).append(span)
        if index not in owners:
            extra_gaps.append({"code": "model_call_owner_unrecorded", "model_call_index": index})

    observed: set[tuple[str, str]] = set()
    for index, row in enumerate(_rows(source.get("tool_calls", []))):
        owner, call_id = _id(row.get("invocation_id")), _id(row.get("tool_call_id"))
        key = (owner, call_id)
        if key in observed:
            raise ValueError("duplicate tool observation")
        observed.add(key)
        span = tools.get(owner, {}).get(call_id)
        if span is None:
            span = Span(
                id=f"gym/observed-tool/{index}",
                kind=SpanKind.TOOL,
                tool_name=row.get("tool_name"),
                tool_call=ToolCall(call_id=call_id, result_count=0),
            )
            if owner in tools:
                tools[owner][call_id] = span
            else:
                roots.append(span)
                extra_gaps.append({"code": "tool_owner_unrecorded", "invocation_id": owner})
        if span.tool_name and row.get("tool_name") and span.tool_name != row["tool_name"]:
            raise ValueError("conflicting tool names")
        span.start_time, span.end_time = (
            _timestamp(row.get("started_at")),
            _timestamp(row.get("completed_at")),
        )
        if span.start_time and span.end_time and span.end_time < span.start_time:
            raise ValueError("tool ends before it starts")
        span.error = _error(row)
        # Gym serializes output=None even when unavailable. A non-null enriched
        # output establishes a result; null needs an explicit conversation item.
        if span.output is UNSET and "output" in row:
            span.output = row["output"]
            if row["output"] is not None:
                assert span.tool_call is not None
                span.tool_call.result_id = call_id
                span.tool_call.result_count = 1
        span.attributes["gym_observation"] = row
    for identifier, calls in tools.items():
        spans[identifier].children.extend(calls.values())

    seen_turns: set[tuple[str, int]] = set()
    for index, turn in enumerate(_rows(source.get("turns", []))):
        owner, number = _id(turn.get("invocation_id")), turn.get("turn_no")
        if turn.get("task_id") != task or turn.get("rollout_id") != rollout:
            raise ValueError("turn identity differs from trajectory")
        if type(number) is not int or number < 1 or (owner, number) in seen_turns:
            raise ValueError("invalid or duplicate turn number")
        step_count = turn.get("step_count")
        if type(step_count) is not int or step_count < 0 or turn.get("timestamp") is None:
            raise ValueError("turn requires a timestamp and nonnegative step_count")
        seen_turns.add((owner, number))
        # Turns are semantic decisions, not additional LLM calls. Preserve their
        # payloads, but do not allocate model tokens or cost to them a second time.
        span = Span(
            id=f"gym/turn/{index}",
            kind=SpanKind.CHAIN,
            start_time=_timestamp(turn.get("timestamp")),
            input=turn.get("question", UNSET),
            output=turn.get("answer", UNSET),
            attributes={"gym": turn},
        )
        (spans[owner].children if owner in spans else roots).append(span)
        if owner not in spans:
            extra_gaps.append({"code": "turn_owner_unrecorded", "invocation_id": owner})

    def sort_children(items: list[Span]) -> None:
        # Recorded times first; equal or unavailable times retain source order.
        items.sort(
            key=lambda item: (
                item.start_time is None,
                item.start_time or datetime.max.replace(tzinfo=timezone.utc),
            )
        )
        for item in items:
            sort_children(item.children)

    sort_children(roots)
    return Trace(
        id=rollout,
        root_spans=roots,
        aggregate=TraceAggregate(),
        evaluator_results={
            f"gym.{key}": record[key] for key in ("reward", "reward_components") if key in record
        },
        attributes={
            "logical_case_id": task,
            "gym": record,
            "gym_loader_gaps": extra_gaps,
            "gym_gap_count": len(gaps) + len(extra_gaps),
        },
    )


@dataclass
class GymTraceLoader:
    """Normalize native Gym rollouts into a cached, disk-backed snapshot.

    The first ``load()`` or ``describe()`` reads and validates the entire file;
    later calls reuse that snapshot even if the source file changes. Empty
    input, duplicate rollout IDs and ambiguous records raise GymTraceLoadError.
    Missing evidence stays missing; whole-rollout totals are not inferred.

    Example:
        loader = GymTraceLoader(GymTraceConfig(path=Path("rollouts.jsonl")))
        snapshot = loader.load()
        description = loader.describe()
    """

    config: GymTraceConfig

    @cached_property
    def _loaded(self) -> tuple[TraceSnapshot, GymTraceDescription]:
        _line_number = calls = gap_count = 0
        cases: set[str] = set()
        path = self.config.path

        def traces() -> Iterator[Trace]:
            nonlocal _line_number, calls, gap_count
            with path.open("rb") as stream:
                records = [stream.read()] if self.config.format == "json" else stream
                for _line_number, raw in enumerate(records, 1):
                    if not raw.strip():
                        continue
                    trace = _normalize(_object(_decode(raw)))
                    trace.source_url = file_source_url(
                        path, _line_number if self.config.format == "jsonl" else None
                    )
                    pending = list(trace.root_spans)
                    while pending:
                        span = pending.pop()
                        calls += span.kind is SpanKind.TOOL
                        pending.extend(span.children)
                    cases.add(str(trace.attributes["logical_case_id"]))
                    count = trace.attributes["gym_gap_count"]
                    assert isinstance(count, int)
                    gap_count += count
                    yield trace

        try:
            snapshot = TraceSnapshot(traces())
        except GymTraceLoadError:
            raise
        except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError) as error:
            raise GymTraceLoadError(
                f"{path}:{_line_number}: invalid Gym rollout ({type(error).__name__})"
            ) from None
        if not snapshot:
            raise GymTraceLoadError(f"{path}: contains no Gym rollouts")
        return snapshot, GymTraceDescription(
            source=f"gym:{path.resolve()}",
            path=str(path.resolve()),
            format=self.config.format,
            trace_count=len(snapshot),
            call_count=calls,
            distinct_logical_cases=len(cases),
            gap_count=gap_count,
        )

    def load(self) -> TraceSnapshot:
        """Return all normalized rollouts, validating the file on first access."""
        return self._loaded[0]

    def describe(self) -> GymTraceDescription:
        """Return corpus counts and file identity, loading on first access."""
        return self._loaded[1]
