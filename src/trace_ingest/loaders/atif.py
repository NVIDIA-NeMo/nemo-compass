# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load ATIF JSONL into canonical traces.

Source contract: https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, with_config

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

__all__ = ["ATIFTraceConfig", "ATIFTraceDescription", "ATIFTraceLoadError", "ATIFTraceLoader"]


class ATIFTraceLoadError(ValueError):
    """An ATIF JSONL corpus cannot be normalized unambiguously."""


class ATIFTraceConfig(BaseModel):
    """Settings for an ATIF JSONL file."""

    model_config = ConfigDict(extra="forbid")
    path: Path


class ATIFTraceDescription(TraceDescription):
    path: str


# Validate source dictionaries without filling absent fields or discarding extensions.
@with_config(ConfigDict(extra="allow", strict=True))
class _Source(TypedDict):
    pass


_Identifier = Annotated[str, Field(min_length=1)]


class _Agent(_Source):
    name: str
    version: str
    model_name: NotRequired[str | None]


class _Call(_Source):
    tool_call_id: _Identifier
    function_name: _Identifier
    arguments: NotRequired[dict[str, Any] | None]


class _Reference(_Source, total=False):
    trajectory_id: str | None
    trajectory_path: str | None
    session_id: str | None


class _Result(_Source, total=False):
    source_call_id: str | None
    content: Any
    subagent_trajectory_ref: list[_Reference] | None


class _Observation(_Source):
    results: list[_Result]


class _Step(_Source):
    step_id: int
    source: Literal["system", "user", "agent"]
    message: str | list[dict[str, Any]]
    timestamp: NotRequired[str | None]
    model_name: NotRequired[str | None]
    llm_call_count: NotRequired[Annotated[int, Field(ge=0)] | None]
    tool_calls: NotRequired[list[_Call] | None]
    observation: NotRequired[_Observation | None]
    metrics: NotRequired[dict[str, Any] | None]


class _Trajectory(_Source):
    schema_version: Annotated[str, Field(pattern=r"^ATIF-v1\.[0-8]$")]
    trajectory_id: NotRequired[_Identifier | None]
    session_id: NotRequired[_Identifier | None]
    agent: _Agent
    steps: list[_Step]
    final_metrics: NotRequired[dict[str, Any] | None]
    subagent_trajectories: NotRequired[list[_Trajectory] | None]
    continued_trajectory_ref: NotRequired[str | None]


_ATIF = TypeAdapter(_Trajectory)


def _metadata(source: Mapping[str, Any], *exclude: str) -> dict[str, Any]:
    return {key: value for key, value in source.items() if key not in exclude}


def _tokens(metrics: dict[str, Any], prefix: str = "") -> TokenCounts | None:
    prompt, cached, completion = (
        metrics.get(prefix + key) for key in ("prompt_tokens", "cached_tokens", "completion_tokens")
    )
    for value in (prompt, cached, completion):
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("ATIF token counts must be non-negative integers")
    if prompt is not None and cached is not None and cached > prompt:
        raise ValueError("cached_tokens exceeds prompt_tokens")
    # Partial accounting remains in metadata; unknown counts are not zero.
    if prompt is None or cached is None or completion is None:
        return None
    return TokenCounts(
        input_tokens=prompt - cached, cached_input_tokens=cached, output_tokens=completion
    )


def _cost(value: object) -> float | None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
        raise ValueError("ATIF cost must be a number")
    return value


def _step_span(
    step: _Step,
    *,
    span_id: str,
    model: str | None,
    prior_user_text: str | None,
    subagents: Callable[[_Result], list[Span]],
) -> Span:
    is_agent = step["source"] == "agent"
    if not is_agent and any(
        step.get(key) is not None
        for key in ("model_name", "tool_calls", "metrics", "reasoning_content", "reasoning_effort")
    ):
        raise ValueError("model, tool calls, reasoning, and metrics require source='agent'")
    is_llm = is_agent and step.get("llm_call_count") != 0
    if (
        is_agent
        and not is_llm
        and any(step.get(key) is not None for key in ("metrics", "reasoning_content"))
    ):
        raise ValueError("llm_call_count=0 cannot have metrics or reasoning_content")
    raw_timestamp = step.get("timestamp")
    timestamp = (
        datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
        if raw_timestamp is not None
        else None
    )
    metrics = step.get("metrics") or {}
    span = Span(
        id=span_id,
        kind=SpanKind.LLM if is_llm else SpanKind.CHAIN,
        # Keep naive timestamps in metadata rather than inventing a timezone.
        start_time=timestamp if timestamp and timestamp.utcoffset() is not None else None,
        input=UNSET
        if is_agent
        else {"messages": [{"role": step["source"], "content": step["message"]}]},
        output={"role": "assistant", "content": step["message"]} if is_agent else UNSET,
        model=(step.get("model_name") or model) if is_llm else None,
        cost_usd=_cost(metrics.get("cost_usd")),
        token_counts=_tokens(metrics),
        attributes={"atif": _metadata(step, "message", "tool_calls", "observation")},
    )
    calls = step.get("tool_calls") or []
    matched: dict[str | None, list[_Result]] = {call["tool_call_id"]: [] for call in calls}
    if len(matched) != len(calls):
        raise ValueError(f"step {step['step_id']} contains duplicate tool_call_id values")
    matched[None] = []  # Observations not associated with a tool call.
    for result in (step.get("observation") or {}).get("results", []):
        call_id = result.get("source_call_id")
        if call_id not in matched:
            raise ValueError(f"step {step['step_id']}: unknown source_call_id {call_id!r}")
        matched[call_id].append(result)
    for index, call in enumerate(calls):
        results = matched[call["tool_call_id"]]
        output = results[0].get("content", UNSET) if len(results) == 1 else results or UNSET
        span.children.append(
            Span(
                id=f"{span_id}/tool/{index}",
                kind=SpanKind.TOOL,
                input=call.get("arguments", UNSET),
                output=output,
                tool_name=call["function_name"],
                tool_call=ToolCall(
                    call_id=call["tool_call_id"],
                    result_id=call["tool_call_id"] if results else None,
                    result_count=len(results),
                    prior_user_text=prior_user_text,
                ),
                children=[child for result in results for child in subagents(result)],
                attributes={"atif": {"call": call, "results": results}},
            )
        )
    for index, result in enumerate(matched[None]):
        span.children.append(
            Span(
                id=f"{span_id}/observation/{index}",
                kind=SpanKind.CHAIN,
                output=result.get("content", UNSET),
                children=subagents(result),
                attributes={"atif": result},
            )
        )
    return span


def _trajectory_span(source: _Trajectory, prefix: str = "trajectory") -> Span:
    if not (source.get("trajectory_id") or source.get("session_id")):
        raise ValueError("trajectory_id or session_id is required for stable trace identity")
    if source.get("continued_trajectory_ref"):
        raise ValueError(
            "external continuation is not supported; combine the complete trajectory first"
        )
    embedded: dict[str, tuple[int, _Trajectory]] = {}
    for index, child in enumerate(source.get("subagent_trajectories") or []):
        child_id = child.get("trajectory_id")
        if child_id is None or child_id in embedded:
            raise ValueError("embedded subagents require unique trajectory_id values")
        embedded[child_id] = (index, child)
    attached: set[str] = set()

    def subagents(result: _Result) -> list[Span]:
        children = []
        for ref in result.get("subagent_trajectory_ref") or []:
            child_id = ref.get("trajectory_id")
            if child_id is None or child_id not in embedded:
                raise ValueError(
                    f"unresolved subagent reference {ref!r}; embed it in subagent_trajectories with a matching trajectory_id"
                )
            if child_id in attached:
                raise ValueError(f"subagent {child_id!r} has multiple parents")
            attached.add(child_id)
            index, child = embedded[child_id]
            children.append(_trajectory_span(child, f"{prefix}/subagent/{index}"))
        return children

    root = Span(
        id=prefix,
        kind=SpanKind.AGENT,
        attributes={"atif": _metadata(source, "steps", "subagent_trajectories")},
    )
    prior_user_text = None
    for index, step in enumerate(source["steps"], start=1):
        if step["step_id"] != index:
            raise ValueError(f"step_id must be sequential from 1; expected {index}")
        if step["source"] == "user":
            message = step["message"]
            prior_user_text = (
                message
                if isinstance(message, str)
                else "\n".join(
                    part["text"]
                    for part in message
                    if part.get("type") == "text" and isinstance(part.get("text"), str)
                )
            )
        root.children.append(
            _step_span(
                step,
                span_id=f"{prefix}/step/{index}",
                model=source["agent"].get("model_name"),
                prior_user_text=prior_user_text,
                subagents=subagents,
            )
        )
    # Unreferenced subagents belong to the trajectory, not an invented calling step.
    root.children.extend(
        _trajectory_span(child, f"{prefix}/subagent/{index}")
        for child_id, (index, child) in embedded.items()
        if child_id not in attached
    )
    return root


def _normalize(source: _Trajectory) -> Trace:
    root = _trajectory_span(source)
    metrics = source.get("final_metrics") or {}
    attributes = dict(root.attributes)
    if source.get("session_id") is not None:
        attributes["logical_case_id"] = source["session_id"]
    trace_id = source.get("trajectory_id") or source.get("session_id")
    assert trace_id is not None
    return Trace(
        id=trace_id,
        root_spans=[root],
        aggregate=TraceAggregate(
            cost_usd=_cost(metrics.get("total_cost_usd")), token_counts=_tokens(metrics, "total_")
        ),
        attributes=attributes,
    )


@dataclass
class ATIFTraceLoader:
    """Stream complete ATIF trajectories into a disk-backed canonical snapshot."""

    config: ATIFTraceConfig

    @cached_property
    def _loaded(self) -> tuple[TraceSnapshot, ATIFTraceDescription]:
        path = self.config.path
        _line_number = call_count = 0
        cases: set[str] = set()

        def traces() -> Iterator[Trace]:
            nonlocal _line_number, call_count
            with path.open("rb") as corpus:
                for _line_number, raw in enumerate(corpus, 1):
                    if not raw.strip():
                        continue
                    trace = _normalize(_ATIF.validate_json(raw))
                    trace.source_url = file_source_url(path, _line_number)
                    pending = list(trace.root_spans)
                    while pending:
                        span = pending.pop()
                        call_count += span.kind is SpanKind.TOOL
                        pending.extend(span.children)
                    cases.add(str(trace.attributes.get("logical_case_id") or trace.id))
                    yield trace

        try:
            snapshot = TraceSnapshot(traces())
        except (OSError, ValueError) as error:
            raise ATIFTraceLoadError(f"{path}:{_line_number}: invalid ATIF: {error}") from error
        if not snapshot:
            raise ATIFTraceLoadError(f"{path}: contains no ATIF trajectories")
        return snapshot, ATIFTraceDescription(
            source=f"atif:{path.resolve()}",
            path=str(path.resolve()),
            trace_count=len(snapshot),
            call_count=call_count,
            distinct_logical_cases=len(cases),
        )

    def load(self) -> TraceSnapshot:
        return self._loaded[0]

    def describe(self) -> ATIFTraceDescription:
        return self._loaded[1]
