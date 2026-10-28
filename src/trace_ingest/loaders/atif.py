# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load ATIF JSONL into canonical traces.

Source contract: https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md

Each non-blank line is one trajectory with a ``trajectory_id`` or ``session_id``.
``Trace.session_id`` is the ``session_id``, when recorded (``session_basis``
``atif.session_id``).

Accepted input is Harbor's ``Trajectory`` field set for ATIF-v1.0 through v1.8,
validated strictly (no type coercion, unknown keys rejected), plus
``isError``/``is_error`` flags on a result's root. Every field lands in a typed
field or an ``extra`` slot:

* Each trajectory (root ``trajectory``, embedded subagent ``<parent>/subagent/{i}``)
  is an AGENT span with ``agent`` (``AgentInfo``), ``tool_definitions``, ``run``
  (``RunInfo``), and ``turns``.
* Each step is a ``Turn`` with id ``<agent>/turn/{step_id}``.
* An agent step is a span ``<agent>/step/{step_id}``: LLM, or CHAIN when
  ``llm_call_count`` is 0 (the turn's ``dispatch_span_id``). Its output is the
  message; it carries the model, cost, token counts (``prompt_tokens`` is the
  ``input_total``, cached tokens included), and ``llm_call`` (``LLMCallInfo``).
  ``reasoning_effort`` is a request parameter in ``llm_params`` of an LLM span,
  or ``attributes["reasoning_effort"]`` of a CHAIN span, which made no request.
  User and system steps have no span.
* ``final_metrics`` totals are the AGENT span's ``cost_usd`` and ``token_counts``;
  ``total_steps`` and ``extra`` go to ``RunInfo``. The root's totals are also
  the trace aggregate.
* Turn authorship follows the source only: ``copied_context`` when
  ``is_copied_context``, ``model`` for an agent step that records at least one
  model call (``llm_call_count`` >= 1), else ``unknown``.
* Tool calls are TOOL spans ``<step span>/tool/{k}`` under the step span, with
  ``prior_user_text`` from the nearest preceding user step. Results with
  ``source_call_id`` go to that call's ``ToolCall.results`` and the span's
  ``output`` (the content, or a list of contents for several results); the rest
  go to ``Turn.unlinked_results``. A step's ``tool_call_id`` values must be unique.
* A result whose error flags are not one boolean keeps them as recorded in
  ``attributes["is_error_flags"]``, keyed by location (``is_error``,
  ``extra.isError``, ...), beside ``attributes["is_error_status"]``.
* A ``subagent_trajectory_ref`` resolves to an embedded trajectory by
  ``trajectory_id``. The subagent AGENT span is a child of the referencing TOOL
  span (or of the step span, or the AGENT span, for an unlinked result).
  Unreferenced embedded subagents follow the steps. External
  ``trajectory_path`` files are not read.

``Trace.losses`` records what the source did not keep or the loader did not
follow. Each ``Loss.detail`` is structured:

* ``llm_calls_aggregated``: ``{"llm_call_count": n}``.
* ``unresolved_subagent_ref`` (no embedded trajectory matches): ``{"ref": {...}}``,
  the reference object as recorded.
* ``continued_trajectory_not_followed``: ``{"continued_trajectory_ref": value}``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from trace_ingest.loaders.helper.llm_span.token_usage import token_counts as _token_counts
from trace_ingest.loaders.helper.tool_definitions import tool_definitions
from trace_ingest.loaders.trace_loaders import TraceDescription
from trace_ingest.models import (
    UNSET,
    AgentInfo,
    AudioPart,
    Authorship,
    ImagePart,
    Initiator,
    LLMCallInfo,
    Loss,
    RunInfo,
    Span,
    SpanKind,
    SubagentRef,
    TextPart,
    TokenCounts,
    ToolCall,
    ToolDefinition,
    ToolResult,
    Trace,
    TraceAggregate,
    TraceSnapshot,
    Turn,
    TurnRole,
    case_key,
)
from trace_ingest.source_links import file_source_url

__all__ = [
    "ATIFTraceConfig",
    "ATIFTraceDescription",
    "ATIFTraceLoadError",
    "ATIFTraceLoader",
    "error_flags_from_atif",
    "tool_definitions_from_atif",
    "tool_result_from_atif",
]

TOOL_ERROR = "ATIF tool result reported is_error"
_ERROR_FLAGS = ("isError", "is_error")


class ATIFTraceLoadError(ValueError):
    """An ATIF JSONL corpus cannot be normalized unambiguously."""


class ATIFTraceConfig(BaseModel):
    """Settings for an ATIF JSONL file."""

    model_config = ConfigDict(extra="forbid")
    path: Path


class ATIFTraceDescription(TraceDescription):
    path: str


# Source models mirror harbor.models.trajectories (harbor 0.24.0) field for field.
class _Source(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


_Extra = dict[str, JsonValue] | None


class _ImageSource(_Source):
    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
    path: str


class _AudioSource(_Source):
    media_type: Literal[
        "audio/wav",
        "audio/mpeg",
        "audio/mp4",
        "audio/aac",
        "audio/ogg",
        "audio/flac",
        "audio/webm",
        "audio/aiff",
    ]
    path: str
    duration_sec: Annotated[float, Field(ge=0)] | None = None


class _ContentPart(_Source):
    type: Literal["text", "image", "audio"]
    text: str | None = None
    source: _ImageSource | _AudioSource | None = None

    @model_validator(mode="after")
    def _matches_type(self) -> _ContentPart:
        expected = {"text": None, "image": _ImageSource, "audio": _AudioSource}[self.type]
        if (expected is None) != (self.text is not None) or not isinstance(
            self.source, expected or type(None)
        ):
            raise ValueError(
                "a text part needs only 'text'; an image or audio part needs only a matching 'source'"
            )
        return self


_Content = str | list[_ContentPart]


class _Agent(_Source):
    name: str
    version: str
    model_name: str | None = None
    tool_definitions: list[dict[str, JsonValue]] | None = None
    extra: _Extra = None


class _Call(_Source):
    tool_call_id: str
    function_name: str
    arguments: dict[str, JsonValue]
    extra: _Extra = None


class _Reference(_Source):
    trajectory_id: str | None = None
    session_id: str | None = None
    trajectory_path: str | None = None
    extra: _Extra = None

    @model_validator(mode="after")
    def _resolvable(self) -> _Reference:
        if self.trajectory_id is None and self.trajectory_path is None:
            raise ValueError("a subagent reference needs trajectory_id or trajectory_path")
        return self


class _Result(_Source):
    source_call_id: str | None = None
    content: _Content | None = None
    subagent_trajectory_ref: list[_Reference] | None = None
    extra: _Extra = None
    # Not in Harbor's model; tolerated so error flags on the result root are interpreted.
    isError: JsonValue = None
    is_error: JsonValue = None


class _Observation(_Source):
    results: list[_Result]


class _Metrics(_Source):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    cost_usd: float | None = None
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    logprobs: list[float] | None = None
    extra: _Extra = None


class _FinalMetrics(_Source):
    total_prompt_tokens: int | None = None
    total_completion_tokens: int | None = None
    total_cached_tokens: int | None = None
    total_cost_usd: float | None = None
    total_steps: Annotated[int, Field(ge=0)] | None = None
    extra: _Extra = None


class _Step(_Source):
    step_id: Annotated[int, Field(ge=1)]
    timestamp: str | None = None
    source: Literal["system", "user", "agent"]
    model_name: str | None = None
    reasoning_effort: str | float | None = None
    message: _Content
    reasoning_content: str | None = None
    tool_calls: list[_Call] | None = None
    observation: _Observation | None = None
    metrics: _Metrics | None = None
    is_copied_context: bool | None = None
    llm_call_count: Annotated[int, Field(ge=0)] | None = None
    extra: _Extra = None

    @model_validator(mode="after")
    def _source_fields(self) -> _Step:
        if self.timestamp is not None:
            datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        agent_only = (
            "model_name",
            "reasoning_effort",
            "reasoning_content",
            "tool_calls",
            "metrics",
        )
        if self.source != "agent" and any(getattr(self, key) is not None for key in agent_only):
            raise ValueError(f"{', '.join(agent_only)} require source='agent'")
        if (
            self.source == "agent"
            and self.llm_call_count == 0
            and (self.metrics is not None or self.reasoning_content is not None)
        ):
            raise ValueError("llm_call_count=0 cannot have metrics or reasoning_content")
        return self


class _Trajectory(_Source):
    schema_version: (
        Literal[
            "ATIF-v1.0",
            "ATIF-v1.1",
            "ATIF-v1.2",
            "ATIF-v1.3",
            "ATIF-v1.4",
            "ATIF-v1.5",
            "ATIF-v1.6",
            "ATIF-v1.7",
            "ATIF-v1.8",
        ]
        | None
    ) = None
    session_id: str | None = None
    trajectory_id: str | None = None
    agent: _Agent
    steps: Annotated[list[_Step], Field(min_length=1)]
    notes: str | None = None
    final_metrics: _FinalMetrics | None = None
    continued_trajectory_ref: str | None = None
    extra: _Extra = None
    subagent_trajectories: list[_Trajectory] | None = None

    @model_validator(mode="after")
    def _relationships(self) -> _Trajectory:
        for index, step in enumerate(self.steps, start=1):
            if step.step_id != index:
                raise ValueError(f"step_id must be sequential from 1; expected {index}")
            ids = [call.tool_call_id for call in step.tool_calls or []]
            calls = set(ids)
            if len(calls) != len(ids):
                raise ValueError(f"step {step.step_id} contains duplicate tool_call_id values")
            for result in step.observation.results if step.observation else []:
                if result.source_call_id is not None and result.source_call_id not in calls:
                    raise ValueError(
                        f"step {step.step_id}: unknown source_call_id {result.source_call_id!r}"
                    )
        ids = [child.trajectory_id for child in self.subagent_trajectories or []]
        if None in ids or len(set(ids)) != len(ids):
            raise ValueError("embedded subagents require unique trajectory_id values")
        return self


def tool_definitions_from_atif(items: object) -> list[ToolDefinition]:
    """Preserve the ATIF agent's recorded definitions, including unknown entries."""
    return tool_definitions(items) or []


def _error_verdict(flags: dict[str, JsonValue]) -> tuple[bool | None, dict[str, JsonValue]]:
    """Interpret error flags keyed by location; return the flag and result attributes.

    Flags that are not one boolean are kept as recorded, since ``is_error`` cannot hold them.
    """

    values = list(flags.values())
    verdicts = [flag for flag in values if type(flag) is bool]
    if len(verdicts) != len(values):
        return None, {"is_error_status": "invalid", "is_error_flags": flags}
    if len(set(verdicts)) > 1:
        return None, {"is_error_status": "conflicting", "is_error_flags": flags}
    return (verdicts[0] if verdicts else None), {}


def error_flags_from_atif(result: Mapping[str, Any]) -> dict[str, JsonValue]:
    """Error flags on an ATIF observation result and its ``extra``, keyed by location."""
    flags: dict[str, JsonValue] = {key: result[key] for key in _ERROR_FLAGS if key in result}
    extra = result.get("extra")
    if isinstance(extra, Mapping):
        flags |= {f"extra.{key}": extra[key] for key in _ERROR_FLAGS if key in extra}
    return flags


def tool_result_from_atif(result: Mapping[str, Any]) -> ToolResult:
    """Normalize one ATIF observation result, reading ``is_error`` from it and its ``extra``."""
    is_error, attributes = _error_verdict(error_flags_from_atif(result))
    return ToolResult(
        content=result.get("content", UNSET), is_error=is_error, attributes=attributes
    )


def _is_error(result: _Result) -> tuple[bool | None, dict[str, JsonValue]]:
    extra = result.extra or {}
    return _error_verdict(
        {
            **{key: getattr(result, key) for key in _ERROR_FLAGS if key in result.model_fields_set},
            **{f"extra.{key}": extra[key] for key in _ERROR_FLAGS if key in extra},
        }
    )


def _content_json(content: _Content | None) -> JsonValue:
    if content is None or isinstance(content, str):
        return content
    parts = _turn_content(content)
    assert not isinstance(parts, str)
    return [part.model_dump(mode="json") for part in parts]


def _turn_content(content: _Content) -> str | list[TextPart | ImagePart | AudioPart]:
    if isinstance(content, str):
        return content
    parts: list[TextPart | ImagePart | AudioPart] = []
    for part in content:
        if isinstance(part.source, _AudioSource):
            parts.append(
                AudioPart(
                    media_type=part.source.media_type,
                    path=part.source.path,
                    duration_sec=part.source.duration_sec,
                )
            )
        elif part.source is not None:
            parts.append(ImagePart(media_type=part.source.media_type, path=part.source.path))
        else:
            assert part.text is not None
            parts.append(TextPart(text=part.text))
    return parts


def _text(content: _Content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(part.text for part in content if part.text is not None)


def _tokens(prompt: int | None, cached: int | None, completion: int | None) -> TokenCounts | None:
    for value in (prompt, cached, completion):
        if value is not None and value < 0:
            raise ValueError("ATIF token counts must be non-negative integers")
    if prompt is not None and cached is not None and cached > prompt:
        raise ValueError("cached_tokens exceeds prompt_tokens")
    # ATIF prompt_tokens include cached_tokens; each count is read on its own,
    # so a missing one leaves only itself unknown.
    return _token_counts(input_total=prompt, cached_input_tokens=cached, output_tokens=completion)


def _aware(timestamp: str | None) -> datetime | None:
    if timestamp is None:
        return None
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    # Naive timestamps stay on the turn only; a timezone is never invented.
    return parsed if parsed.utcoffset() is not None else None


def _call_scope(extra: _Extra) -> tuple[str | None, dict[str, JsonValue]]:
    """Move ``extra.tool_scope.server`` into a typed server when it is a string."""

    remaining = dict(extra or {})
    scope = remaining.get("tool_scope")
    if not isinstance(scope, dict) or not isinstance(scope.get("server"), str):
        return None, remaining
    rest = {key: value for key, value in scope.items() if key != "server"}
    if rest:
        remaining["tool_scope"] = rest
    else:
        del remaining["tool_scope"]
    server = scope["server"]
    assert isinstance(server, str)
    return server, remaining


@dataclass
class _Subagents:
    """Resolve one trajectory's subagent references to AGENT spans."""

    prefix: str
    source: _Trajectory
    losses: list[Loss]
    attached: set[str] = field(default_factory=set)
    embedded: dict[str | None, tuple[int, _Trajectory]] = field(init=False)

    def __post_init__(self) -> None:
        self.embedded = {
            child.trajectory_id: (index, child)
            for index, child in enumerate(self.source.subagent_trajectories or [])
        }

    def embedded_id(self, index: int) -> str:
        return f"{self.prefix}/subagent/{index}"

    def resolve(
        self, result: _Result, *, owner: str, turn_id: str
    ) -> tuple[list[SubagentRef], list[Span]]:
        refs: list[SubagentRef] = []
        spans: list[Span] = []
        for ref in result.subagent_trajectory_ref or []:
            span_id, span = self._resolve(ref, owner=owner, turn_id=turn_id)
            refs.append(
                SubagentRef(
                    span_id=span_id,
                    trajectory_id=ref.trajectory_id,
                    session_id=ref.session_id,
                    trajectory_path=ref.trajectory_path,
                    extra=ref.extra or {},
                )
            )
            if span is not None:
                spans.append(span)
        return refs, spans

    def _resolve(
        self, ref: _Reference, *, owner: str, turn_id: str
    ) -> tuple[str | None, Span | None]:
        if ref.trajectory_id is not None and ref.trajectory_id in self.embedded:
            index, child = self.embedded[ref.trajectory_id]
            span_id = self.embedded_id(index)
            if span_id in self.attached:
                return span_id, None
            self.attached.add(span_id)
            return span_id, _agent_span(child, span_id, self.losses)
        self.losses.append(
            Loss(
                code="unresolved_subagent_ref",
                detail={"ref": ref.model_dump(mode="json", exclude_none=True)},
                span_id=owner,
                turn_id=turn_id,
            )
        )
        return None, None

    def unattached(self) -> list[Span]:
        return [
            _agent_span(child, self.embedded_id(index), self.losses)
            for index, child in self.embedded.values()
            if self.embedded_id(index) not in self.attached
        ]


def _authorship(step: _Step) -> Authorship:
    """Who wrote the step, only as far as ATIF establishes it."""

    if step.is_copied_context:
        return Authorship.COPIED_CONTEXT
    # Only a recorded model call establishes that a model wrote the step; the
    # source role and a dispatch (llm_call_count=0) do not say who wrote it.
    if step.source == "agent" and step.llm_call_count is not None and step.llm_call_count > 0:
        return Authorship.MODEL
    return Authorship.UNKNOWN


def _step(
    step: _Step,
    *,
    prefix: str,
    agent: _Agent,
    subagents: _Subagents,
    losses: list[Loss],
    prior_user_text: str | None,
) -> tuple[Turn, list[Span]]:
    """Map one step to its turn and the spans it adds under the AGENT span."""

    turn_id = f"{prefix}/turn/{step.step_id}"
    span_id = f"{prefix}/step/{step.step_id}"
    step_span: Span | None = None
    if step.source == "agent":
        dispatch = step.llm_call_count == 0
        metrics = step.metrics or _Metrics()
        token_counts = _tokens(
            metrics.prompt_tokens, metrics.cached_tokens, metrics.completion_tokens
        )
        llm_call = LLMCallInfo(
            call_count=step.llm_call_count,
            prompt_token_ids=metrics.prompt_token_ids,
            completion_token_ids=metrics.completion_token_ids,
            logprobs=metrics.logprobs,
            metrics_extra=metrics.extra or {},
        )
        effort = (
            {} if step.reasoning_effort is None else {"reasoning_effort": step.reasoning_effort}
        )
        step_span = Span(
            id=span_id,
            kind=SpanKind.CHAIN if dispatch else SpanKind.LLM,
            start_time=_aware(step.timestamp),
            model=step.model_name if dispatch else step.model_name or agent.model_name,
            cost_usd=metrics.cost_usd,
            token_counts=token_counts,
            output={"role": "assistant", "content": _content_json(step.message)},
            llm_params=effort if effort and not dispatch else None,
            llm_call=llm_call if dispatch or llm_call != LLMCallInfo() else UNSET,
            # A dispatch made no model request, so its effort is not a request parameter.
            attributes=effort if dispatch else {},
        )
        if step.llm_call_count is not None and step.llm_call_count > 1:
            losses.append(
                Loss(
                    code="llm_calls_aggregated",
                    detail={"llm_call_count": step.llm_call_count},
                    span_id=span_id,
                    turn_id=turn_id,
                )
            )

    calls = step.tool_calls or []
    # _Trajectory guarantees each source_call_id names exactly one call of the step.
    call_index = {call.tool_call_id: index for index, call in enumerate(calls)}
    matched: list[list[_Result]] = [[] for _ in calls]
    unlinked_source: list[_Result] = []
    for result in step.observation.results if step.observation else []:
        if result.source_call_id is None:
            unlinked_source.append(result)
        else:
            matched[call_index[result.source_call_id]].append(result)

    def tool_result(result: _Result, owner: str) -> tuple[ToolResult, list[Span]]:
        is_error, attributes = _is_error(result)
        refs, spans = subagents.resolve(result, owner=owner, turn_id=turn_id)
        return (
            ToolResult(
                content=_content_json(result.content)
                if "content" in result.model_fields_set
                else UNSET,
                is_error=is_error,
                subagent_refs=refs,
                extra=result.extra or {},
                attributes=attributes,
            ),
            spans,
        )

    tool_spans: list[Span] = []
    for index, (call, results) in enumerate(zip(calls, matched, strict=True)):
        tool_id = f"{span_id}/tool/{index}"
        pairs = [tool_result(result, tool_id) for result in results]
        tool_results = [result for result, _ in pairs]
        flags = [result.is_error for result in tool_results]
        attributes: dict[str, JsonValue] = {}
        if True in flags:
            attributes["explicit_error"] = True
        elif flags and all(flag is False for flag in flags):
            attributes["explicit_error"] = False
        server, extra = _call_scope(call.extra)
        tool_spans.append(
            Span(
                id=tool_id,
                kind=SpanKind.TOOL,
                input=call.arguments,
                output=tool_results[0].content
                if len(results) == 1
                else [_content_json(result.content) for result in results] or UNSET,
                tool_name=call.function_name,
                tool_call=ToolCall(
                    call_id=call.tool_call_id,
                    result_id=call.tool_call_id if results else None,
                    result_count=len(results),
                    prior_user_text=prior_user_text,
                    results=tool_results,
                    server=server,
                    initiator=Initiator.HARNESS if step.llm_call_count == 0 else Initiator.MODEL,
                    extra=extra,
                ),
                error=TOOL_ERROR if True in flags else None,
                children=[span for _, spans in pairs for span in spans],
                attributes=attributes,
            )
        )

    unlinked_owner = span_id if step_span is not None else prefix
    unlinked = [tool_result(result, unlinked_owner) for result in unlinked_source]
    unlinked_spans = [span for _, spans in unlinked for span in spans]
    added: list[Span]
    if step_span is not None:
        step_span.children = [*tool_spans, *unlinked_spans]
        added = [step_span]
    else:
        added = unlinked_spans

    turn = Turn(
        id=turn_id,
        index=step.step_id,
        role=TurnRole(step.source),
        authorship=_authorship(step),
        content=_turn_content(step.message),
        reasoning=step.reasoning_content if step.reasoning_content is not None else UNSET,
        timestamp=step.timestamp if step.timestamp is not None else UNSET,
        llm_span_ids=[span_id] if step_span is not None and step_span.kind is SpanKind.LLM else [],
        dispatch_span_id=span_id
        if step_span is not None and step_span.kind is SpanKind.CHAIN
        else None,
        tool_call_span_ids=[span.id for span in tool_spans],
        unlinked_results=[result for result, _ in unlinked],
        extra=step.extra or {},
    )
    return turn, added


def _agent_span(source: _Trajectory, prefix: str, losses: list[Loss]) -> Span:
    subagents = _Subagents(prefix, source, losses)
    if source.continued_trajectory_ref is not None:
        losses.append(
            Loss(
                code="continued_trajectory_not_followed",
                detail={"continued_trajectory_ref": source.continued_trajectory_ref},
                span_id=prefix,
            )
        )
    turns: list[Turn] = []
    children: list[Span] = []
    prior_user_text: str | None = None
    for step in source.steps:
        if step.source == "user":
            prior_user_text = _text(step.message)
        turn, spans = _step(
            step,
            prefix=prefix,
            agent=source.agent,
            subagents=subagents,
            losses=losses,
            prior_user_text=prior_user_text,
        )
        turns.append(turn)
        children.extend(spans)
    # Unreferenced subagents belong to the trajectory, not an invented calling step.
    children.extend(subagents.unattached())
    metrics = source.final_metrics or _FinalMetrics()
    agent = source.agent
    return Span(
        id=prefix,
        kind=SpanKind.AGENT,
        cost_usd=metrics.total_cost_usd,
        token_counts=_tokens(
            metrics.total_prompt_tokens,
            metrics.total_cached_tokens,
            metrics.total_completion_tokens,
        ),
        agent=AgentInfo(
            name=agent.name,
            version=agent.version,
            model_name=agent.model_name,
            extra=agent.extra or {},
        ),
        tool_definitions=tool_definitions_from_atif(agent.tool_definitions)
        if agent.tool_definitions is not None
        else UNSET,
        run=RunInfo(
            schema_version=source.schema_version,
            session_id=source.session_id,
            trajectory_id=source.trajectory_id,
            notes=source.notes,
            continued_trajectory_ref=source.continued_trajectory_ref,
            total_steps=metrics.total_steps,
            final_metrics_extra=metrics.extra or {},
            extra=source.extra or {},
        ),
        turns=turns,
        children=children,
    )


def _normalize(source: _Trajectory) -> Trace:
    trace_id = source.trajectory_id or source.session_id
    if not trace_id:
        raise ValueError("trajectory_id or session_id is required for stable trace identity")
    losses: list[Loss] = []
    root = _agent_span(source, "trajectory", losses)
    return Trace(
        id=trace_id,
        root_spans=[root],
        aggregate=TraceAggregate(cost_usd=root.cost_usd, token_counts=root.token_counts),
        session_id=source.session_id,
        session_basis=None if source.session_id is None else "atif.session_id",
        losses=losses,
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
        sessions: set[str] = set()

        def traces() -> Iterator[Trace]:
            nonlocal _line_number, call_count
            with path.open("rb") as corpus:
                for _line_number, raw in enumerate(corpus, 1):
                    if not raw.strip():
                        continue
                    trace = _normalize(_Trajectory.model_validate_json(raw))
                    trace.source_url = file_source_url(path, _line_number)
                    pending = list(trace.root_spans)
                    while pending:
                        span = pending.pop()
                        call_count += span.kind is SpanKind.TOOL
                        pending.extend(span.children)
                    cases.add(case_key(trace))
                    if trace.session_id:
                        sessions.add(trace.session_id)
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
            distinct_cases=len(cases),
            distinct_sessions=len(sessions),
        )

    def load(self) -> TraceSnapshot:
        return self._loaded[0]

    def describe(self) -> ATIFTraceDescription:
        return self._loaded[1]
