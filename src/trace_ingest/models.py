# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalized trace contracts shared by Compass Signals and insight compilation."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import closing
from datetime import datetime
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator
from pydantic.experimental.missing_sentinel import MISSING

from trace_ingest.source_links import SourceURL

# Pydantic's missing sentinel is omitted by ``model_dump`` and JSON Schema, while
# an explicitly supplied ``None`` remains JSON null. Tool-result analysis depends
# on this distinction.
UNSET = MISSING

TRACE_SCHEMA_VERSION = "2"


class _TraceModel(BaseModel):
    """Base configuration for forward-compatible canonical trace models."""

    model_config = ConfigDict(extra="allow")
    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)


class SpanKind(str, Enum):
    LLM = "LLM"
    TOOL = "TOOL"
    AGENT = "AGENT"
    CHAIN = "CHAIN"
    RETRIEVER = "RETRIEVER"
    EMBEDDING = "EMBEDDING"
    RERANKER = "RERANKER"
    EVALUATOR = "EVALUATOR"
    GUARDRAIL = "GUARDRAIL"
    UNKNOWN = "UNKNOWN"


class TurnRole(str, Enum):
    """Who a conversation turn is attributed to in the source."""

    USER = "user"
    SYSTEM = "system"
    AGENT = "agent"


class Authorship(str, Enum):
    """Who actually wrote a turn's content, when the source says so."""

    HUMAN = "human"
    HARNESS = "harness"
    COPIED_CONTEXT = "copied_context"
    MODEL = "model"
    UNKNOWN = "unknown"


class Initiator(str, Enum):
    """What issued a tool call."""

    MODEL = "model"
    HARNESS = "harness"
    USER = "user"
    ENVIRONMENT = "environment"


class ToolCall(_TraceModel):
    """Source metadata that identifies and joins one tool invocation."""

    call_id: str | None = Field(
        default=None,
        description=(
            "The source's tool-call identifier. Preserve repeated IDs so duplicate "
            "instrumentation remains observable."
        ),
    )
    index: int | None = Field(
        default=None,
        ge=0,
        description="Zero-based tool-call execution order when the source records it.",
    )
    result_id: str | None = Field(
        default=None,
        description="The source result's genuine reference to this call, when available.",
    )
    result_count: int = Field(
        default=1,
        ge=0,
        description="Number of source results matched to this call; use zero when none exists.",
    )
    instrumentation_alias_of: str | None = Field(
        default=None,
        description="Canonical tool name when the observed name is an instrumentation alias.",
    )
    prior_user_text: str | None = Field(
        default=None,
        description="Nearest preceding user text, only when captured from the source.",
    )
    returned_data: bool | UNSET = Field(
        default=UNSET,
        description="Whether the result returned usable data; omit when the source cannot say.",
    )
    results: list[ToolResult] | UNSET = Field(
        default=UNSET,
        description=(
            "Source results matched to this call, in source order; its length equals "
            "result_count. Omit when the results were not recorded, as in traces written "
            "before schema version 2."
        ),
    )
    server: str | None = Field(
        default=None,
        description="Explicit tool scope such as an MCP server; never inferred from the name.",
    )
    initiator: Initiator | None = Field(
        default=None, description="What issued this call; null when the source does not say."
    )
    extra: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Source extension fields of the call, minus values moved into typed fields.",
    )

    @model_validator(mode="after")
    def results_match_count(self) -> ToolCall:
        if self.results is not UNSET and len(self.results) != self.result_count:
            raise ValueError("result_count must equal len(results)")
        return self


class ToolResult(_TraceModel):
    """One source result returned for a tool invocation."""

    content: JsonValue | UNSET = Field(
        default=UNSET,
        description="Raw recorded result content. Omit when absent; explicit JSON null remains null.",
    )
    is_error: bool | None = Field(
        default=None,
        description="Source-recorded error flag; null when the source did not record a usable flag.",
    )
    subagent_refs: list[SubagentRef] = Field(
        default_factory=list,
        description="Subagent runs this result delegated to, in source order.",
    )
    extra: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Source extension fields of the result, minus flags that were interpreted.",
    )
    attributes: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Additional lossless result metadata not represented by a typed field.",
    )


class SubagentRef(_TraceModel):
    """A tool result's reference to a delegated subagent run."""

    span_id: str | None = Field(
        default=None,
        description="The AGENT span holding the subagent run; null when it could not be resolved.",
    )
    trajectory_id: str | None = Field(default=None, description="Referenced run identifier.")
    session_id: str | None = Field(
        default=None,
        description="Informational session identifier.",
    )
    trajectory_path: str | None = Field(
        default=None,
        description="Referenced external file or URL, verbatim.",
    )
    extra: dict[str, JsonValue] = Field(
        default_factory=dict, description="Source extension fields of the reference."
    )


class ToolDefinition(_TraceModel):
    """One tool the source declared as available to an agent."""

    name: str = Field(min_length=1, description="Declared tool name.")
    server: str | None = Field(
        default=None,
        description="Explicit source scope such as an MCP server; never inferred from the name.",
    )
    description: str | None = Field(default=None, description="Declared tool description.")
    input_schema: JsonValue | None = Field(
        default=None, description="Declared input JSON Schema, when recorded."
    )
    raw: JsonValue = Field(description="Verbatim source definition.")


class TokenCounts(_TraceModel):
    """Non-negative token counts using the source's authoritative accounting."""

    input_tokens: int = Field(ge=0, description="Uncached input tokens.")
    cached_input_tokens: int = Field(ge=0, description="Cached input tokens.")
    output_tokens: int = Field(ge=0, description="Generated output tokens.")


class TextPart(_TraceModel):
    """A text part of a multimodal message."""

    type: Literal["text"] = "text"
    text: str


class ImagePart(_TraceModel):
    """An image part of a multimodal message, referenced by location."""

    type: Literal["image"] = "image"
    media_type: str = Field(description="Image MIME type, such as image/png.")
    path: str = Field(description="Relative or absolute file path, or a URL, verbatim.")


class AudioPart(_TraceModel):
    """An audio part of a multimodal message, referenced by location."""

    type: Literal["audio"] = "audio"
    media_type: str = Field(description="Audio MIME type, such as audio/wav.")
    path: str = Field(description="Relative or absolute file path, or a URL, verbatim.")
    duration_sec: float | None = Field(default=None, ge=0, description="Duration, when recorded.")


ContentPart = Annotated[TextPart | ImagePart | AudioPart, Field(discriminator="type")]


class Turn(_TraceModel):
    """One message of an agent's conversation, as recorded by the source.

    Turns are a view over spans: per-call detail such as model, tokens, and cost
    stays on the LLM spans the turn references. Turn order is defined only within
    one AGENT span, not across agents.
    """

    id: str = Field(
        min_length=1,
        description='Stable identifier, "<agent span id>/turn/<index>" for source loaders.',
    )
    index: int = Field(ge=1, description="Position within the agent span, contiguous.")
    role: TurnRole
    authorship: Authorship = Authorship.UNKNOWN
    content: str | list[ContentPart] | UNSET = Field(
        default=UNSET, description="Message as recorded: a string or ordered parts."
    )
    reasoning: str | UNSET = Field(default=UNSET, description="Recorded reasoning text.")
    timestamp: str | UNSET = Field(
        default=UNSET,
        description="ISO 8601 timestamp exactly as recorded; it may lack a timezone.",
    )
    llm_span_ids: list[str] = Field(
        default_factory=list,
        description="LLM spans that produced this turn, in order.",
    )
    dispatch_span_id: str | None = Field(
        default=None,
        description="CHAIN span of an agent step that made no model call, if any.",
    )
    tool_call_span_ids: list[str] = Field(
        default_factory=list,
        description="TOOL spans issued in this turn, in call order.",
    )
    unlinked_results: list[ToolResult] = Field(
        default_factory=list,
        description="Results observed in this turn that name no tool call, in source order.",
    )
    extra: dict[str, JsonValue] = Field(
        default_factory=dict, description="Source extension fields of the turn, verbatim."
    )


class AgentInfo(_TraceModel):
    """Identity of the agent that an AGENT span ran."""

    name: str
    version: str | None = None
    model_name: str | None = Field(default=None, description="Default model of the agent.")
    extra: dict[str, JsonValue] = Field(
        default_factory=dict, description="Source extension fields of the agent, verbatim."
    )


class RunTotals(_TraceModel):
    """Source-reported totals for one agent run, as recorded (any may be missing)."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    cost_usd: float | None = None
    steps: int | None = Field(default=None, ge=0)
    extra: dict[str, JsonValue] = Field(default_factory=dict)


class RunInfo(_TraceModel):
    """Source run metadata recorded for one AGENT span."""

    schema_version: str | None = Field(
        default=None, description="Source format version, such as ATIF-v1.7."
    )
    session_id: str | None = Field(
        default=None,
        description="Source session identifier of the run.",
    )
    trajectory_id: str | None = Field(
        default=None,
        description="Source identifier of the run's trajectory.",
    )
    notes: str | None = None
    continued_trajectory_ref: str | None = Field(
        default=None,
        description="Reference to the run's continuation, verbatim; not followed.",
    )
    final_metrics: RunTotals | None = None
    extra: dict[str, JsonValue] = Field(
        default_factory=dict, description="Source extension fields of the run, verbatim."
    )


class LLMCallInfo(_TraceModel):
    """Per-call model detail beyond the Span's model, token_counts, and cost_usd."""

    call_count: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Model calls the source recorded for this span; >1 means calls were merged, "
            "0 a dispatch without a model call; null when unrecorded."
        ),
    )
    reasoning_effort: str | float | None = None
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    logprobs: list[float] | None = None
    prompt_tokens: int | None = Field(
        default=None, description="Recorded only when token_counts cannot be formed."
    )
    cached_tokens: int | None = Field(
        default=None, description="Recorded only when token_counts cannot be formed."
    )
    completion_tokens: int | None = Field(
        default=None, description="Recorded only when token_counts cannot be formed."
    )
    metrics_extra: dict[str, JsonValue] = Field(
        default_factory=dict, description="Source extension fields of the call metrics."
    )


class Loss(_TraceModel):
    """Information the source lacked or the loader could not represent."""

    code: str = Field(min_length=1)
    detail: dict[str, JsonValue] = Field(
        default_factory=dict,
        description=(
            "Structured loss data, never an encoded string. Keys by code: "
            "unresolved_subagent_ref has ref (the reference as written); "
            "continued_trajectory_not_followed has continued_trajectory_ref; "
            "duplicate_tool_call_id has tool_call_id; tool_definition_skipped has "
            "definition (the raw item); llm_calls_aggregated has llm_call_count."
        ),
    )
    span_id: str | None = Field(default=None, description="Span the loss concerns.")
    turn_id: str | None = Field(default=None, description="Turn the loss concerns.")


class Span(_TraceModel):
    """One canonical, recursively nested unit of work in an agent trace."""

    id: str = Field(
        min_length=1,
        description="Stable span identifier, unique within its containing trace.",
    )
    source_url: SourceURL | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Link to this span in its source, when the loader can resolve one.",
    )
    kind: SpanKind = Field(description="Normalized span category represented by this source span.")
    children: list[Span] = Field(
        default_factory=list,
        description="Direct child spans in deterministic source execution order.",
    )

    start_time: datetime | None = Field(
        default=None,
        description="Timezone-aware source start time, when recorded.",
    )
    end_time: datetime | None = Field(
        default=None,
        description="Timezone-aware source end time, when recorded.",
    )
    # UNSET means the value was not recorded; None means it was explicitly
    # recorded as JSON null. Compass Signals use this to detect missing results.
    input: JsonValue | UNSET = Field(
        default=UNSET,
        description="Raw recorded input. Omit when absent; explicit JSON null remains null.",
    )
    output: JsonValue | UNSET = Field(
        default=UNSET,
        description="Raw recorded output. Omit when absent; explicit JSON null remains null.",
    )
    cost_usd: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
        description="Source-reported span cost in US dollars, when available.",
    )
    token_counts: TokenCounts | None = Field(
        default=None,
        description="Source-reported token counts for this span, when available.",
    )
    model: str | None = Field(
        default=None,
        description="Source-reported model identifier, when applicable.",
    )

    tool_name: str | None = Field(
        default=None,
        description="Observed tool name for a TOOL span; preserve the source value.",
    )
    tool_call: ToolCall | None = Field(
        default=None,
        description="Call/result join metadata; valid only for a TOOL span.",
    )

    error: str | None = Field(
        default=None,
        description="Authoritative source error text or status; do not infer one from output text.",
    )
    attributes: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Additional lossless source metadata not represented by a typed field.",
    )
    tool_definitions: list[ToolDefinition] | UNSET = Field(
        default=UNSET,
        description=(
            "Tools declared to this agent, or offered to this model call; valid only for an "
            "AGENT or LLM span."
        ),
    )
    agent: AgentInfo | UNSET = Field(
        default=UNSET, description="Agent identity; valid only for an AGENT span."
    )
    run: RunInfo | UNSET = Field(
        default=UNSET, description="Source run metadata; valid only for an AGENT span."
    )
    turns: list[Turn] | UNSET = Field(
        default=UNSET,
        description=(
            "The agent's conversation in order; valid only for an AGENT span. Omitted when "
            "the source did not record one. The order is within this agent only, not "
            "relative to turns of other AGENT spans."
        ),
    )
    llm_call: LLMCallInfo | UNSET = Field(
        default=UNSET,
        description=(
            "Per-call model detail; valid for an LLM span, or a CHAIN span whose call_count is 0."
        ),
    )

    @field_validator("start_time", "end_time")
    @classmethod
    def timestamps_include_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("timestamp must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_span(self) -> Span:
        if (
            self.start_time is not None
            and self.end_time is not None
            and self.end_time < self.start_time
        ):
            raise ValueError(f"span {self.id!r} ends before it starts")
        if self.tool_call is not None and self.kind is not SpanKind.TOOL:
            raise ValueError("tool_call metadata requires kind=TOOL")
        if self.tool_definitions is not UNSET and self.kind not in _DECLARING_KINDS:
            raise ValueError("tool_definitions require kind=AGENT or kind=LLM")
        if self.kind is not SpanKind.AGENT and any(
            value is not UNSET for value in (self.agent, self.run, self.turns)
        ):
            raise ValueError("agent, run, and turns require kind=AGENT")
        if self.llm_call is not UNSET and not (
            self.kind is SpanKind.LLM
            or (self.kind is SpanKind.CHAIN and self.llm_call.call_count == 0)
        ):
            raise ValueError("llm_call requires kind=LLM, or kind=CHAIN with call_count=0")
        if self.turns is not UNSET:
            indexes = [turn.index for turn in self.turns]
            if indexes and indexes != list(range(indexes[0], indexes[0] + len(indexes))):
                raise ValueError(f"span {self.id!r} turn indexes must be contiguous")
        return self


_DECLARING_KINDS = frozenset({SpanKind.AGENT, SpanKind.LLM})


class TraceAggregate(_TraceModel):
    """Source-reported or deterministically derived whole-trace measurements."""

    cost_usd: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
        description="Whole-trace cost in US dollars, when available.",
    )
    latency_ms: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
        description="Whole-trace latency in milliseconds, when available.",
    )
    token_counts: TokenCounts | None = Field(
        default=None,
        description="Whole-trace token counts, when available.",
    )


class Trace(_TraceModel):
    """Canonical representation of one complete end-to-end agent run.

    Source loaders deterministically construct this model before any Compass
    Signal runs. Provider-specific parsing or compatibility logic does not
    belong downstream of this boundary.
    """

    id: str = Field(
        min_length=1,
        description="Stable trace identifier, unique within the loaded corpus.",
    )
    schema_version: Literal["2"] = Field(
        default=TRACE_SCHEMA_VERSION,
        description="Canonical trace contract version.",
    )
    source_url: SourceURL | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Loader-resolved link to the original trace or local source file.",
    )
    root_spans: list[Span] = Field(
        description="Complete root span trees in deterministic source execution order."
    )
    aggregate: TraceAggregate = Field(
        description="Whole-trace measurements; use an empty object when none are available."
    )
    evaluator_results: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Normalized evaluation signals keyed by adapter-assigned names.",
    )
    attributes: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Additional lossless trace-level source metadata; do not invent values.",
    )
    losses: list[Loss] = Field(
        default_factory=list,
        description="Information the source lacked or the loader could not represent.",
    )

    @model_validator(mode="after")
    def validate_references(self) -> Trace:
        # span id -> (span, nearest AGENT ancestor id)
        spans: dict[str, tuple[Span, str | None]] = {}
        pending: list[tuple[Span, str | None]] = [
            (span, None) for span in reversed(self.root_spans)
        ]
        while pending:
            span, owner = pending.pop()
            if span.id in spans:
                raise ValueError(f"trace {self.id!r} contains duplicate span id {span.id!r}")
            spans[span.id] = (span, owner)
            inner = span.id if span.kind is SpanKind.AGENT else owner
            pending.extend((child, inner) for child in reversed(span.children))

        turn_ids: set[str] = set()
        claimed: dict[str, str] = {}
        refs: list[SubagentRef] = []
        for span, _ in spans.values():
            if span.tool_call is not None and span.tool_call.results is not UNSET:
                refs.extend(
                    ref for result in span.tool_call.results for ref in result.subagent_refs
                )
            if span.turns is UNSET:
                continue
            for turn in span.turns:
                if turn.id in turn_ids:
                    raise ValueError(f"trace {self.id!r} contains duplicate turn id {turn.id!r}")
                turn_ids.add(turn.id)
                refs.extend(ref for result in turn.unlinked_results for ref in result.subagent_refs)
                for kind, ids in (
                    (SpanKind.LLM, turn.llm_span_ids),
                    (SpanKind.TOOL, turn.tool_call_span_ids),
                    (SpanKind.CHAIN, [turn.dispatch_span_id] if turn.dispatch_span_id else []),
                ):
                    for span_id in ids:
                        target, owner = spans.get(span_id, (None, None))
                        if target is None or target.kind is not kind or owner != span.id:
                            raise ValueError(
                                f"turn {turn.id!r} must reference a {kind.value} span under "
                                f"agent {span.id!r}; got {span_id!r}"
                            )
                for span_id in turn.tool_call_span_ids:
                    if claimed.setdefault(span_id, turn.id) != turn.id:
                        raise ValueError(f"TOOL span {span_id!r} belongs to more than one turn")
        for ref in refs:
            if ref.span_id is not None and (
                ref.span_id not in spans or spans[ref.span_id][0].kind is not SpanKind.AGENT
            ):
                raise ValueError(f"subagent reference {ref.span_id!r} is not an AGENT span")
        for loss in self.losses:
            if loss.span_id is not None and loss.span_id not in spans:
                raise ValueError(f"loss references unknown span {loss.span_id!r}")
            if loss.turn_id is not None and loss.turn_id not in turn_ids:
                raise ValueError(f"loss references unknown turn {loss.turn_id!r}")
        return self


class _DiskTraceIndex(Mapping[str, Trace]):
    """Temporary SQLite storage for normalized traces."""

    def __init__(self, traces: Iterable[Trace]) -> None:
        self._directory = TemporaryDirectory(prefix="compass-traces-")
        self._path = Path(self._directory.name) / "traces.sqlite3"
        try:
            with closing(sqlite3.connect(self._path)) as connection, connection:
                connection.execute(
                    "CREATE TABLE traces (id TEXT PRIMARY KEY, position INTEGER, payload TEXT)"
                )
                connection.execute("CREATE INDEX trace_order ON traces(position)")
                for position, trace in enumerate(traces):
                    try:
                        connection.execute(
                            "INSERT INTO traces VALUES (?, ?, ?)",
                            (trace.id, position, trace.model_dump_json()),
                        )
                    except sqlite3.IntegrityError as error:
                        raise ValueError(
                            f"snapshot contains duplicate trace id {trace.id!r}"
                        ) from error
        except BaseException:
            self._directory.cleanup()
            raise

    def __len__(self) -> int:
        with closing(sqlite3.connect(self._path)) as connection:
            return connection.execute("SELECT COUNT(*) FROM traces").fetchone()[0]

    def __iter__(self) -> Iterator[str]:
        with closing(sqlite3.connect(self._path)) as connection:
            for (trace_id,) in connection.execute("SELECT id FROM traces ORDER BY position"):
                yield trace_id

    def __getitem__(self, trace_id: str) -> Trace:
        with closing(sqlite3.connect(self._path)) as connection:
            row = connection.execute(
                "SELECT payload FROM traces WHERE id = ?", (trace_id,)
            ).fetchone()
            if row is None:
                raise KeyError(trace_id)
            return Trace.model_validate_json(row[0])

    def iter_traces(self) -> Iterator[Trace]:
        with closing(sqlite3.connect(self._path)) as connection:
            for (payload,) in connection.execute("SELECT payload FROM traces ORDER BY position"):
                yield Trace.model_validate_json(payload)

    def sort(self, key: Callable[[str], Any]) -> None:
        trace_ids = sorted(self, key=key)
        with closing(sqlite3.connect(self._path)) as connection, connection:
            connection.executemany(
                "UPDATE traces SET position = ? WHERE id = ?", enumerate(trace_ids)
            )


class TraceSnapshot:
    """A disk-backed, reiterable snapshot of complete normalized traces."""

    def __init__(self, traces: Iterable[Trace]) -> None:
        self.traces_by_id = _DiskTraceIndex(traces)

    def __iter__(self) -> Iterator[Trace]:
        return self.traces_by_id.iter_traces()

    def __len__(self) -> int:
        return len(self.traces_by_id)

    def __repr__(self) -> str:
        return f"TraceSnapshot(trace_count={len(self)})"

    def get_trace_by_id(self, trace_id: str) -> Trace:
        return self.traces_by_id[trace_id]

    def sort(self, key: Callable[[str], Any]) -> None:
        """Order the index using loader metadata, without loading trace payloads."""
        self.traces_by_id.sort(key)

    @property
    def trace_count(self) -> int:
        return len(self)


__all__ = [
    "AgentInfo",
    "AudioPart",
    "Authorship",
    "ContentPart",
    "ImagePart",
    "Initiator",
    "LLMCallInfo",
    "Loss",
    "RunInfo",
    "RunTotals",
    "SpanKind",
    "Span",
    "SubagentRef",
    "TextPart",
    "TokenCounts",
    "TRACE_SCHEMA_VERSION",
    "ToolDefinition",
    "ToolResult",
    "TraceAggregate",
    "TraceSnapshot",
    "Trace",
    "ToolCall",
    "Turn",
    "TurnRole",
    "UNSET",
]
