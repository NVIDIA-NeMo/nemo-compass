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
from typing import Any, Literal

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
    attributes: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Additional lossless result metadata not represented by a typed field.",
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
    """Non-negative token counts as the source records them.

    Each count is copied from a source count with the same meaning, never derived
    from other counts; a count the source does not record is None.
    """

    input_total: int | None = Field(
        default=None,
        ge=0,
        exclude_if=lambda value: value is None,
        description="All input tokens as the source totals them: uncached input plus cache "
        "reads and cache writes.",
    )
    input_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Uncached input tokens; excludes cache reads and cache writes.",
    )
    cached_input_tokens: int | None = Field(
        default=None, ge=0, description="Input tokens read from the cache."
    )
    output_tokens: int | None = Field(
        default=None, ge=0, description="Generated output tokens, reasoning tokens included."
    )
    cache_write_tokens: int | None = Field(
        default=None,
        ge=0,
        exclude_if=lambda value: value is None,
        description="Input tokens written to the cache, across all TTLs.",
    )
    cache_writes_by_ttl: dict[int, int] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Input tokens written to the cache per TTL in seconds, as the response "
        "reports them; may cover only some of cache_write_tokens.",
    )
    source: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Where the counts were read, such as nemo_relay (the provider's usage), "
        "openinference or langfuse_sdk.",
    )
    fallback_reason: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Why a preferred source was passed over for this one; None when no "
        "preferred source recorded counts.",
    )


class Span(_TraceModel):
    """One canonical, recursively nested unit of work in an agent trace."""

    id: str = Field(
        min_length=1,
        description="Stable span identifier, unique within its containing trace.",
    )
    source_url: SourceURL | None = Field(default=None, exclude_if=lambda value: value is None)
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
    llm_tools: list[JsonValue] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Top-level tool list sent to the model, in order; valid only for an LLM span.",
    )
    llm_params: dict[str, JsonValue] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Top-level request parameters sent to the model, without the model, tools "
        "and prompt content; an empty object means all defaults. Valid only for an LLM span.",
    )
    agent_id: str | None = Field(
        default=None,
        min_length=1,
        exclude_if=lambda value: value is None,
        description="Source-recorded agent ID, unique within the logical case: an AGENT span's "
        "own agent, or the agent an LLM call belongs to. Never a name shared by parallel agents.",
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
        if self.llm_tools is not None and self.kind is not SpanKind.LLM:
            raise ValueError("llm_tools requires kind=LLM")
        if self.llm_params is not None and self.kind is not SpanKind.LLM:
            raise ValueError("llm_params requires kind=LLM")
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

    @model_validator(mode="after")
    def span_ids_are_unique(self) -> Trace:
        seen: set[str] = set()
        pending = list(reversed(self.root_spans))
        while pending:
            span = pending.pop()
            if span.id in seen:
                raise ValueError(f"trace {self.id!r} contains duplicate span id {span.id!r}")
            seen.add(span.id)
            pending.extend(reversed(span.children))
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
    "SpanKind",
    "Span",
    "TokenCounts",
    "TRACE_SCHEMA_VERSION",
    "ToolDefinition",
    "ToolResult",
    "TraceAggregate",
    "TraceSnapshot",
    "Trace",
    "ToolCall",
    "UNSET",
]
