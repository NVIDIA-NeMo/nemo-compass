# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Merge provider records that share a session key into one conversation trace.

Each record's spans are namespaced as ``<record trace id>/<span id>`` and hung under
a synthesized session-root AGENT span that owns the conversation's turns. Record
root AGENT spans are demoted to CHAIN so the session root is their nearest agent.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from typing import Literal

from pydantic import JsonValue

from trace_ingest.models import (
    UNSET,
    Authorship,
    Loss,
    RunInfo,
    Span,
    SpanKind,
    SubagentRef,
    TokenCounts,
    ToolDefinition,
    Trace,
    TraceAggregate,
    Turn,
)

_SUMMED_TOKEN_FIELDS = (
    "input_total",
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "cache_write_tokens",
)


def session_root_id(session_id: str) -> str:
    """Id shared by a merged session's ``Trace`` and its session-root span."""
    return f"session:{session_id}"


def _walk(spans: Sequence[Span]) -> Iterator[Span]:
    pending = list(reversed(spans))
    while pending:
        span = pending.pop()
        yield span
        pending.extend(reversed(span.children))


def _refs(spans: Sequence[Span]) -> Iterator[SubagentRef]:
    for span in _walk(spans):
        if span.tool_call is not None and span.tool_call.results is not UNSET:
            for result in span.tool_call.results:
                yield from result.subagent_refs
        if span.turns is not UNSET:
            for turn in span.turns:
                for result in turn.unlinked_results:
                    yield from result.subagent_refs


def prepare_record(record: Trace) -> tuple[Trace, list[ToolDefinition]]:
    """Copy ``record`` with namespaced ids and demoted roots, ready for ``merge_session``.

    Returns the copy and the tool definitions moved off its demoted AGENT roots,
    because CHAIN spans may not declare tools.
    """
    prepared = record.model_copy(deep=True)
    definitions: list[ToolDefinition] = []
    for root in prepared.root_spans:
        if root.kind is not SpanKind.AGENT:
            continue
        if any(value is not UNSET for value in (root.agent, root.run, root.turns)):
            raise ValueError(f"record {record.id} root {root.id} already owns a run")
        if root.tool_definitions is not UNSET:
            definitions.extend(root.tool_definitions)
            root.tool_definitions = UNSET
        root.kind = SpanKind.CHAIN
        root.attributes["subtype"] = root.attributes.get("subtype") or SpanKind.AGENT.value

    def span_id(value: str) -> str:
        return f"{record.id}/{value}"

    turn_ids: dict[str, str] = {}
    for span in _walk(prepared.root_spans):
        span.id = span_id(span.id)
        if span.turns is UNSET:
            continue
        for turn in span.turns:
            # Turn ids stay "<agent span id>/turn/<index>" under the renamed agent.
            turn_ids[turn.id] = new_turn_id = span_id(turn.id)
            turn.id = new_turn_id
            turn.llm_span_ids = [span_id(value) for value in turn.llm_span_ids]
            turn.tool_call_span_ids = [span_id(value) for value in turn.tool_call_span_ids]
            if turn.dispatch_span_id is not None:
                turn.dispatch_span_id = span_id(turn.dispatch_span_id)
    for ref in _refs(prepared.root_spans):
        if ref.span_id is not None:
            ref.span_id = span_id(ref.span_id)
    for loss in prepared.losses:
        if loss.span_id is not None:
            loss.span_id = span_id(loss.span_id)
        if loss.turn_id is not None:
            loss.turn_id = turn_ids[loss.turn_id]
    return prepared, definitions


def join_turns(records: Sequence[Sequence[Turn]]) -> list[Turn]:
    """Join per-record conversations by position.

    A record's copied-context turns re-send earlier history; only those beyond
    the turns already joined are kept, followed by the record's own turns.
    """
    turns: list[Turn] = []
    for record in records:
        copies = [turn for turn in record if turn.authorship is Authorship.COPIED_CONTEXT]
        own = [turn for turn in record if turn.authorship is not Authorship.COPIED_CONTEXT]
        turns += copies[len(turns) :]
        turns += own
    return turns


def _span_window(spans: Sequence[Span]) -> tuple[datetime | None, datetime | None]:
    """Earliest start and latest end over spans that record both times."""
    starts: list[datetime] = []
    ends: list[datetime] = []
    for span in _walk(spans):
        if span.start_time is not None and span.end_time is not None:
            starts.append(span.start_time)
            ends.append(span.end_time)
    return min(starts, default=None), max(ends, default=None)


def _earliest_start(record: Trace) -> datetime:
    starts = [span.start_time for span in _walk(record.root_spans) if span.start_time is not None]
    return min(starts, default=datetime.max.replace(tzinfo=UTC))


def _token_counts(records: Sequence[Trace]) -> TokenCounts | None:
    """Session totals; a count any record does not report is None, never a partial sum."""
    reported = [
        record.aggregate.token_counts
        for record in records
        if record.aggregate.token_counts is not None
    ]
    if len(reported) != len(records):
        return None
    values: dict[str, object] = {}
    for field in _SUMMED_TOKEN_FIELDS:
        counts = [getattr(count, field) for count in reported]
        values[field] = None if None in counts else sum(counts)
    by_ttl = [count.cache_writes_by_ttl for count in reported if count.cache_writes_by_ttl]
    if len(by_ttl) == len(reported):
        ttls = set.intersection(*(set(entry) for entry in by_ttl))
        values["cache_writes_by_ttl"] = {
            ttl: sum(entry[ttl] for entry in by_ttl) for ttl in sorted(ttls)
        } or None
    sources = {count.source for count in reported if count.source is not None}
    values["source"] = sources.pop() if len(sources) == 1 else None
    return TokenCounts.model_validate(values)


def _tool_definitions(definitions: Sequence[ToolDefinition]) -> list[ToolDefinition]:
    """Drop exact repeats only; same-name entries that differ are all kept, in order."""
    union: list[ToolDefinition] = []
    for definition in definitions:
        if definition not in union:
            union.append(definition)
    return union


def merge_session(
    records: Sequence[Trace],
    *,
    session_id: str,
    session_basis: str,
    definitions: Sequence[ToolDefinition] = (),
    record_turns: Sequence[Sequence[Turn]] | None = None,
    turn_losses: Sequence[Loss] = (),
    partial_reason: Literal["max_traces"] | None = None,
) -> Trace:
    """Merge records prepared by ``prepare_record`` into one session trace.

    ``record_turns`` is aligned with ``records``; ``None`` leaves the session
    root without turns.
    """
    if not records:
        raise ValueError(f"session {session_id!r} has no records")
    if record_turns is not None and len(record_turns) != len(records):
        raise ValueError("record_turns must align with records")
    order = sorted(range(len(records)), key=lambda i: (_earliest_start(records[i]), records[i].id))
    ordered = [records[i] for i in order]
    root_id = session_root_id(session_id)

    children: list[Span] = []
    for record in ordered:
        for root in record.root_spans:
            if root.source_url is None:
                root.source_url = record.source_url
            children.append(root)

    start, end = _span_window(children)

    turns: list[Turn] | UNSET = UNSET
    if record_turns is not None:
        joined = join_turns([record_turns[i] for i in order])
        turns = [
            turn.model_copy(update={"id": f"{root_id}/turn/{n}", "index": n})
            for n, turn in enumerate(joined, start=1)
        ]
    tools = _tool_definitions(definitions)
    root = Span(
        id=root_id,
        kind=SpanKind.AGENT,
        start_time=start,
        end_time=end,
        children=children,
        run=RunInfo(session_id=session_id),
        tool_definitions=tools if tools else UNSET,
        turns=turns,
        attributes={"name": "session", "subtype": "session"},
    )

    costs = [
        record.aggregate.cost_usd for record in ordered if record.aggregate.cost_usd is not None
    ]
    latency = (
        (end - start).total_seconds() * 1000 if start is not None and end is not None else None
    )

    case_ids = {record.case_id for record in ordered}
    case_id = case_ids.pop() if len(case_ids) == 1 else None

    scores: dict[str, list[JsonValue]] = {}
    for record in ordered:
        for name, value in record.evaluator_results.items():
            scores.setdefault(name, []).append({"trace_id": record.id, "value": value})
    evaluator_results: dict[str, JsonValue] = dict(scores)

    losses = [loss for record in ordered for loss in record.losses]
    losses += turn_losses
    if partial_reason is not None:
        losses.append(
            Loss(
                code="session_partial",
                detail={"reason": partial_reason, "loaded_record_count": len(ordered)},
                span_id=root_id,
            )
        )

    pointer = ordered[0].attributes.get("source_pointer")
    attributes: dict[str, JsonValue] = {
        "source_pointer": {
            "provider": pointer.get("provider") if isinstance(pointer, dict) else None,
            "session_id": session_id,
            "session_basis": session_basis,
        },
        "records": [
            {
                "trace_id": record.id,
                "source_url": record.source_url,
                "root_span_ids": [span.id for span in record.root_spans],
                "attributes": record.attributes,
            }
            for record in ordered
        ],
    }
    if "task_text" in ordered[0].attributes:
        attributes["task_text"] = ordered[0].attributes["task_text"]

    return Trace(
        id=root_id,
        session_id=session_id,
        session_basis=session_basis,
        case_id=case_id,
        root_spans=[root],
        aggregate=TraceAggregate(
            cost_usd=sum(costs) if len(costs) == len(ordered) else None,
            latency_ms=latency,
            token_counts=_token_counts(ordered),
        ),
        evaluator_results=evaluator_results,
        attributes=attributes,
        losses=losses,
    )


__all__ = ["join_turns", "merge_session", "prepare_record", "session_root_id"]
