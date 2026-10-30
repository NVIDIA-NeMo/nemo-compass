# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Merge provider records that share a session key into one conversation trace.

Each record's spans are namespaced as ``<record trace id>/<span id>`` and hung under
a synthesized root AGENT span that owns the conversation's turns. Record root AGENT
spans are demoted to CHAIN so the trace's root is their nearest agent.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from pydantic import JsonValue

from trace_ingest.loaders.helper.turns import NoTurns, agent_turns, run_root, with_run_root_turns
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


def merged_trace_id(session_id: str) -> str:
    """Id shared by a merged ``Trace`` and its root span."""
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


def _own_calls(spans: Sequence[Span]) -> Iterator[Span]:
    """LLM spans under ``spans`` that no nested AGENT span owns."""
    pending = list(spans)
    while pending:
        span = pending.pop()
        if span.kind is SpanKind.AGENT:
            continue
        if span.kind is SpanKind.LLM:
            yield span
        pending.extend(span.children)


def prepare_record(record: Trace) -> tuple[Trace, list[ToolDefinition]]:
    """Copy ``record`` with namespaced ids and demoted roots, ready for ``merge_records``.

    Returns the copy and the tool definitions moved off its demoted AGENT roots,
    because CHAIN spans may not declare tools. A demoted root's ``agent_id`` is copied
    to the LLM calls it owns that record none, so they keep the agent they ran under.
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
        if root.agent_id is not None:
            for call in _own_calls(root.children):
                call.agent_id = call.agent_id or root.agent_id
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
    """Join per-record conversations in record order, keeping each record's turn order.

    A record's copied-context turns re-send earlier history. They are matched in
    order, by role and content, against the turns already joined; copies up to the
    last match are dropped and later ones kept. When none match, only the copies
    beyond the number of turns already joined are kept.
    """
    turns: list[Turn] = []
    for record in records:
        copies = [
            i for i, turn in enumerate(record) if turn.authorship is Authorship.COPIED_CONTEXT
        ]
        start, last = 0, None
        for i in copies:
            match = next(
                (
                    j
                    for j in range(start, len(turns))
                    if (turns[j].role, turns[j].content) == (record[i].role, record[i].content)
                ),
                None,
            )
            if match is not None:
                start, last = match + 1, i
        dropped = set(copies[: len(turns)]) if last is None else {i for i in copies if i <= last}
        turns += [turn for i, turn in enumerate(record) if i not in dropped]
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


def merge_records(
    records: Sequence[Trace],
    *,
    session_id: str,
    session_basis: str,
    definitions: Sequence[ToolDefinition] = (),
    record_turns: Sequence[Sequence[Turn]] | None = None,
    turn_losses: Sequence[Loss] = (),
    partial_reason: Literal["max_traces"] | None = None,
) -> Trace:
    """Merge records prepared by ``prepare_record`` into one trace.

    ``record_turns`` is aligned with ``records``; ``None`` leaves the trace's
    root without turns. Recorded tool-call indexes continue across records in
    conversation order, so each record's calls follow the previous record's.
    """
    if not records:
        raise ValueError(f"session {session_id!r} has no records")
    if record_turns is not None and len(record_turns) != len(records):
        raise ValueError("record_turns must align with records")
    order = sorted(range(len(records)), key=lambda i: (_earliest_start(records[i]), records[i].id))
    ordered = [records[i] for i in order]
    root_id = merged_trace_id(session_id)

    children: list[Span] = []
    offset = 0
    for record in ordered:
        for root in record.root_spans:
            if root.source_url is None:
                root.source_url = record.source_url
            children.append(root)
        indexes: list[int] = []
        for span in _walk(record.root_spans):
            if span.tool_call is not None and span.tool_call.index is not None:
                span.tool_call.index += offset
                indexes.append(span.tool_call.index)
        offset = max(indexes) + 1 if indexes else offset

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
        attributes={"name": "conversation", "subtype": "conversation"},
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
                code="merge_incomplete",
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


@dataclass(frozen=True)
class Record:
    """One normalized provider record and its recorded conversation input and output."""

    trace: Trace
    conversation_input: object = None
    conversation_output: object = None


def _merge_group(
    group: Sequence[Record], *, orchestration_subtypes: frozenset[str], capped: bool
) -> Trace:
    first = group[0].trace
    assert first.session_id is not None and first.session_basis is not None
    prepared = [prepare_record(record.trace) for record in group]
    owner_id = merged_trace_id(first.session_id)
    record_turns: list[list[Turn]] | None = []
    turn_losses: list[Loss] = []
    try:
        for (trace, _), record in zip(prepared, group, strict=True):
            # A record owns turns only as it would on its own: through a single run root.
            root = run_root(trace, orchestration_subtypes=orchestration_subtypes)
            if root is None:
                raise NoTurns
            turns, losses = agent_turns(
                root.children,
                owner_id=owner_id,
                conversation_input=record.conversation_input,
                conversation_output=record.conversation_output,
            )
            record_turns.append(turns)
            turn_losses += losses
    except NoTurns:
        record_turns, turn_losses = None, []
    return merge_records(
        [trace for trace, _ in prepared],
        session_id=first.session_id,
        session_basis=first.session_basis,
        definitions=[definition for _, definitions in prepared for definition in definitions],
        record_turns=record_turns,
        turn_losses=turn_losses,
        partial_reason="max_traces" if capped else None,
    )


def assemble(
    records: Sequence[Record],
    *,
    orchestration_subtypes: frozenset[str] = frozenset(),
    capped: bool = False,
) -> list[Trace]:
    """One trace per ``session_id``, in order of each key's first record.

    Records sharing a ``session_id`` merge into one trace whose turns are built
    per record and joined; when any record has no single run root (see
    ``run_root``) or no recoverable conversation, the merged trace has no turns.
    A record without a ``session_id`` stays its own trace, with turns on its
    promoted run root. ``capped`` marks every merged trace incomplete because
    selection stopped at the loader's limit. Raises ``ValueError`` naming the
    ``session_id`` when its records cannot be merged.
    """
    groups: dict[str | int, list[Record]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record.trace.session_id or index, []).append(record)
    traces: list[Trace] = []
    for key, group in groups.items():
        if isinstance(key, int):
            (record,) = group
            traces.append(
                with_run_root_turns(
                    record.trace,
                    conversation_input=record.conversation_input,
                    conversation_output=record.conversation_output,
                    orchestration_subtypes=orchestration_subtypes,
                )
            )
            continue
        try:
            traces.append(
                _merge_group(group, orchestration_subtypes=orchestration_subtypes, capped=capped)
            )
        except ValueError as error:
            raise ValueError(f"session {key!r} could not be merged: {error}") from error
    return traces


__all__ = [
    "Record",
    "assemble",
    "join_turns",
    "merge_records",
    "merged_trace_id",
    "prepare_record",
]
