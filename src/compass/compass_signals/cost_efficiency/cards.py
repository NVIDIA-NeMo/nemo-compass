# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Group findings into one card per rule and kind, and promote recurring cards to Problems.

Nothing here knows a rule: each rule supplies its wording as a ``RuleSpec``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, JsonValue

from compass.compass_signals.compass_signals import Problem

SUPPORTING_TRACES = 5


class Finding(BaseModel):
    rule: str
    kind: str  # one of the rule's RuleSpec.kinds
    trace_id: str
    session: str
    session_id: str | None  # source-recorded; session falls back to the trace ID
    span_ids: tuple[str, ...]
    model: str | None
    tokens: int  # what the rule counts, not a known saving; RuleSpec.waste names it
    change_summary: dict[str, JsonValue] = {}
    reference_trace_id: str | None = None  # the call compared against, for rules that compare
    reference_span_id: str | None = None
    details: dict[str, JsonValue] = {}  # rule-specific values, for audit only


@dataclass(frozen=True)
class RuleSpec:
    """How a rule's Problem reads."""

    rule: str
    title: str
    kinds: dict[str, tuple[str, str]]  # kind -> (label, suggested fix)
    waste: str  # what Finding.tokens counts, e.g. "previously cached tokens"
    evidence: Callable[[Finding], str]  # one sentence on one finding
    summary: Callable[[Finding], str]  # what changed in one finding of a kind; "" for nothing


class CostCard(BaseModel):
    card_id: str
    rule: str
    title: str
    waste: str
    traces: int
    sessions: int
    calls: int
    tokens: int
    kind: str
    models: dict[str, int]  # model -> traces
    representatives: tuple[Finding, ...]
    eligible: bool
    audit_reason: str | None
    supporting_trace_ids: tuple[str, ...]
    candidate_trace_ids: tuple[str, ...]
    session_ids: tuple[str, ...]  # source-recorded sessions of all candidate traces


def build_cards(
    findings: Iterable[Finding], specs: Mapping[str, RuleSpec], *, minimum_independent_traces: int
) -> list[CostCard]:
    # Each kind has its own cause and fix, so each becomes its own card.
    groups: dict[tuple[str, str], list[Finding]] = defaultdict(list)
    for finding in findings:
        groups[finding.rule, finding.kind].append(finding)
    cards = []
    for (rule, kind), members in groups.items():
        trace_tokens: dict[str, int] = defaultdict(int)
        for member in members:
            trace_tokens[member.trace_id] += member.tokens
        ranked = sorted(trace_tokens, key=lambda trace_id: (-trace_tokens[trace_id], trace_id))
        supporting = tuple(ranked[:SUPPORTING_TRACES])
        representatives = tuple(
            next(m for m in members if m.trace_id == trace_id) for trace_id in supporting
        )
        audit_reason = (
            "below_minimum_traces" if len(trace_tokens) < minimum_independent_traces else None
        )
        cards.append(
            CostCard(
                card_id=f"ce:{rule}:{kind}",
                rule=rule,
                title=specs[rule].title,
                waste=specs[rule].waste,
                traces=len(trace_tokens),
                sessions=len({member.session for member in members}),
                calls=len(members),
                tokens=sum(trace_tokens.values()),
                kind=kind,
                models=_traces_by(members, lambda m: m.model or "-"),
                representatives=representatives,
                eligible=audit_reason is None,
                audit_reason=audit_reason,
                supporting_trace_ids=supporting,
                candidate_trace_ids=tuple(sorted(trace_tokens)),
                session_ids=tuple(sorted({m.session_id for m in members if m.session_id})),
            )
        )
    return sorted(cards, key=lambda card: (-card.traces, -card.tokens, card.card_id))


def _traces_by(members: Sequence[Finding], key: Callable[[Finding], str]) -> dict[str, int]:
    traces: dict[str, set[str]] = defaultdict(set)
    for member in members:
        traces[key(member)].add(member.trace_id)
    return {name: len(ids) for name, ids in sorted(traces.items(), key=lambda i: -len(i[1]))}


def problems_from_cards(
    cards: Sequence[CostCard],
    findings: Sequence[Finding],
    specs: Mapping[str, RuleSpec],
    totals: Mapping[str, int],
) -> tuple[Problem, ...]:
    return tuple(
        Problem(
            description=_describe(
                card,
                specs[card.rule],
                [f for f in findings if (f.rule, f.kind) == (card.rule, card.kind)],
                totals,
            ),
            supporting_trace_ids=card.supporting_trace_ids,
            candidate_trace_ids=card.candidate_trace_ids,
        )
        for card in cards
        if card.eligible
    )


def _describe(
    card: CostCard, spec: RuleSpec, members: Sequence[Finding], totals: Mapping[str, int]
) -> str:
    # A kind the spec does not list is a detector bug: the lookup raises.
    label, fix = spec.kinds[card.kind]
    parts = [
        f"{card.title} ({label}) in {_count(card.traces, 'trace')} "
        f"({_percent(card.traces, totals['total_traces'])} of all traces, "
        f"{_percent(card.traces, totals['traces_with_counts'])} of traces with token counts; "
        f"{_count(card.sessions, 'session')}, "
        f"{_percent(card.sessions, totals['sessions_with_counts'])} of sessions "
        f"with token counts), {_count(card.calls, 'call')} "
        f"({_percent(card.calls, totals['calls_with_counts'])} of LLM calls with token counts).",
        spec.evidence(card.representatives[0]),
        f"({card.tokens:,} {card.waste}.)",
    ]
    # The costliest finding stands for the card.
    if summary := spec.summary(max(members, key=lambda m: m.tokens)):
        parts.append(summary)
    parts.append(f"Suggested fix: {fix}")
    parts.append("Figures cover only the observed calls.")
    return " ".join(parts)


def names(values: Sequence[str]) -> str:
    """Up to five names, sorted, then how many more."""
    ordered = sorted(values)
    shown = ", ".join(ordered[:5])
    return shown + (f", +{len(ordered) - 5} more" if len(ordered) > 5 else "")


def _count(value: int, singular: str) -> str:
    return f"{value:,} {singular}{'' if value == 1 else 's'}"


def _percent(part: int, whole: int) -> str:
    if whole <= 0:
        return "n/a"
    value = f"{100 * part / whole:.1f}".rstrip("0").rstrip(".")
    return f"{value}%"


__all__ = [
    "CostCard",
    "Finding",
    "RuleSpec",
    "build_cards",
    "names",
    "problems_from_cards",
]
