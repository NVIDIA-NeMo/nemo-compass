# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rule cache_miss_after_tool_change: a cache miss whose tool list differs from its reference call's.

Only compares the tools; ``misses.cache_misses`` has already ruled out expiry.
Other changes in the same miss are other rules' findings.

This module reads tool definitions but records only tool names, never their text.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from compass.compass_signals._hashing import stable_json
from compass.compass_signals.cost_efficiency.cards import Finding, RuleSpec, names
from compass.traces import UNSET, Span
from trace_ingest.models import ToolDefinition

RULE = "cache_miss_after_tool_change"
ToolChangeKind = Literal[
    "tools_added", "tools_removed", "tools_added_and_removed", "definition_changed", "other"
]


@dataclass(frozen=True)
class ToolChange:
    """How the tool list changed: the kind of change and what changed.

    ``change_summary`` holds tool names only, never definitions.
    """

    kind: ToolChangeKind
    change_summary: dict[str, Any] = field(default_factory=dict)


def check(call: Span, reference: Span) -> ToolChange | None:
    """The first difference from the reference's tool list to the call's, coarsest first.

    ``tools_added``, ``tools_removed``, ``tools_added_and_removed``: the set of tool names
    changed, with the added and removed names.
    ``definition_changed``: same names, a definition differs, with the changed names.
    ``other``: the lists differ in any other way, such as tool order, JSON key order,
    cache breakpoints or duplicate tool names.
    None when either call has no recorded tool list or the lists are identical.
    """

    sent, reference_sent = _sent(call), _sent(reference)
    if sent is None or reference_sent is None:
        return None
    old, new = _by_name(reference_sent), _by_name(sent)
    if old.keys() != new.keys():
        added, removed = sorted(new.keys() - old.keys()), sorted(old.keys() - new.keys())
        kind = (
            "tools_added_and_removed"
            if added and removed
            else "tools_added"
            if added
            else "tools_removed"
        )
        return ToolChange(kind, {"added": added, "removed": removed})
    # stable_json ignores key order but, unlike ==, tells 1 from true.
    changed = sorted(name for name in new if stable_json(old[name]) != stable_json(new[name]))
    if changed:
        return ToolChange("definition_changed", {"changed": changed})
    if _raw_json(reference_sent) != _raw_json(sent):
        return ToolChange("other")
    return None


def _sent(span: Span) -> list[ToolDefinition] | None:
    """The tool list as the request sent it; None when the span records none."""

    if span.tool_definitions is UNSET:
        return None
    return span.tool_definitions


def _by_name(definitions: list[ToolDefinition]) -> dict[str, Any]:
    """Tool name -> definition, in sent order, without breakpoint markers the model never sees."""

    return {
        _tool_name(definition, position): (
            {key: value for key, value in definition.raw.items() if key != "cache_control"}
            if isinstance(definition.raw, Mapping)
            else definition.raw
        )
        for position, definition in enumerate(definitions)
    }


def _tool_name(definition: ToolDefinition, position: int) -> str:
    """The loader's name, else the built-in tool type, else the position."""

    if definition.name:
        return definition.name
    raw = definition.raw
    tool_type = raw.get("type") if isinstance(raw, Mapping) else None
    return tool_type if isinstance(tool_type, str) and tool_type else f"#{position}"


def _raw_json(definitions: list[ToolDefinition]) -> str:
    """The tool list as sent, key order included."""

    raw = [definition.raw for definition in definitions]
    return json.dumps(raw, ensure_ascii=False, separators=(",", ":"))


def _evidence(finding: Finding) -> str:
    return f"The previous call left {finding.tokens:,} cached tokens; this call read 0."


def _summary(finding: Finding) -> str:
    summary = cast(dict[str, list[str]], finding.change_summary)  # as check() records it
    if finding.kind in ("tools_added", "tools_removed", "tools_added_and_removed"):
        parts = []
        if summary.get("added"):
            parts.append(f"Tools added: {names(summary['added'])}")
        if summary.get("removed"):
            parts.append(f"Tools removed: {names(summary['removed'])}")
        return "; ".join(parts) + "." if parts else ""
    if finding.kind == "definition_changed":
        return f"Changed tools: {names(summary['changed'])}."
    return ""


SPEC = RuleSpec(
    rule=RULE,
    kinds={
        "tools_added": (
            "Cache miss after tools were added",
            "Check whether added tools were available earlier. If so, consider registering "
            "them before the first call.",
        ),
        "tools_removed": (
            "Cache miss after tools were removed",
            "Check whether removed tools must leave the list. If not, consider keeping them "
            "registered.",
        ),
        "tools_added_and_removed": (
            "Cache miss after tools were added and removed",
            "Check whether added tools were available earlier and whether removed tools must "
            "leave the list. If not, consider keeping one tool list from the first call.",
        ),
        "definition_changed": (
            "Cache miss after a tool definition changed",
            "Avoid changes that do not change the tool's meaning; keep changes needed for "
            "correct tool use.",
        ),
        "other": (
            "Cache miss after the tool list was reordered or reformatted",
            "Avoid unnecessary tool-list changes.",
        ),
    },
    waste="previously cached tokens; savings from a fix are unknown",
    evidence=_evidence,
    summary=_summary,
)


__all__ = ["RULE", "SPEC", "ToolChange", "ToolChangeKind", "check"]
