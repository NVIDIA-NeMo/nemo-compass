# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool-issue Compass Signal.

The implementation follows the current seven-category, nineteen-finding
catalog. It is deterministic, capability-gated, and independent of source
loaders. Missing evidence produces no finding. Every finding retains the
original pointer supplied by the caller.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from jsonschema import validators
from pydantic import BaseModel, Field

from compass.compass_signals._hashing import digest, stable_json
from compass.compass_signals._trace import walk_spans
from compass.compass_signals.compass_signals import CompassSignalResult, Problem
from compass.traces import UNSET, SpanKind, Trace, TraceSnapshot
from trace_ingest.catalog import record_tool_catalogs

ToolCatalog = Mapping[str, Mapping[str, Any] | bool | None]

DETECTOR_VERSION = "tid-v1"
CARD_MINIMUM_CASES = 3
RETRY_THRESHOLD = 3
MISSING = object()

FINDING_TYPES = frozenset(
    {
        "unknown_tool",
        "malformed_tool_call",
        "missing_required_argument",
        "unknown_argument",
        "argument_type_mismatch",
        "argument_enum_violation",
        "json_schema_violation",
        "duplicate_call_id",
        "missing_tool_result",
        "duplicate_tool_result",
        "orphan_tool_result",
        "call_result_id_mismatch",
        "explicit_tool_failure",
        "repeated_identical_failed_call",
        "modified_retry_same_failure",
        "mapped_instrumentation_alias",
        "unresolved_placeholder_argument",
        "explicitly_rejected_ungrounded_identifier",
        "explicit_prerequisite_or_state_failure",
    }
)

FAMILY = {
    "unknown_tool": "tool_contract_and_arguments",
    "malformed_tool_call": "tool_contract_and_arguments",
    "missing_required_argument": "tool_contract_and_arguments",
    "unknown_argument": "tool_contract_and_arguments",
    "argument_type_mismatch": "tool_contract_and_arguments",
    "argument_enum_violation": "tool_contract_and_arguments",
    "json_schema_violation": "tool_contract_and_arguments",
    "duplicate_call_id": "call_result_integrity",
    "missing_tool_result": "call_result_integrity",
    "duplicate_tool_result": "call_result_integrity",
    "orphan_tool_result": "call_result_integrity",
    "call_result_id_mismatch": "call_result_integrity",
    "explicit_tool_failure": "explicit_tool_outcome",
    "repeated_identical_failed_call": "recovery_and_retries",
    "modified_retry_same_failure": "recovery_and_retries",
    "mapped_instrumentation_alias": "trace_instrumentation",
    "unresolved_placeholder_argument": "argument_provenance",
    "explicitly_rejected_ungrounded_identifier": "argument_provenance",
    "explicit_prerequisite_or_state_failure": "explicit_prerequisite_or_state",
}

SHELL_EXIT = re.compile(r"Process exited with code\s+(-?\d+)\b", re.IGNORECASE)
STRICT_ERROR = re.compile(r"\A\s*(?:error|failed|failure)\s*[:\-]", re.IGNORECASE)
TRACEBACK = re.compile(r"(?m)^Traceback \(most recent call last\):")
REJECTION = re.compile(
    r"(?i)(?:invalid|unknown|not found|does not exist|is not existing|missing input|not a valid)"
)
PLACEHOLDER = re.compile(
    r"(?ix)^(?:<[a-z][a-z0-9 _-]{1,40}>|\{\{[^{}]{1,60}\}\}|"
    r"(?:todo|tbd|placeholder|replace[_-]?me|your[_-][a-z0-9_-]+)|"
    r"[a-z]+\d+[_-](?:ref[_-]?)?id)$"
)
IDENTIFIER_FIELD = re.compile(r"(?i)(?:^|_)(?:id|refid|reference|handle|key)$")
IDENTIFIER_VALUE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@/\-]{2,127}$")
CODE_EXECUTION_TOOLS = frozenset({"CodeExecutionTool"})
STATE_PATTERNS = tuple(
    (name, re.compile(source))
    for name, source in (
        (
            "no_active_session",
            r"(?i)(?:\[NO_ACTIVE_SESSION\]|no active [a-z0-9 _-]{1,40}session)",
        ),
        (
            "initialize_before_use",
            r"(?i)(?:not initialized|please .{0,80}\bbefore (?:running|using|calling))",
        ),
        (
            "required_type_or_target",
            r"(?i)(?:must be (?:a|an|the) [a-z][a-z0-9 _-]{1,50}"
            r"|cannot be created without (?:a|an|the) [a-z][a-z0-9 _-]{1,50})",
        ),
        (
            "required_first_step",
            r"(?i)(?:must|need(?:s)? to|require(?:s|d)?) .{0,100}\bfirst\b",
        ),
    )
)


def parse_json(value: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def result_text(value: object) -> str:
    parsed = parse_json(value)
    if isinstance(parsed, Mapping):
        content = parsed.get("content")
        return content if isinstance(content, str) else stable_json(parsed)
    return value if isinstance(value, str) else stable_json(value)


def compact(value: object, limit: int = 900) -> str:
    text = " ".join(result_text(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass(frozen=True)
class CallRecord:
    trace_id: str
    call_index: int
    call_id: str
    tool_name: str
    arguments: Any
    result: Any = MISSING
    source_pointer: Mapping[str, Any] = field(default_factory=dict)
    result_id: str | None = None
    result_count: int = 1
    explicit_error: bool | None = None
    outcome_marker: str | None = None
    instrumentation_alias_of: str | None = None
    prior_user_text: str = ""
    tool_catalog: ToolCatalog | None = None


@dataclass(frozen=True)
class TraceRecord:
    trace_id: str
    calls: Sequence[CallRecord]
    case_id: str | None = None
    orphan_results: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    complete_provenance_context: bool = False


def schema_errors(catalog: ToolCatalog, tool_name: str, arguments: object) -> list[dict[str, Any]]:
    """Apply the active catalog/JSON Schema without coercing evidence."""

    if tool_name not in catalog:
        return [
            {
                "type": "unknown_tool",
                "path": "tool_name",
                "message": "Tool is absent from the active catalog.",
            }
        ]
    # Missing input is unavailable evidence, not an observed malformed call.
    if arguments is UNSET:
        return []
    if not isinstance(arguments, Mapping):
        return [
            {"type": "malformed_tool_call", "path": "$", "message": "Arguments are not an object."}
        ]
    schema = catalog[tool_name]
    if schema is None:
        return []
    validator_cls = validators.validator_for(schema)
    validator_cls.check_schema(schema)
    output: list[dict[str, Any]] = []
    for error in sorted(
        validator_cls(schema).iter_errors(arguments), key=lambda item: list(item.path)
    ):
        kind = {
            "required": "missing_required_argument",
            "additionalProperties": "unknown_argument",
            "type": "argument_type_mismatch",
            "enum": "argument_enum_violation",
        }.get(str(error.validator), "json_schema_violation")
        output.append(
            {
                "type": kind,
                "path": ".".join(str(part) for part in error.path) or "$",
                "validator": str(error.validator),
                "message": error.message,
            }
        )
    return output


def strict_failure(call: CallRecord) -> tuple[bool, str | None]:
    """Decode failure only from structured or strict tool-native evidence."""

    if call.explicit_error is True:
        return True, call.outcome_marker or "explicit_error_field"
    if call.explicit_error is False:
        return False, None
    if call.result is MISSING:
        return False, None
    parsed = parse_json(call.result)
    if isinstance(parsed, Mapping):
        if (
            parsed.get("isError") is True
            or parsed.get("success") is False
            or parsed.get("called") is False
        ):
            return True, "structured_error_flag"
        if parsed.get("error") not in (None, False, "", []):
            return True, "structured_error_value"
        if str(parsed.get("status", "")).lower() in {"error", "failed", "failure"}:
            return True, "structured_error_status"
        nested = parsed.get("result")
        if isinstance(nested, Mapping) and (
            nested.get("isError") is True or nested.get("success") is False
        ):
            return True, "nested_error_flag"
    text = result_text(call.result)
    match = SHELL_EXIT.search(text)
    if match:
        code = int(match.group(1))
        return code != 0, f"shell_exit_code_{code}"
    if STRICT_ERROR.search(text):
        return True, "error_prefix"
    if call.tool_name in CODE_EXECUTION_TOOLS and TRACEBACK.search(text):
        return True, "python_traceback"
    return False, None


def _finding(
    trace: TraceRecord,
    call: CallRecord,
    issue_type: str,
    summary: str,
    *,
    attribution: str,
    evidence: Mapping[str, Any] | None = None,
    mechanism_key: str = "",
) -> dict[str, Any]:
    if issue_type not in FINDING_TYPES:
        raise ValueError(f"unknown TID finding type: {issue_type}")
    payload = {
        "trace_id": trace.trace_id,
        "case_id": trace.case_id or trace.trace_id,
        "call_index": call.call_index,
        "call_id": call.call_id,
        "tool_name": call.tool_name,
        "issue_family": FAMILY[issue_type],
        "issue_type": issue_type,
        "mechanism_key": mechanism_key or issue_type,
        "evidence_state": "confirmed",
        "attribution": attribution,
        "summary": summary,
        "evidence": dict(evidence or {}),
        "source_pointer": dict(call.source_pointer),
        "detector_version": DETECTOR_VERSION,
    }
    payload["issue_id"] = digest(payload)[:24]
    return payload


def _flatten_strings(value: object, prefix: str = "$") -> Iterable[tuple[str, str, str]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _flatten_strings(child, f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _flatten_strings(child, f"{prefix}[{index}]")
    elif isinstance(value, str):
        leaf = re.split(r"[.\[]", prefix)[-1].rstrip("]")
        yield prefix, leaf, value


def detect_trace(
    trace: TraceRecord,
    *,
    retry_threshold: int = RETRY_THRESHOLD,
) -> list[dict[str, Any]]:
    """Run all evaluable TID v1 rules on one ordered trace."""

    findings: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    failed_by_exact_call: dict[tuple[str, str], list[CallRecord]] = defaultdict(list)
    modified_retries: dict[tuple[str, str], list[tuple[CallRecord, str]]] = defaultdict(list)
    prior_results: list[str] = []

    for call in sorted(trace.calls, key=lambda item: (item.call_index, item.call_id)):
        arguments = call.arguments if isinstance(call.arguments, Mapping) else {}
        contract_root = False
        if call.tool_catalog is not None:
            for error in schema_errors(call.tool_catalog, call.tool_name, call.arguments):
                contract_root = True
                findings.append(
                    _finding(
                        trace,
                        call,
                        str(error["type"]),
                        str(error["message"]),
                        attribution="agent_call",
                        evidence={
                            key: value
                            for key, value in error.items()
                            if key not in {"type", "message"}
                        },
                    )
                )
        elif call.arguments is not UNSET and not isinstance(call.arguments, Mapping):
            contract_root = True
            findings.append(
                _finding(
                    trace,
                    call,
                    "malformed_tool_call",
                    "Arguments are not an object.",
                    attribution="agent_call",
                )
            )

        if call.call_id in seen_ids:
            findings.append(
                _finding(
                    trace,
                    call,
                    "duplicate_call_id",
                    "The trace reuses a call ID.",
                    attribution="instrumentation",
                )
            )
        seen_ids.add(call.call_id)
        if call.result is MISSING:
            findings.append(
                _finding(
                    trace,
                    call,
                    "missing_tool_result",
                    "The call has no linked result.",
                    attribution="instrumentation",
                )
            )
        if call.result_count > 1:
            findings.append(
                _finding(
                    trace,
                    call,
                    "duplicate_tool_result",
                    "The call has more than one linked result.",
                    attribution="instrumentation",
                    evidence={"result_count": call.result_count},
                )
            )
        if call.result_id is not None and call.result_id != call.call_id:
            findings.append(
                _finding(
                    trace,
                    call,
                    "call_result_id_mismatch",
                    "The result ID does not match the call ID.",
                    attribution="instrumentation",
                    evidence={"result_id": call.result_id},
                )
            )
        if call.instrumentation_alias_of:
            findings.append(
                _finding(
                    trace,
                    call,
                    "mapped_instrumentation_alias",
                    "The recorded tool name is a known instrumentation alias.",
                    attribution="instrumentation",
                    evidence={"original_tool_name": call.instrumentation_alias_of},
                )
            )

        failed, marker = strict_failure(call)
        if failed and not contract_root:
            findings.append(
                _finding(
                    trace,
                    call,
                    "explicit_tool_failure",
                    f"The tool reported an explicit failure ({marker}).",
                    attribution="tool_or_environment",
                    evidence={"failure_marker": marker},
                    mechanism_key=str(marker),
                )
            )
        call_failed = failed or contract_root or call.result is MISSING or call.result_count > 1
        if call_failed:
            exact_key = (call.tool_name, stable_json(arguments))
            failed_by_exact_call[exact_key].append(call)
            failure_text = compact(call.result) if call.result is not MISSING else "missing_result"
            failure_class = str(marker or digest(failure_text)[:12])
            modified_retries[(call.tool_name, failure_class)].append((call, stable_json(arguments)))

        leaves = list(_flatten_strings(arguments))
        output = result_text(call.result) if call.result is not MISSING else ""
        rejected = bool(REJECTION.search(output))
        placeholders = [
            (path, value) for path, _, value in leaves if PLACEHOLDER.fullmatch(value.strip())
        ]
        if placeholders and (failed or rejected):
            findings.append(
                _finding(
                    trace,
                    call,
                    "unresolved_placeholder_argument",
                    "A failed call contains unresolved placeholder arguments.",
                    attribution="agent_call",
                    evidence={
                        "argument_paths_and_values": placeholders[:20],
                        "result_excerpt": compact(output),
                    },
                )
            )
        if trace.complete_provenance_context and rejected:
            prior = "\n".join([call.prior_user_text, *prior_results])
            ungrounded = [
                (path, value)
                for path, leaf, value in leaves
                if IDENTIFIER_FIELD.search(leaf)
                and IDENTIFIER_VALUE.fullmatch(value)
                and value in output
                and value not in prior
            ]
            if ungrounded:
                findings.append(
                    _finding(
                        trace,
                        call,
                        "explicitly_rejected_ungrounded_identifier",
                        "The tool rejected an identifier with no observable prior provenance.",
                        attribution="agent_call",
                        evidence={
                            "argument_paths_and_values": ungrounded[:20],
                            "result_excerpt": compact(output),
                        },
                    )
                )
        state_matches = [name for name, pattern in STATE_PATTERNS if pattern.search(output)]
        if state_matches:
            findings.append(
                _finding(
                    trace,
                    call,
                    "explicit_prerequisite_or_state_failure",
                    "The result explicitly reports an unmet prerequisite or invalid state.",
                    attribution="tool_or_environment",
                    evidence={"state_class": state_matches[0], "result_excerpt": compact(output)},
                    mechanism_key=state_matches[0],
                )
            )
        if trace.complete_provenance_context and call.result is not MISSING:
            prior_results.append(output)

    for orphan in trace.orphan_results:
        synthetic = CallRecord(
            trace_id=trace.trace_id,
            call_index=-1,
            call_id=str(orphan.get("result_id") or "<missing>"),
            tool_name="<unknown>",
            arguments={},
            source_pointer=orphan,
        )
        findings.append(
            _finding(
                trace,
                synthetic,
                "orphan_tool_result",
                "A result has no matching call.",
                attribution="instrumentation",
            )
        )

    for (tool_name, _), calls in failed_by_exact_call.items():
        if len(calls) >= retry_threshold:
            findings.append(
                _finding(
                    trace,
                    calls[0],
                    "repeated_identical_failed_call",
                    f"The same failed {tool_name!r} call was repeated {len(calls)} times.",
                    attribution="agent_recovery",
                    evidence={
                        "repeat_count": len(calls),
                        "call_indices": [call.call_index for call in calls],
                    },
                )
            )
    for (tool_name, failure_class), attempts in modified_retries.items():
        argument_sets = {arguments for _, arguments in attempts}
        if len(attempts) >= retry_threshold and len(argument_sets) >= 2:
            findings.append(
                _finding(
                    trace,
                    attempts[0][0],
                    "modified_retry_same_failure",
                    f"{len(attempts)} failed {tool_name!r} attempts changed arguments but preserved one failure.",
                    attribution="agent_recovery",
                    evidence={
                        "attempt_count": len(attempts),
                        "distinct_argument_count": len(argument_sets),
                        "call_indices": [call.call_index for call, _ in attempts],
                    },
                    mechanism_key=failure_class,
                )
            )
    return findings


def detect(
    traces: Iterable[TraceRecord],
    *,
    retry_threshold: int = RETRY_THRESHOLD,
) -> list[dict[str, Any]]:
    """Run TID over traces and return a stable evidence-first ordering."""

    findings = [
        finding
        for trace in traces
        for finding in detect_trace(trace, retry_threshold=retry_threshold)
    ]
    return sorted(
        findings,
        key=lambda item: (
            item["trace_id"],
            item["call_index"],
            item["issue_type"],
            item["issue_id"],
        ),
    )


class RepresentativeEvidence(BaseModel):
    trace_id: str = Field(min_length=1)
    call_id: str = Field(min_length=1)
    call_index: int = Field(
        ge=-1,
        description="Zero-based tool-call index; -1 identifies an orphan tool result.",
    )
    tool_name: str = Field(min_length=1)
    source_pointer: Mapping[str, Any]
    observation: str


class ToolIssueCard(BaseModel):
    card_id: str = Field(min_length=1)
    issue_type: str = Field(min_length=1)
    issue_family: str = Field(min_length=1)
    mechanism_key: str = Field(min_length=1)
    finding_count: int = Field(ge=1)
    independent_case_count: int = Field(ge=1)
    eligible_for_insights: bool
    trace_ids: tuple[str, ...] = Field(min_length=1)
    representative_evidence: tuple[RepresentativeEvidence, ...] = Field(min_length=1)
    impact_status: Literal["not_established"]
    impact_boundary: str


def build_cards(
    findings: Iterable[Mapping[str, Any]], *, minimum_independent_cases: int = CARD_MINIMUM_CASES
) -> list[ToolIssueCard]:
    """Promote recurring findings into compact evidence cards for insight compilation."""

    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for finding in findings:
        groups[(str(finding["issue_type"]), str(finding.get("mechanism_key") or ""))].append(
            finding
        )
    cards: list[ToolIssueCard] = []
    for (issue_type, mechanism), members in sorted(groups.items()):
        cases = sorted({str(member["case_id"]) for member in members})
        representatives: dict[str, Mapping[str, Any]] = {}
        for member in members:
            representatives.setdefault(str(member["case_id"]), member)
        eligible = len(cases) >= minimum_independent_cases
        examples = [
            {
                "trace_id": member["trace_id"],
                "call_id": member["call_id"],
                "call_index": member["call_index"],
                "tool_name": member["tool_name"] or "unknown tool",
                "source_pointer": member["source_pointer"],
                "observation": member["summary"],
            }
            for member in list(representatives.values())[:3]
        ]
        cards.append(
            ToolIssueCard(
                card_id=f"tid:{issue_type}:{mechanism}",
                issue_type=issue_type,
                issue_family=FAMILY[issue_type],
                mechanism_key=mechanism,
                finding_count=len(members),
                independent_case_count=len(cases),
                eligible_for_insights=eligible,
                trace_ids=tuple(dict.fromkeys(str(member["trace_id"]) for member in members)),
                representative_evidence=tuple(examples),
                impact_status="not_established",
                impact_boundary="No impact is claimed beyond the directly observed tool-use issue.",
            )
        )
    return cards


def catalog_coverage(findings: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Return finding counts while keeping all nineteen catalog entries visible."""

    counts = Counter(str(finding["issue_type"]) for finding in findings)
    return {name: counts[name] for name in sorted(FINDING_TYPES)}


@dataclass(frozen=True)
class ToolIssueEvidenceArtifacts:
    findings: tuple[Mapping[str, Any], ...]
    cards: tuple[ToolIssueCard, ...]
    catalog_coverage: Mapping[str, int]
    config: ToolIssueConfig


class ToolIssueConfig(BaseModel):
    """Typed configuration owned by tool-issue analysis."""

    minimum_independent_cases: int = Field(default=CARD_MINIMUM_CASES, ge=1)
    retry_threshold: int = Field(default=RETRY_THRESHOLD, ge=1)
    include_audit_problems: bool = False


def problems_from_cards(
    cards: Sequence[ToolIssueCard], *, include_audit: bool = False
) -> tuple[Problem, ...]:
    """Project recurring tool-issue cards into candidate problems for synthesis."""

    problems: list[Problem] = []
    for card in cards:
        if not include_audit and not card.eligible_for_insights:
            continue
        representatives = card.representative_evidence
        trace_ids = tuple(dict.fromkeys(item.trace_id for item in representatives if item.trace_id))
        if not trace_ids:
            continue
        observation_parts = []
        for item in representatives:
            observation = (item.observation or "issue observed").rstrip(". ")
            observation_parts.append(f"{item.tool_name or 'unknown tool'}: {observation}")
        observations = "; ".join(observation_parts)
        problems.append(
            Problem(
                description=(
                    f"Tool issue `{card.issue_type}` with mechanism "
                    f"`{card.mechanism_key}` produced {card.finding_count} finding(s) "
                    f"across {card.independent_case_count} independent case(s). "
                    f"Representative observations: {observations}. "
                    f"{card.impact_boundary}"
                ),
                supporting_trace_ids=trace_ids,
                candidate_trace_ids=card.trace_ids,
            )
        )
    return tuple(problems)


def _input_text(value: object) -> str:
    if isinstance(value, str):
        return value

    values: Sequence[Any]
    if isinstance(value, Mapping):
        messages = value.get("messages")
        values = messages if isinstance(messages, Sequence) else ()
    elif isinstance(value, Sequence):
        values = value
    else:
        values = ()
    for message in reversed(values):
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
    return ""


def to_tool_issue_trace(trace: Trace) -> TraceRecord:
    """Project one normalized trace into the tool-issue analysis model."""

    catalogs: dict[str, ToolCatalog | None] = {}
    for roots, catalog in record_tool_catalogs(trace):
        schemas = (
            {
                name: dict(schema)
                if isinstance(schema, Mapping)
                else schema
                if isinstance(schema, bool)
                else None
                for name, schema in catalog.items()
            }
            if catalog is not None
            else None
        )
        pending = list(roots)
        while pending:
            span = pending.pop()
            pending.extend(span.children)
            catalogs[span.id] = schemas

    calls: list[CallRecord] = []
    prior_user_text = _input_text(trace.attributes.get("task_text"))
    for visit in walk_spans(trace):
        span = visit.span
        attributes = span.attributes
        observed_user_text = _input_text(span.input)
        if observed_user_text:
            prior_user_text = observed_user_text
        if span.kind is not SpanKind.TOOL:
            continue

        details = span.tool_call
        result_count = details.result_count if details is not None else 1
        call_user_text = details.prior_user_text if details is not None else None
        explicit_error = attributes.get("explicit_error")
        outcome_marker = attributes.get("outcome_marker")
        calls.append(
            CallRecord(
                trace_id=trace.id,
                call_index=(
                    details.index
                    if details is not None and details.index is not None
                    else len(calls)
                ),
                call_id=str(details.call_id if details and details.call_id else span.id),
                tool_name=str(span.tool_name or attributes.get("name") or ""),
                arguments=span.input,
                result=MISSING if result_count == 0 or span.output is UNSET else span.output,
                source_pointer=visit.source_pointer,
                result_id=details.result_id if details is not None else None,
                result_count=result_count,
                explicit_error=(
                    explicit_error
                    if isinstance(explicit_error, bool)
                    else True
                    if span.error is not None
                    else None
                ),
                outcome_marker=(str(outcome_marker) if outcome_marker is not None else span.error),
                instrumentation_alias_of=(
                    details.instrumentation_alias_of if details is not None else None
                ),
                prior_user_text=str(call_user_text or prior_user_text),
                tool_catalog=catalogs.get(span.id),
            )
        )

    orphan_results = trace.attributes.get("orphan_results")
    return TraceRecord(
        trace_id=trace.id,
        calls=tuple(calls),
        case_id=trace.case_id,
        orphan_results=tuple(dict(item) for item in orphan_results if isinstance(item, Mapping))
        if isinstance(orphan_results, Sequence)
        else (),
        complete_provenance_context=(trace.attributes.get("complete_provenance_context") is True),
    )


@dataclass(frozen=True)
class ToolIssueCompassSignal:
    name = "tool-issues"

    config: ToolIssueConfig = field(default_factory=ToolIssueConfig)

    def check_prerequisites(self, snapshot: TraceSnapshot) -> str | None:
        if not isinstance(self.config, ToolIssueConfig):
            raise TypeError("tool-issues requires ToolIssueConfig")
        if not any(
            visit.span.kind is SpanKind.TOOL for trace in snapshot for visit in walk_spans(trace)
        ):
            return "No tool calls"
        return None

    async def analyze(self, snapshot: TraceSnapshot) -> CompassSignalResult:
        traces = await asyncio.to_thread(lambda: [to_tool_issue_trace(trace) for trace in snapshot])
        return await asyncio.to_thread(self._analyze, traces)

    def _analyze(self, traces: Sequence[TraceRecord]) -> CompassSignalResult:
        findings = detect(
            traces,
            retry_threshold=self.config.retry_threshold,
        )
        cards = build_cards(
            findings,
            minimum_independent_cases=self.config.minimum_independent_cases,
        )
        problems = problems_from_cards(cards, include_audit=self.config.include_audit_problems)
        calls = [call for trace in traces for call in trace.calls]
        limited = []
        if calls:
            missing_results = sum(
                call.result is MISSING or call.result in (None, "", {}, []) for call in calls
            )
            if missing_results:
                limited.append(f"{missing_results} of {len(calls)} tool calls lack usable results")
            missing_catalogs = sum(call.tool_catalog is None for call in calls)
            if missing_catalogs:
                scope = "all" if missing_catalogs == len(calls) else f"{missing_catalogs} of"
                limited.append(
                    f"tool schemas missing for {scope} {len(calls)} tool call{'s' if len(calls) != 1 else ''}"
                )
        return CompassSignalResult(
            signal_name=self.name,
            problems=problems,
            finding_count=len(findings),
            limitations=tuple(limited),
            artifacts=ToolIssueEvidenceArtifacts(
                findings=tuple(findings),
                cards=tuple(cards),
                catalog_coverage=catalog_coverage(findings),
                config=self.config,
            ),
        )
