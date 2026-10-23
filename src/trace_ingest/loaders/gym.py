# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read Gym rollout JSON/JSONL: ``ng_trajectory`` attachments or Responses envelopes.

Source: NVIDIA-NeMo/Gym@399e6783e0e879424fc20a23f8e2d46ffa4cfdee,
nemo_gym/rollout_observability.py (TrajectoryRecord), rollout_collection.py.
Supported producer paths and required observability configuration:
https://docs.nvidia.com/nemo/gym/reference/trajectory-capabilities/

Rollouts with ng_trajectory schema 1.0 keep the invocation tree, captured model
calls, per-call timing and semantic turns. Coverage is path-dependent; recorded
gaps and partial evidence are preserved.

Records without ``ng_trajectory`` (absent or null) load from the OpenAI
Responses shape instead: a rollout envelope (``responses_create_params`` plus
``response``) or a standalone Responses object (``object == "response"``):

- One AGENT root. ``input`` is the request's message items as a list: string
  input becomes ``[{"role": "user", "content": input}]`` and ``instructions``
  a leading ``{"role": "system", ...}`` message; non-message items are omitted
  there and stay in the raw record. ``output`` is the text of the last
  assistant message in the generated output, else UNSET. Tool definitions come
  from ``responses_create_params.tools``, else ``response.tools``.
- One LLM span for the recorded response: ``model`` from ``response.model``
  (else the request model), ``output`` the response output items, and token
  counts mapped from ``usage`` exactly like ``token_stats`` (input, cached
  input and output tokens must all be present). The response can aggregate
  several model turns of an agent loop; per-call grouping and timing are not
  inferred, and no span has timestamps. The TOOL spans follow the LLM span as
  siblings, so the span tree does not show how messages, model turns and tool
  calls interleaved; ``item_path`` and the LLM output keep the source order.
  ``error`` is ``response.error.code`` or a failed/cancelled status.
- One TOOL span per ``function_call``, paired with ``function_call_output`` by
  ``call_id`` over the request input followed by the generated output, in
  source order. Arguments are decoded JSON, or the raw string when undecodable.
  Unpaired calls keep ``result_count == 0``; duplicate calls, duplicate
  results and results without a preceding call are rejected. Tool results have
  no authoritative error flag, so ``ToolResult.is_error`` is None.
- When ``response.output`` starts with an exact copy of the request input
  (ignoring message ``id``/``status``), the copy is counted once. Other
  prompt-role items in the output, remote conversation state
  (``previous_response_id``/``conversation`` on the request or the response),
  namespaced functions and item types other than message, reasoning,
  function_call and function_call_output are rejected.
- Identity: ``"gym-" + sha256(record bytes)[:16]`` where the record bytes are
  the JSONL line or whole JSON file with surrounding ASCII whitespace stripped,
  so byte-identical records collide and are rejected as duplicates.
  ``_ng_task_index`` sets ``logical_case_id`` to ``gym-task-index-<n>``.
  The gaps ``ng_trajectory_unavailable`` and ``responses_output_aggregated``
  record the missing attachment and the lost interleaving.

ATIF conversion is outside this loader's contract. No Gym runtime, inference,
media fetching, or filesystem reference resolution.
"""

from __future__ import annotations

import hashlib
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
    ToolDefinition,
    ToolResult,
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
    Records carry ``ng_trajectory`` or the Responses rollout envelope.
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


def _tool_definitions(raw: object) -> list[ToolDefinition]:
    """Map ``responses_create_params.tools`` items; skip nameless or non-object items."""

    definitions: list[ToolDefinition] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        function = function if isinstance(function, dict) else {}
        name = item.get("name") or function.get("name")
        if not isinstance(name, str) or not name:
            continue
        description = item.get("description") or function.get("description")
        definitions.append(
            ToolDefinition(
                name=name,
                description=description if isinstance(description, str) else None,
                input_schema=item["parameters"]
                if "parameters" in item
                else function.get("parameters"),
                raw=item,
            )
        )
    return definitions


def _finalize_tool_results(spans: list[Span]) -> None:
    """Record one typed result per counted tool output once errors are known."""

    for span in spans:
        if span.tool_call is not None:
            span.tool_call.results = (
                [ToolResult(content=span.output, is_error=True if span.error else None)]
                if span.tool_call.result_count
                else []
            )
        _finalize_tool_results(span.children)


def _conversation_tools(
    items: list[tuple[dict[str, Any], dict[str, Any]]],
    prefix: str,
    *,
    raw_unparsed_arguments: bool = False,
) -> dict[str, Span]:
    """Pair calls/results in ``(item, location)`` order by scoped call IDs.

    A missing result remains UNSET; an explicit result with null output remains
    a matched result. Observations may enrich these spans later with timing.
    ``raw_unparsed_arguments`` keeps unparseable argument text verbatim and any
    decoded JSON value; ``ng_trajectory`` conversations require a JSON object.
    """

    calls: dict[str, Span] = {}
    user: str | None = None
    for item, location in items:
        if item.get("type", "message") not in (
            "message",
            "reasoning",
            "function_call",
            "function_call_output",
        ):
            raise GymTraceLoadError("unsupported Gym conversation item; add an explicit mapping")
        if item.get("role") == "user":
            user = _text(item.get("content"), ("input_text", "text"))
        if item.get("type") == "function_call":
            if item.get("namespace") is not None:
                raise GymTraceLoadError("namespaced Gym functions require an explicit name mapping")
            call_id = _id(item.get("call_id"))
            if call_id in calls:
                raise ValueError("duplicate invocation-scoped tool call ID")
            arguments = item.get("arguments", UNSET)
            if isinstance(arguments, str):
                if not raw_unparsed_arguments:
                    arguments = _object(_decode(arguments))
                else:
                    try:
                        arguments = _decode(arguments)
                    except (ValueError, RecursionError):
                        pass
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
                attributes={"gym": {"call": item, **location}},
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
            span.attributes["gym_result"] = {"item": item, **location}
    return calls


def _text(content: object, part_types: tuple[str, ...]) -> str:
    """Return string content unchanged, else join text parts of ``part_types``."""

    if isinstance(content, str):
        return content
    return "\n".join(
        part["text"]
        for part in _rows(content)
        if part.get("type") in part_types and isinstance(part.get("text"), str)
    )


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


_PROMPT_ROLES = ("user", "system", "developer")


def _message_signature(item: object) -> object:
    """Compare message items without volatile wrapper fields; other items stay exact."""

    if not isinstance(item, dict) or item.get("type", "message") != "message" or "role" not in item:
        return item
    return {
        **{
            key: value
            for key, value in item.items()
            if key not in ("id", "status") and not (key == "phase" and value is None)
        },
        "type": "message",
    }


def _normalize_responses(record: dict[str, Any], raw: bytes) -> Trace:
    """Map a rollout envelope or standalone Responses object without ``ng_trajectory``.

    One AGENT root holds one LLM span for the recorded response and one TOOL
    span per ``function_call``. Results pair by ``call_id`` over the request
    input followed by the generated response output.
    """

    if "response" in record:
        request, response, shape = (
            record.get("responses_create_params"),
            record["response"],
            ("rollout_envelope"),
        )
        if not isinstance(request, dict):
            raise GymTraceLoadError(
                "Gym rollout without ng_trajectory requires responses_create_params and "
                "response; see https://docs.nvidia.com/nemo/gym/reference/trajectory-capabilities/"
            )
    elif record.get("object") == "response":
        request, response, shape = {}, record, "responses_object"
    else:
        raise GymTraceLoadError(
            "Gym record requires ng_trajectory schema 1.0, a rollout envelope, "
            "or a standalone Responses object"
        )
    response = _object(response)
    output = _rows(response.get("output"))
    if any(
        source.get("previous_response_id") or source.get("conversation")
        for source in (request, response)
    ):
        raise GymTraceLoadError(
            "Gym record references remote conversation state; supply self-contained input"
        )
    incoming = request.get("input", [])
    if isinstance(incoming, str):
        incoming = [{"role": "user", "content": incoming}]
    incoming = _rows(incoming)
    instructions = request.get("instructions")
    if instructions is not None and not isinstance(instructions, str):
        raise ValueError("request instructions must be text")

    # Some agents echo the request input as an exact prefix of response.output;
    # count those items once. Any other prompt-role output is ambiguous history.
    echoed = len(output) >= len(incoming) and all(
        _message_signature(a) == _message_signature(b)
        for a, b in zip(incoming, output[: len(incoming)], strict=True)
    )
    generated = output[len(incoming) :] if echoed else output
    if any(item.get("role") in _PROMPT_ROLES for item in generated):
        raise GymTraceLoadError(
            "Gym response output contains prompt-role items beyond an exact request prefix"
        )
    items = [
        (item, {"item_path": f"$.responses_create_params.input[{i}]"})
        for i, item in enumerate(incoming)
    ]
    offset = len(incoming) if echoed else 0
    items += [
        (item, {"item_path": f"$.response.output[{offset + i}]"})
        for i, item in enumerate(generated)
    ]
    tools = _conversation_tools(items, "gym/response", raw_unparsed_arguments=True)

    root_input = [item for item in incoming if item.get("type", "message") == "message"]
    if instructions:
        root_input.insert(0, {"role": "system", "content": instructions})
    final = next(
        (
            _text(item.get("content"), ("output_text", "text"))
            for item in reversed(generated)
            if item.get("type", "message") == "message" and item.get("role") == "assistant"
        ),
        None,
    )
    usage = response.get("usage")
    token_counts = None
    if usage is not None:
        usage = _object(usage)
        cached = usage.get("input_tokens_details")
        token_counts = _tokens(
            {
                "prompt_tokens": usage.get("input_tokens"),
                "cached_tokens": _object(cached).get("cached_tokens")
                if cached is not None
                else None,
                "completion_tokens": usage.get("output_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
        )
    model = response.get("model") or request.get("model")
    # Responses failures carry ``error: {code, message}``; status covers the rest.
    error = response.get("error")
    if error is not None:
        code = _object(error).get("code")
        error = code if isinstance(code, str) and code else "response_error"
    llm = Span(
        id="gym/response/model",
        kind=SpanKind.LLM,
        input=request if shape == "rollout_envelope" else UNSET,
        output=response["output"],
        model=model if isinstance(model, str) and model else None,
        token_counts=token_counts,
        error=error or _error(response),
        attributes={
            "gym_response_id": response.get("id"),
            "gym_response_status": response.get("status"),
        },
    )
    root = Span(
        id="gym/response",
        kind=SpanKind.AGENT,
        input=root_input,
        output=final if final else UNSET,
        children=[llm, *tools.values()],
    )
    _finalize_tool_results([root])
    definitions = _tool_definitions(request.get("tools", response.get("tools")))
    if definitions:
        root.tool_definitions = definitions
    identity = "gym-" + hashlib.sha256(raw.strip()).hexdigest()[:16]
    task_index = record.get("_ng_task_index")
    attributes: dict[str, Any] = {
        "gym": record,
        "gym_source_shape": shape,
        "gym_loader_gaps": [
            {"code": "ng_trajectory_unavailable"},
            {"code": "responses_output_aggregated"},
        ],
        "gym_gap_count": 2,
    }
    if type(task_index) is int:
        attributes["logical_case_id"] = f"gym-task-index-{task_index}"
    return Trace(
        id=identity,
        root_spans=[root],
        aggregate=TraceAggregate(),
        evaluator_results={
            f"gym.{key}": record[key] for key in ("reward", "reward_components") if key in record
        },
        attributes=attributes,
    )


def _normalize(record: dict[str, Any], raw: bytes) -> Trace:
    """Map one native attachment while retaining the complete source envelope.

    Rollout IDs identify traces; task IDs group logical cases. Invocations,
    captured calls, tools and semantic turns map to separate span kinds.
    Source gaps remain under ``attributes.gym``; additional join gaps are in
    ``gym_loader_gaps``. Rewards stay evaluation signals, not execution errors.
    Records without (or with null) ``ng_trajectory`` use ``_normalize_responses``.
    """

    if record.get("ng_trajectory") is None:
        return _normalize_responses(record, raw)
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
        tools[identifier] = _conversation_tools(
            [
                (item, {"conversation_index": index})
                for index, item in enumerate(_rows(invocation.get("conversation", [])))
            ],
            span.id,
        )

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
    _finalize_tool_results(roots)
    params = record.get("responses_create_params")
    definitions = _tool_definitions(params.get("tools") if isinstance(params, dict) else None)
    if definitions:
        depth_first: list[Span] = []
        pending = list(roots)
        while pending:
            item = pending.pop(0)
            depth_first.append(item)
            pending[:0] = item.children
        owner = next(
            (
                item
                for kind in (SpanKind.AGENT, SpanKind.LLM)
                for candidates in (roots, depth_first)
                for item in candidates
                if item.kind is kind
            ),
            None,
        )
        if owner is None:
            extra_gaps.append({"code": "tool_definitions_without_owner_span"})
        else:
            owner.tool_definitions = definitions
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
                    trace = _normalize(_object(_decode(raw)), raw)
                    trace.source_url = file_source_url(
                        path, _line_number if self.config.format == "jsonl" else None
                    )
                    pending = list(trace.root_spans)
                    while pending:
                        span = pending.pop()
                        calls += span.kind is SpanKind.TOOL
                        pending.extend(span.children)
                    cases.add(str(trace.attributes.get("logical_case_id", trace.id)))
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
