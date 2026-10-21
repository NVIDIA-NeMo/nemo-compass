# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export canonical traces as ATIF trajectories.

Traces loaded from ATIF are rebuilt exactly from the layout the ATIF loader records.
Every other trace is linearized deterministically: spans are visited depth-first in
``children`` order, and each information loss is recorded as a code in
``extra.compass.loss_codes``.

Tool-result error flags are written to ``observation.results[*].extra.is_error`` because
ATIF observation results forbid additional top-level fields; the ATIF loader reads the
flag back from that location.

Source contract: https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md
"""

from __future__ import annotations

import json
from pathlib import PurePosixPath
from urllib.parse import urlsplit

from pydantic import JsonValue

from trace_ingest.models import UNSET, Span, SpanKind, Trace

__all__ = ["ATIF_VERSION", "EXPORTER", "ATIFExportError", "to_atif"]

EXPORTER = "trace_ingest.exporters.atif@1"
ATIF_VERSION = "ATIF-v1.7"

_ROLE_SOURCES = {
    "agent": "agent",
    "assistant": "agent",
    "developer": "system",
    "human": "user",
    "system": "system",
    "user": "user",
}
_ORCHESTRATION = frozenset({SpanKind.AGENT, SpanKind.CHAIN})
_RESPONSE_ITEM_TYPES = frozenset({"message", "reasoning", "function_call", "function_call_output"})
_TEXT_PART_TYPES = frozenset({"text", "input_text", "output_text"})
_IMAGE_MEDIA_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
_SUFFIX_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


class ATIFExportError(ValueError):
    """A trace cannot be represented as an ATIF trajectory."""


def to_atif(
    trace: Trace, *, agent_name: str | None = None, agent_version: str | None = None
) -> dict[str, JsonValue]:
    """Return one ATIF trajectory for ``trace``.

    ``agent_name`` is required unless the trace was loaded from ATIF, whose recorded
    agent is reproduced unchanged.
    """

    root = _atif_root(trace)
    if root is not None:
        return _rebuild(root)
    if agent_name is None:
        raise ATIFExportError("agent_name is required for non-ATIF traces")
    return _Linearizer(trace).export(agent_name=agent_name, agent_version=agent_version)


# ATIF-origin traces ---------------------------------------------------------------------


def _atif_root(trace: Trace) -> Span | None:
    if len(trace.root_spans) != 1:
        return None
    root = trace.root_spans[0]
    raw = root.attributes.get("atif")
    if root.id != "trajectory" or not isinstance(raw, dict) or "schema_version" not in raw:
        return None
    return root


def _rebuild(span: Span) -> dict[str, JsonValue]:
    """Rebuild one trajectory from the ATIF loader's ``atif`` and ``atif_layout`` attributes."""

    trajectory: dict[str, JsonValue] = dict(_raw(span, "atif"))
    steps: list[JsonValue] = []
    for step_span in span.children:
        if step_span.id.startswith(f"{span.id}/step/"):
            steps.append(_rebuild_step(step_span))
    trajectory["steps"] = steps
    subagents = sorted(_subagent_spans(span), key=_subagent_index)
    if subagents:
        trajectory["subagent_trajectories"] = [_rebuild(child) for child in subagents]
    return trajectory


def _rebuild_step(span: Span) -> dict[str, JsonValue]:
    step: dict[str, JsonValue] = dict(_raw(span, "atif"))
    layout = _raw(span, "atif_layout")
    step["message"] = _recorded_message(span, agent=step.get("source") == "agent")
    tools = [child for child in span.children if child.kind is SpanKind.TOOL]
    if layout.get("tool_calls_key"):
        step["tool_calls"] = [_raw(tool, "atif")["call"] for tool in tools]
    observation = layout.get("observation")
    if observation is not None:
        if not isinstance(observation, dict):
            raise ATIFExportError(f"span {span.id!r} has an invalid ATIF observation layout")
        indexed: list[tuple[int, JsonValue]] = []
        for tool in tools:
            indexes = _raw(tool, "atif_layout").get("result_indexes")
            results = _raw(tool, "atif").get("results")
            if not isinstance(indexes, list) or not isinstance(results, list):
                raise ATIFExportError(f"span {tool.id!r} has an invalid ATIF result layout")
            indexed.extend(zip(indexes, results, strict=True))
        for child in span.children:
            if child.kind is SpanKind.CHAIN:
                index = _raw(child, "atif_layout").get("result_index")
                if not isinstance(index, int):
                    raise ATIFExportError(f"span {child.id!r} has an invalid ATIF result index")
                indexed.append((index, child.attributes.get("atif")))
        indexed.sort(key=lambda pair: pair[0])
        step["observation"] = {**observation, "results": [result for _, result in indexed]}
    return step


def _recorded_message(span: Span, *, agent: bool) -> JsonValue:
    """Return the step message the ATIF loader stored in the span input or output."""

    container = span.output if agent else span.input
    if not agent and isinstance(container, dict):
        messages = container.get("messages")
        container = messages[0] if isinstance(messages, list) and messages else None
    if not isinstance(container, dict) or "content" not in container:
        raise ATIFExportError(f"span {span.id!r} has no recoverable ATIF message")
    return container["content"]


def _subagent_spans(span: Span) -> list[Span]:
    """Return embedded subagent spans owned by ``span``, without entering nested subagents."""

    found: list[Span] = []
    pending = list(reversed(span.children))
    while pending:
        child = pending.pop()
        layout = child.attributes.get("atif_layout")
        if child.kind is SpanKind.AGENT and isinstance(layout, dict) and "subagent_index" in layout:
            found.append(child)
        else:
            pending.extend(reversed(child.children))
    return found


def _subagent_index(span: Span) -> int:
    index = _raw(span, "atif_layout").get("subagent_index")
    if not isinstance(index, int):
        raise ATIFExportError(f"span {span.id!r} has an invalid subagent index")
    return index


def _raw(span: Span, key: str) -> dict[str, JsonValue]:
    value = span.attributes.get(key)
    if not isinstance(value, dict):
        raise ATIFExportError(f"span {span.id!r} is missing ATIF {key} attributes")
    return value


# Generic linearization -------------------------------------------------------------------


class _Linearizer:
    def __init__(self, trace: Trace) -> None:
        self.trace = trace
        self.steps: list[dict[str, JsonValue]] = []
        self.loss_codes: set[str] = set()
        self.models: set[str] = set()

    def export(self, *, agent_name: str, agent_version: str | None) -> dict[str, JsonValue]:
        roots = self.trace.root_spans
        if not roots:
            raise ATIFExportError("trace has no recoverable user instruction")
        first = roots[0]
        if len(roots) > 1:
            self.loss_codes.add("multiple_root_spans_linearized")
        self.steps.extend(_instruction_steps(first, self.loss_codes))
        for root in roots:
            self._visit(root, depth=0)
        self._collect_models(roots)
        if first.output is not UNSET and first.output is not None:
            final = _message(first.output, self.loss_codes)
            if final != self._last_agent_message():
                step = _agent_step(first, message=final)
                if first.end_time is not None:
                    step["timestamp"] = first.end_time.isoformat()
                self.steps.append(step)
        for number, step in enumerate(self.steps, start=1):
            step["step_id"] = number

        agent: dict[str, JsonValue] = {"name": agent_name, "version": agent_version or "unknown"}
        if len(self.models) == 1:
            agent["model_name"] = next(iter(self.models))
        if first.kind is SpanKind.AGENT and first.tool_definitions is not UNSET:
            definitions = [d.raw for d in first.tool_definitions if isinstance(d.raw, dict)]
            if definitions:
                agent["tool_definitions"] = definitions
        return {
            "schema_version": ATIF_VERSION,
            "session_id": _logical_case_id(self.trace) or self.trace.id,
            "trajectory_id": self.trace.id,
            "agent": agent,
            "steps": self.steps,
            "extra": {
                "compass": {
                    "exporter": EXPORTER,
                    "trace_id": self.trace.id,
                    "trace_schema_version": self.trace.schema_version,
                    "loss_codes": sorted(self.loss_codes),
                }
            },
        }

    def _visit(self, span: Span, *, depth: int) -> None:
        emitted = self._emit(span, depth=depth)
        if emitted and depth > 1:
            self.loss_codes.add("span_tree_linearized")
        for child in span.children:
            self._visit(child, depth=depth + 1)

    def _emit(self, span: Span, *, depth: int) -> bool:
        if span.kind is SpanKind.TOOL:
            self.steps.append(self._tool_step(span))
            return True
        if span.kind is SpanKind.LLM:
            self.steps.append(_llm_step(span, self.loss_codes))
            return True
        if span.kind in _ORCHESTRATION:
            if span.children:
                if depth > 0:
                    self.loss_codes.add("orchestration_span_not_emitted")
                return False
            if span.output is UNSET:
                return False
            self.steps.append(_agent_step(span, message=_message(span.output, self.loss_codes)))
            return True
        self.loss_codes.add(f"{span.kind.value.lower()}_span_not_emitted")
        return False

    def _tool_step(self, span: Span) -> dict[str, JsonValue]:
        call_id = (span.tool_call.call_id if span.tool_call else None) or span.id
        if isinstance(span.input, dict):
            arguments: dict[str, JsonValue] = span.input
        elif span.input is not UNSET and span.input is not None:
            arguments = {"value": span.input}
        else:
            arguments = {}
        step = _agent_step(span, message="")
        step["llm_call_count"] = 0
        step["tool_calls"] = [
            {
                "tool_call_id": call_id,
                "function_name": span.tool_name or "unknown_tool",
                "arguments": arguments,
            }
        ]
        results: list[JsonValue] = []
        for result in span.tool_call.results if span.tool_call else []:
            entry: dict[str, JsonValue] = {
                "source_call_id": call_id,
                "content": ""
                if result.content is UNSET
                else _content(result.content, self.loss_codes),
            }
            is_error = result.is_error
            if is_error is None and span.error is not None:
                is_error = True
            if is_error is not None:
                entry["extra"] = {"is_error": is_error}
            results.append(entry)
        if results:
            step["observation"] = {"results": results}
        else:
            self.loss_codes.add("tool_result_missing")
        return step

    def _collect_models(self, spans: list[Span]) -> None:
        pending = list(spans)
        while pending:
            span = pending.pop()
            if span.model:
                self.models.add(span.model)
            pending.extend(span.children)

    def _last_agent_message(self) -> JsonValue:
        for step in reversed(self.steps):
            if step["source"] == "agent":
                return step["message"]
        return None


def _instruction_steps(root: Span, losses: set[str]) -> list[dict[str, JsonValue]]:
    value = root.input
    if value is UNSET or value is None or value in ("", [], {}):
        raise ATIFExportError("trace has no recoverable user instruction")
    if isinstance(value, str) and not value.strip():
        raise ATIFExportError("trace has no recoverable user instruction")
    messages = _message_records(value)
    if messages is None:
        return [_step(root, source="user", message=_content(value, losses))]
    sources = [_ROLE_SOURCES.get(_role(message), "system") for message in messages]
    last_user = max((i for i, source in enumerate(sources) if source == "user"), default=-1)
    if last_user < 0:
        raise ATIFExportError("trace has no recoverable user instruction")
    steps = []
    for index, (message, source) in enumerate(zip(messages, sources, strict=True)):
        content = message.get("content", message.get("message", message.get("text", "")))
        step = _step(root, source=source, message=_message(content, losses))
        step["is_copied_context"] = index != last_user
        steps.append(step)
    return steps


def _message_records(value: JsonValue) -> list[dict[str, JsonValue]] | None:
    candidate = value.get("messages") if isinstance(value, dict) else value
    if not isinstance(candidate, list) or not candidate:
        return None
    if not all(
        isinstance(item, dict) and ("role" in item or "source" in item) for item in candidate
    ):
        return None
    return candidate


def _role(message: dict[str, JsonValue]) -> str:
    return str(message.get("role") or message.get("source") or "").lower()


def _llm_step(span: Span, losses: set[str]) -> dict[str, JsonValue]:
    message = "" if span.output is UNSET else _message(span.output, losses)
    step = _agent_step(span, message=message)
    step["llm_call_count"] = 1
    if span.model:
        step["model_name"] = span.model
    metrics: dict[str, JsonValue] = {}
    if span.token_counts is not None:
        counts = span.token_counts
        metrics["prompt_tokens"] = counts.input_tokens + counts.cached_input_tokens
        metrics["cached_tokens"] = counts.cached_input_tokens
        metrics["completion_tokens"] = counts.output_tokens
    if span.cost_usd is not None:
        metrics["cost_usd"] = span.cost_usd
    if metrics:
        step["metrics"] = metrics
    return step


def _agent_step(span: Span, *, message: JsonValue) -> dict[str, JsonValue]:
    return _step(span, source="agent", message=message)


def _step(span: Span, *, source: str, message: JsonValue) -> dict[str, JsonValue]:
    step: dict[str, JsonValue] = {"step_id": 0}
    if span.start_time is not None:
        step["timestamp"] = span.start_time.isoformat()
    step["source"] = source
    step["message"] = message
    return step


def _logical_case_id(trace: Trace) -> str | None:
    value = trace.attributes.get("logical_case_id")
    return value if isinstance(value, str) and value else None


def _text(value: JsonValue) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _message(value: JsonValue, losses: set[str]) -> JsonValue:
    """Return the conversational content of a recorded message or model output.

    The result is ATIF message content: text, or a list of text and image parts.
    """

    if isinstance(value, dict):
        for key in ("content", "message", "text", "response", "answer", "output"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
            if key in ("content", "output") and isinstance(candidate, list):
                return _message(candidate, losses)
        choices = value.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            choice = choices[0]
            message = choice.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(choice.get("text"), str):
                return choice["text"]
    if isinstance(value, list) and value and all(_is_item(item) for item in value):
        return _items_message(value, losses)
    return _content(value, losses)


def _is_item(value: JsonValue) -> bool:
    """Whether ``value`` is a chat message or Responses API output item."""

    return isinstance(value, dict) and (
        "role" in value or value.get("type") in _RESPONSE_ITEM_TYPES
    )


def _items_message(items: list[JsonValue], losses: set[str]) -> JsonValue:
    """Keep assistant message content; tool calls are exported from TOOL spans instead."""

    parts: list[JsonValue] = []
    for item in items:
        assert isinstance(item, dict)
        if item.get("type") == "reasoning":
            losses.add("reasoning_not_exported")
            continue
        if item.get("role", "assistant") != "assistant" or item.get("type", "message") != "message":
            continue
        content = _content(item.get("content", ""), losses)
        parts.extend([{"type": "text", "text": content}] if isinstance(content, str) else content)  # ty: ignore[invalid-argument-type]
    return _joined(parts)


def _content(value: JsonValue, losses: set[str]) -> JsonValue:
    """Map recorded content to ATIF content, keeping typed images as image parts."""

    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        parts: list[JsonValue] = []
        for item in value:
            part = _content_part(item, losses)
            if part is None:
                return _text(value)
            parts.append(part)
        return _joined(parts)
    return _text(value)


def _content_part(item: JsonValue, losses: set[str]) -> dict[str, JsonValue] | None:
    if not isinstance(item, dict):
        return None
    kind = item.get("type")
    if kind in _TEXT_PART_TYPES and isinstance(item.get("text"), str):
        return {"type": "text", "text": item["text"]}
    if kind == "refusal" and isinstance(item.get("refusal"), str):
        return {"type": "text", "text": item["refusal"]}
    if kind == "image" and isinstance(item.get("source"), dict):
        source = item["source"]
        media_type, path = source.get("media_type"), source.get("path")
        if media_type in _IMAGE_MEDIA_TYPES and isinstance(path, str) and path:
            return {"type": "image", "source": {"media_type": media_type, "path": path}}
    elif kind in ("input_image", "image_url"):
        url = item.get("image_url")
        url = url.get("url") if isinstance(url, dict) else url
        media_type = _image_media_type(url)
        if isinstance(url, str) and media_type is not None:
            return {"type": "image", "source": {"media_type": media_type, "path": url}}
    else:
        return None
    # An image whose media type was not recorded cannot be represented in ATIF.
    losses.add("image_without_media_type_omitted")
    return {"type": "text", "text": "[image omitted: media type not recorded]"}


def _image_media_type(url: JsonValue) -> str | None:
    if not isinstance(url, str) or not url.strip():
        return None
    if url.startswith("data:"):
        media_type = url[5:].split(";", 1)[0].split(",", 1)[0]
    else:
        media_type = _SUFFIX_MEDIA_TYPES.get(PurePosixPath(urlsplit(url).path).suffix.lower())
    return media_type if media_type in _IMAGE_MEDIA_TYPES else None


def _joined(parts: list[JsonValue]) -> JsonValue:
    """Return plain text when every part is text, else the ATIF part list."""

    if all(isinstance(part, dict) and part.get("type") == "text" for part in parts):
        return "\n".join(part["text"] for part in parts)  # ty: ignore[invalid-argument-type, not-subscriptable]
    return parts
