# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build an agent's conversation turns from a recorded input and normalized spans.

The conversation is the recorded input's messages, then one model turn per LLM
span with the TOOL spans that follow it folded in, then a closing turn when the
recorded output differs from the last agent text. Re-sent history is labeled
``copied_context``; only an input's last user message, when no agent message or
tool result follows it, is the record's own.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, TypeGuard
from urllib.parse import urlsplit

from trace_ingest.models import (
    UNSET,
    AudioPart,
    Authorship,
    ImagePart,
    Loss,
    Span,
    SpanKind,
    TextPart,
    ToolResult,
    Trace,
    Turn,
    TurnRole,
)

IMAGE_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
_IMAGE_SUFFIXES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
_TEXT_KEYS = ("content", "message", "text", "response", "answer", "output")
_TOOL_CALL_BLOCKS = ("tool_call", "tool_use", "function_call")
_CONTENT_KEYS = ("content", "message", "text")
_LANGCHAIN_ROLES = {
    "HumanMessage": "user",
    "AIMessage": "assistant",
    "SystemMessage": "system",
    "ToolMessage": "tool",
}
_ROLES = {
    "user": TurnRole.USER,
    "human": TurnRole.USER,
    "assistant": TurnRole.AGENT,
    "ai": TurnRole.AGENT,
    "agent": TurnRole.AGENT,
    "model": TurnRole.AGENT,
    "system": TurnRole.SYSTEM,
    "developer": TurnRole.SYSTEM,
}
_TOOL_ROLES = frozenset({"tool", "function"})


class NoTurns(Exception):
    """The source records no conversation that the turn mapping can recover."""


def parse_json(value: object) -> object:
    """A JSON-encoded string decoded; any other value unchanged."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def has_input(value: object) -> bool:
    """Whether a value carries content rather than an absent or empty placeholder."""
    if value is UNSET or value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (dict, list, tuple)):
        return bool(value)
    return True


def text(value: object) -> str:
    """A string as is; any other value as sorted-key JSON."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def image_part(url: object) -> ImagePart | None:
    """An image referenced verbatim, typed from a data URI or the path suffix."""
    if not isinstance(url, str) or not url.strip():
        return None
    if url.startswith("data:"):
        media_type = url[5:].split(";", 1)[0].split(",", 1)[0]
    else:
        try:
            path = urlsplit(url).path
        except ValueError:  # e.g. an unbalanced IPv6 bracket
            return None
        media_type = _IMAGE_SUFFIXES.get(Path(path).suffix.lower())
    if media_type not in IMAGE_TYPES:
        return None
    return ImagePart(media_type=media_type, path=url)


def content_parts(value: object) -> list[TextPart | ImagePart | AudioPart] | None:
    """OpenAI chat content parts, when every part is text or an image with a known type."""
    if not isinstance(value, list) or not value:
        return None
    parts: list[TextPart | ImagePart | AudioPart] = []
    for part in value:
        if not isinstance(part, dict):
            return None
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            parts.append(TextPart(text=part["text"]))
            continue
        image = part.get("image_url")
        url = image.get("url") if isinstance(image, dict) else image
        image = image_part(url) if part.get("type") == "image_url" else None
        if image is None:
            return None
        parts.append(image)
    return parts


def _reply(value: object) -> object:
    """The message an output carries, unwrapped from its envelope.

    Unwraps a first choice or generation, a nested ``message``, the last of
    ``messages``, and LangChain's serialized constructor and ``{type, data}`` forms.
    """
    while True:
        if isinstance(value, list) and value and isinstance(value[0], dict):
            value = value[0]
            continue
        if not isinstance(value, dict):
            return value
        generations = value.get("generations")
        messages = value.get("messages")
        if value.get("type") == "constructor" and isinstance(value.get("kwargs"), dict):
            value = value["kwargs"]
        elif isinstance(value.get("type"), str) and isinstance(value.get("data"), dict):
            value = value["data"]
        elif isinstance(value.get("choices"), list) and value["choices"]:
            value = value["choices"][0]
        elif isinstance(generations, list) and generations and isinstance(generations[0], list):
            value = generations[0]
        elif isinstance(value.get("message"), dict):
            value = value["message"]
        elif isinstance(messages, list) and messages:
            value = messages[-1]
        else:
            return value


def _content_blocks(value: object) -> TypeGuard[list[dict[str, Any]]]:
    """Whether a value is a list of typed content blocks holding text or tool calls."""
    return (
        isinstance(value, list)
        and all(isinstance(block, dict) and isinstance(block.get("type"), str) for block in value)
        and any(block["type"] in ("text", *_TOOL_CALL_BLOCKS) for block in value)
    )


def message_text(value: object) -> str | None:
    """The text of a chat message, completion, generation or answer object; else its JSON.

    None for a reply without text: null content, or only tool calls. Content blocks
    (LangChain ``output_version="v1"``, Anthropic) give the text of their text blocks.
    """
    reply = _reply(value)
    if not isinstance(reply, dict):
        return text(reply)
    calls = bool(reply.get("tool_calls") or reply.get("function_call"))
    content = next((reply[key] for key in _TEXT_KEYS if isinstance(reply.get(key), str)), None)
    blocks = reply.get("content")
    if content is None and _content_blocks(blocks):
        content = "".join(
            block["text"]
            for block in blocks
            if block["type"] == "text" and isinstance(block.get("text"), str)
        )
        calls = calls or any(block["type"] in _TOOL_CALL_BLOCKS for block in blocks)
    if content is not None:
        return None if calls and not content else content
    if blocks is None and ("content" in reply or calls):
        return None
    return text(reply)


def _message_record(item: dict[str, Any]) -> dict[str, Any] | None:
    """The message with a resolved ``role``, or None when it names none."""
    if item.get("lc") == 1 and item.get("type") == "constructor":
        path, kwargs = item.get("id"), item.get("kwargs")
        if not isinstance(path, list) or not path or not isinstance(kwargs, dict):
            return None
        role = kwargs.get("role") if path[-1] == "ChatMessage" else _LANGCHAIN_ROLES.get(path[-1])
        if not isinstance(role, str) or not role:
            return None
        record = {"role": role, "content": kwargs.get("content")}
        if kwargs.get("tool_call_id") is not None:
            record["tool_call_id"] = kwargs["tool_call_id"]
        if kwargs.get("tool_calls"):
            record["tool_calls"] = kwargs["tool_calls"]
        return record
    data = item.get("data")
    if isinstance(item.get("type"), str) and isinstance(data, dict):
        # LangChain ``messages_to_dict``: the message's fields under ``data``.
        return _message_record({**data, "type": item["type"]})
    for key in ("role", "source", "type"):
        role = item.get(key)
        if isinstance(role, str) and role:
            return {**item, "role": role}
    return None


def message_records(value: object) -> list[dict[str, Any]] | None:
    """A chat input's messages, each with a resolved ``role``; None when not a message list.

    Accepts OpenAI messages, LangChain ``type`` messages and serialized LangChain
    constructors, optionally under ``messages`` and wrapped in a one-item batch.
    """
    candidate = value["messages"] if isinstance(value, dict) and "messages" in value else value
    if isinstance(candidate, list) and len(candidate) == 1 and isinstance(candidate[0], list):
        candidate = candidate[0]
    if not isinstance(candidate, list) or not candidate:
        return None
    records: list[dict[str, Any]] = []
    for item in candidate:
        record = _message_record(item) if isinstance(item, dict) else None
        if record is None:
            return None
        records.append(record)
    return records


def is_chat_shaped(value: object) -> bool:
    """Whether a value looks like a message list, even one ``message_records`` rejects."""
    if isinstance(value, dict):
        return "messages" in value
    return isinstance(value, list) and any(
        isinstance(item, dict) and ("role" in item or "source" in item or "type" in item)
        for item in value
    )


def input_turns(value: object) -> list[Turn]:
    """Turns for a recorded conversation input, with placeholder ids.

    Every message is copied context except the last user message when no agent
    message or tool result follows it; a reply after it shows an earlier call sent
    it. Tool results join the preceding agent turn. A non-chat input is one user
    turn holding its text.
    """
    messages = message_records(value)
    if messages is None:
        if is_chat_shaped(value):
            raise NoTurns  # an empty or malformed message list
        return [Turn(id="-", index=1, role=TurnRole.USER, content=text(value))]
    roles: list[TurnRole | None] = []
    for item in messages:
        source_role = str(item["role"]).lower()
        if source_role in _TOOL_ROLES:
            roles.append(None)
        elif source_role in _ROLES:
            roles.append(_ROLES[source_role])
        else:
            raise NoTurns
    if TurnRole.USER not in roles:
        raise NoTurns
    last_user = len(roles) - 1 - roles[::-1].index(TurnRole.USER)
    replied = any(role is not TurnRole.SYSTEM for role in roles[last_user + 1 :])
    own = None if replied else last_user
    turns: list[Turn] = []
    for index, (item, role) in enumerate(zip(messages, roles, strict=True)):
        key = next((key for key in _CONTENT_KEYS if key in item), None)
        if key is None and "parts" in item:
            raise NoTurns  # OpenTelemetry GenAI ``parts`` messages are not mapped.
        if role is None:
            result = ToolResult(
                content=UNSET if key is None else parse_json(item[key]),
                extra={"tool_call_id": item["tool_call_id"]}
                if isinstance(item.get("tool_call_id"), str)
                else {},
            )
            if turns and turns[-1].role is TurnRole.AGENT:
                turns[-1].unlinked_results.append(result)
            else:
                turns.append(
                    Turn(
                        id="-",
                        index=1,
                        role=TurnRole.AGENT,
                        authorship=Authorship.COPIED_CONTEXT,
                        unlinked_results=[result],
                    )
                )
            continue
        raw = None if key is None else item[key]
        content: list[TextPart | ImagePart | AudioPart] | str | None = content_parts(raw)
        if content is None and raw is not None:
            # A part list content_parts cannot map stays whole, not cut to its first part.
            content = text(raw) if isinstance(raw, list) else message_text(raw)
        turns.append(
            Turn(
                id="-",
                index=1,
                role=role,
                authorship=Authorship.UNKNOWN if index == own else Authorship.COPIED_CONTEXT,
                content=UNSET if content is None else content,
            )
        )
    return turns


def _walk(spans: Sequence[Span]) -> Iterator[Span]:
    pending = list(reversed(spans))
    while pending:
        span = pending.pop()
        yield span
        pending.extend(reversed(span.children))


def agent_turns(
    spans: Sequence[Span],
    *,
    owner_id: str,
    conversation_input: object,
    conversation_output: object,
) -> tuple[list[Turn], list[Loss]]:
    """The conversation owned by ``owner_id``: input turns, span turns, closing output.

    ``spans`` are the owner's children. Each LLM span is a model turn; TOOL spans
    join the latest model turn. Nested AGENT spans are not entered; their LLM and
    TOOL steps are reported as ``steps_outside_agent_not_in_turns``. Raises
    ``NoTurns`` when no conversation input was recorded.
    """
    if not has_input(conversation_input):
        raise NoTurns
    turns = input_turns(conversation_input)
    last_text = next(
        (
            turn.content
            for turn in reversed(turns)
            if turn.role is TurnRole.AGENT and isinstance(turn.content, str)
        ),
        None,
    )
    own: int | None = None
    outside: dict[str, list[str]] = {}
    pending = list(reversed(spans))
    while pending:
        span = pending.pop()
        if span.kind is SpanKind.AGENT:
            steps = [
                step.id
                for step in _walk(span.children)
                if step.kind in (SpanKind.LLM, SpanKind.TOOL)
            ]
            if steps:
                outside.setdefault(span.id, []).extend(steps)
            continue
        if span.kind is SpanKind.LLM:
            reply = (
                None if span.output is UNSET or span.output is None else message_text(span.output)
            )
            content = UNSET if reply is None else reply
            turns.append(
                Turn(
                    id="-",
                    index=1,
                    role=TurnRole.AGENT,
                    authorship=Authorship.MODEL,
                    content=content,
                    llm_span_ids=[span.id],
                )
            )
            own = len(turns) - 1
            last_text = "" if reply is None else reply
        elif span.kind is SpanKind.TOOL:
            if own is not None:
                turns[own].tool_call_span_ids.append(span.id)
            else:
                turns.append(
                    Turn(id="-", index=1, role=TurnRole.AGENT, tool_call_span_ids=[span.id])
                )
                own = len(turns) - 1
                last_text = ""
        pending.extend(reversed(span.children))
    if has_input(conversation_output):
        # A message-list output (e.g. LangGraph state) closes with its last message.
        records = message_records(conversation_output)
        closing = message_text(records[-1] if records else conversation_output)
        # A reply without text (e.g. only tool calls) is the last model turn, not an answer.
        if closing is not None and closing != last_text:
            turns.append(Turn(id="-", index=1, role=TurnRole.AGENT, content=closing))
    turns = [
        turn.model_copy(update={"id": f"{owner_id}/turn/{index}", "index": index})
        for index, turn in enumerate(turns, start=1)
    ]
    losses = [
        Loss(
            code="steps_outside_agent_not_in_turns",
            detail={"agent_span_id": owner_id, "span_ids": list(span_ids)},
            span_id=agent_id,
        )
        for agent_id, span_ids in outside.items()
    ]
    return turns, losses


def run_root(trace: Trace, *, orchestration_subtypes: frozenset[str] = frozenset()) -> Span | None:
    """The trace's single root when it orchestrates the run.

    An AGENT, CHAIN, or UNKNOWN root whose subtype is in ``orchestration_subtypes``
    qualifies, unless it records an LLM call.
    """
    if len(trace.root_spans) != 1:
        return None
    root = trace.root_spans[0]
    orchestrates = root.kind in (SpanKind.AGENT, SpanKind.CHAIN) or (
        root.kind is SpanKind.UNKNOWN and root.attributes.get("subtype") in orchestration_subtypes
    )
    return root if orchestrates and root.llm_call is UNSET else None


def promote_run_root(
    trace: Trace, *, orchestration_subtypes: frozenset[str] = frozenset()
) -> Span | None:
    """The trace's ``run_root`` as the AGENT span that owns the run.

    A non-AGENT root keeps its source kind as its subtype.
    """
    root = run_root(trace, orchestration_subtypes=orchestration_subtypes)
    if root is not None and root.kind is not SpanKind.AGENT:
        root.attributes["subtype"] = root.attributes.get("subtype") or root.kind.value
        root.kind = SpanKind.AGENT
    return root


def with_run_root_turns(
    trace: Trace,
    *,
    conversation_input: object,
    conversation_output: object,
    orchestration_subtypes: frozenset[str] = frozenset(),
) -> Trace:
    """Promote the trace's run root and attach its turns, when a conversation is recorded."""
    root = promote_run_root(trace, orchestration_subtypes=orchestration_subtypes)
    if root is None or root.turns is not UNSET:
        return trace
    try:
        root.turns, losses = agent_turns(
            root.children,
            owner_id=root.id,
            conversation_input=conversation_input,
            conversation_output=conversation_output,
        )
    except NoTurns:
        return trace  # No recoverable conversation: turns stay unset.
    trace.losses.extend(losses)
    return trace


__all__ = [
    "IMAGE_TYPES",
    "NoTurns",
    "agent_turns",
    "content_parts",
    "has_input",
    "image_part",
    "input_turns",
    "is_chat_shaped",
    "message_records",
    "message_text",
    "parse_json",
    "promote_run_root",
    "run_root",
    "text",
    "with_run_root_turns",
]
