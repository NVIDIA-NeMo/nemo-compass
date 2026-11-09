# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Present framework operations without changing execution identity."""

import json
import logging
from collections.abc import Sequence
from operator import attrgetter

from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import SpanContext

_logger = logging.getLogger(__name__)
# Bound serialized requests before compression, below the SDK's whole-batch rejection limit.
_MAX_BATCH_BYTES = 5 * 1024 * 1024

_OPERATION_NAMES = {
    "nooa.tracing": {"generation": "agent.execute", "context_snapshot": "agent.context"},
    "openinference.instrumentation.litellm": {
        "acompletion": "llm.chat_completion",
        "completion": "llm.chat_completion",
        "aresponses": "llm.responses",
        "responses": "llm.responses",
    },
}


class _ExportSpan(ReadableSpan):
    """Read-only name and tag overrides; preserve the original SDK span."""

    def __init__(self, original: ReadableSpan, name: str, tags: Sequence[str]) -> None:
        self._original = original
        self._export_name = name
        self._export_attributes = dict(original.attributes or {})
        if tags:
            self._export_attributes["langfuse.trace.tags"] = tuple(tags)

    name = property(attrgetter("_export_name"))
    context = property(attrgetter("_original.context"))
    parent = property(attrgetter("_original.parent"))
    resource = property(attrgetter("_original.resource"))
    attributes = property(attrgetter("_export_attributes"))
    events = property(attrgetter("_original.events"))
    links = property(attrgetter("_original.links"))
    kind = property(attrgetter("_original.kind"))
    instrumentation_info = property(attrgetter("_original.instrumentation_info"))
    instrumentation_scope = property(attrgetter("_original.instrumentation_scope"))
    status = property(attrgetter("_original.status"))
    start_time = property(attrgetter("_original.start_time"))
    end_time = property(attrgetter("_original.end_time"))
    dropped_attributes = property(attrgetter("_original.dropped_attributes"))
    dropped_events = property(attrgetter("_original.dropped_events"))
    dropped_links = property(attrgetter("_original.dropped_links"))

    def to_json(self, indent: int | None = 4) -> str:
        document = json.loads(self._original.to_json(indent=indent))
        document["name"] = self.name
        document["attributes"] = self._export_attributes
        return json.dumps(document, indent=indent)

    def get_span_context(self) -> SpanContext | None:
        return self._original.get_span_context()


def _with_operation_name(span: ReadableSpan, tags: Sequence[str]) -> ReadableSpan:
    """Name the API layer, rather than suggesting nested SDK calls are extra requests."""
    scope = span.instrumentation_scope.name if span.instrumentation_scope else ""
    name = _OPERATION_NAMES.get(scope, {}).get(span.name)
    if scope == "nooa.tracing" and span.name == "generation":
        method = (span.attributes or {}).get("agent.method")
        if isinstance(method, str) and method:
            name = f"agent.execute.{method}"
    if scope == "nooa.tracing" and span.name == "context_snapshot":
        is_diff = (span.attributes or {}).get("nooa.system_message.is_diff")
        if isinstance(is_diff, bool):
            name = "agent.context_diff" if is_diff else "agent.context_snapshot"
    return _ExportSpan(span, name=name or span.name, tags=tags) if name or tags else span


class CompassSpanExporter(SpanExporter):
    """Forward readable operations and Langfuse tags in bounded OTLP batches."""

    def __init__(self, delegate: SpanExporter, tags: Sequence[str] = ()) -> None:
        self._delegate = delegate
        self._tags = tuple(tags)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self._export_batch(tuple(_with_operation_name(span, self._tags) for span in spans))

    def _export_batch(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        size = encode_spans(spans).ByteSize()
        if size <= _MAX_BATCH_BYTES:
            return self._delegate.export(spans)
        if len(spans) == 1:
            _logger.error(
                "Cannot export span %s: %d bytes exceeds the %d-byte batch limit",
                spans[0].name,
                size,
                _MAX_BATCH_BYTES,
            )
            return SpanExportResult.FAILURE
        middle = len(spans) // 2
        left = self._export_batch(spans[:middle])
        right = self._export_batch(spans[middle:])
        if left is SpanExportResult.FAILURE or right is SpanExportResult.FAILURE:
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self._delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._delegate.force_flush(timeout_millis)
