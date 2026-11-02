# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Present framework operations without changing execution identity."""

import hashlib
import json
import logging
from collections.abc import Sequence
from operator import attrgetter
from threading import Lock

from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import SpanContext

logger = logging.getLogger(__name__)

# Leave room below the OTLP exporter's 64 MiB ceiling and bound each HTTP request.
MAX_BATCH_BYTES = 16 * 1024 * 1024

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
    """Read-only name override; delegate all other fields to the ended SDK span."""

    def __init__(self, original: ReadableSpan, name: str, attributes: dict | None = None) -> None:
        self._original = original
        self._export_name = name
        self._export_attributes = original.attributes if attributes is None else attributes

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
        document["attributes"] = dict(self.attributes or {})
        return json.dumps(document, indent=indent)

    def get_span_context(self) -> SpanContext | None:
        return self._original.get_span_context()


def _with_operation_name(span: ReadableSpan) -> ReadableSpan:
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
    return _ExportSpan(span, name=name) if name else span


def span_id_digest(span_ids: set[int]) -> str:
    """Fingerprint an inventory independently of export or arrival order."""
    return hashlib.sha256(
        "".join(f"{span_id:016x}" for span_id in sorted(span_ids)).encode()
    ).hexdigest()


class SpanInventory(SpanProcessor):
    """Count spans before the batch queue, so queue losses remain detectable."""

    def __init__(self) -> None:
        self.started: set[int] = set()
        self.ended: set[int] = set()
        self._lock = Lock()

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        with self._lock:
            self.started.add(span.get_span_context().span_id)

    def on_end(self, span: ReadableSpan) -> None:
        if span.context:
            with self._lock:
                self.ended.add(span.context.span_id)

    def metadata(self) -> dict[str, int | str]:
        with self._lock:
            return {
                "expected_span_count": len(self.started),
                "expected_span_id_sha256": span_id_digest(self.started),
            }


def _request_size(spans: Sequence[ReadableSpan]) -> int:
    return encode_spans(spans).ByteSize()


def _bound_span(span: ReadableSpan, max_bytes: int) -> ReadableSpan:
    """Summarize only oversized attribute values, keeping identity and usage intact.

    Normal spans are lossless. A single oversized span cannot be split into
    separate requests without changing its identity. Replace its largest values
    with explicit size/hash summaries until it fits; never slice JSON mid-value.
    """
    attributes = dict(span.attributes or {})
    sizes = sorted(
        (
            (len(json.dumps(value, ensure_ascii=False).encode()), key)
            for key, value in attributes.items()
        ),
        reverse=True,
    )
    summarized = []
    for size, key in sizes:
        if size < 1024:
            break  # Small execution metadata and token counts are never sacrificed.
        value = attributes[key]
        data = (
            value.encode()
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False).encode()
        )
        attributes[key] = json.dumps(
            {
                "compass_telemetry_omitted": True,
                "original_utf8_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        summarized.append(key)
        attributes["compass.telemetry.summarized_attributes"] = tuple(summarized)
        bounded = _ExportSpan(span, span.name, attributes)
        if _request_size((bounded,)) <= max_bytes:
            logger.warning(
                "Summarized oversized telemetry attributes on span %s: %s",
                span.name,
                ", ".join(summarized),
            )
            return bounded
    raise ValueError(
        "Span exceeds telemetry request budget even after summarizing large attributes"
    )


class CompassSpanExporter(SpanExporter):
    """Preserve span identity while exporting requests within a byte budget."""

    def __init__(self, delegate: SpanExporter, *, max_batch_bytes: int = MAX_BATCH_BYTES) -> None:
        if max_batch_bytes <= 0:
            raise ValueError("max_batch_bytes must be positive")
        self._delegate = delegate
        self._max_batch_bytes = max_batch_bytes
        self.exported_span_ids: set[int] = set()
        self.failed_span_count = 0

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if not spans:
            return SpanExportResult.SUCCESS
        return self._export_batch(tuple(_with_operation_name(span) for span in spans))

    def _export_batch(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            if _request_size(spans) > self._max_batch_bytes:
                if len(spans) > 1:
                    middle = len(spans) // 2
                    left = self._export_batch(spans[:middle])
                    right = self._export_batch(spans[middle:])
                    return (
                        SpanExportResult.SUCCESS
                        if left == right == SpanExportResult.SUCCESS
                        else SpanExportResult.FAILURE
                    )
                spans = (_bound_span(spans[0], self._max_batch_bytes),)
            result = self._delegate.export(spans)
        except Exception as error:
            # Transport errors can contain credentials. Keep only the exception type.
            logger.error("Compass telemetry export failed (%s)", type(error).__name__)
            result = SpanExportResult.FAILURE
        if result == SpanExportResult.SUCCESS:
            self.exported_span_ids.update(span.context.span_id for span in spans if span.context)
        else:
            self.failed_span_count += len(spans)
            logger.error("Compass telemetry incomplete: failed to export %d spans", len(spans))
        return result

    def shutdown(self) -> None:
        self._delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._delegate.force_flush(timeout_millis)
