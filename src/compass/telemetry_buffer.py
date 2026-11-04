# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep completed spans on disk until their parents can be exported first."""

import logging
import pickle
import tempfile
import zlib
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path

from opentelemetry.attributes import BoundedAttributes
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.trace import Link
from opentelemetry.util.types import Attributes

logger = logging.getLogger(__name__)


class _RestoredSpan(ReadableSpan):
    def __init__(self, dropped: tuple[int, int, int], values: dict) -> None:
        super().__init__(**values)
        self._dropped = dropped

    dropped_attributes = property(lambda self: self._dropped[0])
    dropped_events = property(lambda self: self._dropped[1])
    dropped_links = property(lambda self: self._dropped[2])


def _attributes(values: Attributes, dropped: int = 0) -> BoundedAttributes:
    attributes = BoundedAttributes(maxlen=None, attributes=values)
    attributes.dropped = dropped
    return attributes


def _pack_span(span: ReadableSpan) -> bytes:
    # Store public values, not SDK locks or live mutable attribute containers.
    scope = span.instrumentation_scope
    data = {
        "name": span.name,
        "context": span.context,
        "parent": span.parent,
        "kind": span.kind,
        "status": span.status,
        "start_time": span.start_time,
        "end_time": span.end_time,
        "attributes": dict(span.attributes or {}),
        "resource": (dict(span.resource.attributes), span.resource.schema_url),
        "instrumentation_scope": (
            (scope.name, scope.version, scope.schema_url, dict(scope.attributes or {}))
            if scope
            else None
        ),
        "events": [
            (e.name, e.timestamp, dict(e.attributes or {}), e.dropped_attributes)
            for e in span.events
        ],
        "links": [
            (link.context, dict(link.attributes or {}), link.dropped_attributes)
            for link in span.links
        ],
        "dropped": (span.dropped_attributes, span.dropped_events, span.dropped_links),
    }
    return zlib.compress(pickle.dumps(data, protocol=5), level=1)


def _unpack_span(data: bytes) -> ReadableSpan:
    # These files are created by this process in its private 0700 directory.
    # They are not an import format for externally supplied telemetry.
    values = pickle.loads(zlib.decompress(data))
    values["resource"] = Resource(*values["resource"])
    if scope := values["instrumentation_scope"]:
        values["instrumentation_scope"] = InstrumentationScope(*scope)
    values["events"] = [
        Event(name, _attributes(attributes, dropped), timestamp)
        for name, timestamp, attributes, dropped in values["events"]
    ]
    values["links"] = [
        Link(context, _attributes(attributes, dropped))
        for context, attributes, dropped in values["links"]
    ]
    return _RestoredSpan(values.pop("dropped"), values)


class BufferedSpanExporter(SpanExporter):
    """Export completed parents before children without retaining payloads in RAM.

    An OTLP acknowledgement can precede persistence. In particular, children
    waiting for a long-running root may disappear inside the receiving service.
    Buffer until shutdown and send each parent before any of its children.
    Failed records stay on disk; successful records are removed after delivery.
    """

    def __init__(self, delegate: SpanExporter) -> None:
        self._delegate = delegate
        self.directory = Path(tempfile.mkdtemp(prefix="compass-telemetry-"))
        self._parents: dict[int, int | None] = {}
        self._exported: set[int] = set()
        self._write_failed = False

    @property
    def _incomplete(self) -> bool:
        return self._write_failed or bool(self._parents.keys() - self._exported)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        result = SpanExportResult.SUCCESS
        for span in spans:
            try:
                if span.context is None:
                    raise ValueError("Cannot buffer a span without its identity")
                span_id = span.context.span_id
                path = self.directory / f"{span_id:016x}.span"
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(_pack_span(span))
                temporary.replace(path)
                self._parents[span_id] = span.parent.span_id if span.parent else None
            except Exception as error:
                self._write_failed = True
                logger.error("Compass telemetry buffering failed (%s)", type(error).__name__)
                result = SpanExportResult.FAILURE
        return result

    def _parent_first_levels(self) -> Iterator[list[int]]:
        children: dict[int | None, list[int]] = defaultdict(list)
        for span_id, parent_id in self._parents.items():
            children[parent_id if parent_id in self._parents else None].append(span_id)
        pending = children[None]
        while pending:
            yield pending
            pending = [child for span_id in pending for child in children[span_id]]

    def _deliver(self) -> None:
        # Batch siblings, but never a parent and child together: the receiver
        # may process spans within one request in parallel.
        for level in self._parent_first_levels():
            eligible = [
                span_id
                for span_id in level
                if span_id not in self._exported
                and (
                    self._parents[span_id] not in self._parents
                    or self._parents[span_id] in self._exported
                )
            ]
            for offset in range(0, len(eligible), 8):
                paths = [
                    self.directory / f"{span_id:016x}.span"
                    for span_id in eligible[offset : offset + 8]
                ]
                try:
                    result = self._delegate.export(
                        [_unpack_span(path.read_bytes()) for path in paths]
                    )
                    if result == SpanExportResult.SUCCESS:
                        for path in paths:
                            self._exported.add(int(path.stem, 16))
                            path.unlink()
                except Exception as error:
                    logger.error("Compass telemetry replay failed (%s)", type(error).__name__)

    def shutdown(self) -> None:
        try:
            self._deliver()
            if self._incomplete:
                self._deliver()  # Retry transient failures without repeating acknowledged spans.
            if self._incomplete:
                logger.error("Compass telemetry incomplete; retained spans in %s", self.directory)
            else:
                self.directory.rmdir()
        finally:
            self._delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        # SDK batch flushing writes completed spans to disk. Delivering here
        # could send children while their parent is still running.
        return True
