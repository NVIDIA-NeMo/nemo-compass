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

from compass.telemetry_export import CompassSpanExporter

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

    def __init__(self, delegate: CompassSpanExporter, *, archive_dir: Path | None = None) -> None:
        self._delegate = delegate
        if archive_dir is not None:
            archive_dir.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix="compass-telemetry-", dir=archive_dir))
        self._retain_archive = archive_dir is not None
        self._parents: dict[int, int | None] = {}
        self._write_failures = 0

    @property
    def exported_span_ids(self) -> set[int]:
        return self._delegate.exported_span_ids

    @property
    def failed_span_count(self) -> int:
        return self._write_failures + len(self._parents.keys() - self.exported_span_ids)

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
                self._write_failures += 1
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
                if span_id not in self.exported_span_ids
                and (
                    self._parents[span_id] not in self._parents
                    or self._parents[span_id] in self.exported_span_ids
                )
            ]
            for offset in range(0, len(eligible), 8):
                paths = [
                    self.directory / f"{span_id:016x}.span"
                    for span_id in eligible[offset : offset + 8]
                ]
                try:
                    self._delegate.export([_unpack_span(path.read_bytes()) for path in paths])
                    for path in paths:
                        if (
                            not self._retain_archive
                            and int(path.stem, 16) in self.exported_span_ids
                        ):
                            path.unlink()
                except Exception as error:
                    logger.error("Compass telemetry replay failed (%s)", type(error).__name__)

    def shutdown(self) -> None:
        try:
            self._deliver()
            if self.failed_span_count:
                self._deliver()  # Retry transient failures without repeating acknowledged spans.
            if self.failed_span_count:
                logger.error("Compass telemetry incomplete; retained spans in %s", self.directory)
            elif not self._retain_archive:
                self.directory.rmdir()
            if self._retain_archive:
                logger.warning("Compass telemetry archive retained in %s", self.directory)
        finally:
            self._delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        # SDK batch flushing writes completed spans to disk. Delivering here
        # could send children while their parent is still running.
        return True
