# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Link, SpanContext, StatusCode, TraceFlags

from compass.telemetry_buffer import BufferedSpanExporter, _pack_span, _unpack_span
from compass.telemetry_export import CompassSpanExporter


def records():
    provider = TracerProvider(
        resource=Resource({"service.name": "buffer-test"}, schema_url="https://resource.test"),
        span_limits=SpanLimits(max_attributes=2, max_events=1, max_links=1),
    )
    memory = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    tracer = provider.get_tracer("buffer-test", "1.0", "https://scope.test")
    context = SpanContext(42, 43, True, TraceFlags(1))
    with tracer.start_as_current_span("root") as root:
        root.set_attributes({"a": 1, "b": "🧭", "c": (0.1, 0.2)})
        root.add_event("discarded")
        root.add_event("kept", {"x": "y"})
        with tracer.start_as_current_span("parent", links=[Link(context)]) as parent:
            parent.set_status(StatusCode.ERROR, "example failure")
            for index in range(3):
                with tracer.start_as_current_span(f"child-{index}"):
                    pass
    provider.shutdown()
    return memory.get_finished_spans()


def test_disk_roundtrip_preserves_the_entire_otlp_message():
    spans = records()
    restored = [_unpack_span(_pack_span(span)) for span in spans]
    assert encode_spans(spans) == encode_spans(restored)


class ParentCheckingExporter(SpanExporter):
    def __init__(self, failures):
        self.failures = failures
        self.acknowledged = set()
        self.calls = []

    def export(self, spans):
        self.calls.append([span.name for span in spans])
        # Parents must be acknowledged in an earlier request, not merely be
        # somewhere in the same batch that the receiver may process in parallel.
        assert all(not span.parent or span.parent.span_id in self.acknowledged for span in spans)
        if self.failures:
            self.failures -= 1
            return SpanExportResult.FAILURE
        self.acknowledged.update(span.context.span_id for span in spans)
        return SpanExportResult.SUCCESS


@pytest.mark.parametrize("retain_archive", [False, True])
def test_buffer_waits_for_root_and_recovers_failed_parent_before_children(tmp_path, retain_archive):
    receiver = ParentCheckingExporter(failures=1)
    buffer = BufferedSpanExporter(
        CompassSpanExporter(receiver), archive_dir=tmp_path if retain_archive else None
    )
    spans = records()  # End order is children, parent, root.
    for span in spans:
        assert buffer.export([span]) == SpanExportResult.SUCCESS
    assert buffer.force_flush()
    assert not receiver.calls
    assert len(list(buffer.directory.glob("*.span"))) == len(spans)
    buffer.shutdown()
    assert receiver.calls == [["root"], ["root"], ["parent"], ["child-0", "child-1", "child-2"]]
    assert buffer.exported_span_ids == {span.context.span_id for span in spans}
    assert buffer.failed_span_count == 0
    if retain_archive:
        assert len(list(buffer.directory.glob("*.span"))) == len(spans)
    else:
        assert not buffer.directory.exists()


def test_failed_parent_keeps_all_undelivered_spans_on_disk():
    receiver = ParentCheckingExporter(failures=2)
    buffer = BufferedSpanExporter(CompassSpanExporter(receiver))
    spans = records()
    buffer.export(spans)
    buffer.shutdown()
    assert receiver.calls == [["root"], ["root"]]
    assert buffer.failed_span_count == len(spans)
    assert len(list(buffer.directory.glob("*.span"))) == len(spans)
    for path in buffer.directory.iterdir():
        path.unlink()
    buffer.directory.rmdir()
