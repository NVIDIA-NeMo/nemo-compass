# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import SpanContext, TraceFlags

from compass.telemetry_export import CompassSpanExporter, SpanInventory, span_id_digest


def span(index, **attributes):
    return ReadableSpan(
        name=f"signal.{index}",
        context=SpanContext(1, index, False, TraceFlags(1)),
        parent=SpanContext(1, 999, False, TraceFlags(1)),
        attributes=attributes,
        start_time=1,
        end_time=2,
    )


class RecordingExporter(SpanExporter):
    def __init__(self, limit, fail_call=None, raises=False):
        self.limit = limit
        self.batches = []
        self.fail_call = fail_call
        self.raises = raises

    def export(self, spans):
        # Measure the actual OTLP wire payload, including resources and envelopes.
        assert len(encode_spans(spans).SerializeToString()) <= self.limit
        self.batches.append(spans)
        if len(self.batches) == self.fail_call:
            if self.raises:
                raise RuntimeError("secret transport details")
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS


def test_oversized_batch_is_split_without_changing_any_span():
    limit = 16 * 1024 * 1024
    spans = [span(i, payload="x" * 1024 * 1024) for i in range(1, 86)]
    delegate = RecordingExporter(limit)
    exporter = CompassSpanExporter(delegate)
    assert exporter.export(spans) == SpanExportResult.SUCCESS
    assert [s for batch in delegate.batches for s in batch] == spans
    assert len(delegate.batches) > 1
    assert exporter.exported_span_ids == set(range(1, 86))


def test_single_large_span_summarizes_payload_but_preserves_identity_and_usage():
    original = span(
        1,
        **{
            "input.value": json.dumps({"large": "🧭" * 10000}, ensure_ascii=False),
            "llm.token_count.prompt": 1234,
            "llm.token_count.completion": 56,
            "openinference.span.kind": "LLM",
        },
    )
    delegate = RecordingExporter(4096)
    exporter = CompassSpanExporter(delegate, max_batch_bytes=4096)
    assert exporter.export([original]) == SpanExportResult.SUCCESS
    exported = delegate.batches[0][0]
    assert exported.context == original.context
    assert exported.parent == original.parent
    assert exported.name == original.name
    assert (exported.start_time, exported.end_time) == (1, 2)
    assert exported.attributes["llm.token_count.prompt"] == 1234
    assert exported.attributes["llm.token_count.completion"] == 56
    summary = json.loads(exported.attributes["input.value"])
    assert summary["compass_telemetry_omitted"] is True
    assert summary["original_utf8_bytes"] == len(original.attributes["input.value"].encode())
    assert len(summary["sha256"]) == 64
    assert exported.attributes["compass.telemetry.summarized_attributes"] == ("input.value",)
    assert "🧭" in original.attributes["input.value"]
    assert (
        json.loads(exported.to_json())["attributes"]["input.value"]
        == exported.attributes["input.value"]
    )


@pytest.mark.parametrize("raises", [False, True])
def test_failed_split_batch_does_not_hide_loss_or_skip_other_batches(raises, caplog):
    spans = [span(i, payload="x" * 2000) for i in range(1, 5)]
    delegate = RecordingExporter(3000, fail_call=2, raises=raises)
    exporter = CompassSpanExporter(delegate, max_batch_bytes=3000)
    assert exporter.export(spans) == SpanExportResult.FAILURE
    assert len(delegate.batches) == 4
    assert exporter.exported_span_ids == {1, 3, 4}
    assert "incomplete" in caplog.text
    assert "secret transport details" not in caplog.text


def test_inventory_covers_spans_that_never_reach_exporter():
    inventory = SpanInventory()
    spans = [span(i) for i in range(1, 4)]
    for record in spans:
        inventory.on_start(record)
    delegate = RecordingExporter(4096)
    exporter = CompassSpanExporter(delegate)
    exporter.export(spans[:1])
    assert inventory.metadata() == {
        "expected_span_count": 3,
        "expected_span_id_sha256": span_id_digest({1, 2, 3}),
    }
    assert inventory.started == {1, 2, 3}
    assert exporter.exported_span_ids == {1}
