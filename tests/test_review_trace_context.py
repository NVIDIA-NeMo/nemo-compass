# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nooa.agentdoc import _structured

from compass.insights_generation import confidence, validation
from compass.traces import Span, Trace, TraceAggregate


@pytest.mark.parametrize("review", [validation, confidence])
def test_trace_context_preserves_evidence_when_class_introspection_fails(monkeypatch, review):
    trace = Trace(
        id="failed-test-run",
        root_spans=[
            Span(
                id="pytest-call",
                kind="TOOL",
                tool_name="terminal",
                input={"command": "pytest"},
                output={"exit_code": 1, "stderr": "AssertionError: regression remains"},
            )
        ],
        aggregate=TraceAggregate(),
    )

    def fail_class_inspection(*args, **kwargs):
        raise RuntimeError("dictionary changed size during iteration")

    monkeypatch.setattr(_structured, "extract_type_info", fail_class_inspection)

    context = review._trace_context((trace,))

    assert "failed-test-run" in context
    assert "pytest-call" in context
    assert "terminal" in context
    assert "AssertionError: regression remains" in context
