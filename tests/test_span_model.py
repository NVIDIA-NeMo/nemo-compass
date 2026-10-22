# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from trace_ingest.models import Span, SpanKind


def test_llm_params_round_trip() -> None:
    # An empty object is a recorded request with default parameters, not an unknown one.
    llm = Span(id="llm", kind=SpanKind.LLM, llm_params={})
    assert Span.model_validate_json(llm.model_dump_json()).llm_params == {}
    unknown = Span(id="llm", kind=SpanKind.LLM)
    assert "llm_params" not in unknown.model_dump()


def test_llm_fields_require_an_llm_span() -> None:
    with pytest.raises(ValueError, match="llm_tools requires kind=LLM"):
        Span(id="tool", kind=SpanKind.TOOL, llm_tools=[])
    with pytest.raises(ValueError, match="llm_params requires kind=LLM"):
        Span(id="tool", kind=SpanKind.TOOL, llm_params={})
