# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json

from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from compass.insight import Insight
from compass.insights_generation.confidence import InsightConfidence
from compass.traces import Trace, TraceAggregate


def _response(name: str, arguments: dict, call_id: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=json.dumps(arguments))],
        finish_reason="tool_calls",
        assistant_message={},
    )


def _read_agent_code() -> LLMResponse:
    return _response(
        "execute_python",
        {"code": "print(self.codebase.read_file('agent.py'))"},
        "read",
    )


def _insight() -> Insight:
    return Insight(
        name="Requests time out too quickly",
        description="The agent's HTTP client times out before slow tools can respond.",
        evidence=[{"trace_id": "t1"}, {"trace_id": "t2"}],
    )


def _rate(tmp_path, llm: FakeLLMClient) -> str | None:
    trace = Trace(id="t1", root_spans=[], aggregate=TraceAggregate())
    return asyncio.run(InsightConfidence(tmp_path, llm).rate(_insight(), (trace,)))


def test_insight_confidence_reads_code_before_rating(tmp_path) -> None:
    (tmp_path / "agent.py").write_text("TIMEOUT_SECONDS = 1\n", encoding="utf-8")
    llm = FakeLLMClient(
        [_read_agent_code(), _response("return_result", {"result": {"confidence": "high"}}, "done")]
    )

    assert _rate(tmp_path, llm) == "high"
    assert llm.call_count == 2
    assert "TIMEOUT_SECONDS = 1" in str(llm.last_messages)


def test_insight_confidence_can_be_low_after_code_inspection(tmp_path) -> None:
    (tmp_path / "agent.py").write_text("TIMEOUT_SECONDS = 60\n", encoding="utf-8")
    llm = FakeLLMClient(
        [_read_agent_code(), _response("return_result", {"result": {"confidence": "low"}}, "done")]
    )

    assert _rate(tmp_path, llm) == "low"


def test_insight_confidence_is_unset_when_step_budget_is_exhausted(tmp_path) -> None:
    (tmp_path / "agent.py").write_text("TIMEOUT_SECONDS = 1\n", encoding="utf-8")
    llm = FakeLLMClient([_read_agent_code() for _ in range(40)])

    assert _rate(tmp_path, llm) is None
