# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json

from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from compass.compass_signals.compass_signals import Problem
from compass.insights_generation.validation import ProblemValidation
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


def _validate(tmp_path, llm: FakeLLMClient) -> bool | None:
    problem = Problem(description="Requests time out too quickly", supporting_trace_ids=("t1",))
    trace = Trace(id="t1", root_spans=[], aggregate=TraceAggregate())
    return asyncio.run(ProblemValidation(tmp_path, llm).is_supported(problem, (trace,)))


def test_problem_validation_reads_code_before_accepting_problem(tmp_path) -> None:
    (tmp_path / "agent.py").write_text("TIMEOUT_SECONDS = 1\n", encoding="utf-8")
    llm = FakeLLMClient(
        [_read_agent_code(), _response("return_result", {"result": {"supported": True}}, "done")]
    )

    assert _validate(tmp_path, llm) is True
    assert llm.call_count == 2
    assert "TIMEOUT_SECONDS = 1" in str(llm.last_messages)


def test_problem_validation_rejects_code_contradicted_problem(tmp_path) -> None:
    (tmp_path / "agent.py").write_text("TIMEOUT_SECONDS = 60\n", encoding="utf-8")
    llm = FakeLLMClient(
        [_read_agent_code(), _response("return_result", {"result": {"supported": False}}, "done")]
    )

    assert _validate(tmp_path, llm) is False


def test_problem_validation_can_return_not_applicable(tmp_path) -> None:
    llm = FakeLLMClient([_response("return_result", {"result": {"supported": None}}, "done")])

    assert _validate(tmp_path, llm) is None


def test_problem_validation_retains_problem_when_step_budget_is_exhausted(tmp_path) -> None:
    (tmp_path / "agent.py").write_text("TIMEOUT_SECONDS = 1\n", encoding="utf-8")
    llm = FakeLLMClient([_read_agent_code() for _ in range(40)])

    assert _validate(tmp_path, llm) is None
