# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json

from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from compass.compass_signals.compass_signals import Problem
from compass.insights_generation.validation import CandidateProblemValidator
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
    return asyncio.run(
        CandidateProblemValidator(tmp_path, llm).validate(problem, (trace,))
    ).supported


def test_problem_validation_reads_code_before_accepting_problem(tmp_path) -> None:
    (tmp_path / "agent.py").write_text("TIMEOUT_SECONDS = 1\n", encoding="utf-8")
    llm = FakeLLMClient(
        [
            _read_agent_code(),
            _response(
                "return_result",
                {"result": {"supported": True, "explanation": "The timeout is too short."}},
                "done",
            ),
        ]
    )

    assert _validate(tmp_path, llm) is True
    assert llm.call_count == 2
    assert "TIMEOUT_SECONDS = 1" in str(llm.last_messages)


def test_problem_validation_rejects_code_contradicted_problem(tmp_path) -> None:
    (tmp_path / "agent.py").write_text("TIMEOUT_SECONDS = 60\n", encoding="utf-8")
    llm = FakeLLMClient(
        [
            _read_agent_code(),
            _response(
                "return_result",
                {
                    "result": {
                        "supported": False,
                        "explanation": "The configured timeout contradicts the claim.",
                        "rejection_basis": "code",
                    }
                },
                "done",
            ),
        ]
    )

    assert _validate(tmp_path, llm) is False


def test_problem_validation_can_return_not_applicable(tmp_path) -> None:
    llm = FakeLLMClient(
        [
            _response(
                "return_result",
                {"result": {"supported": None, "explanation": "Insufficient evidence."}},
                "done",
            )
        ]
    )

    assert _validate(tmp_path, llm) is None


def test_problem_validation_retains_problem_when_step_budget_is_exhausted(tmp_path) -> None:
    (tmp_path / "agent.py").write_text("TIMEOUT_SECONDS = 1\n", encoding="utf-8")
    llm = FakeLLMClient([_read_agent_code() for _ in range(40)])

    assert _validate(tmp_path, llm) is None


def test_combined_validation_receives_ethos_and_reads_code(tmp_path):
    from compass.insights_generation.validation import CandidateProblemValidator

    (tmp_path / "agent.py").write_text("MAX_ATTEMPTS = 2\n")
    ethos = "Skip the update only after two failed attempts."
    decision = {
        "supported": False,
        "explanation": "The observed bounded skip follows the explicit contract and implementation.",
        "code_assessment": "The implementation permits two attempts before skipping.",
        "ethos_assessment": "Skipping is permitted once both attempts fail.",
        "ethos_passage": ethos,
        "rejection_basis": "code_and_ethos",
    }
    llm = FakeLLMClient(
        [_read_agent_code(), _response("return_result", {"result": decision}, "done")]
    )
    problem = Problem(description="The agent skipped a failed update", supporting_trace_ids=("t",))
    result = asyncio.run(CandidateProblemValidator(tmp_path, llm, ethos).validate(problem, ()))
    assert result.supported is False
    assert result.code_assessment and result.ethos_assessment
    assert ethos in str(llm.last_messages)
    assert "MAX_ATTEMPTS = 2" in str(llm.last_messages)


def test_missing_code_cannot_ground_a_rejection():
    from compass.insights_generation.validation import CandidateProblemValidator

    llm = FakeLLMClient(
        [
            _response(
                "return_result",
                {
                    "result": {
                        "supported": False,
                        "explanation": "Unsupported code claim",
                        "rejection_basis": "code",
                    }
                },
                "done",
            )
        ]
    )
    problem = Problem(description="Problem", supporting_trace_ids=("t",))
    result = asyncio.run(
        CandidateProblemValidator(None, llm, "Allow bounded retries.").validate(problem, ())
    )
    assert result.supported is None
    assert "not supplied" in result.explanation


def test_candidate_problem_validation_preserves_cancellation(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import pytest
    from nooa.errors import GenerationAborted

    from compass.insights_generation import validation

    async def abort(*args):
        raise GenerationAborted("Interrupted")

    monkeypatch.setattr(validation, "_ValidationAgent", lambda *args: SimpleNamespace(decide=abort))
    with pytest.raises(GenerationAborted):
        _validate(tmp_path, FakeLLMClient())
