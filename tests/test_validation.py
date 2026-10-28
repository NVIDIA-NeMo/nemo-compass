# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json

import pytest
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


def _validate(tmp_path, llm: FakeLLMClient, ethos: str | None = None) -> bool | None:
    problem = Problem(description="Requests time out too quickly", supporting_trace_ids=("t1",))
    trace = Trace(id="t1", root_spans=[], aggregate=TraceAggregate())
    return asyncio.run(
        CandidateProblemValidator(tmp_path, llm, ethos).validate(problem, (trace,))
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

    ethos = "Allow requests at least ten seconds to complete."
    assert _validate(tmp_path, llm, ethos) is True
    assert ethos in str(llm.last_messages)
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


@pytest.mark.parametrize(
    "basis,passage,expected",
    [
        ("ethos", "Require approval.", False),
        ("ethos", "Approval is always required.", None),
        ("code", None, None),
    ],
)
def test_ethos_only_validation_requires_grounded_rejections(basis, passage, expected):
    ethos = "Require approval."
    llm = FakeLLMClient(
        [
            _response(
                "return_result",
                {
                    "result": {
                        "supported": False,
                        "explanation": "The candidate criticizes required approval.",
                        "rejection_basis": basis,
                        "ethos_passage": passage,
                    }
                },
                "done",
            )
        ]
    )
    problem = Problem(description="Approval adds friction", supporting_trace_ids=("t",))
    decision = asyncio.run(CandidateProblemValidator(None, llm, ethos).validate(problem, ()))
    assert decision.supported is expected
    assert ethos in str(llm.last_messages)


def test_candidate_problem_validation_preserves_cancellation(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from nooa.errors import GenerationAborted

    from compass.insights_generation import validation

    async def abort(*args):
        raise GenerationAborted("Interrupted")

    monkeypatch.setattr(
        validation,
        "_ValidationAgent",
        lambda *args: SimpleNamespace(decide=abort, traces=SimpleNamespace(overview=lambda: "")),
    )
    with pytest.raises(GenerationAborted):
        _validate(tmp_path, FakeLLMClient())
