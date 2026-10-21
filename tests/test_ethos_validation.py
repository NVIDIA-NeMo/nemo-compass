# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json

import pytest
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall
from pydantic import ValidationError

from compass.compass_signals.compass_signals import Problem
from compass.insights_generation.validation import CandidateDecision, CandidateValidation
from compass.traces import Trace, TraceAggregate

ethos = "Ask for approval before issuing any refund."


def response(decision):
    return LLMResponse(
        raw_response=None,
        content="",
        finish_reason="tool_calls",
        assistant_message={},
        tool_calls=[
            ToolCall(id="done", name="return_result", arguments=json.dumps({"result": decision}))
        ],
    )


@pytest.mark.parametrize(
    "contradicted,passage,expected",
    [
        (True, ethos, True),
        (False, None, False),
        (True, "All refunds require approval.", False),
    ],
)
def test_ethos_decision_requires_grounding(contradicted, passage, expected):
    llm = FakeLLMClient(
        [
            response(
                {
                    "supported": not contradicted,
                    "rejection_basis": "ethos" if contradicted else None,
                    "ethos_passage": passage,
                    "explanation": "The candidate criticizes required approval.",
                }
            )
        ]
    )
    problem = Problem(description="Approval adds unnecessary friction", supporting_trace_ids=("t",))
    trace = Trace(id="t", root_spans=[], aggregate=TraceAggregate())
    decision = asyncio.run(CandidateValidation(None, llm, ethos).validate(problem, (trace,)))
    assert (decision.supported is False) is expected
    assert decision.explanation
    assert ethos in str(llm.last_messages)
    assert problem.description in str(llm.last_messages)


def test_ethos_rejection_requires_nonempty_passage():
    for passage in (None, "", "  "):
        with pytest.raises(ValidationError):
            CandidateDecision(
                supported=False,
                rejection_basis="ethos",
                explanation="Allowed",
                ethos_passage=passage,
            )


def test_empty_ethos_is_configuration_error():
    with pytest.raises(ValueError, match="non-empty"):
        CandidateValidation(None, FakeLLMClient(), "  ")


def test_generation_error_retains_candidate(monkeypatch):
    from nooa.errors import GenerationError

    from compass.insights_generation import validation

    async def fail(*args, **kwargs):
        raise GenerationError("Unable to complete")

    from types import SimpleNamespace

    monkeypatch.setattr(
        validation, "_ValidationAgent", lambda *args, **kwargs: SimpleNamespace(decide=fail)
    )
    problem = Problem(description="Problem", supporting_trace_ids=("t",))
    decision = asyncio.run(CandidateValidation(None, FakeLLMClient(), ethos).validate(problem, ()))
    assert decision.supported is None
    assert "inconclusive" in decision.explanation
