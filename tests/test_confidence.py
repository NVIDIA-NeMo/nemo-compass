# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from typing import cast

from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from compass.insight import CommitEvidence, Insight
from compass.insights_generation.confidence import InsightConfidence
from compass.insights_generation.repository_history import GitHistory
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


def test_commit_evidence_loads_legacy_why_but_emits_evidence_reason() -> None:
    evidence = CommitEvidence.model_validate(
        {
            "commit": "a" * 40,
            "why": "This commit adds the missing validation.",
            "resolution": "resolved",
        }
    )

    assert evidence.evidence_reason == "This commit adds the missing validation."
    assert evidence.model_dump() == {
        "commit": "a" * 40,
        "evidence_reason": "This commit adds the missing validation.",
        "resolution": "resolved",
    }


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


def test_repository_investigation_shares_confidence_model_call(tmp_path) -> None:
    (tmp_path / "agent.py").write_text("TIMEOUT_SECONDS = 1\n", encoding="utf-8")
    sha = "a" * 40
    captured = {}

    class FakeHistory:
        def git_history(self, path, query=""):
            captured["path"] = path
            return f"commit {sha}"

        def evidence(self, commits):
            captured["commits"] = commits
            return commits

    llm = FakeLLMClient(
        [
            _response(
                "execute_python",
                {"code": "print(self.repository_history.git_history('agent.py'))"},
                "history",
            ),
            _response(
                "return_result",
                {
                    "result": {
                        "confidence": "high",
                        "repository_evidence": [
                            {
                                "commit": sha,
                                "evidence_reason": "This commit introduces the short timeout.",
                                "resolution": "unresolved",
                            },
                            {
                                "commit": "b" * 40,
                                "evidence_reason": "This commit increases the timeout.",
                                "resolution": "resolved",
                            },
                        ],
                    }
                },
                "done",
            ),
        ]
    )
    trace = Trace(id="t1", root_spans=[], aggregate=TraceAggregate())

    review = asyncio.run(
        InsightConfidence(
            tmp_path,
            llm,
            cast(GitHistory, FakeHistory()),
        ).investigate(_insight(), (trace,))
    )

    assert review.confidence == "high"
    assert review.repository_evidence is not None
    assert captured["path"] == "agent.py"
    assert captured["commits"] == [
        CommitEvidence(
            commit=sha,
            evidence_reason="This commit introduces the short timeout.",
            resolution="unresolved",
        ),
        CommitEvidence(
            commit="b" * 40,
            evidence_reason="This commit increases the timeout.",
            resolution="resolved",
        ),
    ]
    assert llm.call_count == 2
