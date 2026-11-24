# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import Mock

import pytest
from nooa.unifiedllm import FakeLLMClient

from compass.cli.main import _run_compass_signals
from compass.compass_signals.tool_issues.signal import build_cards, problems_from_cards
from compass.compass_signals.user_sentiment import signal as sentiment
from compass.config import CompassSignalsConfig
from compass.traces import Trace, TraceAggregate, TraceSnapshot


def test_tool_signal_keeps_all_candidates_and_limits_review_examples():
    preview_size = 3
    expected = tuple(f"trace-{i}" for i in range(preview_size + 2))
    findings = [
        {
            "trace_id": trace_id,
            "case_id": trace_id if i <= preview_size else expected[0],
            "issue_type": "missing_tool_result",
            "mechanism_key": "missing-result",
            "call_id": f"call-{i}",
            "call_index": i,
            "tool_name": "lookup",
            "source_pointer": {},
            "summary": "No result recorded",
        }
        for i, trace_id in enumerate(expected)
    ]
    # Keep different traces of the same case. Remove duplicate trace IDs.
    findings.append(findings[0] | {"call_id": "another-call"})
    findings.extend(
        finding | {"trace_id": f"sibling-{i}", **other_group}
        for i, finding in enumerate(findings[:3])
        for other_group in (
            {"mechanism_key": "other-mechanism"},
            {"issue_type": "duplicate_tool_result"},
        )
    )
    cards = build_cards(findings)
    card = next(
        card
        for card in cards
        if card.issue_type == "missing_tool_result" and card.mechanism_key == "missing-result"
    )
    (problem,) = problems_from_cards([card])

    assert problem.candidate_trace_ids == expected
    assert problem.supporting_trace_ids == expected[:preview_size]


def test_missing_prerequisites_skip_analysis_without_llm_calls(monkeypatch):
    dependencies = Mock()
    monkeypatch.setattr(sentiment, "validate_embedding_dependencies", dependencies)
    snapshot = TraceSnapshot([Trace(id="one", root_spans=[], aggregate=TraceAggregate())])
    llm = FakeLLMClient()
    progress = []
    results = asyncio.run(
        _run_compass_signals(CompassSignalsConfig(), snapshot, lambda: llm, progress.append)
    )
    assert {item.signal_name for item in results if item.skip_reason} == {
        "tool-issues",
        "cost-efficiency",
        "ethos-divergence",
        "grader-failure-patterns",
        "user-sentiment",
    }
    assert progress == []
    assert llm.call_count == 0
    dependencies.assert_not_called()


def test_sentiment_explicit_backends_check_only_local_dependencies(monkeypatch):
    dependencies = Mock()
    monkeypatch.setattr(sentiment, "validate_embedding_dependencies", dependencies)
    snapshot = TraceSnapshot([])
    local = sentiment.UserSentimentCompassSignal(
        sentiment.UserSentimentConfig(device="cpu"), FakeLLMClient()
    )
    assert local.check_prerequisites(snapshot) == "No embedding backend configured"
    dependencies.assert_not_called()
    local.config.local_embeddings = True
    assert local.check_prerequisites(snapshot) is None
    dependencies.assert_called_once_with()
    assert local.embedding_generator.device == "cpu"
    assert local.embedding_generator.litellm is None
    dependencies.side_effect = ValueError("Install local-embedding")
    with pytest.raises(ValueError, match="local-embedding"):
        local.check_prerequisites(snapshot)

    remote = sentiment.UserSentimentCompassSignal(
        sentiment.UserSentimentConfig.model_validate(
            {"local_embeddings": True, "litellm": {"model": "openai/qwen"}}
        ),
        FakeLLMClient(),
    )
    assert remote.check_prerequisites(snapshot) is None
    assert remote.embedding_generator.litellm == remote.config.litellm
