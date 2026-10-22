# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import warnings
from unittest.mock import Mock

import pytest
from nooa.unifiedllm import FakeLLMClient

from compass.cli.main import _run_compass_signals
from compass.compass_signals.anomaly_and_patterns.signal import (
    AnomalyAndPatternsAnalysis,
    NormalizedCall,
    NormalizedTrace,
    problems_from_analysis,
    run_anomaly_and_patterns,
)
from compass.compass_signals.tool_issues.signal import build_cards, problems_from_cards
from compass.compass_signals.user_sentiment import signal as sentiment
from compass.config import CompassSignalsConfig
from compass.traces import Trace, TraceAggregate, TraceSnapshot


@pytest.mark.parametrize("signal,preview_size", [("tool", 3), ("anomaly", 50)])
def test_signals_keep_all_candidates_and_limit_review_examples(signal, preview_size):
    expected = tuple(f"trace-{i}" for i in range(preview_size + 2))
    if signal == "tool":
        findings = [
            {
                "trace_id": trace_id,
                "logical_case_id": trace_id if i <= preview_size else expected[0],
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
    else:
        rows = [
            {
                "trace_id": trace_id,
                "anomaly_score": len(expected) - i,
                "is_anomaly": True,
                "anomaly_reasons": ("duration",),
                "pca_x": 0,
                "pca_y": 0,
                "source_pointer": {},
            }
            for i, trace_id in enumerate(expected)
        ]
        rows.extend(
            [
                rows[0],
                rows[0] | {"trace_id": "not-flagged", "is_anomaly": False, "anomaly_score": 100},
            ]
        )
        result = AnomalyAndPatternsAnalysis.model_validate(
            dict(
                prepared=(),
                failure_events=(),
                anomalies=rows[::-1],
                trajectory_groups=None,
                verdict_groups=(),
                failure_groups=(),
                cross_tool_failure_groups=(),
                digest="",
            )
        )
        (problem,) = problems_from_analysis(result)

    assert problem.candidate_trace_ids == expected
    assert problem.supporting_trace_ids == expected[:preview_size]


@pytest.mark.parametrize("count,patterns", [(0, 1), (2, 2), (10, 1), (10, 2)])
def test_clustering_handles_small_and_repetitive_corpora(count, patterns):
    traces = [
        NormalizedTrace(str(i), [NormalizedCall(str(i), 0, f"tool-{i % patterns}")])
        for i in range(count)
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.simplefilter("error", RuntimeWarning)
        result = run_anomaly_and_patterns(traces)
    groups = result.trajectory_groups
    assert groups is not None
    if count >= 3 and patterns > 1:
        assert groups["selected_k"] == 2
        assert len(groups["assignments"]) == count
    else:
        assert groups["status"] == "not_evaluable"


def test_missing_prerequisites_skip_analysis_without_llm_calls(monkeypatch):
    dependencies = Mock()
    monkeypatch.setattr(sentiment, "validate_embedding_dependencies", dependencies)
    snapshot = TraceSnapshot([Trace(id="one", root_spans=[], aggregate=TraceAggregate())])
    llm = FakeLLMClient()
    progress = []
    results = asyncio.run(
        _run_compass_signals(
            CompassSignalsConfig(cost_efficiency=True), snapshot, lambda: llm, progress.append
        )
    )
    assert {item.signal_name for item in results if item.skip_reason} == {
        "tool-issues",
        "cost-efficiency",
        "ethos-divergence",
        "grader-failure-patterns",
        "user-sentiment",
    }
    assert set().union(*map(set, progress)) == {"anomaly-and-patterns"}
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
