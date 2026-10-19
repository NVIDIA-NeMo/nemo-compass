# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construction and registration of the repository's built-in Compass Signals."""

from __future__ import annotations

from collections.abc import Callable

from nooa.unifiedllm import UnifiedLLM

from compass.compass_signals.anomaly_and_patterns.signal import (
    AnomalyAndPatternsCompassSignal,
    AnomalyAndPatternsConfig,
)
from compass.compass_signals.ethos_divergence.ethos_divergence_detector import (
    EthosDivergenceCompassSignal,
    EthosDivergenceConfig,
)
from compass.compass_signals.grader_failure_patterns import (
    GraderFailurePatternsCompassSignal,
    GraderFailurePatternsConfig,
)
from compass.compass_signals.registry import CompassSignalRegistry
from compass.compass_signals.tool_issues.signal import (
    ToolIssueCompassSignal,
    ToolIssueConfig,
)
from compass.compass_signals.user_sentiment.signal import (
    UserSentimentCompassSignal,
    UserSentimentConfig,
)

USER_SENTIMENT = UserSentimentCompassSignal.name
ANOMALY_AND_PATTERNS = AnomalyAndPatternsCompassSignal.name
TOOL_ISSUES = ToolIssueCompassSignal.name
ETHOS_DIVERGENCE = EthosDivergenceCompassSignal.name
GRADER_FAILURE_PATTERNS = GraderFailurePatternsCompassSignal.name
BUILTIN_SIGNAL_NAMES = (
    ANOMALY_AND_PATTERNS,
    TOOL_ISSUES,
    ETHOS_DIVERGENCE,
    GRADER_FAILURE_PATTERNS,
    USER_SENTIMENT,
)


def registered_builtin_signals(
    *,
    anomaly_and_patterns: AnomalyAndPatternsConfig | None = None,
    tool_issues: ToolIssueConfig | None = None,
    ethos_divergence: EthosDivergenceConfig | None = None,
    grader_failure_patterns: GraderFailurePatternsConfig | None = None,
    user_sentiment: UserSentimentConfig | None = None,
    llm_factory: Callable[[], UnifiedLLM] | None = None,
) -> CompassSignalRegistry:
    """Construct and register the built-ins with supplied typed configuration."""

    registry = CompassSignalRegistry()
    if anomaly_and_patterns is not None:
        registry.register(
            AnomalyAndPatternsCompassSignal(config=anomaly_and_patterns),
            config=anomaly_and_patterns,
        )
    if tool_issues is not None:
        registry.register(ToolIssueCompassSignal(config=tool_issues), config=tool_issues)
    if ethos_divergence is not None:
        if llm_factory is None:
            raise ValueError("ethos-divergence requires an LLM client factory")
        registry.register(
            EthosDivergenceCompassSignal(config=ethos_divergence, llm=llm_factory()),
            config=ethos_divergence,
        )
    if grader_failure_patterns is not None:
        if llm_factory is None:
            raise ValueError("grader-failure-patterns requires an LLM client factory")
        registry.register(
            GraderFailurePatternsCompassSignal(llm=llm_factory(), config=grader_failure_patterns),
            config=grader_failure_patterns,
        )
    if user_sentiment is not None:
        if llm_factory is None:
            raise ValueError("user-sentiment requires an LLM client factory")
        registry.register(
            UserSentimentCompassSignal(config=user_sentiment, llm=llm_factory()),
            config=user_sentiment,
        )
    return registry


__all__ = [
    "ANOMALY_AND_PATTERNS",
    "BUILTIN_SIGNAL_NAMES",
    "ETHOS_DIVERGENCE",
    "GRADER_FAILURE_PATTERNS",
    "TOOL_ISSUES",
    "USER_SENTIMENT",
    "registered_builtin_signals",
]
