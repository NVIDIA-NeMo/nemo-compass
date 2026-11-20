# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from compass.compass_signals.anomaly_and_patterns.signal import AnomalyAndPatternsConfig
from compass.compass_signals.cost_efficiency.signal import CostEfficiencyConfig
from compass.compass_signals.ethos_divergence.ethos_divergence_detector import EthosDivergenceConfig
from compass.compass_signals.grader_failure_patterns import GraderFailurePatternsConfig
from compass.compass_signals.tool_issues.signal import ToolIssueConfig
from compass.compass_signals.user_sentiment.signal import UserSentimentConfig


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnomalyAndPatternsSignalConfig(AnomalyAndPatternsConfig):
    """Strict YAML settings for anomaly-and-pattern evidence."""

    model_config = ConfigDict(extra="forbid")


class ToolIssueSignalConfig(ToolIssueConfig):
    """Strict YAML settings for deterministic tool-issue evidence."""

    model_config = ConfigDict(extra="forbid")


class CostEfficiencySignalConfig(CostEfficiencyConfig):
    """Strict YAML settings for deterministic cost-efficiency evidence."""

    model_config = ConfigDict(extra="forbid")


class CompassSignalsConfig(ConfigModel):
    """All signals run by default; use false to disable, true or {} for defaults."""

    anomaly_and_patterns: AnomalyAndPatternsSignalConfig | None = Field(
        default_factory=AnomalyAndPatternsSignalConfig,
        description="Anomaly and recurring-pattern evidence",
    )
    tool_issues: ToolIssueSignalConfig | None = Field(
        default_factory=ToolIssueSignalConfig,
        description="Deterministic tool-issue evidence",
    )
    cost_efficiency: CostEfficiencySignalConfig | None = Field(
        default_factory=CostEfficiencySignalConfig,
        description="Deterministic prompt-cache and token-efficiency evidence",
    )
    grader_failure_patterns: GraderFailurePatternsConfig | None = Field(
        default_factory=GraderFailurePatternsConfig,
        description="LLM review of failures linked to recorded grader results",
    )

    ethos_divergence: EthosDivergenceConfig | None = Field(default_factory=EthosDivergenceConfig)
    user_sentiment: UserSentimentConfig | None = Field(default_factory=UserSentimentConfig)

    @field_validator("*", mode="before")
    @classmethod
    def parse_enabled(cls, value: object) -> object:
        """Translate YAML switches to the config model (enabled) or None (disabled)."""
        if value is True:
            return {}
        return None if value is False else value

    @model_validator(mode="after")
    def at_least_one_signal_is_configured(self) -> CompassSignalsConfig:
        if all(
            signal is None
            for signal in (
                self.anomaly_and_patterns,
                self.tool_issues,
                self.cost_efficiency,
                self.ethos_divergence,
                self.grader_failure_patterns,
                self.user_sentiment,
            )
        ):
            raise ValueError("at least one Compass Signal must be configured")
        return self


class AnalysisOptions(ConfigModel):
    """Reusable analysis settings, independent of source loading and host credentials."""

    compass_signals: CompassSignalsConfig = Field(default_factory=CompassSignalsConfig)
