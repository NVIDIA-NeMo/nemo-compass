# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FilePath, field_validator, model_validator
from pydantic_core import PydanticCustomError
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from compass.compass_signals.anomaly_and_patterns.signal import AnomalyAndPatternsConfig
from compass.compass_signals.ethos_divergence.ethos_divergence_detector import (
    EthosDivergenceConfig,
)
from compass.compass_signals.grader_failure_patterns import GraderFailurePatternsConfig
from compass.compass_signals.tool_issues.signal import ToolIssueConfig
from compass.compass_signals.user_sentiment.signal import UserSentimentConfig
from compass.trace_loaders.atif import ATIFTraceConfig
from compass.trace_loaders.braintrust import validate_braintrust_selection
from compass.trace_loaders.intake import IntakeTraceLoaderConfig
from compass.trace_loaders.langfuse import validate_langfuse_time_window
from trace_ingest.loaders.gym import GymTraceConfig

_CONFIG_MODEL_SETTINGS = SettingsConfigDict(
    extra="forbid",
    cli_kebab_case=True,
    cli_implicit_flags=True,
    cli_hide_none_type=True,
    cli_avoid_json=True,
    cli_use_class_docs_for_groups=True,
)


class ConfigModel(BaseModel):
    """Strict base for the shareable YAML contract."""

    model_config = _CONFIG_MODEL_SETTINGS


class _ConfigFileSettingsSource(YamlConfigSettingsSource):
    """Load YAML from the path already parsed from the higher-priority CLI source."""

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        # Defer reading until Pydantic has populated ``current_state`` from the
        # CLI source, which is automatically given the highest priority.
        super().__init__(settings_cls, yaml_file=None)

    def __call__(self) -> dict[str, object]:
        config_path = self.current_state.get("config")
        if config_path is None:
            return {}
        return YamlConfigSettingsSource(self.settings_cls, yaml_file=config_path)()


class FilesystemConfig(ConfigModel):
    """Settings owned by the canonical JSONL filesystem loader."""

    path: Path


class MLflowExperimentConfig(ConfigModel):
    """Settings owned by the live MLflow experiment loader."""

    experiment: str
    tracking_uri: str | None = None
    filter: str | None = None


class MLflowExportConfig(ConfigModel):
    """Settings owned by the native MLflow export loader."""

    path: Path


class LangfuseTraceExportFileConfig(ConfigModel):
    """Settings for the documented Langfuse v3/v4 export files."""

    api_version: Literal["v3", "v4"]
    path: Path
    scores_path: Path | None = None
    base_url: str | None = None

    @model_validator(mode="after")
    def validate_version_options(self) -> LangfuseTraceExportFileConfig:
        if self.api_version == "v3" and any(
            value is not None for value in (self.scores_path, self.base_url)
        ):
            raise ValueError("scores_path and base_url require api_version=v4")
        return self


class BraintrustConfig(ConfigModel):
    """Settings owned by the live Braintrust SQL trace loader."""

    project_id: str | None = None
    experiment_id: str | None = None
    org_name: str | None = None
    app_url: str | None = None
    api_url: str | None = None
    from_timestamp: datetime
    to_timestamp: datetime

    @model_validator(mode="after")
    def selection_is_valid(self) -> BraintrustConfig:
        validate_braintrust_selection(
            self.project_id, self.experiment_id, self.from_timestamp, self.to_timestamp
        )
        return self


class LangfuseConfig(ConfigModel):
    """Settings owned by the live Langfuse v3/v4 trace loader."""

    base_url: str | None = None
    from_timestamp: datetime
    to_timestamp: datetime
    filter: str | None = None

    @model_validator(mode="after")
    def time_window_is_valid(self) -> LangfuseConfig:
        validate_langfuse_time_window(self.from_timestamp, self.to_timestamp)
        return self


class LangSmithTraceSourceConfig(ConfigModel):
    """Settings owned by the LangSmith Trace Loader."""

    project: str
    api_url: str | None = None
    filter: str | None = None
    tree_filter: str | None = None
    start_time: datetime | None = None

    @model_validator(mode="after")
    def start_time_has_timezone(self) -> LangSmithTraceSourceConfig:
        if self.start_time is not None and (
            self.start_time.tzinfo is None or self.start_time.utcoffset() is None
        ):
            raise ValueError("start_time must include a timezone")
        return self


class LangSmithTraceExportFileSourceConfig(ConfigModel):
    """Settings owned by the LangSmith Trace Export File Loader."""

    path: Path


class TraceConfig(ConfigModel):
    """Shared trace settings and exactly one configured loader."""

    max_traces: int | None = Field(
        default=None,
        ge=1,
        description="Maximum complete traces loaded by a provider source",
    )
    filesystem: FilesystemConfig | None = Field(
        default=None,
        description="Canonical JSONL filesystem loader",
    )
    gym: GymTraceConfig | None = Field(
        default=None, description="Gym rollout JSON/JSONL with ng_trajectory"
    )
    atif: ATIFTraceConfig | None = Field(
        default=None,
        description="ATIF trajectory JSONL loader",
    )
    mlflow_experiment: MLflowExperimentConfig | None = Field(
        default=None,
        description="Live MLflow experiment loader",
    )
    mlflow_export: MLflowExportConfig | None = Field(
        default=None,
        description="Native MLflow export loader",
    )
    intake: IntakeTraceLoaderConfig | None = Field(
        default=None,
        description="Bounded NeMo Platform Intake query",
    )
    braintrust: BraintrustConfig | None = Field(
        default=None,
        description="Live Braintrust project logs or experiment traces",
    )
    langfuse: LangfuseConfig | None = Field(
        default=None,
        description="Live Langfuse v3/v4 trace loader",
    )
    langfuse_trace_export_file: LangfuseTraceExportFileConfig | None = Field(
        default=None,
        description="Langfuse v3/v4 trace export file loader",
    )
    langsmith: LangSmithTraceSourceConfig | None = Field(
        default=None,
        description="LangSmith Trace Loader",
    )
    langsmith_trace_export_file: LangSmithTraceExportFileSourceConfig | None = Field(
        default=None,
        description="LangSmith Trace Export File Loader",
    )

    @model_validator(mode="after")
    def source_has_required_settings(self) -> TraceConfig:
        configured = sum(
            source is not None
            for source in (
                self.filesystem,
                self.atif,
                self.gym,
                self.mlflow_experiment,
                self.mlflow_export,
                self.intake,
                self.braintrust,
                self.langfuse,
                self.langfuse_trace_export_file,
                self.langsmith,
                self.langsmith_trace_export_file,
            )
        )
        if configured != 1:
            raise ValueError("trace must configure exactly one loader")
        if self.filesystem is not None and self.max_traces is not None:
            raise ValueError("trace.max_traces is not supported by the filesystem loader")
        if self.gym is not None and self.max_traces is not None:
            raise ValueError("trace.max_traces is not supported by the Gym loader")
        if self.atif is not None and self.max_traces is not None:
            raise ValueError("trace.max_traces is not supported by the ATIF loader")
        return self


class AnomalyAndPatternsSignalConfig(AnomalyAndPatternsConfig):
    """Strict YAML settings for anomaly-and-pattern evidence."""

    model_config = ConfigDict(extra="forbid")


class ToolIssueSignalConfig(ToolIssueConfig):
    """Strict YAML settings for deterministic tool-issue evidence."""

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
                self.ethos_divergence,
                self.grader_failure_patterns,
                self.user_sentiment,
            )
        ):
            raise ValueError("at least one Compass Signal must be configured")
        return self


class RunConfig(BaseSettings):
    """The complete public configuration contract for an analysis run."""

    model_config = _CONFIG_MODEL_SETTINGS

    config: Path | None = Field(
        default=None,
        exclude=True,
        description="Optional YAML configuration file",
    )
    trace: TraceConfig = Field(description="Trace source and loading settings")
    ethos_path: FilePath | None = Field(
        default=None,
        description="Agent purpose and behavioral requirements document; defaults to ETHOS.md or ethos.md in the working directory",
    )
    output_path: Path = Field(default=Path("insights.yml"), description="Output path")
    compass_signals: CompassSignalsConfig = Field(
        default_factory=CompassSignalsConfig, description="Compass Signal selection and settings"
    )
    code_base: Path | None = Field(
        default=None,
        description="Optional local codebase used to validate trace-derived problems",
    )

    model: str | None = Field(default=None, description="Inference model")
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Maximum output tokens per model response. "
            "Set this when your model supports a smaller output limit."
        ),
    )
    api_base: str | None = None
    existing_insights: Path | None = Field(
        default=None,
        description="Existing insights.json to reconcile with newly generated Insights",
    )

    @field_validator("ethos_path", mode="before")
    @classmethod
    def discover_ethos_document(cls, value: object) -> object:
        if value is None:
            for name in ("ETHOS.md", "ethos.md"):
                path = Path.cwd() / name
                if path.is_file():
                    return path
        return value

    @field_validator("ethos_path")
    @classmethod
    def validate_ethos_content(cls, path: Path | None) -> Path | None:
        if path is not None and not path.read_text(encoding="utf-8").strip():
            raise PydanticCustomError(
                "ethos_empty",
                "ETHOS.md is empty or contains only whitespace: {path}. "
                "Add your agent's purpose and behavioral requirements, or select a non-empty file "
                "with ethos_path in YAML or --ethos-path.",
                {"path": str(path)},
            )
        return path

    @model_validator(mode="after")
    def explicit_ethos_requires_document(self) -> RunConfig:
        if (
            "ethos_divergence" in self.compass_signals.model_fields_set
            and self.compass_signals.ethos_divergence is not None
            and self.ethos_path is None
        ):
            raise PydanticCustomError(
                "ethos_path_required",
                "Ethos divergence was explicitly enabled but no ETHOS.md was found. "
                "Place ETHOS.md or ethos.md in the working directory, set ethos_path in YAML, "
                "or pass --ethos-path to an existing ETHOS.md.",
            )
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Read explicit values first, followed by the selected YAML configuration."""

        return init_settings, _ConfigFileSettingsSource(settings_cls)
