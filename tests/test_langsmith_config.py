# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

import compass.config as app_config
from compass.cli.main import _configured_trace_loader
from compass.config import (
    CompassSignalsConfig,
    RunConfig,
    TraceConfig,
)
from compass.trace_loaders.langsmith import (
    LANGSMITH_DEFAULT_MAX_TRACES,
    LangSmithTraceConfig,
    LangSmithTraceExportFileConfig,
    LangSmithTraceExportFileLoader,
    LangSmithTraceLoader,
)


def test_run_config_selects_langsmith_trace_loader(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """trace:
  max_traces: 25
  langsmith:
    project: glamr-ux
    api_url: https://langsmith.example/api/v1
    filter: 'eq(status, "success")'
    tree_filter: 'eq(run_type, "tool")'
    start_time: 2026-09-01T00:00:00Z
compass_signals:
  anomaly_and_patterns: {}
""",
        encoding="utf-8",
    )

    config = RunConfig(_cli_parse_args=["--config", str(config_file)])
    loader = _configured_trace_loader(config.trace)

    assert isinstance(loader, LangSmithTraceLoader)
    assert loader.config == LangSmithTraceConfig(
        project_name="glamr-ux",
        api_url="https://langsmith.example/api/v1",
        filter='eq(status, "success")',
        tree_filter='eq(run_type, "tool")',
        start_time=datetime(2026, 9, 1, tzinfo=timezone.utc),
        max_traces=25,
    )


def test_run_config_loads_a_langsmith_trace_export_file(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """trace:
  max_traces: 25
  langsmith_trace_export_file:
    path: exports/langsmith-traces
compass_signals:
  anomaly_and_patterns: {}
""",
        encoding="utf-8",
    )

    config = RunConfig(_cli_parse_args=["--config", str(config_file)])
    loader = _configured_trace_loader(config.trace)

    assert isinstance(loader, LangSmithTraceExportFileLoader)
    assert loader.config == LangSmithTraceExportFileConfig(
        path=Path("exports/langsmith-traces"),
        max_traces=25,
    )


def test_langsmith_trace_loader_uses_provider_default_trace_bound():
    config = RunConfig.model_validate(
        {
            "trace": {"langsmith": {"project": "glamr-ux"}},
            "compass_signals": {"anomaly_and_patterns": {}},
        }
    )

    loader = _configured_trace_loader(config.trace)

    assert isinstance(loader, LangSmithTraceLoader)
    assert loader.config.max_traces == LANGSMITH_DEFAULT_MAX_TRACES


def test_cli_selects_langsmith_trace_export_file_loader():
    config = RunConfig(
        _cli_parse_args=[
            "--trace.langsmith-trace-export-file.path",
            "exports/langsmith-traces",
            "--compass-signals.anomaly-and-patterns.contamination",
            "0.02",
        ]
    )

    loader = _configured_trace_loader(config.trace)

    assert isinstance(loader, LangSmithTraceExportFileLoader)
    assert loader.config.path.as_posix() == "exports/langsmith-traces"


def test_langsmith_start_time_requires_a_timezone():
    with pytest.raises(ValidationError, match="timezone"):
        app_config.LangSmithTraceSourceConfig(
            project="glamr-ux",
            start_time=datetime(2026, 9, 1),
        )


def test_trace_config_rejects_multiple_sources():
    with pytest.raises(ValidationError, match="exactly one loader"):
        RunConfig(
            trace=TraceConfig(
                langsmith=app_config.LangSmithTraceSourceConfig(project="glamr-ux"),
                langsmith_trace_export_file=app_config.LangSmithTraceExportFileSourceConfig(
                    path="traces"
                ),
            ),
            compass_signals=CompassSignalsConfig(anomaly_and_patterns={}),
        )
