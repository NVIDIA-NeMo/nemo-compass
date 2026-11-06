# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load a Langfuse CLI observation response and optional score response."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, JsonValue

from trace_ingest.loaders.helper.merge import Record
from trace_ingest.loaders.langfuse.config import (
    LANGFUSE_DEFAULT_MAX_TRACES,
    LangfuseTraceLoadError,
    _describe_corpus,
    _LangfuseCorpusDescription,
)
from trace_ingest.loaders.langfuse.file_io import _newest_conversations
from trace_ingest.loaders.langfuse.normalization import _normalize_v4_trace, _started_at
from trace_ingest.loaders.langfuse.v4_records import _group_observations, _group_scores
from trace_ingest.models import TraceSnapshot
from trace_ingest.source_links import file_source_url


class _Meta(BaseModel):
    truncated: bool = Field(strict=True)
    cursor: str | None = None


class _Body(BaseModel):
    data: list[dict[str, JsonValue]]
    meta: _Meta


class _Response(BaseModel):
    status: Literal[200]
    body: _Body


def _read_cli_data(path: Path) -> list[dict[str, JsonValue]]:
    try:
        response = _Response.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as error:
        raise LangfuseTraceLoadError(
            f"Invalid Langfuse CLI export {path}; expected a successful --all --json response: {error}"
        ) from error
    if response.body.meta.truncated or response.body.meta.cursor:
        raise LangfuseTraceLoadError(
            f"Incomplete Langfuse CLI export {path}; increase --max-items or narrow the query and export again"
        )
    return response.body.data


@dataclass(frozen=True)
class LangfuseV4FileTraceConfig:
    """Observation and score CLI response files, with optional platform links."""

    path: Path
    scores_path: Path | None = None
    max_traces: int = LANGFUSE_DEFAULT_MAX_TRACES
    base_url: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if self.scores_path is not None:
            object.__setattr__(self, "scores_path", Path(self.scores_path))
        if self.max_traces < 1:
            raise ValueError("max_traces must be at least 1")


class LangfuseV4FileTraceDescription(_LangfuseCorpusDescription):
    """Offline v4 corpus and score coverage."""

    path: str
    scores_path: str | None
    export_trace_count: int
    max_traces: int
    truncated: bool
    base_url: str | None
    ignored_score_count: int


@dataclass
class LangfuseV4FileTraceLoader:
    """Join observation and score records, reuse online normalization, and merge by session."""

    config: LangfuseV4FileTraceConfig
    _snapshot: TraceSnapshot = field(
        default_factory=lambda: TraceSnapshot([]), init=False, repr=False
    )
    _export_trace_count: int = field(default=0, init=False)
    _ignored_score_count: int = field(default=0, init=False)
    _merged_trace_count: int = field(default=0, init=False)

    def load(self) -> TraceSnapshot:
        from langfuse.api.commons.types.observation_v2 import ObservationV2
        from langfuse.api.core.pydantic_utilities import parse_obj_as
        from langfuse.api.scores_v3.types.score_v3 import ScoreV3

        try:
            observations = parse_obj_as(list[ObservationV2], _read_cli_data(self.config.path))
            scores = (
                parse_obj_as(list[ScoreV3], _read_cli_data(self.config.scores_path))
                if self.config.scores_path is not None
                else []
            )
        except ValueError as error:
            raise LangfuseTraceLoadError(f"Invalid Langfuse export records: {error}") from error
        groups = _group_observations(observations)
        by_trace, ignored = _group_scores(groups, scores)
        records: list[tuple[datetime, str, Record]] = []
        for trace_id, items in groups.items():
            trace = _normalize_v4_trace(trace_id, items, by_trace[trace_id], self.config.base_url)
            if trace.source_url is None:
                trace.source_url = file_source_url(self.config.path)
            trace.attributes["export_path"] = str(self.config.path.resolve())
            langfuse = trace.attributes["langfuse"]
            assert isinstance(langfuse, dict)
            record = Record(trace, langfuse.get("input"), langfuse.get("output"))
            # Live selection lists observations newest first, so a trace ranks by its newest.
            records.append((max(_started_at(obs) for obs in items), trace_id, record))
        traces, self._merged_trace_count = _newest_conversations(
            [(time, record) for time, _, record in sorted(records, key=lambda item: item[:2])],
            self.config.max_traces,
        )
        self._snapshot = TraceSnapshot(traces)
        self._export_trace_count = len(groups)
        self._ignored_score_count = ignored
        return self._snapshot

    def describe(self) -> LangfuseV4FileTraceDescription:
        return {
            **_describe_corpus(f"langfuse-v4-export:{self.config.path.resolve()}", self._snapshot),
            "path": str(self.config.path.resolve()),
            "scores_path": str(self.config.scores_path.resolve())
            if self.config.scores_path is not None
            else None,
            "export_trace_count": self._export_trace_count,
            "max_traces": self.config.max_traces,
            "truncated": self._merged_trace_count > self._snapshot.trace_count,
            "base_url": self.config.base_url,
            "ignored_score_count": self._ignored_score_count,
        }
