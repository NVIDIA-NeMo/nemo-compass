# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reconstruct live Langfuse v4 traces from observations and scores."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from trace_ingest.loaders.helper.merge import Record, assemble
from trace_ingest.loaders.langfuse.config import (
    LangfuseTraceConfig,
    LangfuseTraceDescription,
    LangfuseTraceLoadError,
    _describe_live_source,
)
from trace_ingest.loaders.langfuse.filters import _bounded_filter
from trace_ingest.loaders.langfuse.normalization import (
    _ORCHESTRATION,
    _normalize_v4_trace,
    _started_at,
)
from trace_ingest.loaders.langfuse.v4_records import _group_observations, _group_scores
from trace_ingest.models import TraceSnapshot

if TYPE_CHECKING:
    from langfuse import Langfuse
    from langfuse.api.commons.types.observation_v2 import ObservationV2
    from langfuse.api.scores_v3.types.score_v3 import ScoreV3


_API_PAGE_SIZE = 100
_MAX_CONCURRENT_TRACES = 4
_SCORE_BATCH_SIZE = 100


@dataclass
class LangfuseV4TraceLoader:
    """Load observations and scores from a detected v4 server."""

    config: LangfuseTraceConfig
    client: Langfuse = field(repr=False)
    base_url: str
    _snapshot: TraceSnapshot = field(
        default_factory=lambda: TraceSnapshot([]), init=False, repr=False
    )

    def load(self) -> TraceSnapshot:
        """Reconstruct bounded traces from observations and scores, merged by session."""
        client = self.client
        base_url = self.base_url
        selected: dict[str, datetime] = {}
        group_keys: set[str] = set()
        capped = False
        cursor: str | None = None
        seen_cursors: set[str] = set()
        bounded_filter = _bounded_filter(
            self.config.filter_string,
            from_timestamp=self.config.from_timestamp,
            to_timestamp=self.config.to_timestamp,
            time_column="startTime",
        )
        while True:
            response = client.api.observations.get_many(
                limit=_API_PAGE_SIZE,
                fields="core,basic",
                from_start_time=self.config.from_timestamp,
                to_start_time=self.config.to_timestamp,
                filter=bounded_filter,
                cursor=cursor,
            )
            for observation in response.data:
                if not observation.trace_id:
                    raise LangfuseTraceLoadError("Langfuse returned an observation without traceId")
                if observation.trace_id in selected:
                    continue
                group = observation.session_id or observation.trace_id
                if group not in group_keys and len(group_keys) == self.config.max_traces:
                    capped = True
                    break
                group_keys.add(group)
                selected[observation.trace_id] = _started_at(observation)
            if capped:
                break
            cursor = _next_cursor(response.meta.cursor, seen_cursors, bool(response.data))
            if cursor is None:
                break

        trace_ids = sorted(selected, key=lambda key: (selected[key], key))
        # map preserves selection order even when requests complete out of order.
        with ThreadPoolExecutor(max_workers=_MAX_CONCURRENT_TRACES) as executor:
            groups = _group_observations(
                [
                    observation
                    for observations in executor.map(self._fetch_observations, trace_ids)
                    for observation in observations
                ]
            )
        scores: list[ScoreV3] = []
        for offset in range(0, len(trace_ids), _SCORE_BATCH_SIZE):
            cursor = None
            seen_cursors = set()
            while True:
                score_response = client.api.scores_v3.get_many_v3(
                    trace_id=",".join(trace_ids[offset : offset + _SCORE_BATCH_SIZE]),
                    fields="details,subject,annotation",
                    limit=_API_PAGE_SIZE,
                    cursor=cursor,
                )
                scores.extend(score_response.data)
                cursor = _next_cursor(
                    score_response.meta.cursor, seen_cursors, bool(score_response.data)
                )
                if cursor is None:
                    break
        by_trace, _ = _group_scores(groups, scores)
        records: list[Record] = []
        for trace_id in trace_ids:
            trace = _normalize_v4_trace(trace_id, groups[trace_id], by_trace[trace_id], base_url)
            trace.attributes["observation_window"] = {
                "from_timestamp": self.config.from_timestamp.isoformat(),
                "to_timestamp": self.config.to_timestamp.isoformat(),
                "may_exclude_trace_steps": True,
            }
            langfuse = trace.attributes["langfuse"]
            assert isinstance(langfuse, dict)
            records.append(Record(trace, langfuse.get("input"), langfuse.get("output")))
        try:
            traces = assemble(records, orchestration_subtypes=_ORCHESTRATION, capped=capped)
        except ValueError as error:
            raise LangfuseTraceLoadError(str(error)) from error
        self._snapshot = TraceSnapshot(traces)
        return self._snapshot

    def _fetch_observations(self, trace_id: str) -> list[ObservationV2]:
        observations: list[ObservationV2] = []
        cursor = None
        seen_cursors = set()
        while True:
            # Keep the time bounds so Langfuse can prune its time partitions.
            # Drop the selection filter to retain other steps of the selected
            # trace inside this window.
            response = self.client.api.observations.get_many(
                trace_id=trace_id,
                from_start_time=self.config.from_timestamp,
                to_start_time=self.config.to_timestamp,
                fields="core,basic,io,metadata,model,usage,prompt,trace_context",
                limit=_API_PAGE_SIZE,
                cursor=cursor,
            )
            observations.extend(response.data)
            cursor = _next_cursor(response.meta.cursor, seen_cursors, bool(response.data))
            if cursor is None:
                break
        if not observations or any(obs.trace_id != trace_id for obs in observations):
            raise LangfuseTraceLoadError(
                f"Langfuse returned missing or mismatched observations for trace {trace_id!r}"
            )
        return observations

    def describe(self) -> LangfuseTraceDescription:
        """Describe the configured source and most recently loaded corpus."""
        return _describe_live_source(self.config, self.base_url, "v4", self._snapshot)


def _next_cursor(cursor: str | None, seen: set[str], has_data: bool) -> str | None:
    if cursor is not None:
        if cursor in seen or not has_data:
            raise LangfuseTraceLoadError("Langfuse returned non-progressing cursor pagination")
        seen.add(cursor)
    return cursor
