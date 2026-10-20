# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Group v4 observations and associate their evaluation scores."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING

from trace_ingest.loaders.langfuse.config import LangfuseTraceLoadError

if TYPE_CHECKING:
    from langfuse.api.commons.types.observation_v2 import ObservationV2
    from langfuse.api.scores_v3.types.score_v3 import ScoreV3


def _group_observations(observations: Sequence[ObservationV2]) -> dict[str, list[ObservationV2]]:
    groups: dict[str, list[ObservationV2]] = defaultdict(list)
    for observation in observations:
        if not observation.id or not observation.trace_id or not observation.project_id:
            raise LangfuseTraceLoadError("observation id, traceId, and projectId must be nonempty")
        groups[observation.trace_id].append(observation)
    projects = {obs.project_id for obs in observations}
    if len(projects) > 1:
        raise LangfuseTraceLoadError("Langfuse observations contain multiple projects")
    return groups


def _group_scores(
    groups: dict[str, list[ObservationV2]], scores: Sequence[ScoreV3]
) -> tuple[dict[str, list[ScoreV3]], int]:
    trace_ids_by_span: dict[str, set[str]] = defaultdict(set)
    projects = set()
    for trace_id, observations in groups.items():
        for observation in observations:
            trace_ids_by_span[observation.id].add(trace_id)
            projects.add(observation.project_id)
    by_trace: dict[str, list[ScoreV3]] = defaultdict(list)
    score_ids: set[str] = set()
    ignored = 0
    for score in scores:
        if score.id in score_ids:
            raise LangfuseTraceLoadError(f"duplicate score id {score.id!r}")
        score_ids.add(score.id)
        if projects and score.project_id not in projects:
            raise LangfuseTraceLoadError("score projectId does not match observations")
        subject = score.subject
        trace_id = None
        if subject is not None and subject.kind == "trace":
            trace_id = subject.id
        elif subject is not None and subject.kind == "observation":
            trace_id = subject.trace_id
            candidates = trace_ids_by_span.get(subject.id, set())
            if trace_id is None and len(candidates) == 1:
                trace_id = next(iter(candidates))
        if trace_id in groups:
            by_trace[trace_id].append(score)
        else:
            ignored += 1
    return by_trace, ignored
