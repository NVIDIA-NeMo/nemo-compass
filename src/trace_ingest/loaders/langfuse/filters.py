# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose version-specific Langfuse selection filters."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime

from trace_ingest.loaders.langfuse.config import LangfuseTraceLoadError


def _bounded_filter(
    filter_string: str | None,
    *,
    from_timestamp: datetime,
    to_timestamp: datetime,
    time_column: str = "timestamp",
) -> str | None:
    if filter_string is None:
        return None
    try:
        conditions = json.loads(filter_string)
    except json.JSONDecodeError as error:
        raise LangfuseTraceLoadError(
            f"Langfuse filter must be a JSON array: {error.msg}"
        ) from error
    if not isinstance(conditions, list) or not all(
        isinstance(condition, Mapping) for condition in conditions
    ):
        raise LangfuseTraceLoadError("Langfuse filter must be a JSON array of conditions")
    time_conditions = [
        {
            "type": "datetime",
            "column": time_column,
            "operator": ">=",
            "value": from_timestamp.isoformat(),
        },
        {
            "type": "datetime",
            "column": time_column,
            "operator": "<",
            "value": to_timestamp.isoformat(),
        },
    ]
    return json.dumps([*conditions, *time_conditions], separators=(",", ":"))
