# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Anthropic Messages usage.

``input_tokens`` excludes cache reads (``cache_read_input_tokens``) and writes
(``cache_creation_input_tokens``); ``cache_creation`` splits the writes by TTL.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trace_ingest.models import TokenCounts
from trace_ingest.token_usage._core import token_counts as _token_counts

_TTL_KEYS = {"ephemeral_5m_input_tokens": 300, "ephemeral_1h_input_tokens": 3600}


def token_counts(usage: Mapping[str, Any]) -> TokenCounts | None:
    creation = _cache_creation(usage)
    return _token_counts(
        input_tokens=usage.get("input_tokens"),
        cached_input_tokens=usage.get("cache_read_input_tokens"),
        output_tokens=usage.get("output_tokens"),
        cache_write_tokens=usage.get("cache_creation_input_tokens"),
        cache_writes_by_ttl={ttl: creation.get(key) for key, ttl in _TTL_KEYS.items()},
    )


def _cache_creation(usage: Mapping[str, Any]) -> Mapping[str, Any]:
    """The ``cache_creation`` breakdown.

    Some gateways keep it only per entry of ``usage.iterations``; it is read
    there only when there is a single entry, since several would need summing.
    """

    creation = usage.get("cache_creation")
    iterations = usage.get("iterations")
    if creation is None and isinstance(iterations, list) and len(iterations) == 1:
        item = iterations[0]
        creation = item.get("cache_creation") if isinstance(item, Mapping) else None
    return creation if isinstance(creation, Mapping) else {}


__all__ = ["token_counts"]
