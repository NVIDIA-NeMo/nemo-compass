# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Anthropic Messages usage and prompt cache TTL.

``input_tokens`` excludes cache reads (``cache_read_input_tokens``) and writes
(``cache_creation_input_tokens``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from trace_ingest.models import TokenCounts
from trace_ingest.token_usage._core import count
from trace_ingest.token_usage._core import token_counts as _token_counts

_TTL_SECONDS = {"5m": 300, "1h": 3600}
_USAGE_TTL_KEYS = (("ephemeral_5m_input_tokens", 300), ("ephemeral_1h_input_tokens", 3600))


def token_counts(
    usage: Mapping[str, Any], *, request: Mapping[str, Any] | None = None
) -> TokenCounts | None:
    return _token_counts(
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        cache_read=usage.get("cache_read_input_tokens"),
        cache_write=usage.get("cache_creation_input_tokens"),
        input_includes_cache=False,
        ttl=ttl(request, usage),
    )


def ttl(request: Mapping[str, Any] | None, usage: Mapping[str, Any] | None) -> int | None:
    """Shortest TTL the request asks for, else the shortest the provider reports writing to."""

    return _request_ttl_seconds(request) or _usage_ttl_seconds(usage)


def _request_ttl_seconds(request: Mapping[str, Any] | None) -> int | None:
    """Shortest explicit breakpoint TTL; None when any breakpoint does not set a known one."""

    ttls: list[int] = []
    for control in _cache_controls(request):
        ttl_seconds = _TTL_SECONDS.get(str(control.get("ttl")))
        if ttl_seconds is None:
            return None
        ttls.append(ttl_seconds)
    return min(ttls) if ttls else None


def _usage_ttl_seconds(usage: Mapping[str, Any] | None) -> int | None:
    """Shortest TTL written to, from ``cache_creation``.

    Only calls that write the cache carry it; the request body is the better
    source. Some gateways keep the breakdown only per entry of ``usage.iterations``.
    """

    if usage is None:
        return None
    iterations = usage.get("iterations")
    creations = [
        creation
        for creation in [
            usage.get("cache_creation"),
            *(
                item.get("cache_creation")
                for item in (iterations if isinstance(iterations, list) else [])
                if isinstance(item, Mapping)
            ),
        ]
        if isinstance(creation, Mapping)
    ]
    for key, ttl_seconds in _USAGE_TTL_KEYS:
        if any(count(creation.get(key)) for creation in creations):
            return ttl_seconds
    return None


def _cache_controls(value: object) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        control = value.get("cache_control")
        if isinstance(control, Mapping):
            yield control
        for child in value.values():
            yield from _cache_controls(child)
    elif isinstance(value, list):
        for child in value:
            yield from _cache_controls(child)


__all__ = ["token_counts", "ttl"]
