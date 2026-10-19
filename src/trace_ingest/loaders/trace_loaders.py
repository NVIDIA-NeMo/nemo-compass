# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trace source boundary."""

from __future__ import annotations

from typing import Protocol, TypedDict

from trace_ingest.models import TraceSnapshot


class TraceDescription(TypedDict):
    """Source-independent corpus facts recorded with analysis results."""

    source: str
    trace_count: int
    call_count: int
    distinct_logical_cases: int


class TraceLoader(Protocol):
    """Load one source into a normalized, reiterable trace snapshot."""

    def load(self) -> TraceSnapshot:
        """Return the configured source as normalized traces."""
        ...

    def describe(self) -> TraceDescription:
        """Describe the configured source and most recently loaded corpus."""
        ...
