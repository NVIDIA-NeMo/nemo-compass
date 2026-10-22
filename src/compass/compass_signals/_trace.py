# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private traversal primitives shared by Compass Signals."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from compass.traces import Span, Trace


@dataclass(frozen=True)
class SpanVisit:
    """One span and its stable position in a depth-first trace traversal."""

    trace_id: str
    span: Span
    path: tuple[int, ...]
    index: int

    @property
    def source_pointer(self) -> dict[str, object]:
        source_pointer = self.span.attributes.get("source_pointer")
        if isinstance(source_pointer, Mapping):
            return dict(source_pointer)
        return {
            "trace_id": self.trace_id,
            "span_id": self.span.id,
            "span_path": list(self.path),
        }


def walk_spans(trace: Trace) -> Iterator[SpanVisit]:
    """Yield every span once in root-first, depth-first list order."""

    index = 0

    def walk(children: list[Span], parent_path: tuple[int, ...]) -> Iterator[SpanVisit]:
        nonlocal index
        for child_index, span in enumerate(children):
            path = (*parent_path, child_index)
            visit = SpanVisit(trace_id=trace.id, span=span, path=path, index=index)
            index += 1
            yield visit
            yield from walk(span.children, path)

    yield from walk(trace.root_spans, ())


def session_of(trace: Trace) -> str | None:
    """The source-recorded session ID (logical_case_id) of a trace, if any."""

    case = trace.attributes.get("logical_case_id")
    return str(case) if case is not None else None


__all__ = ["SpanVisit", "session_of", "walk_spans"]
