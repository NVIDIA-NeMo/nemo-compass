# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load canonical ``Trace`` objects from JSONL files."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from trace_ingest.loaders.trace_loaders import TraceDescription
from trace_ingest.models import Span, SpanKind, Trace, TraceSnapshot
from trace_ingest.source_links import file_source_url

__all__ = ["FSDataLoadError", "FSDataLoader"]


class FSDataLoadError(ValueError):
    """A filesystem trace corpus is not valid canonical Trace JSONL."""


def _spans(trace: Trace) -> Iterator[Span]:
    pending = list(reversed(trace.root_spans))
    while pending:
        span = pending.pop()
        yield span
        pending.extend(reversed(span.children))


def _parse_trace(raw: str | bytes, *, path: Path, line_number: int) -> Trace:
    try:
        return Trace.model_validate_json(raw)
    except ValidationError as error:
        problem = (
            "malformed JSON"
            if any(detail["type"] == "json_invalid" for detail in error.errors())
            else "invalid trace"
        )
        raise FSDataLoadError(f"{path}:{line_number}: {problem}: {error}") from error


@dataclass
class FSDataLoader:
    """Read one canonical ``Trace`` JSON object per non-empty line."""

    path: Path
    _snapshot: TraceSnapshot | None = field(default=None, init=False, repr=False)
    _description: TraceDescription | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def load(self) -> TraceSnapshot:
        if self._snapshot is not None:
            return self._snapshot

        call_count = 0
        cases: set[str] = set()

        def traces() -> Iterator[Trace]:
            nonlocal call_count
            with self.path.open("rb") as corpus:
                for line_number, raw in enumerate(corpus, start=1):
                    if raw.isspace():
                        continue
                    trace = _parse_trace(raw, path=self.path, line_number=line_number)
                    trace.source_url = file_source_url(self.path, line_number)
                    for span in _spans(trace):
                        span.source_url = None
                        call_count += span.kind is SpanKind.TOOL
                    cases.add(str(trace.attributes.get("logical_case_id") or trace.id))
                    yield trace

        try:
            snapshot = TraceSnapshot(traces())
        except FSDataLoadError:
            raise
        except ValueError as error:
            raise FSDataLoadError(f"{self.path}: {error}") from error
        if not snapshot:
            raise FSDataLoadError(f"{self.path}: contains no traces")
        self._snapshot = snapshot
        self._description = {
            "source": f"fs:{self.path.resolve()}",
            "trace_count": len(snapshot),
            "call_count": call_count,
            "distinct_logical_cases": len(cases),
        }
        return snapshot

    def describe(self) -> TraceDescription:
        self.load()
        assert self._description is not None
        return self._description
