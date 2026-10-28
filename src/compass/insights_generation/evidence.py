# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small review context with explicit access to the original span evidence."""

from nooa.agentdoc import truncating_pformat
from nooa.skill import Skill

from compass.compass_signals._trace import walk_spans
from compass.traces import SpanKind, Trace


class TraceEvidence(Skill):
    def __init__(self, traces: tuple[Trace, ...]) -> None:
        self._spans = {
            (trace.id, visit.span.id): visit.span for trace in traces for visit in walk_spans(trace)
        }

    def overview(self) -> str:
        """Summarize span identities and tool activity; detail can be omitted."""
        rows = [
            {
                "trace_id": trace_id,
                "span_id": span_id,
                "kind": span.kind.value,
                "tool": span.tool_name,
                "error": span.error,
                "user_request": span.tool_call.prior_user_text if span.tool_call else None,
                "input": span.input if span.kind is SpanKind.TOOL else None,
                "output": span.output if span.kind is SpanKind.TOOL else None,
            }
            for (trace_id, span_id), span in self._spans.items()
        ]
        return (
            "Summary only. Inputs, outputs, prompts, and recovery may be omitted. "
            "Use read_span for relevant spans before drawing conclusions.\n"
            + truncating_pformat(
                rows, max_chars=24_000, max_depth=5, max_length=200, max_string=800
            )
        )

    def read_span(self, trace_id: str, span_id: str) -> str:
        """Read original span fields, excluding child spans (each has its own ID).

        Includes prompts and metadata omitted from the summary. Output can truncate;
        missing evidence leaves uncertainty. Batch related span reads in one step.
        """
        span = self._spans[trace_id, span_id]
        return truncating_pformat(
            span.model_dump(mode="json", exclude={"children"}, exclude_unset=True),
            max_chars=16_000,
            max_depth=12,
            max_length=200,
            max_string=8_000,
        )
