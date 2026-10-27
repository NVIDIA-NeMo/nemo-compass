# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from collections.abc import Callable
from datetime import datetime
from typing import Annotated, Literal

from nooa import Agent, no_trace, strategy
from nooa.config import CodeActConfig
from nooa.strategies import CodeActStrategy
from nooa.strategy_validation import InvariantError
from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema

from compass import telemetry_phases as telemetry
from compass.compass_signals._trace import walk_spans
from compass.compass_signals.compass_signals import CompassSignalResult
from compass.insight import Insight, SpanEvidence, TraceEvidence
from compass.traces import Trace, TraceSnapshot

Decision = Literal["match", "no_match", "unknown"]


class EvidenceDecision(BaseModel):
    """Whether a trace matches the insight, with IDs of the spans that support it."""

    model_config = ConfigDict(extra="forbid", revalidate_instances="always")

    status: Decision
    witness_span_ids: tuple[str, ...] = ()


class EvidenceCompletion(BaseModel):
    """An insight with a function and examples for checking its evidence.

    The application saves only the insight. The function and checks stay in memory.
    """

    insight: Insight
    # JSON cannot carry a function. Return this object from the agent's Python session.
    predicate: SkipJsonSchema[Callable[[Trace], EvidenceDecision | dict[str, object]] | None] = (
        Field(default=None, exclude=True, repr=False)
    )
    checks: list[tuple[Trace, Decision]] = Field(default_factory=list, exclude=True, repr=False)
    unresolved_reason: str | None = None

    def _evaluate(self, trace: Trace) -> EvidenceDecision:
        assert self.predicate is not None
        span_ids = {visit.span.id for visit in walk_spans(trace)}
        decision = EvidenceDecision.model_validate(self.predicate(trace))
        if decision.status == "match" and not decision.witness_span_ids:
            raise ValueError(f"Match in trace {trace.id} must cite at least one span ID")
        if set(decision.witness_span_ids) - span_ids:
            raise ValueError(f"Cited span ID does not exist in trace {trace.id}")
        return decision

    def validate_result(self) -> None:
        """Check the function against examples, or require a reason for skipping the scan."""
        if self.predicate is None:
            if not self.unresolved_reason or not self.unresolved_reason.strip() or self.checks:
                raise ValueError(
                    "Without a predicate, leave checks empty. Explain why in unresolved_reason. "
                    "The reason must not be blank."
                )
            return
        if not callable(self.predicate) or self.unresolved_reason is not None:
            raise ValueError("Provide a Python function as predicate. Set unresolved_reason=None.")
        if {expected for _, expected in self.checks} != {"match", "no_match", "unknown"}:
            raise ValueError(
                "Checks must include expected results for match, no_match, and unknown"
            )
        for trace, expected in self.checks:
            actual = self._evaluate(trace).status
            if actual != expected:
                raise ValueError(f"Check {trace.id}: expected {expected}, got {actual}")

    def scan(self, snapshot: TraceSnapshot) -> tuple[list[TraceEvidence], int]:
        """Scan all traces after the example checks pass.

        Return matching traces with their supporting spans and the count of unknown traces.
        """
        self.validate_result()
        if self.predicate is None:
            raise ValueError(self.unresolved_reason)
        matches = []
        unknown = 0
        for trace in snapshot:
            trace_id = trace.id
            decision = self._evaluate(trace)
            if decision.status == "match":
                matches.append(
                    TraceEvidence(
                        trace_id=trace_id,
                        spans=[
                            SpanEvidence(span_id=sid)
                            for sid in dict.fromkeys(decision.witness_span_ids)
                        ],
                    )
                )
            unknown += decision.status == "unknown"
        if not matches:
            raise ValueError(
                "No traces matched despite a passing positive check. Review the function and example."
            )
        return matches, unknown

    def apply(
        self,
        snapshot: TraceSnapshot,
        run_timestamp: datetime,
    ) -> Insight:
        """Add matching evidence, or keep the insight unchanged if the scan fails."""
        with telemetry.evidence_scan(self.insight, len(snapshot)) as record:
            try:
                matches, unknown = self.scan(snapshot)
            except Exception as error:
                skipped = (
                    self.predicate is None
                    and bool(self.unresolved_reason and self.unresolved_reason.strip())
                    and not self.checks
                )
                reason = str(error)[:300]
                telemetry.record_scan_failure(record, error, skipped=skipped)
                warnings.warn(
                    f"The evidence scan did not complete for {self.insight.name!r}: "
                    f"{type(error).__name__}: {reason}. "
                    "The insight keeps its existing references. The scan added no new references.",
                    stacklevel=2,
                )
                return self.insight
            telemetry.record_scan_result(record, len(matches), unknown)
        if unknown:
            warnings.warn(
                f"Evidence scan incomplete for {self.insight.name!r}: "
                f"The function could not decide whether {unknown} of {len(snapshot)} traces match. "
                "The scan added its matches, but other supporting traces may remain without references.",
                stacklevel=2,
            )
        evidence = {item.trace_id: item for item in self.insight.evidence}
        added_trace = any(item.trace_id not in evidence for item in matches)
        for item in matches:
            previous = evidence.get(item.trace_id, item)
            spans = {span.span_id: span for span in previous.spans}
            for evidence_span in item.spans:
                spans.setdefault(evidence_span.span_id, evidence_span)
            evidence[item.trace_id] = previous.model_copy(update={"spans": list(spans.values())})
        return self.insight.model_copy(
            update={
                "evidence": list(evidence.values()),
                "updated_date": run_timestamp if added_trace else self.insight.updated_date,
            }
        )


def _validate_completions(
    _agent: Agent, completions: list[EvidenceCompletion], _call: object
) -> None:
    """Reject invalid returns while the agent can still fix them in Python."""
    for completion in completions:
        try:
            completion.validate_result()
        except Exception as error:
            raise InvariantError(
                f"{completion.insight.name}: {error}. "
                "Inside Python, call return_result with the EvidenceCompletion objects you "
                "created. JSON cannot preserve their functions. If you cannot check the evidence, "
                "return predicate=None and checks=[]. Explain why in unresolved_reason."
            ) from error


class InsightCompilation(Agent):
    @no_trace
    async def compile_insights(
        self,
        compass_signals: list[CompassSignalResult],
        trace_snapshot: TraceSnapshot,
        existing_insights: list[Insight],
        run_timestamp: datetime,
    ) -> list[Insight]:
        completions = await self._compile_insights(
            compass_signals, trace_snapshot, existing_insights, run_timestamp
        )
        return [completion.apply(trace_snapshot, run_timestamp) for completion in completions]

    @no_trace
    @strategy(CodeActStrategy(config=CodeActConfig(postconditions=[_validate_completions])))
    async def _compile_insights(
        self,
        compass_signals: list[CompassSignalResult],
        trace_snapshot: TraceSnapshot,
        existing_insights: list[Insight],
        run_timestamp: datetime,
    ) -> Annotated[  # ty: ignore[empty-body] -- Nooa generates the ellipsis method at runtime.
        list[EvidenceCompletion],
        "Contains live Python functions. Finish inside execute_python with "
        "return_result(completions), where completions is the actual list. "
        "Do not call the standalone return_result tool for this result. "
        "Never serialize the list or quote its name, an expression, or an object repr.",
    ]:
        """
        Completion protocol: your result contains live Python functions that JSON
        cannot preserve. Finish this task inside execute_python by calling
        return_result(completions), where completions is the actual list of
        EvidenceCompletion objects. Do not call the standalone return_result tool.

        Your job is to be the last step of the insight creation process. An
        insight is a bug report or problem that is identified by looking at the
        runtime traces of an AI agent, either in production or an offline
        evaluation context.

        The goal of an insight is to show a developer a problem with their agent
        that they can fix by making a change to their code. Often times this
        could be updating the prompt, changing time outs, fixing code logic,
        updating their agent to use a different model, etc.

        The Compass Signal is a set of potential problems that have been
        surfaced by earlier stages. These may be real problems, and they may not
        be fixable.

        Insights should be ranked holistically based on the following factors:
        1. How fixable is this issue by the agent developer?

        2. How high is the  impact of this issue? For example if the formatting
        is slightly incorrect or if there is a contradiction that's not the end
        of the world. If the agent is failing to even produce a response for the
        user 30% of the time though because of an error, that's a huge deal!

        Your job is to validate that these insights are all impactful and
        fixable. If any proposed problem from a Compass Signal doesn't meet
        this bar, it should be discarded. Some things are not ideal, but are
        recovered by an agent -- for example sometimes an agent will call a
        coding tool that fails, but then go on to recover. There's not anything
        we can do about this issue! And it's OK. Experts do this too. You can
        validate by examining the referenced traces. Every new insight we create
        must have more than one trace that supports it. We want to identify
        problems that are broader in scope than a one-off.

        Inspect short excerpts of supporting traces with
        trace_snapshot.get_trace_by_id(trace_id). Use Python for snapshot scans.

        After validating, you must merge the new insights with the existing
        insights, and across Compass Signals.

        Here's an example of two insights that are semantic duplicates and
        should be merged:

        1. When users request their airline ticket to be canceled, the system
        prompt allows it but there isn't a tool for it, so the agent tells the
        user that the ticket can't be canceled

        2. Ticket cancelation tool is missing from tool schema

        Here's an example of two insights that should remain independent:

        1. Ticket cancellation calls time out for users that are already checked
        in

        2. Ticket cancellation causes an error when the user has a pending
        refund

        Here's an example of an insight that is overly broad. This kind of
        insight should never be created by you, but if there is an existing one
        in the existing_insights, you should keep it and not modify it.

        1. Bash tool call is returning errors

        Preserve each existing insight's id exactly. Never invent an id or
        assign an existing id to a new insight; new insights have id=None.
        Existing insights should never be removed or modified, except that you
        can update their evidence to match new traces you identified and fill
        a missing severity_reason as described below. Keep existing references,
        including those outside this snapshot.
        When you merge insights, check that added references support the merged
        claim. Merge evidence by trace ID. Do not copy every candidate into evidence.

        You are given run_timestamp, the current time for this run. Set an
        insight's updated_date to run_timestamp whenever you create it, or
        whenever you add a trace to an existing insight that it did not
        already support. If an existing insight gains no new trace this run,
        leave its updated_date exactly as given, including leaving it unset
        if it was already unset -- never invent one and never clear one.

        After merging findings, assign severity to each new insight based on
        the observed consequence in its supporting traces, including recovery:
        - high: the agent fails its primary task, silently loses state, or
          produces materially incorrect user-visible output without recovery.
        - med: the agent completes the task but incurs unnecessary turns,
          delay, partial recovery, or degraded output quality.
        - low: minor inconvenience or cosmetic impact with no meaningful
          degradation of the completed task.
        Leave severity=None when the traces do not establish the outcome.
        Judge consequence independently of occurrence count and confidence;
        do not reduce a severe consequence because it occurs in few traces.
        Use the merged supporting evidence, not just the failing span. Source
        code access is not required. Preserve existing insights' ratings.

        For each new insight with a severity rating, add severity_reason as one
        short, plain sentence stating the observed consequence that supports
        the rating. You may also fill a missing severity_reason on an existing
        insight with a severity rating. Leave severity_reason=None when severity
        is unknown. Preserve existing ratings and reasons; never invent an
        unobserved task outcome.

        Leave confidence=None on new insights; an optional code review sets it.

        When inspecting supporting traces, record the specific relevant Span.id values
        in evidence[].spans as span_id, grouped by trace_id. A trace may have several supporting spans
        at any nesting depth. Use only IDs observed in that trace, never event row IDs.
        Do not select unrelated spans or every span automatically. Omit spans when
        the evidence concerns the whole trace or no specific span can be identified.
        Preserve existing evidence spans when their traces are unavailable.
        Leave all evidence url fields unset; the application resolves URLs after compilation.

        After validating, narrowing, and merging insights, find all traces that support
        each final claim. Keep only checked examples and existing references in evidence.
        The application adds further matches. Use this workflow for every snapshot
        size. Review examples to develop the function below. Manual review and
        similarity search cannot replace it.

        1. Define the claim. State in your working analysis what the recorded data
           must show:
           - The operation, trigger, and behavior.
           - Required links between events, order, recovery, duration, and impact.
           - Conditions that exclude a match, and evidence needed to decide.

           An error alone cannot prove its cause or that the task failed. Never
           weaken the claim to gain references. If you cannot check required
           conditions, return the insight with predicate=None and checks=[].
           Explain why in unresolved_reason.

        2. Write predicate(trace), a function that checks whether one trace supports
           the claim. Work with the actual Trace fields. Use supporting_trace_ids,
           candidate_trace_ids, and signal artifacts to find examples and investigate
           disagreements. Read traces with trace_snapshot.get_trace_by_id(id).
           walk_spans(trace) yields visits with .span.

           Print only short excerpts. Keep full data in Python, including span IDs
           and event order. Do not change source traces. The function's result must
           depend only on the trace. Put needed imports inside the function so it
           can run on its own. Never hardcode trace IDs, expected counts, or answer lists.

           Return EvidenceDecision(status=..., witness_span_ids=(...)):
           - match: recorded events establish every required condition. Cite the
             span.id values that show it. A shared group or keyword is insufficient.
             Copied history alone is also insufficient. Link copied records to
             their original events.
           - no_match: recorded evidence shows that the claim does not apply.
           - unknown: required evidence is missing or uninterpretable. Use no_match
             if another recorded condition already shows that the claim does not apply.

        3. Build a small checks=[(Trace, expected_status), ...] list. Include a known
           match, similar cases that fail a required condition, and missing required data.
           Use model_copy(deep=True) to make test copies. Change one condition while
           keeping the others satisfied. Remove individual required fields from
           matching examples instead of testing only empty traces.

           If a supplied example fails, investigate the function, extracted data,
           and reference. Do not relax the claim just to make the example pass.

        4. Create EvidenceCompletion(insight=initial_insight, predicate=predicate,
           checks=checks). Run .scan(trace_snapshot) on this object. It runs the
           example checks and checks the returned fields and cited span IDs. It
           scans every trace, including those outside candidate_trace_ids.

           If execution or checks fail, change the predicate at most once. Rerun
           the checks and scan after that change. If it still fails, return the
           insight with predicate=None and checks=[]. Explain why in unresolved_reason.
           Discard all proposed additions. A broken function returning zero matches
           does not prove there are none.

        Finish inside execute_python using the objects and functions you created:

            completions = [
                EvidenceCompletion(insight=insight, predicate=predicate, checks=checks)
            ]
            return_result(completions)

        Include every final insight in completions. For no final insights, call
        return_result([]) inside execute_python. Never pass a string such as
        "[ec]", "completions", or "[EvidenceCompletion(...)]". For example, the
        standalone tool call {"result": "[ec]"} sends text, not your Python objects,
        and fails validation. Do not serialize the completion objects.

        Do not add their matches to evidence yourself. You can
        fix a rejected return while your Python variables still exist. Fixing how
        you return an existing function does not count as changing it.

        The application reruns checks and scans before adding trace IDs without
        duplicates. It keeps the supporting span IDs and sets updated_date to
        run_timestamp if the scan adds a new trace. If a scan fails, it keeps existing
        references. If some traces are unknown, it adds matches and warns that
        coverage is incomplete.
        Only set unresolved_reason when returning no predicate or checks.

        A span's existence does not prove it supports the claim. A completed scan
        covers this snapshot under the checked function. It does not prove that
        the function found every supporting trace. Note these limits in your working analysis.

        Return one EvidenceCompletion per final insight.
        Include existing insights.
        """
        ...


__all__ = ["Insight", "InsightCompilation"]
