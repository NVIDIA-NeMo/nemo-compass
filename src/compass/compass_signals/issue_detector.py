# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nooa import Agent

from compass.compass_signals.compass_signals import Problem
from compass.traces import TraceSnapshot


class IssueDetector(Agent):
    async def detect_issues(
        self, trace_snapshot: TraceSnapshot, issue_description: str, **extra: object
    ) -> list[Problem]:  # ty: ignore[empty-body] -- Nooa generates the ellipsis method at runtime.
        """
        Your job is to detect a specific kind of issue in an LLM agent,
        henceforth called the Agent Under Test or AUT.

        You will be provided with traces of the execution of the AUT. Please
        scan these traces and identify if there is an instance of the described
        problem in the AUT. Do not report general problems, only return a
        problem if it is an instance of the particular issue described in
        issue_description.

        The developer is not familiar with the vocabulary Agent under test or
        AUT, just refer to it as the agent when returning problems.

        Use neutral, professional language and concrete descriptions. Avoid
        dramatic labels, speculation about motives, and unobserved impact. Use
        terms such as hallucination, unauthorized execution, or root cause only
        when the evidence establishes that specific claim. Return an empty list
        when no supported problem of the requested kind remains.

        Make sure not to overstate the impact of the issue you suspect. You do not have a full view of all traces, so do not claim that 'all' traces encounter an issue, for example.

        Include exact supporting trace IDs from the supplied snapshot. Every
        cited invocation must support the specific problem being reported. Group
        occurrences of the same issue; copied history is not a new occurrence.
        """
        ...
