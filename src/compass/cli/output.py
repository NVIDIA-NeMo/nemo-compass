# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transient progress and a single, outcome-first report for an analysis run."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from compass.compass_signals.compass_signals import CompassSignalResult
from compass.insight import Insight
from compass.insights_generation.ethos_validation import EthosRejection

_SIGNAL_LABELS = {
    "anomaly-and-patterns": "Anomalies and patterns",
    "tool-issues": "Tool issues",
    "ethos-divergence": "Ethos divergence",
    "grader-failure-patterns": "Grader failures",
    "user-sentiment": "Sentiment",
}
_SKIP_LABELS = {
    "No ETHOS.md": "no ETHOS.md found",
    "No evaluator results": "no results",
    "No embedding backend configured": "no embedding backend",
}
_LOG_INTERVAL = 10
_SEVERITY_STYLES = {"high": "bold red", "med": "bold yellow", "low": "bold cyan"}


def display_name(name: str) -> str:
    return _SIGNAL_LABELS.get(name, name.replace("-", " ").capitalize())


def count(value: int, singular: str) -> str:
    return f"{value:,} {singular}{'' if value == 1 else 's'}"


@dataclass
class RunResult:
    trace_count: int
    evidence: list[CompassSignalResult]
    insights: list[Insight]
    existing_insights: list[Insight] = field(default_factory=list)
    rejected_by_code: int = 0
    ethos_rejections: list[EthosRejection] = field(default_factory=list)


class RunOutput:
    def __init__(self, console: Console) -> None:
        self.console = console
        self.trace_count: int | None = None
        self.activity = "Loading traces"
        self.started = time.monotonic()

    def render_progress(self) -> Group:
        lines = []
        if self.trace_count is not None:
            lines.append(Text(f"Loaded {count(self.trace_count, 'trace')}."))
        elapsed = f" · {int(time.monotonic() - self.started)}s"
        activity = Text(self.activity, no_wrap=True, overflow="ellipsis")
        activity.truncate(max(0, self.console.width - len(elapsed)), overflow="ellipsis")
        activity.append(elapsed)
        lines.append(activity)
        return Group(*lines)

    @asynccontextmanager
    async def progress(self) -> AsyncIterator[None]:
        if self.console.is_terminal and not self.console.is_dumb_terminal:
            with Live(
                console=self.console,
                get_renderable=self.render_progress,
                transient=True,
                refresh_per_second=4,
                redirect_stdout=False,
                redirect_stderr=False,
            ):
                yield
        else:
            # Logs and pipes get occasional status lines without terminal control codes.
            async def heartbeat() -> None:
                while True:
                    await asyncio.sleep(_LOG_INTERVAL)
                    self.console.print(f"{self.activity} · {int(time.monotonic() - self.started)}s")

            task = asyncio.create_task(heartbeat())
            try:
                yield
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    def report(self, result: RunResult, output_path: Path) -> None:
        previous = {(item.name, item.description) for item in result.existing_insights}
        new_count = sum((item.name, item.description) not in previous for item in result.insights)
        if new_count:
            outcome = count(new_count, "new insight" if previous else "insight")
        else:
            outcome = "No new insights" if previous else "No insights"
        self.console.print(f"{outcome} from {count(result.trace_count, 'trace')}")
        candidate_count = sum(len(item.problems) for item in result.evidence)
        if not result.insights and candidate_count:
            reason = (
                "none were supported by the codebase"
                if result.rejected_by_code == candidate_count
                else "none were retained after validation and review"
                if result.rejected_by_code or result.ethos_rejections
                else "none met the actionability criteria"
            )
            self.console.print(f"{count(candidate_count, 'candidate issue')} found; {reason}.")
        if result.ethos_rejections:
            self.console.print(
                f"ethos excluded {count(len(result.ethos_rejections), 'candidate issue')}."
            )
            for rejection in result.ethos_rejections:
                self.console.print(Text(f"  {rejection.problem.description}"))
                self.console.print(Text(f"  {rejection.decision.explanation}"))
                self.console.print(Text(f"  ethos: {rejection.decision.passage}"))
        retained = len(result.insights) - new_count
        if retained:
            self.console.print(f"{count(retained, 'existing insight')} retained.")

        for index, insight in enumerate(result.insights, start=1):
            self.console.print(Text(f"\n{index}. {insight.name}", style="bold"))
            trace_count = len({item.trace_id for item in insight.evidence})
            self.console.print(
                Text(f"   Evidence: {count(trace_count, 'supporting trace')}", style="dim")
            )
            if insight.severity is not None:
                severity = Text("   Severity: ", style="dim")
                severity.append(insight.severity.upper(), style=_SEVERITY_STYLES[insight.severity])
                if insight.severity_reason is not None:
                    severity.append(f" — {insight.severity_reason}", style="dim")
                self.console.print(severity)
        self._report_checks(result.evidence)

        if result.insights and output_path != Path("-"):
            self.console.print(Text(f"\nSaved: {output_path}"), soft_wrap=True)

    def _report_checks(self, evidence: list[CompassSignalResult]) -> None:
        if not evidence:
            return
        completed = []
        skipped = []
        for item in evidence:
            name = display_name(item.signal_name)
            if item.skip_reason is not None:
                reason = _SKIP_LABELS.get(item.skip_reason, item.skip_reason)
                skipped.append(f"{name} — {reason}")
                continue
            detail = (
                count(len(item.problems), "candidate")
                if item.problems
                else f"{count(item.finding_count, 'finding')}; no candidates"
                if item.finding_count
                else "no findings"
            )
            if item.limitations:
                detail += "; " + "; ".join(item.limitations)
            if completed:
                name = name.lower()
            completed.append(f"{name} ({detail})")

        self.console.print()
        table = Table.grid(padding=(0, 2))
        table.add_column(width=8)
        table.add_column(ratio=1)
        table.add_row(Text("Ran:"), Text(", ".join(completed) or "None"))
        for index, reason in enumerate(skipped):
            table.add_row(Text("Skipped:" if index == 0 else ""), Text(reason))
        self.console.print(table)
        if any(item.skip_reason == "No ETHOS.md" for item in evidence):
            self.console.print(
                Text(
                    "\nTo enable ethos divergence, set ethos_path: ETHOS.md in YAML "
                    "or pass --ethos-path ETHOS.md, or place ETHOS.md or ethos.md in the working directory."
                )
            )
        if skipped:
            self.console.print(
                "\nTo run skipped analyses: "
                "https://github.com/NVIDIA-NeMo/labs-nemo-compass/blob/main/docs/compass-signals.md",
                soft_wrap=True,
            )
