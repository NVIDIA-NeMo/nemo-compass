# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent tools. Handlers take (args, **kwargs), return a JSON string, and never raise."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from typing import Any

from . import hermes_api, runner, settings

COMPASS_ANALYZE = {
    "name": "compass_analyze",
    "description": (
        "Start NVIDIA NeMo Compass on the user's recent Hermes sessions (NeMo Relay traces) to find "
        "recurring problems: failing or misused tools, outlier sessions, cost waste, user frustration, "
        "and conflicts with the agent's ETHOS.md. Runs in the background and usually takes 10-30 "
        "minutes; this returns at once with an estimate. Use when the user asks what is going wrong, "
        "what to improve, or for a Compass report. Then tell the user it's running and roughly how "
        "long it will take. Check later with compass_latest_report. Do not change anything yourself."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "max_sessions": {
                "type": "integer",
                "description": "Analyze at most N sessions. Omit to use the plugin setting (64).",
                "minimum": 2,
            }
        },
    },
}

COMPASS_LATEST_REPORT = {
    "name": "compass_latest_report",
    "description": (
        "Check the current or most recent Compass run. While it runs, returns progress and time left. "
        "When done, returns the insights and the path to the HTML report. Present findings as "
        "hypotheses with their evidence."
    ),
    "parameters": {"type": "object", "properties": {}},
}


def _summary(result: runner.RunResult, message: str | None = None) -> str:
    data = json.loads(result.to_json())
    # Keep the model's context small: names, severity, a short description, evidence count.
    data["insights"] = [
        {
            "name": i.get("name"),
            "severity": i.get("severity"),
            "description": (i.get("description") or "")[:600],
            "evidence_sessions": len(i.get("evidence") or []),
        }
        for i in result.insights
    ]
    if not result.skipped_files:
        data.pop("skipped_files", None)
    if message:
        data["message"] = message
    return json.dumps(data, ensure_ascii=False)


APPROVAL_RULE_KEY = "nvidia-compass:compass_analyze"


def make_approval_hook(get_config: Callable[..., Any]) -> Callable[..., dict | None]:
    """pre_tool_call hook: send every compass_analyze call through Hermes's approval gate.

    Hermes asks the user (CLI prompt or gateway), honors approvals.cron_mode and the other
    unattended settings, and blocks when nobody can answer. "Always" sticks via the fixed rule_key.
    """

    def approve_compass_analyze(
        tool_name: str = "", args: dict | None = None, **_: object
    ) -> dict | None:
        if tool_name != "compass_analyze":
            return None
        n, where = "your recent", ""
        try:
            s = settings.load(get_config)
            m = (args or {}).get("max_sessions")
            n = m if isinstance(m, int) and m >= 2 else s.max_sessions
            where = f" ({s.api_base or s.model})"
        except Exception:  # noqa: BLE001 - the prompt must still appear
            pass
        return {
            "action": "approve",
            "rule_key": APPROVAL_RULE_KEY,
            "message": (
                f"Start NeMo Compass in the background on up to {n} sessions? It sends their "
                f"content to the Compass model endpoint{where} and usually runs 5 to 15 minutes."
            ),
        }

    return approve_compass_analyze


def make_analyze(get_config: Callable[..., Any]) -> Callable[..., str]:
    def compass_analyze(args: dict, **kwargs: object) -> str:
        try:
            s = settings.load(get_config)
            n = args.get("max_sessions")
            if isinstance(n, int) and n >= 2:
                s = dataclasses.replace(s, max_sessions=n)
            result = runner.start(s, hermes_api.data_dir())
            if result.state == "running" and result.ok:
                return _summary(
                    result,
                    f"Compass started on {result.trace_count} sessions. "
                    "It usually takes 5 to 15 minutes.",
                )
            return _summary(result)
        except Exception as exc:  # noqa: BLE001 - a tool must return JSON, never raise
            return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    return compass_analyze


def compass_latest_report(args: dict, **kwargs: object) -> str:
    try:
        base = hermes_api.data_dir()
        current = runner.status(base)
        if current is None:
            return json.dumps({"ok": False, "error": "No Compass run yet. Use compass_analyze."})
        if current.state == "running":
            msg = f"Still running: {round(current.seconds / 60)} min elapsed."
            previous = runner.latest(base)
            if previous is not None:
                msg += " Results from the previous run are included."
                return _summary(previous, msg + f" Current step: {current.progress or 'starting'}.")
            return _summary(current, msg)
        if current.state == "failed":
            previous = runner.latest(base)
            return _summary(
                current,
                "The latest run failed."
                + (
                    " Previous results: " + previous.report_html
                    if previous and previous.report_html
                    else ""
                ),
            )
        return _summary(current)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
