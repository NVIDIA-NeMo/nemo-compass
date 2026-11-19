# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`hermes compass <setup|status|run|report>` terminal commands."""

from __future__ import annotations

import argparse
import dataclasses
import webbrowser
from collections.abc import Callable
from typing import Any

from . import hermes_api, runner, settings, traces

RELAY_SNIPPET = """\
# Add to your NeMo Relay plugins.toml (see `hermes doctor` for which file Hermes loads),
# then restart Hermes. Compass reads the ATIF files this writes.
[components.config.atif]
enabled = true
agent_name = "Hermes Agent"
output_directory = "{atif_dir}"
filename_template = "trajectory-{{session_id}}.json"
"""


def _setup(s: settings.Settings) -> None:
    print("nvidia-compass setup\n")
    print("1. Turn on NeMo Relay ATIF export (Compass reads these files):\n")
    print(RELAY_SNIPPET.format(atif_dir=s.atif_dir))
    print("   Long-lived gateway sessions only export when they close. To export per chunk,")
    print("   set gateway.telemetry.session_segments.max_turns in config.yaml.")
    print("   Compass needs message text in these files. If its findings look empty, check")
    print("   Relay's enable_full_payloads setting (it writes full conversations to disk).\n")
    print(f"2. Set {settings.API_KEY_ENV}: Desktop > Settings > Plugins > nvidia-compass,")
    print(f"   or add {settings.API_KEY_ENV}=... to your Hermes .env.\n")
    print("3. Check with: hermes compass status")


def _status(s: settings.Settings) -> None:
    files = traces.find_files(s.atif_dir)
    print(f"Relay ATIF folder : {s.atif_dir} ({len(files)} files)")
    print(f"Compass model     : {s.model or '(not set)'}")
    print(f"API base          : {s.api_base or '(provider default)'}")
    print(f"API key           : {'set' if s.api_key else 'missing'}")
    print(f"ETHOS.md          : {s.ethos_path or '(none)'}")
    base = hermes_api.data_dir()
    current = runner.status(base)
    if current is not None and current.state == "running":
        print(
            f"Running now       : {round(current.seconds / 60)} min elapsed, {current.progress or 'starting'}"
        )
    last = runner.latest(base)
    print(f"Last report       : {last.report_html if last else '(none)'}")
    for p in s.problems():
        print(f"! {p}")
    if not files:
        print("! No Relay trajectories yet. Run `hermes compass setup`.")


def _run(s: settings.Settings) -> None:
    base = hermes_api.data_dir()
    print(
        f"Running Compass on up to {s.max_sessions} sessions ({s.sample}), "
        f"stopping after {s.timeout_s // 60} minutes."
    )
    print(f"Progress is written to {base / 'runs'}/<newest>/compass.log")
    result = runner.run(s, base)
    if result.state != "done":
        print(f"Compass did not finish: {result.error}")
        return
    print(
        f"{len(result.insights)} insights from {result.trace_count} sessions in {result.seconds} s."
    )
    for i in result.insights:
        print(f"  [{i.get('severity') or '-'}] {i.get('name')}")
    print(f"Report: {result.report_html}")


def _report(_: settings.Settings) -> None:
    last = runner.latest(hermes_api.data_dir())
    if last is None:
        print("No report yet. Run: hermes compass run")
        return
    print(last.report_html)
    webbrowser.open(f"file://{last.report_html}")


_COMMANDS = {"setup": _setup, "status": _status, "run": _run, "report": _report}


def setup_argparse(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="compass_command")
    subs.add_parser("setup", help="Show how to turn on Relay traces and set the API key")
    subs.add_parser("status", help="Check traces, model and key")
    run = subs.add_parser("run", help="Analyze recent sessions and write a report")
    run.add_argument("--max-sessions", type=int, help="Override the sessions-per-run setting")
    run.add_argument("--sample", choices=["newest", "random"], help="Which sessions to pick")
    subs.add_parser("report", help="Open the latest HTML report")


def make_handler(get_config: Callable[..., Any]) -> Callable[[argparse.Namespace], None]:
    def handle(args: argparse.Namespace) -> None:
        name = getattr(args, "compass_command", None) or "status"
        s = settings.load(get_config)
        overrides = {}
        if getattr(args, "max_sessions", None):
            overrides["max_sessions"] = max(2, args.max_sessions)
        if getattr(args, "sample", None):
            overrides["sample"] = args.sample
        _COMMANDS[name](dataclasses.replace(s, **overrides) if overrides else s)

    return handle
