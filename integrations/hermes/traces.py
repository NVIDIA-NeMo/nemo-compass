# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect NeMo Relay ATIF trajectories into the single JSONL file Compass reads."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

SAMPLE_SEED = 1234  # fixed, so a "random" run is repeatable


@dataclass
class Collected:
    path: Path
    count: int
    available: int = 0
    skipped: list[str] = field(default_factory=list)


def _trajectories(file: Path) -> list[dict]:
    """A Relay file holds one pretty-printed trajectory, or JSONL with one per line."""
    text = file.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        doc = json.loads(text)
        return [doc] if isinstance(doc, dict) else [d for d in doc if isinstance(d, dict)]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


# Relay bulk Compass doesn't need: its raw event log (top-level) and a full copy of each LLM
# request/response (per step). Together they are ~98% of a Relay file.
_DROP_TOP_EXTRA = ("observed_events",)
_DROP_STEP_EXTRA = ("llm_request", "llm_response")


def slim(doc: dict) -> dict:
    """Return a trimmed copy of a Relay trajectory for Compass. The file on disk is not changed.

    Before dropping the request copies, lift the tools the agent was offered into
    `agent.tool_definitions` (Relay leaves that field empty), so Compass can check tool calls
    against their schemas.
    """
    out = dict(doc)
    steps = []
    tools: dict[str, dict] = {}
    for step in doc.get("steps") or []:
        extra = step.get("extra")
        if isinstance(extra, dict):
            request = extra.get("llm_request")
            if isinstance(request, dict):
                for tool in request.get("tools") or []:
                    if isinstance(tool, dict):
                        fn = (
                            tool.get("function") if isinstance(tool.get("function"), dict) else tool
                        )
                        name = fn.get("name")
                        if isinstance(name, str) and name not in tools:
                            tools[name] = tool
            step = {
                **step,
                "extra": {k: v for k, v in extra.items() if k not in _DROP_STEP_EXTRA} or None,
            }
        steps.append(step)
    out["steps"] = steps
    agent = dict(doc.get("agent") or {})
    if tools and not agent.get("tool_definitions"):
        agent["tool_definitions"] = list(tools.values())
    out["agent"] = agent
    if isinstance(doc.get("extra"), dict):
        out["extra"] = {k: v for k, v in doc["extra"].items() if k not in _DROP_TOP_EXTRA} or None
    return out


def find_files(atif_dir: Path) -> list[Path]:
    if not atif_dir.is_dir():
        return []
    files = [p for p in atif_dir.iterdir() if p.is_file() and p.suffix in (".json", ".jsonl")]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def _all_lines(atif_dir: Path, stop_at: int | None, skipped: list[str]) -> list[str]:
    """Newest-first compact lines, deduped. Stops early once `stop_at` lines are found."""
    seen: set[str] = set()
    lines: list[str] = []
    for file in find_files(atif_dir):
        try:
            docs = _trajectories(file)
        except (OSError, ValueError) as exc:
            skipped.append(f"{file.name}: {exc}")
            continue
        for doc in docs:
            if not doc.get("steps"):
                skipped.append(f"{file.name}: no steps (session still open?)")
                continue
            line = json.dumps(slim(doc), separators=(",", ":"), ensure_ascii=False)
            # Compass rejects duplicate trajectories, so drop repeats across files.
            key = doc.get("trajectory_id") or line
            if key in seen:
                continue
            seen.add(key)
            lines.append(line)
            if stop_at is not None and len(lines) >= stop_at:
                return lines
    return lines


def collect(atif_dir: Path, out: Path, max_sessions: int, sample: str = "newest") -> Collected:
    """Write up to `max_sessions` trajectories to `out`, one compact JSON per line.

    newest: the most recent ones. random: a seeded sample from everything available.
    """
    skipped: list[str] = []
    if sample == "random":
        pool = _all_lines(atif_dir, None, skipped)
        chosen = random.Random(SAMPLE_SEED).sample(pool, min(max_sessions, len(pool)))
        available = len(pool)
    else:
        chosen = _all_lines(atif_dir, max_sessions, skipped)
        available = len(chosen)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(chosen) + ("\n" if chosen else ""), encoding="utf-8")
    return Collected(path=out, count=len(chosen), available=available, skipped=skipped)
