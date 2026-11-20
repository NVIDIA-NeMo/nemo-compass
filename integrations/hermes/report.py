# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render Compass insights as one self-contained HTML page (no scripts, no network)."""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path

_SEVERITY_ORDER = {"high": 0, "med": 1, "low": 2, None: 3}

_CSS = """
:root{--bg:#fff;--fg:#1b1b1b;--muted:#666;--line:#e3e3e3;--card:#fafafa;--high:#c0392b;--med:#b9770e;--low:#2e7d32}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--fg:#eee;--muted:#a0a0a0;--line:#2c2c2c;--card:#1c1c1c}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,sans-serif}
main{max-width:860px;margin:0 auto;padding:32px 16px}h1{font-size:22px;margin:0 0 4px}
.meta{color:var(--muted);font-size:13px;margin-bottom:24px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px;margin:12px 0}
.card h2{font-size:16px;margin:0 0 6px}.sev{font-size:12px;font-weight:600;text-transform:uppercase;margin-right:8px}
.sev.high{color:var(--high)}.sev.med{color:var(--med)}.sev.low{color:var(--low)}
details{margin-top:8px}summary{cursor:pointer;color:var(--muted);font-size:13px}
code{font-size:12px;word-break:break-all}ul{margin:6px 0;padding-left:20px}
.empty{color:var(--muted)}
"""


def _evidence(items: list[dict]) -> str:
    rows = []
    for ev in items or []:
        spans = ", ".join(escape(str(s.get("span_id", ""))) for s in ev.get("spans") or [])
        rows.append(
            f"<li><code>{escape(str(ev.get('trace_id', '')))}</code>"
            + (f" (spans: <code>{spans}</code>)" if spans else "")
            + "</li>"
        )
    return "<ul>" + "".join(rows) + "</ul>"


def render(insights: list[dict], trace_count: int, model: str) -> str:
    ordered = sorted(insights, key=lambda i: _SEVERITY_ORDER.get(i.get("severity"), 3))
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cards = []
    for ins in ordered:
        sev = ins.get("severity")
        reason = ins.get("severity_reason")
        badge = f'<span class="sev {escape(sev)}">{escape(sev)}</span>' if sev else ""
        cards.append(
            '<section class="card">'
            f"<h2>{badge}{escape(str(ins.get('name', 'Untitled')))}</h2>"
            f"<p>{escape(str(ins.get('description', '')))}</p>"
            + (f"<p class='meta'>{escape(reason)}</p>" if reason else "")
            + f"<details><summary>Evidence: {len(ins.get('evidence') or [])} sessions</summary>"
            f"{_evidence(ins.get('evidence') or [])}</details></section>"
        )
    body = "".join(cards) or "<p class='empty'>No recurring problems found in these sessions.</p>"
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Compass report</title><style>{_CSS}</style></head><body><main>"
        f"<h1>Compass insights</h1><div class='meta'>{len(ordered)} insights from {trace_count} "
        f"Hermes sessions · model {escape(model)} · {when}<br>Research preview: check each "
        "finding against its sessions before acting on it.</div>"
        f"{body}</main></body></html>"
    )


def write(path: Path, insights: list[dict], trace_count: int, model: str) -> Path:
    path.write_text(render(insights, trace_count, model), encoding="utf-8")
    return path
