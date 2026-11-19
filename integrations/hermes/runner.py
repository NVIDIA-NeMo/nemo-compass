# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Compass as a child process, in the background or blocking, and collect its outputs.

A child process keeps Compass's asyncio loop, LiteLLM globals and long runtime out of the Hermes
process. It runs the `compass` module from the same interpreter Hermes installed it into.

Each run lives in <data dir>/runs/<UTC stamp>/:
  traces.jsonl   what Compass analyzes
  compass.log    Compass's progress output, written live
  status.json    running | done | failed, start time, timeout
  insights.yaml  Compass's findings      report.html  the rendered report
  exit.json      how Compass ended, written by watchdog.py

The run's detached process is watchdog.py, which enforces the timeout itself, so the limit holds
even if Hermes exits mid-run.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import report, traces
from .settings import API_KEY_ENV, Settings

# Only these parent variables reach Compass. Provider keys from Hermes's environment never do.
_PASS_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT", "SSL_CERT_FILE")
# Extra time past the watchdog's deadline before the plugin itself steps in (see _overdue).
GRACE_S = 120
WATCHDOG = Path(__file__).with_name("watchdog.py")
_PROGRESS = re.compile(r"^(.+?) · (\d+)s\s*$")


@dataclass
class RunResult:
    ok: bool
    run_dir: str
    state: str = "failed"  # running | done | failed
    trace_count: int = 0
    insights: list[dict] = field(default_factory=list)
    insights_yaml: str | None = None
    report_html: str | None = None
    error: str | None = None
    skipped_files: list[str] = field(default_factory=list)
    seconds: float = 0.0
    progress: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# --- small helpers --------------------------------------------------------------------


def child_env(settings: Settings) -> dict[str, str]:
    env = {k: os.environ[k] for k in _PASS_ENV if k in os.environ}
    # Hermes runs on a bare interpreter and reaches its package environment through sys.path,
    # so sys.executable alone can't import compass. Hand the child the same import path.
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p and os.path.isdir(p))
    # LiteLLM otherwise downloads its model price list from GitHub each time Compass starts.
    env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    env[API_KEY_ENV] = settings.api_key
    env["COMPASS_MODEL"] = settings.model
    if settings.api_base:
        env["COMPASS_API_BASE"] = settings.api_base
    return env


def command(settings: Settings, traces_path: Path, out_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "compass.cli.main",
        "--trace.atif.path",
        str(traces_path),
        "--output-path",
        str(out_path),
        "--max-tokens",
        str(settings.max_tokens),
    ]
    if settings.ethos_path is not None:
        cmd += ["--ethos-path", str(settings.ethos_path)]
    return cmd


def new_run_dir(base: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = base / "runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    # Compass looks for a .env starting at its working directory and walking up.
    # An empty one here stops that walk before it can reach Hermes's own .env.
    (run_dir / ".env").write_text("# intentionally empty: see runner.py\n", encoding="utf-8")
    return run_dir


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


# --- the lock -------------------------------------------------------------------------


def _lock(base: Path) -> Path | None:
    lock = base / "run.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            owner = int(lock.read_text().strip() or 0)
        except (OSError, ValueError):
            owner = 0
        # Stale if the Hermes process that took it is gone, or it is older than 6 hours.
        if not _alive(owner) or time.time() - lock.stat().st_mtime > 6 * 3600:
            lock.unlink(missing_ok=True)
            return _lock(base)
        return None
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return lock


# --- running --------------------------------------------------------------------------


def _prepare(settings: Settings, base: Path) -> tuple[RunResult, Path | None, dict]:
    """Checks, lock, run folder and traces. Returns (result, lock, status)."""
    problems = settings.problems()
    if problems:
        return RunResult(ok=False, run_dir="", error=" ".join(problems)), None, {}
    base.mkdir(parents=True, exist_ok=True)
    lock = _lock(base)
    if lock is None:
        current = status(base)
        if current is not None and current.state == "running":
            current.ok = False
            current.error = "A Compass run is already in progress."
            return current, None, {}
        return (
            RunResult(ok=False, run_dir="", error="A Compass run is already in progress."),
            None,
            {},
        )
    run_dir = new_run_dir(base)
    collected = traces.collect(
        settings.atif_dir, run_dir / "traces.jsonl", settings.max_sessions, settings.sample
    )
    st = {
        "state": "running",
        "started_at": time.time(),
        "traces": collected.count,
        "model": settings.model,
        "timeout_s": settings.timeout_s,
        "skipped_files": collected.skipped,
    }
    result = RunResult(
        ok=False,
        run_dir=str(run_dir),
        state="running",
        trace_count=collected.count,
        skipped_files=collected.skipped,
    )
    if collected.count < 2:
        error = (
            f"Found {collected.count} Relay trajectories in {settings.atif_dir}. Compass needs at least 2 "
            "(insights must be backed by several sessions). Is Relay ATIF export on? "
            "See `hermes compass setup`."
        )
        st.update(state="failed", error=error)
        _write_json(run_dir / "status.json", st)
        lock.unlink(missing_ok=True)
        result.state, result.error = "failed", error
        return result, None, st
    _write_json(run_dir / "status.json", st)
    return result, lock, st


def _launch(settings: Settings, run_dir: Path, timeout_s: int) -> subprocess.Popen:
    log = (run_dir / "compass.log").open("w", encoding="utf-8")
    compass = command(settings, run_dir / "traces.jsonl", run_dir / "insights.yaml")
    return subprocess.Popen(
        # -I: the watchdog needs only the standard library, and keeps the plugin folder off its path.
        [
            sys.executable,
            "-I",
            str(WATCHDOG),
            str(timeout_s),
            str(run_dir / "exit.json"),
            "--",
            *compass,
        ],
        cwd=run_dir,
        env=child_env(settings),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=log,
        text=True,
        start_new_session=True,  # keeps running if Hermes exits; status() finishes the report later
    )


def _kill_tree(pid: int) -> None:
    """Stop the watchdog and Compass. They share a process group on macOS and Linux."""
    try:
        if hasattr(os, "killpg"):
            os.killpg(pid, signal.SIGKILL)
        else:
            os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass


def _exit_code(run_dir: Path) -> int | None:
    """Compass's exit code from exit.json; None if it timed out. 1 if the watchdog itself died."""
    ended = _read_json(run_dir / "exit.json")
    if not ended:
        return 1
    return None if ended.get("timed_out") else int(ended.get("rc", 1))


def _wait_and_finish(
    proc: subprocess.Popen, run_dir: Path, base: Path, lock: Path, st: dict
) -> None:
    try:
        try:
            proc.wait(timeout=st["timeout_s"] + GRACE_S)
            rc = _exit_code(run_dir)
        except subprocess.TimeoutExpired:  # the watchdog should have ended it by now
            _kill_tree(proc.pid)
            proc.wait()
            rc = None
        _finish(run_dir, st, rc)
    finally:
        lock.unlink(missing_ok=True)


def _finish(run_dir: Path, st: dict, rc: int | None) -> None:
    """Turn a finished child into report + status. Safe to call once per run."""
    st["total_s"] = round(time.time() - st["started_at"], 1)
    if rc is None:
        st.update(state="failed", error=f"Compass timed out after {st['timeout_s'] // 60} minutes.")
    elif rc != 0:
        tail = "\n".join(_log_tail(run_dir, 8))
        st.update(state="failed", error=f"Compass exited with {rc}.\n{tail}")
    else:
        out = run_dir / "insights.yaml"
        insights = (yaml.safe_load(out.read_text(encoding="utf-8")) or []) if out.is_file() else []
        report.write(run_dir / "report.html", insights, st.get("traces", 0), st.get("model") or "")
        st.update(state="done", insights=len(insights))
    _write_json(run_dir / "status.json", st)


def _launched(settings: Settings, run_dir: Path, lock: Path, st: dict) -> subprocess.Popen:
    proc = _launch(settings, run_dir, st["timeout_s"])
    st["pid"] = proc.pid
    _write_json(run_dir / "status.json", st)
    # The lock now follows the Compass process, so it stays held if Hermes exits mid-run.
    lock.write_text(str(proc.pid))
    return proc


def start(settings: Settings, base: Path) -> RunResult:
    """Start a run in the background and return at once (agent tool)."""
    result, lock, st = _prepare(settings, base)
    if lock is None:
        return result
    run_dir = Path(result.run_dir)
    proc = _launched(settings, run_dir, lock, st)
    threading.Thread(
        target=_wait_and_finish,
        args=(proc, run_dir, base, lock, st),
        name="nvidia-compass-run",
        daemon=True,
    ).start()
    result.ok = True
    return result


def run(settings: Settings, base: Path) -> RunResult:
    """Run and wait (terminal command)."""
    result, lock, st = _prepare(settings, base)
    if lock is None:
        return result
    run_dir = Path(result.run_dir)
    proc = _launched(settings, run_dir, lock, st)
    _wait_and_finish(proc, run_dir, base, lock, st)
    return load(run_dir, base)


# --- reading runs back ----------------------------------------------------------------


def _log_tail(run_dir: Path, n: int) -> list[str]:
    try:
        lines = (run_dir / "compass.log").read_text(encoding="utf-8").strip().splitlines()
    except OSError:
        return []
    return lines[-n:]


def _progress(run_dir: Path) -> str | None:
    for line in reversed(_log_tail(run_dir, 40)):
        m = _PROGRESS.match(line.strip())
        if m:
            return f"{m.group(1)} ({m.group(2)} s)"
    return None


def load(run_dir: Path, base: Path) -> RunResult:
    st = _read_json(run_dir / "status.json")
    if st.get("state") == "running":
        if not _alive(st.get("pid")):
            # The watcher thread died with its Hermes process; the run ended on its own.
            _finish(run_dir, st, _exit_code(run_dir))
        elif time.time() > st.get("started_at", 0) + st.get("timeout_s", 0) + GRACE_S:
            # Backstop in case the watchdog itself is stuck.
            _kill_tree(st["pid"])
            _finish(run_dir, st, None)
    state = st.get("state", "failed")
    out, html = run_dir / "insights.yaml", run_dir / "report.html"
    insights = (yaml.safe_load(out.read_text(encoding="utf-8")) or []) if out.is_file() else []
    elapsed = round(time.time() - st.get("started_at", time.time()), 1)
    return RunResult(
        ok=state != "failed",
        run_dir=str(run_dir),
        state=state,
        trace_count=st.get("traces", 0),
        insights=insights if state == "done" else [],
        insights_yaml=str(out) if out.is_file() and state == "done" else None,
        report_html=str(html) if html.is_file() else None,
        error=st.get("error"),
        skipped_files=st.get("skipped_files", []),
        seconds=st.get("total_s", elapsed),
        progress=_progress(run_dir) if state == "running" else None,
    )


def _runs(base: Path) -> list[Path]:
    root = base / "runs"
    return sorted((p for p in root.iterdir() if p.is_dir()), reverse=True) if root.is_dir() else []


def status(base: Path) -> RunResult | None:
    """The newest run, whatever its state."""
    runs = _runs(base)
    return load(runs[0], base) if runs else None


def latest(base: Path) -> RunResult | None:
    """The newest finished report."""
    for run_dir in _runs(base):
        r = load(run_dir, base)
        if r.state == "done":
            return r
    return None
