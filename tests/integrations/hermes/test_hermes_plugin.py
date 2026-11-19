# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import pytest
import yaml

from tests.integrations.hermes.support import FakeCtx, atif, plugin

settings = sys.modules["nvidia_compass.settings"]
traces = sys.modules["nvidia_compass.traces"]
runner = sys.modules["nvidia_compass.runner"]
report = sys.modules["nvidia_compass.report"]


def manifest() -> dict:
    return yaml.safe_load((Path(plugin.__file__).parent / "plugin.yaml").read_text())


# --- registration matches the manifest (catalog rule 6) -------------------------------


def test_registers_exactly_the_declared_tools():
    ctx = FakeCtx()
    plugin.register(ctx)
    assert sorted(ctx.tools) == sorted(manifest()["provides_tools"])
    assert [h[0] for h in ctx.hooks] == manifest()["provides_hooks"] == ["pre_tool_call"]
    assert "compass" in ctx.cli


# --- approval gate (catalog rule 12) --------------------------------------------------


def test_analyze_asks_for_approval_and_nothing_else_does(hermes):
    ctx = FakeCtx()
    plugin.register(ctx)
    name, hook = ctx.hooks[0]
    out = hook(tool_name="compass_analyze", args={"max_sessions": 7}, session_id="s")
    assert out["action"] == "approve" and out["rule_key"] == "nvidia-compass:compass_analyze"
    assert "7 sessions" in out["message"] and "http://127.0.0.1:8000/v1" in out["message"]
    assert hook(tool_name="compass_latest_report", args={}) is None
    assert hook(tool_name="terminal", args={"command": "ls"}) is None


def test_manifest_has_vendor_prefix_and_semver_floor():
    m = manifest()
    assert m["name"].startswith("nvidia-")
    assert m["requires_hermes"].startswith(">=0.")


# --- settings -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "section,expected",
    [
        (
            {"default": "claude-sonnet-5.5", "provider": "anthropic"},
            ("anthropic/claude-sonnet-5.5", ""),
        ),
        (
            {"default": "anthropic/claude-opus", "provider": "openrouter"},
            ("openrouter/anthropic/claude-opus", ""),
        ),
        (
            {"default": "nemotron", "provider": "custom", "base_url": "http://x/v1"},
            ("openai/nemotron", "http://x/v1"),
        ),
        (
            {"default": "nvidia/nemotron-3.5-super", "provider": "nvidia"},
            ("openai/nvidia/nemotron-3.5-super", "https://integrate.api.nvidia.com/v1"),
        ),
        ({}, ("", "")),
    ],
)
def test_model_from_hermes(section, expected):
    assert settings.model_from_hermes({"model": section}) == expected


def test_settings_prefill_from_hermes_and_override(hermes):
    s = settings.load(FakeCtx().get_config)
    assert s.model == "openai/nvidia/nemotron-3.5-super"
    assert s.api_base == "http://127.0.0.1:8000/v1"
    assert s.atif_dir == hermes["home"] / "telemetry" / "nemo-relay" / "atif"
    s2 = settings.load(FakeCtx({"compass_model": "anthropic/claude-sonnet-5.5"}).get_config)
    assert s2.model == "anthropic/claude-sonnet-5.5" and s2.api_base == ""


def test_missing_key_is_reported_not_raised(hermes):
    hermes["secrets"].clear()
    out = json.loads(FakeCtx_tool("compass_analyze")({}))
    assert out["ok"] is False and "COMPASS_API_KEY" in out["error"]


# --- trace collection -----------------------------------------------------------------


def test_collect_newest_first_dedupes_and_skips_empty(relay_dir, tmp_path):
    (relay_dir / "trajectory-dupe.json").write_text(json.dumps(atif("s0")))
    (relay_dir / "trajectory-open.json").write_text(json.dumps({**atif("x"), "steps": []}))
    (relay_dir / "notes.txt").write_text("ignored")
    out = traces.collect(relay_dir, tmp_path / "t.jsonl", max_sessions=10)
    lines = (tmp_path / "t.jsonl").read_text().splitlines()
    assert out.count == len(lines) == 3
    assert all(json.loads(line)["steps"] for line in lines)
    assert any("no steps" in s for s in out.skipped)


def test_collect_respects_limit_and_jsonl_input(relay_dir, tmp_path):
    (relay_dir / "batch.jsonl").write_text("\n".join(json.dumps(atif(f"b{i}")) for i in range(4)))
    assert traces.collect(relay_dir, tmp_path / "t.jsonl", max_sessions=2).count == 2


def test_collected_file_passes_compass_atif_loader(relay_dir, tmp_path):
    """Strict check against Compass's own ATIF models, when Compass is installed."""
    loader_mod = pytest.importorskip("trace_ingest.loaders.atif")
    out = traces.collect(relay_dir, tmp_path / "t.jsonl", max_sessions=10)
    loader = loader_mod.ATIFTraceLoader(loader_mod.ATIFTraceConfig(path=out.path))
    assert len(loader.load()) == 3


# --- the Compass child process --------------------------------------------------------


def test_child_env_carries_only_compass_settings(hermes, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-leak")
    env = runner.child_env(settings.load(FakeCtx().get_config))
    assert "OPENAI_API_KEY" not in env and "ANTHROPIC_API_KEY" not in env
    assert env["COMPASS_API_KEY"] == "test-key"
    assert env["COMPASS_MODEL"] == "openai/nvidia/nemotron-3.5-super"
    assert env["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"  # no price-list download at startup
    assert not any(k.startswith("COMPASS_TELEMETRY") for k in env)
    # The child must see the same packages as Hermes (bare interpreter + sys.path).
    import yaml as _y

    assert str(Path(_y.__file__).parent.parent) in env["PYTHONPATH"].split(os.pathsep)


def test_run_dir_has_empty_dotenv_guard(hermes):
    (hermes["home"] / ".env").write_text("OPENAI_API_KEY=secret\n")
    run_dir = runner.new_run_dir(hermes["home"] / "plugin-data" / "nvidia-compass")
    guard = run_dir / ".env"
    assert guard.is_file() and "=" not in guard.read_text()


def test_too_few_traces_explains_relay_setup(hermes):
    out = json.loads(FakeCtx_tool("compass_analyze")({}))
    assert out["ok"] is False and "Relay" in out["error"]


# A stand-in for `python -m compass.cli.main`: prints progress like Compass, writes insights.
FAKE_COMPASS = r"""
import sys, time, yaml, os
args = sys.argv[1:]
out = args[args.index("--output-path") + 1]
mode = os.environ.get("FAKE_MODE", "ok")
print("Compiling candidate problems into insights · 10s", file=sys.stderr, flush=True)
if mode == "slow":
    time.sleep(30)
if mode == "fail":
    print("Missing required environment settings", file=sys.stderr); sys.exit(2)
yaml.safe_dump([{"name": "web_search hits rate limits", "severity": "high",
                 "description": "Search returns HTTP 429 in most sessions.",
                 "evidence": [{"trace_id": "traj-s0"}, {"trace_id": "traj-s1"}]}], open(out, "w"))
print("Saved: insights.yaml", file=sys.stderr)
"""


@pytest.fixture
def fake_compass(monkeypatch, tmp_path):
    captured = {}
    real_command = runner.command

    def cmd(settings_, traces_path, out_path):
        captured["real"] = real_command(settings_, traces_path, out_path)
        return [sys.executable, "-c", FAKE_COMPASS, *captured["real"][3:]]

    real_env = runner.child_env

    def env(settings_):
        e = real_env(settings_)
        e["FAKE_MODE"] = captured.get("mode", "ok")
        captured["env"] = e
        return e

    monkeypatch.setattr(runner, "command", cmd)
    monkeypatch.setattr(runner, "child_env", env)
    return captured


def _wait_done(base, timeout=20):
    for _ in range(timeout * 10):
        r = runner.status(base)
        if r is not None and r.state != "running":
            return r
        time.sleep(0.1)
    raise AssertionError("run did not finish")


BASE = lambda hermes: hermes["home"] / "plugin-data" / "nvidia-compass"  # noqa: E731


def test_analyze_starts_in_background_then_reports(hermes, relay_dir, fake_compass):
    out = json.loads(FakeCtx_tool("compass_analyze")({"max_sessions": 3}))
    assert out["ok"] is True and out["state"] == "running" and out["trace_count"] == 3
    assert "started" in out["message"]
    assert fake_compass["real"][1:3] == ["-m", "compass.cli.main"]
    assert Path(out["run_dir"], ".env").is_file()
    done = _wait_done(BASE(hermes))
    assert done.state == "done"
    latest = json.loads(plugin.tools.compass_latest_report({}))
    assert latest["insights"][0]["name"] == "web_search hits rate limits"
    assert Path(latest["report_html"]).read_text().count("web_search hits rate limits") == 1
    assert not (BASE(hermes) / "run.lock").exists()


def test_second_start_while_running_is_refused(hermes, relay_dir, fake_compass):
    fake_compass["mode"] = "slow"
    first = json.loads(FakeCtx_tool("compass_analyze")({}))
    assert first["state"] == "running"
    second = json.loads(FakeCtx_tool("compass_analyze")({}))
    assert second["ok"] is False and "already in progress" in second["error"]
    progress = json.loads(plugin.tools.compass_latest_report({}))
    assert progress["state"] == "running" and "Still running" in progress["message"]
    pid = json.loads((Path(first["run_dir"]) / "status.json").read_text())["pid"]
    os.kill(pid, 9)
    assert _wait_done(BASE(hermes)).state == "failed"


def test_compass_failure_is_reported(hermes, relay_dir, fake_compass):
    fake_compass["mode"] = "fail"
    FakeCtx_tool("compass_analyze")({})
    r = _wait_done(BASE(hermes))
    assert r.state == "failed" and "exited with 2" in r.error
    out = json.loads(plugin.tools.compass_latest_report({}))
    assert "failed" in out["message"]


def test_blocking_run_for_cli(hermes, relay_dir, fake_compass):
    r = runner.run(settings.load(FakeCtx().get_config), BASE(hermes))
    assert r.state == "done" and r.insights[0]["severity"] == "high"


def test_finishes_run_whose_watcher_died(hermes, relay_dir, fake_compass):
    """Hermes exited mid-run: the child finished alone, status() writes the report later."""
    s = settings.load(FakeCtx().get_config)
    result, lock, st = runner._prepare(s, BASE(hermes))
    run_dir = Path(result.run_dir)
    proc = runner._launched(s, run_dir, lock, st)
    proc.wait()  # no watcher thread, as if Hermes had died
    lock.unlink()
    r = runner.status(BASE(hermes))
    assert r.state == "done" and Path(r.report_html).is_file()


def test_watchdog_enforces_timeout_without_hermes(hermes, relay_dir, fake_compass):
    """Hermes gone (no watcher thread): the watchdog alone stops Compass at the deadline."""
    fake_compass["mode"] = "slow"
    s = dataclasses.replace(settings.load(FakeCtx().get_config), timeout_s=1)
    result, lock, st = runner._prepare(s, BASE(hermes))
    run_dir = Path(result.run_dir)
    proc = runner._launched(s, run_dir, lock, st)
    t0 = time.monotonic()
    proc.wait(timeout=15)
    assert time.monotonic() - t0 < 10  # the fake sleeps 30 s
    assert json.loads((run_dir / "exit.json").read_text())["timed_out"] is True
    lock.unlink()
    r = runner.status(BASE(hermes))
    assert r.state == "failed" and "timed out" in r.error


def test_overdue_run_is_killed_on_status_check(hermes, relay_dir, fake_compass, monkeypatch):
    """Backstop if the watchdog itself hangs: a status check past deadline + grace kills the run."""
    fake_compass["mode"] = "slow"
    s = settings.load(FakeCtx().get_config)
    result, lock, st = runner._prepare(s, BASE(hermes))
    run_dir = Path(result.run_dir)
    proc = runner._launched(s, run_dir, lock, st)
    st_file = run_dir / "status.json"
    data = json.loads(st_file.read_text())
    data["started_at"] -= data["timeout_s"] + runner.GRACE_S + 1
    st_file.write_text(json.dumps(data))
    r = runner.status(BASE(hermes))
    assert r.state == "failed" and "timed out" in r.error
    proc.wait(timeout=10)
    lock.unlink()


def test_report_escapes_html():
    html = report.render(
        [
            {
                "name": "<script>x</script>",
                "description": "a & b",
                "evidence": [{"trace_id": "t1"}, {"trace_id": "t2"}],
            }
        ],
        2,
        "m",
    )
    assert "<script>x" not in html and "&lt;script&gt;" in html


# helper ------------------------------------------------------------------------------


def FakeCtx_tool(name):
    ctx = FakeCtx()
    plugin.register(ctx)
    return ctx.tools[name]["handler"]


# --- sampling and timing --------------------------------------------------------------


def test_random_sample_is_seeded_and_capped(relay_dir, tmp_path):
    for i in range(10):
        (relay_dir / f"trajectory-r{i}.json").write_text(json.dumps(atif(f"r{i}")))
    a = traces.collect(relay_dir, tmp_path / "a.jsonl", max_sessions=4, sample="random")
    traces.collect(relay_dir, tmp_path / "b.jsonl", max_sessions=4, sample="random")
    assert a.count == 4 and a.available == 13
    assert (tmp_path / "a.jsonl").read_text() == (tmp_path / "b.jsonl").read_text()


def test_default_cap_is_64(hermes):
    assert settings.load(FakeCtx().get_config).max_sessions == 64


def test_timeout_is_30_minutes_unless_set(hermes):
    assert settings.load(FakeCtx().get_config).timeout_s == 30 * 60
    assert settings.load(FakeCtx({"timeout_minutes": 10}).get_config).timeout_s == 10 * 60


# --- trimming Relay bulk --------------------------------------------------------------


def _relay_doc(session="big"):
    doc = atif(session)
    tools = [
        {"type": "function", "function": {"name": "web_search", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}},
    ]
    for step in doc["steps"]:
        step["extra"] = {
            "ancestry": {"id": "a"},
            "llm_request": {"messages": ["x" * 5000], "tools": tools},
            "llm_response": {"raw": "y" * 2000},
        }
    doc["extra"] = {"nemo_relay": {"v": 1}, "observed_events": ["e" * 20000]}
    return doc


def test_slim_drops_bulk_and_lifts_tool_definitions():
    doc = _relay_doc()
    out = traces.slim(doc)
    assert "observed_events" not in out["extra"] and out["extra"]["nemo_relay"] == {"v": 1}
    assert all(
        "llm_request" not in s["extra"] and "llm_response" not in s["extra"] for s in out["steps"]
    )
    assert all(s["extra"]["ancestry"] == {"id": "a"} for s in out["steps"])
    assert [t["function"]["name"] for t in out["agent"]["tool_definitions"]] == [
        "web_search",
        "read_file",
    ]
    assert len(json.dumps(out)) < len(json.dumps(doc)) / 5
    assert "observed_events" in doc["extra"]  # the original is untouched


def test_slimmed_relay_doc_passes_compass_loader_with_tool_schemas(tmp_path):
    loader_mod = pytest.importorskip("trace_ingest.loaders.atif")
    d = tmp_path / "atif"
    d.mkdir()
    for i in range(2):
        (d / f"trajectory-{i}.json").write_text(json.dumps(_relay_doc(f"s{i}")))
    out = traces.collect(d, tmp_path / "t.jsonl", max_sessions=10)
    loaded = list(loader_mod.ATIFTraceLoader(loader_mod.ATIFTraceConfig(path=out.path)).load())
    assert len(loaded) == 2
    agent_span = loaded[0].root_spans[0]
    assert [t.name for t in agent_span.tool_definitions] == ["web_search", "read_file"]
