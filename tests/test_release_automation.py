# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release identity, stale acceptance, and retries against a real local Git history."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "release", Path(__file__).resolve().parents[1] / "tools/release.py"
)
assert spec and spec.loader
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


@pytest.fixture
def history(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def git(*args):
        return subprocess.check_output(["git", *args], text=True).strip()

    git("init", "-b", "main")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    git("commit", "--allow-empty", "-m", "cut")
    git("tag", "release-base/1.2.3")
    git("branch", "release/1.2.3")
    return git


def test_candidates_follow_history_and_reject_stale_acceptance(history, monkeypatch):
    old = history("rev-parse", "HEAD")
    assert release.candidate("release/1.2.3", old) == "1.2.3rc1"
    history("commit", "--allow-empty", "-m", "fix")
    new = history("rev-parse", "HEAD")
    assert release.candidate("release/1.2.3", new) == "1.2.3rc2"
    assert release.candidate("release/1.2.3", new) == "1.2.3rc2"  # Retry, no allocation.
    monkeypatch.setattr(release, "api", lambda path: {"object": {"sha": new}})
    original = release.run
    monkeypatch.setattr(
        release, "run", lambda *args: "" if args[:2] == ("git", "fetch") else original(*args)
    )
    release.current("release/1.2.3", new)
    with pytest.raises(ValueError, match="validate 1.2.3rc2"):
        release.current("release/1.2.3", old)
    for invalid in ["1.2", "v1.2.3", "1.2.3rc1", "1.2.3\nextra=value"]:
        with pytest.raises(ValueError):
            release.release_version(invalid)


@pytest.mark.parametrize("existing", [None, "matching", "different"])
def test_release_upload_retry_preserves_bytes(tmp_path, monkeypatch, existing):
    monkeypatch.chdir(tmp_path)
    for key, value in {
        "SOURCE": "a" * 40,
        "BRANCH": "release/1.2.3",
        "PACKAGE_VERSION": "1.2.3rc1",
        "ACTION": "candidate",
        "GITHUB_RUN_ID": "123",
        "GITHUB_REPOSITORY": "org/repo",
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
    }.items():
        monkeypatch.setenv(key, value)
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel = dist / "example.whl"
    wheel.write_bytes(b"validated wheel")
    monkeypatch.setattr(release, "current", lambda *args: None)
    monkeypatch.setattr(release, "tag_source", lambda tag: "a" * 40)
    calls = []

    def command(*args):
        calls.append(args)
        if args[:3] == ("gh", "release", "list"):
            return json.dumps([{"tagName": "v1.2.3rc1"}])
        if args[:3] == ("gh", "release", "view"):
            return json.dumps(
                {
                    "assets": [1] if existing else [],
                    "body": "notes",
                    "isDraft": True,
                    "url": "https://example.com",
                }
            )
        return ""

    monkeypatch.setattr(release, "run", command)

    def download(tag, directory):
        (directory / wheel.name).write_bytes(
            wheel.read_bytes() if existing == "matching" else b"other"
        )

    monkeypatch.setattr(release, "download", download)
    if existing == "different":
        with pytest.raises(ValueError, match="differs"):
            release.finish()
    else:
        release.finish()
        uploaded = [args[-1] for args in calls if args[:3] == ("gh", "release", "upload")]
        assert ("dist/example.whl" in uploaded) == (existing is None)
        release.verify(dist)
        wheel.write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="checksum mismatch"):
            release.verify(dist)


def test_publication_requires_complete_current_candidate(history, tmp_path, monkeypatch):
    source = history("rev-parse", "HEAD")
    for key, value in {
        "ACTION": "publish",
        "CANDIDATE": "v1.2.3rc1",
        "READY": "true",
        "QA_EVIDENCE": "https://example.com/qa",
        "GITHUB_OUTPUT": str(tmp_path / "outputs"),
    }.items():
        monkeypatch.setenv(key, value)
    completed = {"conclusion": "success", "path": ".github/workflows/prepare-release.yaml"}
    monkeypatch.setattr(
        release,
        "api",
        lambda path: completed if path.startswith("actions/") else {"object": {"sha": source}},
    )
    monkeypatch.setattr(release, "tag_source", lambda tag: source)
    import hashlib

    def download(tag, directory):
        (directory / "example.whl").write_bytes(b"wheel")
        (directory / "candidate.json").write_text(
            json.dumps(
                {
                    "source": source,
                    "branch": "release/1.2.3",
                    "version": "1.2.3rc1",
                    "run_id": "123",
                    "sha256": {"example.whl": hashlib.sha256(b"wheel").hexdigest()},
                }
            )
        )

    monkeypatch.setattr(release, "download", download)
    release.plan()
    assert "version=1.2.3\n" in (tmp_path / "outputs").read_text()
    completed["conclusion"] = "failure"
    with pytest.raises(ValueError, match="has not succeeded"):
        release.plan()
    monkeypatch.setenv("READY", "false")
    with pytest.raises(ValueError, match="Confirm readiness"):
        release.plan()


def test_workflow_version_step_preserves_dependency_lock(tmp_path, monkeypatch):
    import os
    import shutil

    import tomllib
    import yaml

    root = Path(__file__).resolve().parents[1]
    for name in ["pyproject.toml", "uv.lock", "README.md"]:
        shutil.copy(root / name, tmp_path / name)
    original = tomllib.loads((tmp_path / "uv.lock").read_text())
    workflow = yaml.safe_load((root / ".github/workflows/ci.yaml").read_text())
    step = next(
        s
        for s in workflow["jobs"]["quality"]["steps"]
        if s.get("name") == "Set release package version"
    )
    for version in ["0.3.0rc1", "0.3.0"]:
        monkeypatch.setenv("PACKAGE_VERSION", version)
        subprocess.run(["bash", "-e", "-c", step["run"]], cwd=tmp_path, env=os.environ, check=True)
        updated = tomllib.loads((tmp_path / "uv.lock").read_text())
        for package in updated["package"]:
            if package["name"] == "nemo-compass":
                assert package["version"] == version
                package["version"] = tomllib.loads((root / "pyproject.toml").read_text())[
                    "project"
                ]["version"]
        assert updated == original
