# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release identity, artifact integrity, and dependency-lock preservation."""

import hashlib
import importlib.util
import shutil
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
    git("remote", "add", "origin", ".")
    return git


def test_candidates_follow_history_and_reject_stale_acceptance(history, monkeypatch):
    old = history("rev-parse", "HEAD")
    assert release.candidate("release/1.2.3", old) == "1.2.3rc1"
    history("commit", "--allow-empty", "-m", "fix")
    new = history("rev-parse", "HEAD")
    assert release.candidate("release/1.2.3", new) == "1.2.3rc2"
    monkeypatch.setattr(release, "api", lambda path: {"object": {"sha": new}})
    release.current("release/1.2.3", new)
    with pytest.raises(ValueError, match="validate 1.2.3rc2"):
        release.current("release/1.2.3", old)


@pytest.fixture
def build(history, tmp_path, monkeypatch):
    source = history("rev-parse", "HEAD")
    for key, value in {
        "SOURCE": source,
        "BRANCH": "release/1.2.3",
        "PACKAGE_VERSION": "1.2.3rc1",
        "ACTION": "publish",
        "CANDIDATE": "v1.2.3rc1",
        "GITHUB_ACTOR": "captain",
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_REPOSITORY": "org/repo",
        "GITHUB_OUTPUT": str(tmp_path / "outputs"),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
    }.items():
        monkeypatch.setenv(key, value)
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "example.whl").write_bytes(b"wheel")
    (dist / "SHA256SUMS").write_text(hashlib.sha256(b"wheel").hexdigest() + "  example.whl\n")
    release.record()
    monkeypatch.setattr(release, "tag_source", lambda tag: source)
    return dist, source


def test_publication_stops_on_existing_release_or_corrupt_artifacts(build, monkeypatch):
    dist, source = build
    monkeypatch.setenv("PACKAGE_VERSION", "1.2.3")
    release.record()
    monkeypatch.setattr(release, "api", lambda path: {"object": {"sha": source}})
    calls = []
    create_fails = False
    original = release.run

    def command(*args):
        if args[0] != "gh":
            return original(*args)
        calls.append(args)
        if create_fails:
            raise subprocess.CalledProcessError(1, args)
        return "https://github.com/org/repo/releases/tag/v1.2.3"

    monkeypatch.setattr(release, "run", command)
    release.finish()
    assert "dist/example.whl" in calls[0]
    assert "--draft" in calls[0] and calls[-1][-1] == "--draft=false"

    create_fails = True
    calls.clear()
    with pytest.raises(ValueError, match="delete only that draft, keep the tag") as error:
        release.finish()
    assert "Re-run failed jobs" in str(error.value)
    assert "Never delete or modify a published release" in str(error.value)
    assert [args[2] for args in calls] == ["create"]  # No edit after a failed create.
    (dist / "example.whl").write_bytes(b"corrupt")
    calls.clear()
    with pytest.raises(ValueError, match="checksum mismatch"):
        release.finish()
    assert not calls


def test_publication_requires_successful_candidate_run(build, tmp_path, monkeypatch):
    dist, source = build
    completed = {"conclusion": "success", "path": ".github/workflows/start-release.yaml"}
    monkeypatch.setattr(
        release,
        "api",
        lambda path: completed if path.startswith("actions/") else {"object": {"sha": source}},
    )

    def command(*args):
        if args[:3] == ("gh", "release", "download"):
            return (dist / "candidate.json").read_text()
        return subprocess.check_output(args, text=True).strip()

    monkeypatch.setattr(release, "run", command)
    release.plan()
    assert "version=1.2.3\n" in (tmp_path / "outputs").read_text()
    completed["conclusion"] = "failure"
    with pytest.raises(ValueError, match="has not succeeded"):
        release.plan()


def test_release_version_preserves_dependency_lock(tmp_path, monkeypatch):
    import tomllib

    root = Path(__file__).resolve().parents[1]
    for name in ["pyproject.toml", "uv.lock", "README.md"]:
        shutil.copy(root / name, tmp_path / name)
    original = tomllib.loads((tmp_path / "uv.lock").read_text())
    monkeypatch.chdir(tmp_path)
    for version in ["0.3.0rc1", "0.3.0"]:
        monkeypatch.setenv("PACKAGE_VERSION", version)
        release.set_version()
        updated = tomllib.loads((tmp_path / "uv.lock").read_text())
        for package in updated["package"]:
            if package["name"] == "nemo-compass":
                assert package["version"] == version
                package["version"] = tomllib.loads((root / "pyproject.toml").read_text())[
                    "project"
                ]["version"]
        assert updated == original
