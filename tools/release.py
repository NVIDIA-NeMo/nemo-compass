# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release orchestration; CI owns building and validating package bytes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def run(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def api(path: str) -> dict:
    return json.loads(run("gh", "api", f"repos/{os.environ['GITHUB_REPOSITORY']}/{path}"))


def sha(ref: str) -> str:
    return run("git", "rev-parse", "--verify", f"{ref}^{{commit}}")


def tag_source(tag: str) -> str:
    # Tags may have been created by a previous attempt after this checkout.
    run("git", "fetch", "origin", f"refs/tags/{tag}")
    return sha("FETCH_HEAD")


def release_version(value: str) -> str:
    if not re.fullmatch(r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)", value):
        raise ValueError("Use a final version such as 0.3.0")
    return value


def candidate(branch: str, commit: str) -> str:
    version = release_version(branch.removeprefix("release/"))
    base = f"refs/tags/release-base/{version}"
    run("git", "merge-base", "--is-ancestor", base, commit)
    count = int(run("git", "rev-list", "--first-parent", "--count", f"{base}..{commit}"))
    return f"{version}rc{count + 1}"


def current(branch: str, commit: str) -> None:
    head = api(f"git/ref/heads/{branch}")["object"]["sha"]
    if head != commit:
        run("git", "fetch", "origin", head)
        newer = candidate(branch, head)
        raise ValueError(f"Branch changed: validate {newer} at {head} before publishing.")


def download(tag: str, directory: Path) -> None:
    run("gh", "release", "download", tag, "--dir", str(directory))


def verify(directory: Path) -> dict:
    manifest = json.loads((directory / "candidate.json").read_text())
    expected = manifest["sha256"]
    if not expected or any(Path(name).name != name for name in expected):
        raise ValueError("Invalid artifact manifest")
    for name, digest in expected.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    return manifest


def set_version() -> None:
    """Set build metadata without re-resolving locked dependencies."""
    version = os.environ["PACKAGE_VERSION"]
    run("uv", "version", "--frozen", version)
    lock = Path("uv.lock")
    updated, count = re.subn(
        r'(name = "nemo-compass"\nversion = ")[^"]+("\n)',
        lambda match: match[1] + version + match[2],
        lock.read_text(),
    )
    assert count == 1, "Expected exactly one Compass lock entry"
    lock.write_text(updated)


def record() -> None:
    """Keep source, checksums and the original CI attempt with the built artifacts."""
    version = os.environ["PACKAGE_VERSION"]
    manifest = {
        "source": os.environ["SOURCE"],
        "branch": "release/" + version.split("rc")[0],
        "version": version,
        "run_id": os.environ["GITHUB_RUN_ID"],
        "run_attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
        "sha256": dict(
            line.split()[::-1] for line in Path("dist/SHA256SUMS").read_text().splitlines()
        ),
    }
    Path("dist/candidate.json").write_text(json.dumps(manifest, indent=2) + "\n")


def plan() -> None:
    action = os.environ.get("ACTION", "") or "candidate"
    if action == "start":
        version = release_version(os.environ["VERSION"])
        branch = f"release/{version}"
        selected = os.environ.get("COMMIT", "") or os.environ["GITHUB_SHA"]
        if not re.fullmatch(r"[0-9a-f]{40}", selected):
            raise ValueError("Commit must be a full main commit SHA")
        commit = sha(selected)
        run("git", "merge-base", "--is-ancestor", commit, "origin/main")
        try:
            run("git", "cat-file", "-e", f"{commit}:tools/release.py")
        except subprocess.CalledProcessError as error:
            raise ValueError(
                "Selected commit predates release automation; choose a newer main commit"
            ) from error
        # A retry may reuse the branch only while it is still at the original cut.
        existing = run("git", "ls-remote", "origin", f"refs/heads/{branch}")
        if existing and existing.split()[0] != commit:
            raise ValueError("Release branch already advanced; use its current candidate")
        package_version = version + "rc1"
    elif action == "publish":
        tag = os.environ["CANDIDATE"]
        if not re.fullmatch(r"v\d+\.\d+\.\d+rc[1-9]\d*", tag):
            raise ValueError("Select an exact candidate tag, e.g. v0.3.0rc2")
        if os.environ.get("READY") != "true" or not os.environ.get("QA_EVIDENCE", "").startswith(
            "https://"
        ):
            raise ValueError("Confirm readiness and supply the QA evidence HTTPS URL")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            download(tag, directory)
            accepted = verify(directory)
        commit, branch = accepted["source"], accepted["branch"]
        if accepted["version"] != tag[1:] or tag_source(tag) != commit:
            raise ValueError("Candidate identity differs from its tag")
        completed = api(f"actions/runs/{accepted['run_id']}")
        if (
            completed["conclusion"] != "success"
            or completed["path"] != ".github/workflows/release.yaml"
        ):
            raise ValueError("Candidate preparation/staging has not succeeded")
        current(branch, commit)
        if candidate(branch, commit) != tag[1:]:
            raise ValueError("Selected candidate does not match branch history")
        package_version = release_version(branch.removeprefix("release/"))
    else:
        branch = os.environ["GITHUB_REF_NAME"]
        commit = os.environ["GITHUB_SHA"]
        current(branch, commit)
        package_version = candidate(branch, commit)
    values = {"source": commit, "branch": branch, "version": package_version, "action": action}
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        for key, value in values.items():
            print(f"{key}={value}", file=output)


def finish() -> None:
    source, branch, version = (os.environ[key] for key in ("SOURCE", "BRANCH", "PACKAGE_VERSION"))
    action = os.environ["ACTION"]
    tag = f"v{version}"
    if action == "start":
        base = f"refs/tags/release-base/{version.split('rc')[0]}"
        run(
            "git", "push", "--atomic", "origin", f"{source}:refs/heads/{branch}", f"{source}:{base}"
        )
    current(branch, source)
    directory = Path("dist")
    manifest = verify(directory)
    if (manifest["source"], manifest["branch"], manifest["version"], manifest["run_id"]) != (
        source,
        branch,
        version,
        os.environ["GITHUB_RUN_ID"],
    ):
        raise ValueError("Build record differs from the planned release")
    existing = run("gh", "release", "list", "--limit", "1000", "--json", "tagName")
    if tag not in {r["tagName"] for r in json.loads(existing)}:
        run("git", "push", "origin", f"{source}:refs/tags/{tag}")
        run(
            "gh",
            "release",
            "create",
            tag,
            "--verify-tag",
            "--draft",
            "--generate-notes",
            "--title",
            f"nemo-compass {tag}",
            "--prerelease=" + str(action != "publish").lower(),
        )
    if tag_source(tag) != source:
        raise ValueError("Existing tag points to another source commit")
    release = json.loads(run("gh", "release", "view", tag, "--json", "assets,body,isDraft,url"))
    # Resume only missing uploads. Existing bytes must match, including the manifest.
    with tempfile.TemporaryDirectory() as temporary:
        if release["assets"]:
            download(tag, Path(temporary))
        for path in sorted(directory.iterdir()):
            remote = Path(temporary) / path.name
            if remote.exists():
                if remote.read_bytes() != path.read_bytes():
                    raise ValueError(
                        f"Existing {path.name} differs; rerun failed jobs from the original run"
                    )
            else:
                if not release["isDraft"]:
                    raise ValueError(
                        "Published release is incomplete; never mutate published artifacts"
                    )
                run("gh", "release", "upload", tag, str(path))
    marker = "<!-- compass-candidate -->"
    notes = release["body"].split(marker)[0].rstrip()
    notes += f"\n\n{marker}\nSource: `{source}`\n\nVersion: `{version}`\n\n"
    notes += f"[Validated build](https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{manifest['run_id']}/attempts/{manifest['run_attempt']})\n"
    if action == "publish":
        notes += f"\nAccepted {os.environ['CANDIDATE']} by @{os.environ['GITHUB_ACTOR']}: {os.environ['QA_EVIDENCE']}\n"
    Path("release-notes.md").write_text(notes)
    if release["isDraft"]:
        current(branch, source)
        run(
            "gh",
            "release",
            "edit",
            tag,
            "--notes-file",
            "release-notes.md",
            "--draft=" + str(action != "publish").lower(),
        )
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as summary:
        print(f"[{tag}]({release['url']}) — source `{source}`\n", file=summary)
        if action == "publish":
            for name in manifest["sha256"]:
                print(
                    f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/releases/download/{tag}/{name}",
                    file=summary,
                )


if __name__ == "__main__":
    try:
        {"set-version": set_version, "record": record, "plan": plan, "finish": finish}[
            sys.argv[1]
        ]()
    except (ValueError, KeyError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
