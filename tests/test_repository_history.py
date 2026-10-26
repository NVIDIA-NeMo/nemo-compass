# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from compass.insight import CommitEvidence
from compass.insights_generation.repository_history import GitHistory


def _git(path: Path, *args: str, env: dict[str, str] | None = None) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), *args],
        text=True,
        env={**os.environ, "GIT_TEMPLATE_DIR": "", **(env or {})},
    ).strip()


def _commit(path: Path, text: str, subject: str, date: str) -> str:
    (path / "agent.py").write_text(text, encoding="utf-8")
    _git(path, "add", "agent.py")
    _git(
        path,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        subject,
        env={"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date},
    )
    return _git(path, "rev-parse", "HEAD")


def _checkout(tmp_path: Path) -> tuple[Path, Path, str, str]:
    environment = {**os.environ, "GIT_TEMPLATE_DIR": ""}
    remote, seed, checkout = (
        tmp_path / "remote.git",
        tmp_path / "seed",
        tmp_path / "checkout",
    )
    subprocess.check_call(["git", "init", "--bare", "-q", str(remote)], env=environment)
    subprocess.check_call(["git", "init", "-q", str(seed)], env=environment)
    initial = _commit(seed, "TIMEOUT = 1\n", "Add short timeout", "2024-01-01T00:00:00Z")
    _git(seed, "remote", "add", "origin", str(remote))
    _git(seed, "push", "-qu", "origin", "HEAD")
    subprocess.check_call(["git", "clone", "-q", str(remote), str(checkout)], env=environment)
    fixed = _commit(seed, "TIMEOUT = 30\n", "Fix timeout failures", "2025-01-01T00:00:00Z")
    _git(seed, "push", "-q")
    return checkout, seed, initial, fixed


def test_refresh_finds_remote_commit_without_changing_checkout(tmp_path) -> None:
    checkout, _seed, initial, fixed = _checkout(tmp_path)
    history = GitHistory(checkout)

    assert history.refresh_remote() is True
    context = history.git_history(
        "agent.py",
        observed_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )

    assert fixed in context
    assert "upstream=behind" in context
    assert _git(checkout, "rev-parse", "HEAD") == initial
    assert (checkout / "agent.py").read_text(encoding="utf-8") == "TIMEOUT = 1\n"


def test_failed_fetch_uses_local_history(tmp_path, monkeypatch) -> None:
    checkout, _seed, initial, _fixed = _checkout(tmp_path)
    history = GitHistory(checkout)
    original_run = history._run

    def fail_fetch(*args: str, timeout: float = 2):
        return None if args and args[0] == "fetch" else original_run(*args, timeout=timeout)

    monkeypatch.setattr(history, "_run", fail_fetch)

    assert history.refresh_remote() is False
    assert initial in history.git_history("agent.py")
    assert history.evidence([]) == []
    assert history._environment()["GIT_TERMINAL_PROMPT"] == "0"


def test_commit_evidence_supports_multiple_verified_commits(tmp_path) -> None:
    checkout, _seed, initial, fixed = _checkout(tmp_path)
    history = GitHistory(checkout)
    history.refresh_remote()
    evidence = history.evidence(
        [
            CommitEvidence(
                commit=initial,
                evidence_reason="This commit introduces the short timeout.",
                resolution="unresolved",
            ),
            CommitEvidence(
                commit=fixed,
                evidence_reason="This commit increases the timeout to prevent the failure.",
                resolution="resolved",
            ),
            CommitEvidence(
                commit="f" * 40,
                evidence_reason="This commit is not available in the repository.",
                resolution="partial",
            ),
        ]
    )

    assert [item.commit for item in evidence] == [initial, fixed]
    assert [item.resolution for item in evidence] == ["unresolved", "resolved"]
