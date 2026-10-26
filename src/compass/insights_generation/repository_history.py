# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small, bounded Git history tool for code-aware insight review."""

from __future__ import annotations

import os
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Annotated, Literal

from nooa.agentdoc import spec
from nooa.skill import Skill

from compass.insight import CommitEvidence

_COMMAND_TIMEOUT = 2
_FETCH_TIMEOUT = 5
_LOOKBACK_DAYS = 730
_MAX_OUTPUT_BYTES = 100_000
_MAX_COMMITS = 20
_SHA = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_Freshness = Literal["aligned", "ahead", "behind", "diverged", "unknown"]


class GitHistory(Skill):
    """Read bounded history from one repository and optionally refresh its upstream."""

    def __init__(self, code_base: Path) -> None:
        requested = code_base.expanduser().resolve()
        root = self._run_at(requested, "rev-parse", "--show-toplevel")
        if root is None:
            raise ValueError(f"code_base is not inside a Git repository: {code_base}")
        self._root = Path(root.strip()).resolve()
        self._prefix = requested.relative_to(self._root)
        self._refresh_lock = Lock()
        self._refresh_attempted = False
        self._remote_refreshed = False

    @staticmethod
    def _environment() -> dict[str, str]:
        return {
            **os.environ,
            "GIT_TERMINAL_PROMPT": "0",
            "GCM_INTERACTIVE": "Never",
            "GIT_PAGER": "cat",
            "GIT_SSH_COMMAND": "ssh -o BatchMode=yes",
            "LC_ALL": "C",
        }

    @classmethod
    def _run_at(
        cls,
        root: Path,
        *args: str,
        timeout: float = _COMMAND_TIMEOUT,
    ) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), "-c", "credential.interactive=false", *args],
                check=False,
                capture_output=True,
                env=cls._environment(),
                timeout=timeout,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        if result.returncode:
            return None
        return result.stdout[:_MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")

    def _run(self, *args: str, timeout: float = _COMMAND_TIMEOUT) -> str | None:
        return self._run_at(self._root, *args, timeout=timeout)

    def refresh_remote(self) -> bool:
        """Try one noninteractive upstream fetch without changing checked-out files."""

        with self._refresh_lock:
            if self._refresh_attempted:
                return self._remote_refreshed
            self._refresh_attempted = True
            upstream = self._run(
                "rev-parse",
                "--abbrev-ref",
                "--symbolic-full-name",
                "@{upstream}",
            )
            if not upstream or "/" not in upstream:
                return False
            remote, branch = upstream.strip().split("/", 1)
            destination = f"refs/remotes/{remote}/{branch}"
            self._remote_refreshed = (
                self._run(
                    "fetch",
                    "--no-tags",
                    "--no-recurse-submodules",
                    "--quiet",
                    remote,
                    f"+refs/heads/{branch}:{destination}",
                    timeout=_FETCH_TIMEOUT,
                )
                is not None
            )
            return self._remote_refreshed

    def _path(self, path: str) -> str:
        relative = PurePosixPath(path)
        if relative.is_absolute() or ".." in relative.parts or ".git" in relative.parts:
            raise ValueError(f"path is not repository-relative: {path}")
        combined = self._prefix / Path(relative)
        return combined.as_posix()

    def _state(self) -> tuple[str, _Freshness]:
        head = (self._run("rev-parse", "HEAD") or "").strip()
        if not _SHA.fullmatch(head):
            raise ValueError("Git HEAD is unavailable")
        upstream = self._run(
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        )
        counts = (
            self._run(
                "rev-list",
                "--left-right",
                "--count",
                f"HEAD...{upstream.strip()}",
            )
            if upstream
            else None
        )
        try:
            ahead, behind = (int(value) for value in (counts or "").split())
        except ValueError:
            return head, "unknown"
        if ahead and behind:
            freshness: _Freshness = "diverged"
        elif ahead:
            freshness = "ahead"
        elif behind:
            freshness = "behind"
        else:
            freshness = "aligned"
        return head, freshness

    def git_history(
        self,
        path: Annotated[str, spec(description="Repository-relative source path")],
        query: Annotated[
            str,
            spec(description="Optional literal text whose additions or removals are relevant"),
        ] = "",
        observed_at: Annotated[
            datetime | None,
            spec(description="Earliest supporting trace timestamp, when available"),
        ] = None,
    ) -> str:
        """Inspect bounded commits and patches for one relevant source path."""

        head, freshness = self._state()
        args = [
            "log",
            "--all",
            f"--max-count={_MAX_COMMITS}",
            "--format=commit %H%nDate: %cI%nSubject: %s",
            "--patch",
            "--unified=3",
        ]
        if observed_at is not None:
            start = observed_at - timedelta(days=_LOOKBACK_DAYS)
            args.append(f"--since={start.isoformat()}")
        if query:
            args.extend(["-S", query])
        args.extend(["--", self._path(path)])
        history = self._run(*args) or "No matching local history was available."
        return (
            f"Analyzed HEAD {head}; upstream={freshness}; "
            f"remote_refreshed={self._remote_refreshed}.\n{history}"
        )

    def evidence(self, commits: list[CommitEvidence]) -> list[CommitEvidence]:
        """Keep at most three unique commit objects that exist in this repository."""

        verified = []
        seen = set()
        for item in commits[:3]:
            if item.commit in seen or not _SHA.fullmatch(item.commit):
                continue
            if self._run("cat-file", "-e", f"{item.commit}^{{commit}}") is None:
                continue
            seen.add(item.commit)
            verified.append(item)
        return verified


__all__ = ["GitHistory"]
