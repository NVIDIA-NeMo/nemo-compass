# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Find release fixes and merge their tested forward-ports."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path


def api(path: str, *args: str) -> str:
    return subprocess.check_output(
        ["gh", "api", f"repos/{os.environ['GITHUB_REPOSITORY']}/{path}", *args], text=True
    )


def ready_to_merge(pull: dict, run: dict, repo: str) -> bool:
    head = pull["head"]
    return (
        pull["state"] == "open"
        and not pull["draft"]
        and pull["user"]["login"] == "nemo-llm"
        and pull["base"]["ref"] == "main"
        and head["repo"]["full_name"] == repo
        and head["sha"] == run["head_sha"]
        and head["ref"] == run["head_branch"]
        and re.fullmatch(r"backport-[0-9]+-to-main", head["ref"]) is not None
    )


def main() -> None:
    run = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())["workflow_run"]
    source = run["head_sha"]
    if sys.argv[1] == "find":
        for pull in json.loads(api(f"commits/{source}/pulls")):
            if (
                pull["merged_at"]
                and pull["merge_commit_sha"] == source
                and pull["base"]["ref"] == run["head_branch"]
            ):
                with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
                    print(f"number={pull['number']}", file=output)
                return
    elif sys.argv[1] == "merge" and run["pull_requests"]:
        number = run["pull_requests"][0]["number"]
        pull = json.loads(api(f"pulls/{number}"))
        if ready_to_merge(pull, run, os.environ["GITHUB_REPOSITORY"]):
            # Only review is bypassed; GitHub enforces CI and branch freshness.
            result = json.loads(
                api(
                    f"pulls/{number}/merge",
                    "--method",
                    "PUT",
                    "-f",
                    f"sha={source}",
                    "-f",
                    "merge_method=squash",
                )
            )
            if not result["merged"]:
                sys.exit(result["message"])
            print(result["message"])


if __name__ == "__main__":
    main()
