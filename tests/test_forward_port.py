# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Only the bot's current, non-draft forward-port may merge."""

from copy import deepcopy

from tools.forward_port import ready_to_merge


def test_merge_requires_the_tested_bot_forward_port():
    repo = "org/repo"
    run = {"head_sha": "tested", "head_branch": "backport-132-to-main"}
    pull = {
        "state": "open",
        "draft": False,
        "user": {"login": "nemo-llm"},
        "base": {"ref": "main"},
        "head": {"sha": "tested", "ref": run["head_branch"], "repo": {"full_name": repo}},
    }
    assert ready_to_merge(pull, run, repo)
    for field, value in [
        ("state", "closed"),
        ("draft", True),
        ("user", {"login": "someone-else"}),
        ("base", {"ref": "release/0.2.1"}),
        ("head", {**pull["head"], "sha": "untested"}),
        ("head", {**pull["head"], "repo": {"full_name": "someone/fork"}}),
        ("head", {**pull["head"], "ref": "another-branch"}),
    ]:
        changed = deepcopy(pull)
        changed[field] = value
        assert not ready_to_merge(changed, run, repo), field
    ordinary = {**run, "head_branch": "feature"}
    pull["head"]["ref"] = "feature"
    assert not ready_to_merge(pull, ordinary, repo)
