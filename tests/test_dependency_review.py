# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency review policy, graph coverage, and fail-closed acknowledgment."""

import copy
import importlib.util
import sys
from pathlib import Path
from typing import cast

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
spec = importlib.util.spec_from_file_location(
    "dependency_review", Path(__file__).resolve().parents[1] / "tools/dependency_review.py"
)
assert spec and spec.loader
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def test_graph_tracks_extras_dev_cycles_and_platform_versions():
    lock: dict = {
        "package": [
            {
                "name": "nemo-compass",
                "version": "1",
                "dependencies": [{"name": "a", "extra": ["trace"]}],
                "optional-dependencies": {"sdk": [{"name": "b"}]},
                "dev-dependencies": {"dev": [{"name": "b"}]},
            },
            {
                "name": "a",
                "version": "1",
                "optional-dependencies": {"trace": [{"name": "b"}], "unused": [{"name": "absent"}]},
            },
            {"name": "b", "version": "1", "dependencies": [{"name": "a"}]},
            {"name": "b", "version": "2", "dependencies": []},
        ]
    }
    lock["package"] = [
        {**p, "source": {"registry": "https://pypi.org/simple"}} for p in lock["package"]
    ]
    graph = review.graph(lock)
    assert set(graph) == {("a", "1"), ("b", "1"), ("b", "2")}
    assert graph["b", "2"] == {
        "scopes": ["core", "dev:dev", "extra:sdk"],
        "via": ["a", "nemo-compass"],
        "direct": True,
    }
    cast(dict, lock["package"][0])["dependencies"].append({"name": "missing"})
    with pytest.raises(ValueError, match="Unresolved dependency edge"):
        review.graph(lock)


def test_acknowledgment_covers_only_relevant_license_evidence():
    evidence = {
        "license": "MIT",
        "terms": "MIT terms",
        "shared": {},
        "direct": True,
        "scopes": ["core"],
        "via": ["nemo-compass"],
    }
    base = {("a", "1"): evidence}
    for changed in [{("a", "2"): evidence}, {}, base]:
        assert review.compare(base, changed)[1] is None
    changed = {**base, ("b", "1"): evidence}
    report, fingerprint = review.compare(base, changed)
    assert fingerprint and "added" in report
    assert review.compare(base, copy.deepcopy(changed))[1] == fingerprint
    # An unrelated version-only update does not invalidate acknowledgment of b.
    assert review.compare(base, {("a", "2"): evidence, ("b", "1"): evidence})[1] == fingerprint
    for field, value in [
        ("license", "BSD-3-CLAUSE"),
        ("terms", "new terms"),
        ("scopes", ["dev:test"]),
    ]:
        revised = copy.deepcopy(changed)
        revised["b", "1"][field] = value
        assert review.compare(base, revised)[1] != fingerprint
    assert review.compare(base, {("a", "1"): {**evidence, "license": "BSD-3-CLAUSE"}})[1]


def test_ack_requires_exact_comment_and_current_merge_permission(monkeypatch):
    comment = {
        "body": "Licensing acknowledged: abc",
        "user": {"login": "maintainer", "type": "User"},
        "html_url": "https://example.com/comment",
    }
    monkeypatch.setattr(review, "comments", lambda number: [comment])
    monkeypatch.setattr(review, "api", lambda path: {"user": {"permissions": {"push": True}}})
    assert "@maintainer" in review.acknowledged(1, "abc")
    assert review.acknowledged(1, "abcd") is None
    monkeypatch.setattr(review, "api", lambda path: {"user": {"permissions": {"push": False}}})
    assert review.acknowledged(1, "abc") is None


def test_incomplete_scanner_output_is_unresolved(monkeypatch):
    def scan(args, **kwargs):
        Path(args[-1]).write_text('{"results": []}')

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(review.subprocess, "run", scan)
    with pytest.raises(ValueError, match="incomplete"):
        review.scan({("a", "1")})
