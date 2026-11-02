# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency review policy, graph coverage, and fail-closed acknowledgment."""

import copy
import importlib.util
import json
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
    ambiguous = copy.deepcopy(lock)
    ambiguous["package"].append(
        {**ambiguous["package"][1], "source": {"registry": "https://other.example"}}
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        review.graph(ambiguous)
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
    report, fingerprint = review.compare(
        base, {("a", "1"): {**evidence, "terms": "changed notice"}}
    )
    assert fingerprint and "documents changed (expression unchanged)" in report


@pytest.mark.parametrize(
    "licenses, override, allowed",
    [
        (["mit"], None, True),
        (["Apache-2.0"], None, True),
        (["GPL-3.0-only"], None, False),
        (["MIT AND GPL-3.0-only"], None, False),
        (["MIT OR Apache-2.0"], None, False),
        (["MIT OR BSD-2-Clause"], None, True),
        (["Apache-2.0 WITH LLVM-exception"], None, False),
        (["MIT"], "MPL-2.0", False),
        (["UNKNOWN"], "MIT", True),
        (["UNKNOWN"], None, False),
    ],
)
def test_allowlist_checks_complete_resolved_expressions(monkeypatch, licenses, override, allowed):
    monkeypatch.setattr(
        review, "ALLOWED_LICENSES", review.ALLOWED_LICENSES | {"MIT OR BSD-2-CLAUSE"}
    )
    expression = review.resolve_license(licenses, override)
    state = {("a", "1"): {"license": expression}}
    assert bool(review.disallowed_licenses(state)) is not allowed


def test_incomplete_scanner_output_is_unresolved(monkeypatch):
    def scan(args, **kwargs):
        Path(args[-1]).write_text('{"results": []}')

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(review.subprocess, "run", scan)
    with pytest.raises(ValueError, match="incomplete"):
        review.scan({("a", "1")})


def test_inventory_snapshot_requires_complete_unambiguous_evidence(monkeypatch):
    for invalid in [[], ["UNKNOWN"], ["MIT", "NOASSERTION"], ["MIT OR"], ["(MIT"]]:
        assert review.resolve_license(invalid, None) is None
    expression = review.resolve_license(["mit OR BSD-3-Clause", " Apache-2.0  "], None)
    assert expression == "APACHE-2.0 AND (MIT OR BSD-3-CLAUSE)"
    source = {"registry": "https://pypi.org/simple"}
    record = {
        "name": "a",
        "version": "1",
        "license": expression,
        "source": source,
        "document_hashes": ["abc"],
    }
    files = {
        "third_party/license_overrides.yaml": "overrides: {}",
        "third_party/license_exceptions.yaml": "{}",
        "uv.lock": "[[package]]"
        + '\nname="a"\nversion="1"\nsource={registry="https://pypi.org/simple"}',
    }
    monkeypatch.setattr(
        review,
        "read",
        lambda ref, path: json.dumps(record) if path.endswith("licenses.jsonl") else files[path],
    )
    monkeypatch.setattr(review, "run", lambda *args: "")
    scopes = {("a", "1"): {"scopes": ["core"], "via": ["nemo-compass"], "direct": True}}
    scan = {("a", "1"): ["MIT OR BSD-3-Clause", "Apache-2.0"]}
    original = review.snapshot("head", scopes, scan)
    assert original["a", "1"]["terms"] == ["abc"]
    # Artifact metadata cannot invalidate acknowledgment of unchanged license evidence.
    files["uv.lock"] += '\nwheels=[{url="https://example.com/a.whl",upload-time="2026-01-01"}]'
    revised = review.snapshot("head", scopes, scan)
    assert revised == original
    assert review.compare({}, revised)[1] == review.compare({}, original)[1]
    for key, invalid in [("source", {}), ("document_hashes", []), ("license", "MIT")]:
        previous = record[key]
        record[key] = invalid
        with pytest.raises(ValueError, match="Missing or stale inventory"):
            review.snapshot("head", scopes, scan)
        record[key] = previous


def test_check_lifecycle_rechecks_acknowledgment_and_pr_identity(tmp_path, monkeypatch):
    event = tmp_path / "event.json"
    event.write_text('{"number": 1}')
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    pr = {"state": "open", "head": {"sha": "head"}, "base": {"sha": "base"}}
    comment = {
        "body": "Licensing acknowledged: fingerprint",
        "user": {"type": "User", "login": "maintainer"},
        "html_url": "https://example.com/ack",
    }
    permission = True
    stale = False
    reads = 0
    writes = []

    def api(path, method="GET", **fields):
        nonlocal reads
        if path == "pulls/1":
            reads += 1
            return {**pr, "base": {"sha": "new"}} if stale and reads % 2 == 0 else pr
        if path.startswith("collaborators/"):
            return {"user": {"permissions": {"push": permission}}}
        writes.append((path, fields))
        return {"id": 42}

    monkeypatch.setattr(review, "api", api)
    monkeypatch.setattr(review, "run", lambda *args: "uv.lock")
    monkeypatch.setattr(review, "read", lambda *args: "")
    monkeypatch.setattr(review, "graph", lambda lock: {})
    monkeypatch.setattr(review, "scan", lambda keys: {})
    monkeypatch.setattr(review, "snapshot", lambda *args: {})
    monkeypatch.setattr(review, "compare", lambda *args: ("license changed", "fingerprint"))
    monkeypatch.setattr(review, "comments", lambda number: [comment] if comment else [])
    for body, allowed, changed, expected in [
        ("Licensing acknowledged: fingerprint", True, False, "success"),
        ("edited away", True, False, "failure"),
        ("Licensing acknowledged: wrong fingerprint", True, False, "failure"),
        ("Licensing acknowledged: fingerprint", False, False, "failure"),
        ("Licensing acknowledged: fingerprint", True, True, "failure"),
    ]:
        permission, stale = allowed, changed
        comment["body"] = body
        review.main()
        assert writes[-1][1]["conclusion"] == expected
    comment = None
    stale = False
    review.main()
    assert writes[-1][1]["conclusion"] == "failure"
    # Even a valid acknowledgment cannot clear a disallowed corrected expression.
    comment = {
        "body": "Licensing acknowledged: fingerprint",
        "user": {"type": "User", "login": "maintainer"},
        "html_url": "https://example.com/ack",
    }
    monkeypatch.setattr(review, "snapshot", lambda *args: {("a", "1"): {"license": "MPL-2.0"}})
    review.main()
    assert writes[-1][1]["conclusion"] == "failure"
    summary = writes[-1][1]["output"]["summary"]
    assert "Acknowledged by @maintainer" in summary
    assert "a==1: `MPL-2.0`" in summary and "Acknowledgment cannot clear" in summary
    monkeypatch.setattr(review, "compare", lambda *args: ("Version update", None))
    review.main()
    assert writes[-1][1]["conclusion"] == "failure"
