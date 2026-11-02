# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report dependency scope and require a maintainer acknowledgment of license changes.

Runs from the default branch with write credentials. PR content is read as data only.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections import defaultdict, deque
from pathlib import Path

import tomllib
import yaml
from generate_third_party_licenses import scan_licenses
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

MARKER = "<!-- compass-dependency-review -->"


def run(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def api(path: str, method: str = "GET", **fields: object) -> dict:
    command = ["gh", "api", f"repos/{os.environ['GITHUB_REPOSITORY']}/{path}", "--method", method]
    if method == "GET":
        return json.loads(run(*command))
    return json.loads(
        subprocess.check_output([*command, "--input", "-"], input=json.dumps(fields), text=True)
    )


def read(ref: str, path: str) -> str:
    return run("git", "show", f"{ref}:{path}")


def graph(lock: dict) -> dict:
    """Union platform resolutions; retain scopes, immediate introducers and directness."""
    packages = lock["package"]
    (root,) = [p for p in packages if p["name"] == "nemo-compass"]
    groups = {"core": root.get("dependencies", [])}
    groups.update({f"extra:{k}": v for k, v in root.get("optional-dependencies", {}).items()})
    groups.update({f"dev:{k}": v for k, v in root.get("dev-dependencies", {}).items()})
    result: dict = {}
    for scope, edges in groups.items():
        pending = deque((edge, "nemo-compass") for edge in edges)
        seen = set()
        while pending:
            edge, parent = pending.popleft()
            matches = [
                p
                for p in packages
                if p["name"] == edge["name"]
                and ("version" not in edge or p["version"] == edge["version"])
                and ("source" not in edge or p["source"] == edge["source"])
            ]
            if not matches:
                raise ValueError(f"Unresolved dependency edge: {edge}")
            for package in matches:
                if "registry" not in package.get("source", {}):
                    raise ValueError(
                        f"Unresolved license source (not a registry package): {package['name']}"
                    )
                key = package["name"], package["version"]
                entry = result.setdefault(key, {"scopes": set(), "via": set(), "direct": False})
                entry["scopes"].add(scope)
                entry["via"].add(parent)
                entry["direct"] |= parent == "nemo-compass"
                visit = (*key, tuple(sorted(edge.get("extra", []))))
                if visit in seen:
                    continue
                seen.add(visit)
                children = list(package.get("dependencies", []))
                for extra in edge.get("extra", []):
                    children.extend(package.get("optional-dependencies", {}).get(extra, []))
                pending.extend((child, package["name"]) for child in children)
    return {
        key: {k: sorted(v) if isinstance(v, set) else v for k, v in entry.items()}
        for key, entry in sorted(result.items())
    }


def scan(keys: set[tuple[str, str]]) -> dict:
    """Use the same pinned scanner as disclosure generation, including dev dependencies."""
    if not keys:
        return {}
    with tempfile.TemporaryDirectory() as temporary:
        requirements = Path(temporary) / "requirements.txt"
        output = Path(temporary) / "licenses.json"
        requirements.write_text("".join(f"{name}=={version}\n" for name, version in sorted(keys)))
        scan_licenses(requirements, output)
        data = json.loads(output.read_text())
    found = {}
    for section in data.get("results", []):
        for package in section.get("packages", []):
            identity = package["package"]
            found[identity["name"], identity["version"]] = package.get("licenses", [])
    if keys - found.keys():
        raise ValueError(f"License analysis incomplete: {sorted(keys - found.keys())}")
    return found


def snapshot(ref: str, scopes: dict, scanned: dict) -> dict:
    overrides = yaml.safe_load(read(ref, "third_party/license_overrides.yaml"))["overrides"]
    exceptions = yaml.safe_load(read(ref, "third_party/license_exceptions.yaml")) or {}
    notices = read(ref, "third_party/NOTICES.txt")
    # Ignore archive URLs/version headers for license comparisons; retain actual upstream text.
    blocks = re.split(r"={80}\n([^\n]+)\n={80}\n", notices)
    documents = {
        blocks[i]: re.sub(r"^(Source|SHA256|File):.*\n", "", blocks[i + 1], flags=re.M).strip()
        for i in range(1, len(blocks), 2)
    }
    lock = tomllib.loads(read(ref, "uv.lock"))
    packages = {(p["name"], p["version"]): p for p in lock["package"]}
    project = tomllib.loads(read(ref, "pyproject.toml"))
    requirements = list(project["project"].get("dependencies", []))
    for entries in project["project"].get("optional-dependencies", {}).values():
        requirements.extend(entries)
    for entries in project.get("dependency-groups", {}).values():
        requirements.extend(e for e in entries if isinstance(e, str))
    result = {}
    for key, scope in scopes.items():
        name, version = key
        licenses = scanned[key]
        expression = overrides.get(name) or " AND ".join(sorted(licenses))
        if not expression or re.search(r"\b(UNKNOWN|NOASSERTION|NONE|OTHER)\b", expression, re.I):
            raise ValueError(f"Unresolved license: {name}=={version}")
        text = documents.get(f"{name}=={version}", "")
        if any(not scope_name.startswith("dev:") for scope_name in scope["scopes"]) and not text:
            raise ValueError(f"Missing collected license documents: {name}=={version}")
        identifiers = set(re.findall(r"[A-Za-z0-9][A-Za-z0-9.-]*", expression)) - {
            "AND",
            "OR",
            "WITH",
        }
        shared = {}
        for path in run(
            "git", "ls-tree", "-r", "--name-only", ref, "third_party/license_texts"
        ).splitlines():
            if Path(path).stem.upper() in {i.upper() for i in identifiers}:
                shared[path] = read(ref, path)
        result[key] = {
            **scope,
            "license": expression.upper(),
            "terms": text,
            "shared": shared,
            "override": overrides.get(name),
            "exception": exceptions.get(name),
            "reported": sorted(licenses),
            "locked": packages[key],
            "requirements": sorted(
                r for r in requirements if canonicalize_name(Requirement(r).name) == name
            ),
        }
    return result


def compare(before: dict, after: dict) -> tuple[str, str | None]:
    grouped: dict = defaultdict(lambda: [[], []])
    for side, state in enumerate((before, after)):
        for (name, version), evidence in state.items():
            grouped[name][side].append((version, evidence))
    lines = [
        "| Dependency | Change | License before → after | Scope / introduced by |",
        "| --- | --- | --- | --- |",
    ]
    acknowledgment = []
    for name, (old, new) in sorted(grouped.items()):
        if old == new:
            continue
        change = "added" if not old else "removed" if not new else "updated"
        if len(old) == len(new) == 1 and old[0][0] != new[0][0]:
            change = "upgraded" if Version(new[0][0]) > Version(old[0][0]) else "downgraded"

        def versions(rows: list) -> str:
            return ", ".join(v for v, _ in rows) or "—"

        def expressions(rows: list) -> str:
            return "; ".join(sorted({e["license"] for _, e in rows})) or "—"

        scope = "; ".join(
            sorted(
                {
                    f"{', '.join(e['scopes'])}; {'direct' if e['direct'] else 'transitive'} via {', '.join(e['via'])}"
                    for _, e in new or old
                }
            )
        )
        lines.append(
            f"| {name} | {change}: {versions(old)} → {versions(new)} | {expressions(old)} → {expressions(new)} | {scope} |"
        )

        # Compare licensing evidence independently from version numbers and dependency scope.
        def terms(rows: list) -> set[str]:
            return {
                json.dumps({k: e[k] for k in ("license", "terms", "shared")}, sort_keys=True)
                for _, e in rows
            }

        if new and (not old or terms(old) != terms(new)):
            acknowledgment.append([name, old, new])
    if len(lines) == 2:
        return "No dependency or licensing changes.", None
    report = (
        "\n".join(lines)
        + "\n\nScopes include platform-specific resolutions; this does not determine functional impact."
    )
    if not acknowledgment:
        return (
            report
            + "\n\nNo new or changed licensing; version/scope changes and removals need no acknowledgment.",
            None,
        )
    fingerprint = hashlib.sha256(json.dumps(acknowledgment, sort_keys=True).encode()).hexdigest()
    return report, fingerprint


def comments(number: int) -> list:
    pages = json.loads(
        run(
            "gh",
            "api",
            "--paginate",
            "--slurp",
            f"repos/{os.environ['GITHUB_REPOSITORY']}/issues/{number}/comments",
        )
    )
    return [comment for page in pages for comment in page]


def acknowledged(number: int, fingerprint: str) -> str | None:
    for comment in comments(number):
        if f"Licensing acknowledged: {fingerprint}" not in comment["body"].splitlines():
            continue
        user = comment["user"]
        if user["type"] != "User":
            continue
        permission = api(f"collaborators/{user['login']}/permission")
        if permission["user"]["permissions"].get("push"):
            return f"@{user['login']} ([acknowledgment]({comment['html_url']}))"
    return None


def main() -> None:
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    number = event.get("number") or event["issue"]["number"]
    pr = api(f"pulls/{number}")
    if pr["state"] != "open":
        return
    head, base = pr["head"]["sha"], pr["base"]["sha"]
    check = api(
        "check-runs", "POST", name="Dependency licensing", head_sha=head, status="in_progress"
    )
    conclusion = "failure"
    try:
        run("git", "fetch", "--no-tags", "origin", head, base)
        paths = ["uv.lock", "pyproject.toml", "third_party"]
        changed = run("git", "diff", "--name-only", base, head, "--", *paths)
        if not changed:
            report, fingerprint = "No dependency or licensing changes.", None
        else:
            scopes = [graph(tomllib.loads(read(ref, "uv.lock"))) for ref in (base, head)]
            scanned = scan(set(scopes[0]) | set(scopes[1]))
            before, after = [
                snapshot(ref, scope, scanned)
                for ref, scope in zip((base, head), scopes, strict=True)
            ]
            report, fingerprint = compare(before, after)
        if fingerprint:
            author = acknowledged(number, fingerprint)
            report += f"\n\nEvidence: `{fingerprint}`\n\n"
            report += (
                f"Acknowledged by {author}."
                if author
                else (
                    "A maintainer with merge access must post this separate comment:\n\n"
                    f"```text\nLicensing acknowledged: {fingerprint}\n```"
                )
            )
            report += "\n\nThis is maintainer acknowledgment, not legal approval. Escalation is a maintainer decision."
            conclusion = "success" if author else "failure"
        else:
            conclusion = "success"
        if api(f"pulls/{number}")["head"]["sha"] != head:
            raise ValueError("PR changed during analysis; awaiting a fresh check")
    except Exception as error:
        report = f"Unresolved dependency/license analysis: {error}"
        conclusion = "failure"
    body = MARKER + "\n## Dependency and licensing review\n\n" + report
    existing = next(
        (
            c
            for c in comments(number)
            if c["user"]["login"] == "github-actions[bot]" and c["body"].startswith(MARKER)
        ),
        None,
    )
    if existing:
        api(f"issues/comments/{existing['id']}", "PATCH", body=body)
    else:
        api(f"issues/{number}/comments", "POST", body=body)
    api(
        f"check-runs/{check['id']}",
        "PATCH",
        status="completed",
        conclusion=conclusion,
        output={"title": "Dependency licensing", "summary": report[:65000]},
    )
    print(report)


if __name__ == "__main__":
    main()
