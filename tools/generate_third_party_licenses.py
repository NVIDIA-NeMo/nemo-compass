#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate and validate dependency license disclosures using the NeMo Platform OSV flow."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypedDict

import tomllib
import yaml
from collect_license_texts import Artifact, Document, collect, render_texts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = PROJECT_ROOT / "third_party" / "licenses.jsonl"
NOTICES_PATH = PROJECT_ROOT / "THIRD_PARTY_LICENSES.md"
OVERRIDES_PATH = PROJECT_ROOT / "third_party" / "license_overrides.yaml"
TEXTS_PATH = PROJECT_ROOT / "third_party" / "NOTICES.txt"
EXCEPTIONS_PATH = PROJECT_ROOT / "third_party" / "license_exceptions.yaml"
LICENSE_TEXTS_PATH = PROJECT_ROOT / "third_party" / "license_texts"


class LicenseRecord(TypedDict):
    """Published license metadata for one locked dependency."""

    name: str
    version: str
    license: str
    package_url: str
    license_documents: list[str]


def _canonicalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _scanner() -> str:
    scanner = shutil.which("osv-scanner")
    if scanner is None:
        raise RuntimeError("osv-scanner was not found on PATH")
    return scanner


def _export_requirements(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "uv",
            "export",
            "--frozen",
            "--no-dev",
            "--all-extras",
            "--quiet",
            "--output-file",
            str(output_path.relative_to(PROJECT_ROOT)),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def scan_licenses(requirements_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    result = subprocess.run(
        [
            _scanner(),
            "scan",
            "source",
            "--licenses",
            "--lockfile",
            str(requirements_path),
            "--format",
            "json",
            "--all-packages",
            "--output",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if result.returncode not in {0, 1} or not output_path.exists():
        raise RuntimeError(f"OSV-Scanner failed with exit code {result.returncode}")

    raw = json.loads(output_path.read_text(encoding="utf-8"))
    for scan_result in raw.get("results", []):
        scan_result.get("source", {}).pop("path", None)
    output_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")


def _load_overrides() -> dict[str, str]:
    raw = yaml.safe_load(OVERRIDES_PATH.read_text(encoding="utf-8")) or {}
    return {_canonicalize(name): license_name for name, license_name in raw["overrides"].items()}


def _resolve_license(licenses: list[str], override: str | None) -> str | None:
    if override is not None:
        return override.upper()
    if not licenses or any(
        re.search(r"\b(UNKNOWN|NOASSERTION|NONE|OTHER)\b", name, re.I) for name in licenses
    ):
        return None
    return " AND ".join(sorted(licenses)).upper()


def _license_records(osv_path: Path) -> list[LicenseRecord]:
    raw = json.loads(osv_path.read_text(encoding="utf-8"))
    overrides = _load_overrides()
    records: dict[tuple[str, str], LicenseRecord] = {}
    unresolved = []
    used_overrides = set()
    standard_texts = {path.stem.upper(): path for path in LICENSE_TEXTS_PATH.glob("*.txt")}

    for scan_result in raw.get("results", []):
        for package_data in scan_result.get("packages", []):
            package = package_data["package"]
            name = package["name"]
            key = _canonicalize(name)
            override = overrides.get(key)
            license_name = _resolve_license(package_data.get("licenses", []), override)
            if license_name is None:
                unresolved.append(f"{name}=={package.get('version', '')}")
                continue
            if override is not None:
                used_overrides.add(key)
            identifiers = set(re.findall(r"[A-Z0-9][A-Z0-9.-]*", license_name)) - {
                "AND",
                "OR",
                "WITH",
            }
            missing = identifiers - standard_texts.keys()
            if missing:
                raise RuntimeError("Add shared license texts for: " + ", ".join(sorted(missing)))
            records[key, package["version"]] = {
                "name": name,
                "version": package["version"],
                "license": license_name,
                "package_url": f"https://pypi.org/project/{name}/{package['version']}/",
                "license_documents": ["third_party/NOTICES.txt"]
                + [
                    str(standard_texts[identifier].relative_to(PROJECT_ROOT))
                    for identifier in sorted(identifiers)
                ],
            }

    if unresolved:
        raise RuntimeError(
            "Packages need verified metadata corrections in third_party/license_overrides.yaml: "
            + ", ".join(sorted(unresolved, key=str.lower))
        )
    # Dev-only corrections are consumed by the PR report, not shipped disclosures.
    locked_names = {
        p["name"] for p in tomllib.loads((PROJECT_ROOT / "uv.lock").read_text())["package"]
    }
    unused_overrides = sorted(set(overrides) - used_overrides - locked_names)
    if unused_overrides:
        raise RuntimeError("Unused license overrides: " + ", ".join(unused_overrides))
    return [records[key] for key in sorted(records)]


def _supplements(records: list[LicenseRecord]) -> dict[str, list[Artifact]]:
    exceptions = yaml.safe_load(EXCEPTIONS_PATH.read_text(encoding="utf-8")) or {}
    versions = {record["name"]: record["version"] for record in records}
    supplements = {}
    for name, entry in exceptions.items():
        if name not in versions or entry["version"] != versions[name]:
            raise RuntimeError(f"Review stale license exception: {name}")
        if not entry.get("documents"):
            raise RuntimeError(f"Empty license exception: {name}")
        supplements[name] = entry["documents"]
    return supplements


def _collect_texts(records: list[LicenseRecord]) -> str:
    lock = tomllib.loads((PROJECT_ROOT / "uv.lock").read_text(encoding="utf-8"))
    packages = {(_canonicalize(p["name"]), p["version"]): p for p in lock["package"]}
    selected = [packages[(_canonicalize(r["name"]), r["version"])] for r in records]
    supplements = _supplements(records)
    cache = PROJECT_ROOT / "tmp" / "license-archives"
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda package: collect(package, cache, supplements.get(package["name"])), selected
            )
        )
    texts: list[tuple[str, str, list[Document]]] = [
        (record["name"], record["version"], documents)
        for record, documents in zip(records, results, strict=True)
    ]
    return render_texts(texts)


def _render_summary(records: list[LicenseRecord]) -> str:
    return "\n".join(json.dumps(record) for record in records) + "\n"


def _render_notices(records: list[LicenseRecord]) -> str:
    lines = [
        "<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->",
        "<!-- SPDX-License-Identifier: Apache-2.0 -->",
        "",
        "# Third-Party Software Licenses",
        "",
        "This repository uses the dependencies listed below. Each package link identifies the exact",
        "version resolved in `uv.lock`. Complete collected license and attribution texts are",
        "preserved in [third_party/NOTICES.txt](third_party/NOTICES.txt), grouped by package and",
        "version, with the original archive URLs, checksums, and file paths. SPDX expressions",
        "are inventory metadata only; upstream texts retain their own terms and notices.",
        "Shared standard license texts are also included under `third_party/license_texts/`.",
        "",
        "The inventory covers runtime dependencies and all optional extras, including transitive",
        "and platform-specific dependencies. Tau Bench example data is MIT-licensed; its copyright",
        "and license are preserved in [third_party/tau-bench-LICENSE.txt](third_party/tau-bench-LICENSE.txt).",
        "",
        "Run `make update-licenses` after changing dependencies to collect texts automatically.",
        "`make check-licenses` verifies the generated inventory and collected texts.",
        "See DEVELOPMENT.md for collection scope and exception handling.",
        "",
        "| Package | SPDX license expression | License and attribution documents |",
        "| --- | --- | --- |",
    ]
    for record in records:
        package = f"[`{record['name']} {record['version']}`]({record['package_url']})"
        license_links = ", ".join(
            f"[{url.rsplit('/', 1)[-1]}]({url})" for url in record["license_documents"]
        )
        lines.append(f"| {package} | `{record['license']}` | {license_links} |")
    return "\n".join(lines) + "\n"


def _check_file(path: Path, expected: str) -> bool:
    if not path.exists() or path.read_bytes().decode("utf-8") != expected:
        print(f"{path.relative_to(PROJECT_ROOT)} is missing or out of date")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail instead of updating files")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / "tmp" / "license-check"
    requirements_path = output_dir / "requirements-main.txt"
    osv_path = output_dir / "osv-licenses.json"

    _export_requirements(requirements_path)
    scan_licenses(requirements_path, osv_path)
    records = _license_records(osv_path)
    summary = _render_summary(records)
    notices = _render_notices(records)
    texts = _collect_texts(records)

    if args.check:
        summary_valid = _check_file(SUMMARY_PATH, summary)
        notices_valid = _check_file(NOTICES_PATH, notices)
        texts_valid = _check_file(TEXTS_PATH, texts)
        valid = summary_valid and notices_valid and texts_valid
        if not valid:
            print("Run `make update-licenses` and commit the results.")
        return 0 if valid else 1

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(summary, encoding="utf-8")
    NOTICES_PATH.write_text(notices, encoding="utf-8")
    TEXTS_PATH.write_text(texts, encoding="utf-8")
    print(f"Wrote disclosures for {len(records)} dependencies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
