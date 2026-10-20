#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check both distributions' identity and licenses, including wheels rebuilt from sdists."""

from __future__ import annotations

import email
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path

import tomllib
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_sdist_filename, parse_wheel_filename
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[1]


def check_metadata(contents: bytes, expected: dict[str, bytes], name: str, version: str) -> None:
    metadata = email.message_from_bytes(contents)
    assert canonicalize_name(metadata["Name"]) == canonicalize_name(name), "Package name differs"
    assert metadata["Version"] == str(Version(version)), "Package version differs"
    assert all(
        canonicalize_name(Requirement(req).name) != "trace-ingest"
        for req in metadata.get_all("Requires-Dist", [])
    ), "Compass must not depend on a separate trace-ingest distribution"
    assert metadata["License-Expression"] == "Apache-2.0"
    assert set(metadata.get_all("License-File", [])) == set(expected), "License-File list differs"


def check_wheel(
    path: Path, expected: dict[str, bytes], name: str, version: str, package: str
) -> None:
    assert parse_wheel_filename(path.name)[:2] == (canonicalize_name(name), Version(version)), (
        "Wheel filename differs from project identity"
    )
    with zipfile.ZipFile(path) as archive:
        (metadata_path,) = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        check_metadata(archive.read(metadata_path), expected, name, version)
        directory = metadata_path.removesuffix("METADATA") + "licenses/"
        for license_name, contents in expected.items():
            assert archive.read(directory + license_name) == contents, (
                f"{path.name}: {license_name} differs"
            )
        packages = {
            member.split("/", 1)[0]
            for member in archive.namelist()
            if not member.split("/", 1)[0].endswith(".dist-info")
        }
        assert packages == {package}, f"{path.name}: top-level packages {sorted(packages)}"


def check_distribution(root: Path, package: str) -> tuple[str, str]:
    """Check one project's wheel, sdist, and a wheel rebuilt from that sdist; return its identity."""
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    expected = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for pattern in project["license-files"]
        for path in root.glob(pattern)
        if path.is_file()
    }
    assert "LICENSE" in expected, f"{project['name']} must ship LICENSE"
    assert expected["LICENSE"] == (REPO_ROOT / "LICENSE").read_bytes(), (
        f"{project['name']}: LICENSE differs from the repository LICENSE"
    )
    name, version = project["name"], project["version"]
    stem = f"{canonicalize_name(name).replace('-', '_')}-{Version(version)}"
    (wheel,) = (REPO_ROOT / "dist").glob(f"{stem}-*.whl")
    sdist = REPO_ROOT / "dist" / f"{stem}.tar.gz"
    check_wheel(wheel, expected, name, version, package)
    assert parse_sdist_filename(sdist.name) == (canonicalize_name(name), Version(version)), (
        "Source filename differs from project identity"
    )
    with tempfile.TemporaryDirectory() as temporary:
        isolated = Path(temporary)
        with tarfile.open(sdist) as archive:
            for license_name, contents in expected.items():
                member = archive.getmember(f"{stem}/{license_name}")
                assert member.isfile(), f"{license_name} must contain bytes, not a symlink"
                source = archive.extractfile(member)
                assert source is not None and source.read() == contents, (
                    f"{stem}: {license_name} differs"
                )
            archive.extractall(isolated, filter="data")
        source_tree = isolated / stem
        check_metadata((source_tree / "PKG-INFO").read_bytes(), expected, name, version)
        subprocess.run(
            [
                "uv",
                "build",
                "--no-config",
                "--no-sources",
                "--wheel",
                "--out-dir",
                str(isolated),
            ],
            cwd=source_tree,
            check=True,
        )
        (rebuilt,) = isolated.glob("*.whl")
        check_wheel(rebuilt, expected, name, version, package)
    print(f"{stem}: metadata and licenses match in wheel, sdist, and rebuilt wheel")
    return name, version


def check_requires(wheel: Path, requirement: str) -> None:
    with zipfile.ZipFile(wheel) as archive:
        (metadata_path,) = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        metadata = email.message_from_bytes(archive.read(metadata_path))
    assert requirement in metadata.get_all("Requires-Dist", []), (
        f"{wheel.name} must require {requirement}"
    )


def main() -> None:
    distributions = [
        check_distribution(REPO_ROOT / "packages" / "trace-ingest", "trace_ingest"),
        check_distribution(REPO_ROOT, "compass"),
    ]
    (ingest_name, version), (_, compass_version) = distributions
    assert compass_version == version, "nemo-compass and nemo-compass-trace-ingest versions differ"
    (compass_wheel,) = (REPO_ROOT / "dist").glob(f"nemo_compass-{Version(version)}-*.whl")
    check_requires(compass_wheel, f"{ingest_name}=={version}")
    built = {path.name for path in (REPO_ROOT / "dist").iterdir() if path.suffix in {".whl", ".gz"}}
    assert len(built) == 2 * len(distributions), f"unexpected distributions: {sorted(built)}"


if __name__ == "__main__":
    main()
