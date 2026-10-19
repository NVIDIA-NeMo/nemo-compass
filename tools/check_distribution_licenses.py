#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check license bytes and metadata in the Compass distribution and wheels rebuilt from sdists."""

from __future__ import annotations

import email
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]


def check_metadata(contents: bytes, expected: dict[str, bytes]) -> None:
    metadata = email.message_from_bytes(contents)
    assert metadata["License-Expression"] == "Apache-2.0"
    assert set(metadata.get_all("License-File", [])) == set(expected), "License-File list differs"


def check_wheel(path: Path, expected: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path) as archive:
        (metadata_path,) = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        check_metadata(archive.read(metadata_path), expected)
        directory = metadata_path.removesuffix("METADATA") + "licenses/"
        for name, contents in expected.items():
            assert archive.read(directory + name) == contents, f"{path.name}: {name} differs"


def main() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
    expected = {
        path.relative_to(REPO_ROOT).as_posix(): path.read_bytes()
        for pattern in project["license-files"]
        for path in REPO_ROOT.glob(pattern)
        if path.is_file()
    }
    assert {"LICENSE", "NOTICE"} <= expected.keys()
    stem = f"{project['name'].replace('-', '_')}-{project['version']}"
    (wheel,) = (REPO_ROOT / "dist").glob(f"{stem}-*.whl")
    check_wheel(wheel, expected)
    with tempfile.TemporaryDirectory() as temporary:
        isolated = Path(temporary)
        with tarfile.open(REPO_ROOT / "dist" / f"{stem}.tar.gz") as archive:
            for name, contents in expected.items():
                member = archive.getmember(f"{stem}/{name}")
                assert member.isfile(), f"{name} must contain bytes, not a symlink"
                source = archive.extractfile(member)
                assert source is not None and source.read() == contents, f"{stem}: {name} differs"
            archive.extractall(isolated, filter="data")
        source_tree = isolated / stem
        check_metadata((source_tree / "PKG-INFO").read_bytes(), expected)
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
        check_wheel(rebuilt, expected)
    print(f"{stem}: license files match in wheel, sdist, and rebuilt wheel")


if __name__ == "__main__":
    main()
