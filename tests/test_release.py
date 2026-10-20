# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the release CLI contract; distribution contents are validated by CI."""

import subprocess
import sys
from pathlib import Path

import pytest

CHECK_RELEASE = Path(__file__).resolve().parents[1] / "tools/check_release.py"


def write_projects(root, name, version, ingest_name, ingest_version, pin):
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "{name}"\nversion = "{version}"\ndependencies = ["{pin}"]\n'
    )
    ingest = root / "packages/trace-ingest"
    ingest.mkdir(parents=True)
    (ingest / "pyproject.toml").write_text(
        f'[project]\nname = "{ingest_name}"\nversion = "{ingest_version}"\n'
    )


INGEST = "nemo-compass-trace-ingest"


@pytest.mark.parametrize(
    ("name", "version", "ingest_name", "ingest_version", "pin", "code", "expected"),
    [
        (
            "nemo-compass",
            "0.1.0rc1",
            INGEST,
            "0.1.0rc1",
            f"{INGEST}==0.1.0rc1",
            0,
            "tag=v0.1.0rc1\nprerelease=true\n",
        ),
        (
            "nemo-compass",
            "0.1.0",
            INGEST,
            "0.1.0",
            f"{INGEST}==0.1.0",
            0,
            "tag=v0.1.0\nprerelease=false\n",
        ),
        ("nemo-compass", "invalid", INGEST, "invalid", f"{INGEST}==invalid", 2, "Invalid version"),
        ("compass", "0.1.0", INGEST, "0.1.0", f"{INGEST}==0.1.0", 2, "Rename the distribution"),
        (
            "nemo-compass",
            "0.1.0",
            "trace-ingest",
            "0.1.0",
            f"{INGEST}==0.1.0",
            2,
            "to nemo-compass-trace-ingest",
        ),
        ("nemo-compass", "0.2.0", INGEST, "0.1.0", f"{INGEST}==0.2.0", 2, "versions must match"),
        (
            "nemo-compass",
            "0.2.0",
            INGEST,
            "0.2.0",
            f"{INGEST}>=0.2.0",
            2,
            "must depend on nemo-compass-trace-ingest==0.2.0",
        ),
    ],
)
def test_release_contract(
    tmp_path, name, version, ingest_name, ingest_version, pin, code, expected
):
    write_projects(tmp_path, name, version, ingest_name, ingest_version, pin)
    result = subprocess.run(
        [sys.executable, str(CHECK_RELEASE)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == code, result.stderr
    assert expected in (result.stderr if code else result.stdout)


def test_repository_release_identity():
    result = subprocess.run(
        [sys.executable, str(CHECK_RELEASE)],
        cwd=CHECK_RELEASE.parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
