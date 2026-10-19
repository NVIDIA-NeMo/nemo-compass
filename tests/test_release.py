# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the release CLI contract; distribution contents are validated by CI."""

import subprocess
import sys
from pathlib import Path

import pytest

CHECK_RELEASE = Path(__file__).resolve().parents[1] / "tools/check_release.py"


@pytest.mark.parametrize(
    ("name", "version", "code", "expected"),
    [
        ("nemo-compass", "0.1.0rc1", 0, "tag=v0.1.0rc1\n"),
        ("nemo-compass", "0.1.0", 0, "tag=v0.1.0\n"),
        ("nemo-compass", "invalid", 2, "Invalid version"),
        ("compass", "0.1.0", 2, "Rename the distribution"),
    ],
)
def test_release_contract(tmp_path, name, version, code, expected):
    (tmp_path / "pyproject.toml").write_text(f'[project]\nname = "{name}"\nversion = "{version}"\n')
    result = subprocess.run(
        [sys.executable, str(CHECK_RELEASE)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == code, result.stderr
    assert expected in (result.stderr if code else result.stdout)
