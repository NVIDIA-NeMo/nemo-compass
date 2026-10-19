#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check release identity and print the GitHub tag and prerelease flag."""

from __future__ import annotations

import argparse
from pathlib import Path

import tomllib
from packaging.version import Version


def check_release(root: Path) -> tuple[str, bool]:
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    version = project["version"]
    if project["name"] != "compass":
        raise ValueError("Rename the distribution to compass before preparing a release.")
    parsed_version = Version(version)
    return f"v{parsed_version}", parsed_version.is_prerelease


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        tag, prerelease = check_release(Path.cwd())
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.error(str(error))
    print(f"tag={tag}\nprerelease={str(prerelease).lower()}")


if __name__ == "__main__":
    main()
