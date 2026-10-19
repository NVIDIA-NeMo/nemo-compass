#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check release identity and print the GitHub tag."""

from __future__ import annotations

import argparse
from pathlib import Path

import tomllib
from packaging.version import Version


def check_release(root: Path) -> str:
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    version = project["version"]
    if project["name"] != "nemo-compass":
        raise ValueError("Rename the distribution to nemo-compass before preparing a release.")
    parsed_version = Version(version)
    return f"v{parsed_version}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        tag = check_release(Path.cwd())
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.error(str(error))
    print(f"tag={tag}")


if __name__ == "__main__":
    main()
