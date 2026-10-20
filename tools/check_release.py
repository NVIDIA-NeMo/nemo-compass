#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check release identity and print the GitHub tag and prerelease flag."""

from __future__ import annotations

import argparse
from pathlib import Path

import tomllib
from packaging.version import Version

TRACE_INGEST = Path("packages/trace-ingest")


def check_release(root: Path) -> tuple[str, bool]:
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    version = project["version"]
    if project["name"] != "nemo-compass":
        raise ValueError("Rename the distribution to nemo-compass before preparing a release.")
    ingest = tomllib.loads((root / TRACE_INGEST / "pyproject.toml").read_text())["project"]
    if ingest["name"] != "nemo-compass-trace-ingest":
        raise ValueError(f"Rename {TRACE_INGEST} to nemo-compass-trace-ingest.")
    if ingest["version"] != version:
        raise ValueError("nemo-compass and nemo-compass-trace-ingest versions must match.")
    if f"nemo-compass-trace-ingest=={version}" not in project["dependencies"]:
        raise ValueError(f"nemo-compass must depend on nemo-compass-trace-ingest=={version}.")
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
