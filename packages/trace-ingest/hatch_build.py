# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ship the repository's src/trace_ingest when building inside the Compass repository."""

from pathlib import Path
from typing import Any

# Hatch provides this dependency in the isolated build environment.
from hatchling.builders.hooks.plugin.interface import (  # ty: ignore[unresolved-import]
    BuildHookInterface,
)


class RepositorySource(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        package = Path(self.root)
        if (package / "src" / "trace_ingest").is_dir():
            return  # A standalone source distribution already contains the package.
        source = package.parents[1] / "src" / "trace_ingest"
        if not (source / "__init__.py").is_file():
            raise FileNotFoundError(f"trace_ingest sources not found at {source}")
        if version == "editable":
            return  # The editable path entry comes from dev-mode-dirs.
        build_data["force_include"][str(source)] = "trace_ingest"
