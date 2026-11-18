# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The only Hermes surfaces this plugin touches, all from the public plugin API list.

Kept in one place so a reviewer can audit them and tests can replace them.
Hermes provides these modules at runtime; it is not a Compass dependency, hence the `ty: ignore`s.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

PLUGIN_ID = "nvidia-compass"


def hermes_home() -> Path:
    from hermes_constants import get_hermes_home  # ty: ignore[unresolved-import]

    return Path(get_hermes_home())


def data_dir() -> Path:
    from plugins.plugin_storage import plugin_data_dir  # ty: ignore[unresolved-import]

    return Path(plugin_data_dir(PLUGIN_ID))


def secret(name: str) -> str:
    from agent.secret_scope import get_secret  # ty: ignore[unresolved-import]

    return (get_secret(name) or "").strip()


def hermes_config() -> dict[str, Any]:
    """Read-only view of Hermes's config.yaml (used only for non-secret model defaults)."""
    try:
        from hermes_cli.config import load_config_readonly  # ty: ignore[unresolved-import]

        return load_config_readonly() or {}
    except Exception:  # noqa: BLE001 - defaults are a convenience, never a failure
        return {}
