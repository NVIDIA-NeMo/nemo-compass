# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment configuration loaded before trace analysis and Insights Generation.

The CLI loads a local ``.env`` here before constructing provider clients, so
their SDKs can read their standard credential variables. NeMo Compass settings are
resolved in one order, most explicit first:

1. an explicit CLI flag
2. a variable already in the process environment
3. a variable in a ``.env`` file next to the corpus, the working directory, or
   the repo root

``.env`` is gitignored; ``.env.example`` is the committed template. Nothing here
imports litellm or a trace-provider SDK, and nothing fails when the file is absent.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

__all__ = [
    "ENV_API_BASE",
    "ENV_API_KEY",
    "ENV_MODEL",
    "find_dotenv",
    "load_dotenv",
    "resolve",
]

#: Repo-specific names, so a `.env` can carry NeMo Compass settings without
#: colliding with whatever provider variables litellm resolves on its own.
ENV_MODEL = "COMPASS_MODEL"
ENV_API_BASE = "COMPASS_API_BASE"
ENV_API_KEY = "COMPASS_API_KEY"

#: Fallbacks, so an environment already configured for litellm's OpenAI-
#: compatible path keeps working without a repo-specific `.env`.
_FALLBACKS = {
    ENV_API_BASE: ("OPENAI_API_BASE", "OPENAI_BASE_URL"),
    ENV_API_KEY: ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"),
}


def find_dotenv(start: Path | str | None = None) -> Path | None:
    """Locate a ``.env``: beside ``start``, then the cwd, then upward to a repo root."""

    candidates: list[Path] = []
    if start is not None:
        start = Path(start)
        candidates.append((start.parent if start.is_file() else start) / ".env")

    here = Path.cwd()
    candidates.append(here / ".env")
    for parent in here.parents:
        candidates.append(parent / ".env")
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            break

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_dotenv(path: Path | str | None = None, *, override: bool = False) -> dict[str, str]:
    """Load a ``.env`` into ``os.environ`` and return what it set.

    The real process environment wins by default: a variable already exported
    is not silently replaced by a stale file.
    """

    resolved = Path(path) if path is not None else find_dotenv()
    if resolved is None or not resolved.is_file():
        return {}

    applied: dict[str, str] = {}
    for key, value in dotenv_values(resolved).items():
        if not value:
            # An unfilled template line. Leaving it out means a real
            # environment variable of the same name still applies.
            continue
        if override or key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


def resolve(name: str, default: str | None = None) -> str | None:
    """Read one setting, falling back to the provider-standard variables."""

    value = os.environ.get(name)
    if value:
        return value
    for fallback in _FALLBACKS.get(name, ()):
        value = os.environ.get(fallback)
        if value:
            return value
    return default
