# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve run settings: plugin settings first, then non-secret defaults from Hermes's main model."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import hermes_api

API_KEY_ENV = "COMPASS_API_KEY"

# Hermes provider id -> LiteLLM prefix that talks to it natively (no api_base needed).
_NATIVE_PREFIX = {"anthropic": "anthropic", "openrouter": "openrouter"}
# OpenAI-compatible providers whose endpoint Hermes knows without a base_url in config.
_DEFAULT_BASE = {"nvidia": "https://integrate.api.nvidia.com/v1"}


@dataclass(frozen=True)
class Settings:
    atif_dir: Path
    model: str
    api_base: str
    api_key: str
    max_sessions: int
    sample: str
    max_tokens: int
    ethos_path: Path | None
    timeout_s: int

    def problems(self) -> list[str]:
        out = []
        if not self.api_key:
            out.append(
                f"{API_KEY_ENV} is not set. Add it in Hermes settings for nvidia-compass, "
                f"or put {API_KEY_ENV}=... in your Hermes .env."
            )
        if not self.model:
            out.append(
                "No Compass model. Set the plugin's `compass_model` setting (a LiteLLM string)."
            )
        if self.ethos_path is not None and not self.ethos_path.is_file():
            out.append(f"ethos_path {self.ethos_path} does not exist.")
        return out


def model_from_hermes(cfg: dict[str, Any]) -> tuple[str, str]:
    """Map Hermes's `model:` section to a (LiteLLM model, api_base) pair. Never reads keys."""
    section = cfg.get("model") or {}
    if isinstance(section, str):
        section = {"default": section}
    name = str(section.get("default") or section.get("model") or "").strip()
    provider = str(section.get("provider") or "auto").strip().lower()
    base_url = str(section.get("base_url") or "").strip()
    if not name:
        return "", ""
    prefix = _NATIVE_PREFIX.get(provider)
    if prefix:
        return f"{prefix}/{name}", ""
    # Everything else Hermes talks to over an OpenAI-compatible endpoint.
    return f"openai/{name}", base_url or _DEFAULT_BASE.get(provider, "")


def load(get_config: Callable[..., Any]) -> Settings:
    def opt(key: str, default: str | int) -> str | int:
        value = get_config(key, default=default)
        return default if value in (None, "") else value

    derived_model, derived_base = model_from_hermes(hermes_api.hermes_config())
    model = str(opt("compass_model", "")).strip()
    api_base = str(opt("api_base", "")).strip()
    if not model:
        model, api_base = derived_model, api_base or derived_base
    atif = str(opt("atif_dir", "")).strip()
    ethos = str(opt("ethos_path", "")).strip()
    return Settings(
        atif_dir=Path(atif).expanduser()
        if atif
        else hermes_api.hermes_home() / "telemetry" / "nemo-relay" / "atif",
        model=model,
        api_base=api_base,
        api_key=hermes_api.secret(API_KEY_ENV),
        max_sessions=max(1, int(opt("max_sessions", 64))),
        sample="random" if str(opt("sample", "newest")).lower() == "random" else "newest",
        max_tokens=max(1024, int(opt("max_tokens", 32768))),
        ethos_path=Path(ethos).expanduser() if ethos else None,
        timeout_s=max(1, int(opt("timeout_minutes", 30))) * 60,
    )
