# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures that point the plugin's Hermes surfaces at a temporary home."""

from __future__ import annotations

import json
import sys

import pytest

from tests.integrations.hermes.support import atif


@pytest.fixture
def hermes(tmp_path, monkeypatch):
    """Point every Hermes surface at a temporary home."""
    home = tmp_path / "hermes"
    home.mkdir()
    api = sys.modules["nvidia_compass.hermes_api"]
    secrets = {"COMPASS_API_KEY": "test-key"}
    cfg = {
        "model": {
            "default": "nvidia/nemotron-3.5-super",
            "provider": "custom",
            "base_url": "http://127.0.0.1:8000/v1",
        }
    }
    monkeypatch.setattr(api, "hermes_home", lambda: home)
    monkeypatch.setattr(api, "data_dir", lambda: home / "plugin-data" / "nvidia-compass")
    monkeypatch.setattr(api, "secret", lambda name: secrets.get(name, ""))
    monkeypatch.setattr(api, "hermes_config", lambda: cfg)
    return {"home": home, "secrets": secrets, "config": cfg}


@pytest.fixture
def relay_dir(hermes):
    d = hermes["home"] / "telemetry" / "nemo-relay" / "atif"
    d.mkdir(parents=True)
    for i in range(3):
        (d / f"trajectory-s{i}.json").write_text(json.dumps(atif(f"s{i}"), indent=2))
    return d
