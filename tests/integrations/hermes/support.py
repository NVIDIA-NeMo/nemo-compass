# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the Hermes plugin tests: load the plugin the way Hermes does, a fake
plugin context, and a small ATIF trajectory."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[3] / "integrations" / "hermes"


def _load_plugin():
    spec = importlib.util.spec_from_file_location(
        "nvidia_compass", PLUGIN_DIR / "__init__.py", submodule_search_locations=[str(PLUGIN_DIR)]
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["nvidia_compass"] = module
    spec.loader.exec_module(module)
    return module


plugin = _load_plugin()


class FakeCtx:
    """The PluginContext methods this plugin calls."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.tools: dict[str, dict] = {}
        self.cli: dict[str, dict] = {}
        self.hooks: list = []

    # Same rule as Hermes's hermes_cli/plugins_state.py: these setting names are reserved.
    RESERVED_ROOTS = {"model", "plugins", "security", "settings"}

    def get_config(self, key, default=None):
        if key.split(".")[0].lower() in self.RESERVED_ROOTS:
            raise ValueError(f"reserved plugin setting name: {key}")
        return self.config.get(key, default)

    def register_tool(self, name, toolset, schema, handler, **kw):
        self.tools[name] = {"toolset": toolset, "schema": schema, "handler": handler}

    def register_cli_command(self, name, help, setup_fn, handler_fn, **kw):
        self.cli[name] = {"setup": setup_fn, "handler": handler_fn}

    def register_hook(self, *a, **kw):
        self.hooks.append(a)


def atif(session: str, n_tools: int = 1) -> dict:
    """A small but complete ATIF-v1.6 trajectory, as NeMo Relay writes it."""
    steps = [
        {
            "step_id": 1,
            "source": "user",
            "message": "Find flights to DC",
            "timestamp": "2026-11-14T10:00:00Z",
        }
    ]
    for i in range(n_tools):
        steps.append(
            {
                "step_id": len(steps) + 1,
                "source": "agent",
                "message": "",
                "model_name": "nemotron",
                "tool_calls": [
                    {
                        "tool_call_id": f"c{i}",
                        "function_name": "web_search",
                        "arguments": {"q": "flights DC"},
                    }
                ],
                "observation": {
                    "results": [
                        {"source_call_id": f"c{i}", "content": "HTTP 429", "is_error": True}
                    ]
                },
                "metrics": {"prompt_tokens": 900, "completion_tokens": 40},
            }
        )
    steps.append({"step_id": len(steps) + 1, "source": "agent", "message": "Sorry, search failed."})
    return {
        "schema_version": "ATIF-v1.6",
        "session_id": session,
        "trajectory_id": f"traj-{session}",
        "agent": {"name": "Hermes Agent", "version": "0.22.7"},
        "steps": steps,
    }
