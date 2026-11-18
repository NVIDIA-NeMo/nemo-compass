# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""nvidia-compass: NeMo Compass trace analysis for Hermes Agent.

Registers two agent tools, the `hermes compass` CLI, and one `pre_tool_call` hook that sends
`compass_analyze` through Hermes's approval gate. Compass runs only when the user or the agent
asks; an agent-started run continues in the background, under a hard time limit.
"""

from __future__ import annotations

from . import cli, tools


def register(ctx) -> None:  # noqa: ANN001 - PluginContext
    ctx.register_tool(
        name="compass_analyze",
        toolset="compass",
        schema=tools.COMPASS_ANALYZE,
        handler=tools.make_analyze(ctx.get_config),
    )
    ctx.register_tool(
        name="compass_latest_report",
        toolset="compass",
        schema=tools.COMPASS_LATEST_REPORT,
        handler=tools.compass_latest_report,
    )
    ctx.register_hook("pre_tool_call", tools.make_approval_hook(ctx.get_config))
    ctx.register_cli_command(
        name="compass",
        help="NeMo Compass: find recurring problems in your Hermes sessions",
        setup_fn=cli.setup_argparse,
        handler_fn=cli.make_handler(ctx.get_config),
    )
