# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from pathlib import Path

import pytest

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "trace.schema.json"


def test_committed_schema_matches_generator():
    generated = subprocess.run(
        [sys.executable, "-m", "trace_ingest.schema"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert SCHEMA_PATH.read_text(encoding="utf-8") == generated


@pytest.mark.parametrize("model", ["ToolCall", "ToolResult", "Trace"])
def test_nooa_inspects_cold_trace_models(model):
    # Each entry starts cold: no earlier test, construction, or schema generation
    # can hide a first-use failure in NOOA's inspection of these nested models.
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            f"from nooa.agentdoc import doc; from trace_ingest import {model}; "
            f"doc({model}, inline_depth=4)",
        ],
        check=True,
        timeout=30,
    )
