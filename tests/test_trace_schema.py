# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "trace.schema.json"


def test_committed_schema_matches_generator():
    generated = subprocess.run(
        [sys.executable, "-m", "trace_ingest.schema"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert SCHEMA_PATH.read_text(encoding="utf-8") == generated
