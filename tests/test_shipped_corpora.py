# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from pathlib import Path

import pytest

from compass.traces import Trace
from trace_ingest.catalog import tool_catalog

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CORPUS = REPO_ROOT / "src" / "compass" / "data" / "sample_corpus.jsonl"


@pytest.mark.parametrize(
    "path",
    [REPO_ROOT / "examples" / "tau_bench_traces.jsonl", SAMPLE_CORPUS],
    ids=["tau-bench-example", "sample-corpus"],
)
def test_shipped_corpus_declares_tool_definitions(path: Path) -> None:
    traces = [Trace.model_validate_json(line) for line in path.read_text().splitlines() if line]

    assert traces
    for trace in traces:
        assert "tool_catalog" not in trace.attributes
        assert tool_catalog(trace) is not None, trace.id


def test_sample_corpus_matches_generator(tmp_path: Path) -> None:
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "generate_demo_corpus.py"), "--out", tmp_path],
        check=True,
        capture_output=True,
    )

    assert (tmp_path / "sample_corpus.jsonl").read_bytes() == SAMPLE_CORPUS.read_bytes()
