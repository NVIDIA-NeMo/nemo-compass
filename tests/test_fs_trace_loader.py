# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from trace_ingest.loaders.fs import FSDataLoader, FSDataLoadError


def test_schema_v2_case_key_is_rejected_instead_of_dropped(tmp_path):
    # Schema version 2 writers omit schema_version, so only the old case key reveals them.
    rows = [
        {"id": "attempt-1", "root_spans": [], "aggregate": {}, "case_id": "task-1"},
        {
            "id": "attempt-2",
            "root_spans": [],
            "aggregate": {},
            "attributes": {"logical_case_id": "task-1"},
        },
    ]
    path = tmp_path / "traces.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(
        FSDataLoadError, match=r"traces\.jsonl:2: attributes\.logical_case_id .* case_id"
    ):
        FSDataLoader(path).load()
