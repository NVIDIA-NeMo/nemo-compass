# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from trace_ingest.loaders.helper.token_counts import token_counts


def counts(value) -> dict | None:
    return None if value is None else value.model_dump(exclude_none=True, exclude={"source"})


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        # Counts are copied, never derived from one another.
        (
            {"input_total": 100, "cached_input_tokens": 60, "output_tokens": 5},
            {"input_total": 100, "cached_input_tokens": 60, "output_tokens": 5},
        ),
        # A cache count larger than the total is kept as the source records it.
        (
            {"input_total": 10, "cached_input_tokens": 60},
            {"input_total": 10, "cached_input_tokens": 60},
        ),
        # An invalid count is unknown; the others stay.
        ({"input_tokens": -1, "output_tokens": 5}, {"output_tokens": 5}),
        ({"input_tokens": True, "output_tokens": None}, None),
        ({"cache_writes_by_ttl": {300: 0, 3600: -1}}, {"cache_writes_by_ttl": {300: 0}}),
    ],
)
def test_core_copies_valid_counts(kwargs, expected):
    assert counts(token_counts(**kwargs)) == expected
