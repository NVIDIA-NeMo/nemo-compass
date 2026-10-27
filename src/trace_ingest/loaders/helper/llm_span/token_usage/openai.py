# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI Responses usage.

The Responses API counts cache reads (``cached_tokens``) and, where a gateway
reports them, cache writes (``cache_write_tokens``) inside ``input_tokens``, and
reasoning tokens inside ``output_tokens``. Each count is read on its own, so a
missing one leaves only itself unknown.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trace_ingest.loaders.helper.llm_span.token_usage._core import token_counts as _token_counts
from trace_ingest.models import TokenCounts


def token_counts(usage: Mapping[str, Any]) -> TokenCounts | None:
    details = usage.get("input_tokens_details")
    details = details if isinstance(details, Mapping) else {}
    return _token_counts(
        input_total=usage.get("input_tokens"),
        cached_input_tokens=details.get("cached_tokens"),
        output_tokens=usage.get("output_tokens"),
        cache_write_tokens=details.get("cache_write_tokens"),
    )


__all__ = ["token_counts"]
