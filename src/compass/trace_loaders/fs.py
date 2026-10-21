# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility exports; implementation lives in trace_ingest.loaders.fs."""

from trace_ingest.loaders.fs import (
    FSDataLoader,
    FSDataLoadError,
)

__all__ = [
    "FSDataLoadError",
    "FSDataLoader",
]
