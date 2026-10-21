# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve a query-only client for either Langfuse API generation."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from trace_ingest.loaders.langfuse.config import LangfuseTraceLoadError

if TYPE_CHECKING:
    from langfuse import Langfuse


def _new_langfuse_client(base_url: str) -> Langfuse:
    try:
        from langfuse import Langfuse
    except ImportError as error:
        raise ModuleNotFoundError(
            "Langfuse support is not installed; install the project with the 'langfuse' extra"
        ) from error
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        raise LangfuseTraceLoadError(
            "Langfuse loading requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY"
        )
    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=base_url,
        tracing_enabled=False,
    )


def _resolved_base_url(configured: str | None) -> str:
    base_url = configured or os.environ.get("LANGFUSE_BASE_URL")
    if not base_url:
        raise LangfuseTraceLoadError(
            "Langfuse loading requires trace.langfuse.base_url or LANGFUSE_BASE_URL"
        )
    return base_url
