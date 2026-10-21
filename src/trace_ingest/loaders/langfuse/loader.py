# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detect the Langfuse server version and delegate to its loader."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from trace_ingest.loaders.langfuse.client import _new_langfuse_client, _resolved_base_url
from trace_ingest.loaders.langfuse.config import (
    LangfuseTraceConfig,
    LangfuseTraceDescription,
    LangfuseTraceLoadError,
    _describe_live_source,
)
from trace_ingest.loaders.langfuse.v3 import LangfuseV3TraceLoader
from trace_ingest.loaders.langfuse.v4 import LangfuseV4TraceLoader
from trace_ingest.models import (
    TraceSnapshot,
)

if TYPE_CHECKING:
    from langfuse import Langfuse


@dataclass
class LangfuseTraceLoader:
    """Detect the server version at load time and delegate to its implementation.

    Construction is network-free. The optional SDK is imported only when loading,
    unless a client has been injected.
    """

    config: LangfuseTraceConfig
    client: Langfuse | None = field(default=None, repr=False)
    _loader: LangfuseV3TraceLoader | LangfuseV4TraceLoader | None = field(
        default=None, init=False, repr=False
    )

    def load(self) -> TraceSnapshot:
        """Create the version-specific loader and load its trace snapshot."""
        base_url = _resolved_base_url(self.config.base_url)
        client = self.client or _new_langfuse_client(base_url)
        api_version = self._resolve_api_version(client)
        self._loader = (
            LangfuseV3TraceLoader(self.config, client)
            if api_version == "v3"
            else LangfuseV4TraceLoader(self.config, client, base_url)
        )
        return self._loader.load()

    def _resolve_api_version(self, client: Langfuse) -> str:
        version = client.api.health.health().version
        match = re.fullmatch(r"v?([0-9]+)\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.+-]+)?", version)
        if match is None or match[1] not in {"3", "4"}:
            raise LangfuseTraceLoadError(
                f"Unsupported Langfuse server version {version!r} from /api/public/health; "
                "expected a v3 or v4 release."
            )
        return f"v{match[1]}"

    def describe(self) -> LangfuseTraceDescription:
        """Report the selected implementation's corpus, or the unloaded source."""
        if self._loader is not None:
            return self._loader.describe()
        return _describe_live_source(
            self.config, _resolved_base_url(self.config.base_url), None, TraceSnapshot([])
        )
