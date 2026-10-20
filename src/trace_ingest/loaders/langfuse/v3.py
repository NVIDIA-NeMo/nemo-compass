# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load complete traces through the Langfuse v3 API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urljoin

from trace_ingest.loaders.langfuse.client import _new_langfuse_client, _resolved_base_url
from trace_ingest.loaders.langfuse.config import (
    LangfuseTraceConfig,
    LangfuseTraceDescription,
    LangfuseTraceLoadError,
    _describe_live_source,
)
from trace_ingest.loaders.langfuse.filters import _bounded_filter
from trace_ingest.loaders.langfuse.normalization import (
    _normalize_trace,
    _trace_id,
    _trace_timestamp,
)
from trace_ingest.models import Trace, TraceSnapshot
from trace_ingest.source_links import http_source_url

if TYPE_CHECKING:
    from langfuse import Langfuse
    from langfuse.api.commons.types.trace_with_details import TraceWithDetails
_API_PAGE_SIZE = 100
_LIST_FIELDS = "core"


@dataclass
class LangfuseV3TraceLoader:
    """Load a bounded selection of complete traces from one Langfuse v3 project.

    Langfuse API credentials identify the project. A client can be injected for
    tests; otherwise the optional SDK is imported only when this loader runs.
    """

    config: LangfuseTraceConfig
    client: Langfuse | None = field(default=None, repr=False)
    _snapshot: TraceSnapshot = field(
        default_factory=lambda: TraceSnapshot([]), init=False, repr=False
    )
    _base_url: str | None = field(default=None, init=False, repr=False)

    def load(self) -> TraceSnapshot:
        """Select trace summaries, fetch their complete details, and normalize them."""

        base_url = _resolved_base_url(self.config.base_url)
        client = self.client or _new_langfuse_client(base_url)
        selected = self._select_traces(client)

        traces: list[Trace] = []
        # Query-summary timestamps determine ordering, even if detail timestamps differ.
        for summary in sorted(
            selected, key=lambda trace: (_trace_timestamp(trace), _trace_id(trace))
        ):
            trace_id = _trace_id(summary)
            provider_trace = client.api.trace.get(trace_id)
            trace = _normalize_trace(
                provider_trace,
                source_pointer={
                    "provider": "langfuse",
                    "base_url": base_url,
                    "trace_id": trace_id,
                },
            )
            if provider_trace.html_path:
                trace.source_url = http_source_url(
                    urljoin(base_url.rstrip("/") + "/", provider_trace.html_path)
                )
            traces.append(trace)

        snapshot = TraceSnapshot(traces)
        self._base_url = base_url
        self._snapshot = snapshot
        return snapshot

    def _select_traces(self, client: Langfuse) -> list[TraceWithDetails]:
        selected: dict[str, TraceWithDetails] = {}
        page = 1
        page_size = min(self.config.max_traces, _API_PAGE_SIZE)
        filter_string = _bounded_filter(
            self.config.filter_string,
            from_timestamp=self.config.from_timestamp,
            to_timestamp=self.config.to_timestamp,
        )

        while len(selected) < self.config.max_traces:
            response = client.api.trace.list(
                page=page,
                limit=page_size,
                from_timestamp=self.config.from_timestamp,
                to_timestamp=self.config.to_timestamp,
                order_by="timestamp.desc",
                fields=_LIST_FIELDS,
                filter=filter_string,
            )
            for trace in response.data:
                selected.setdefault(_trace_id(trace), trace)
                if len(selected) == self.config.max_traces:
                    break

            if page >= response.meta.total_pages or len(selected) == self.config.max_traces:
                break
            if not response.data:
                raise LangfuseTraceLoadError(
                    "Langfuse returned an empty trace page before the final page"
                )
            page += 1

        return list(selected.values())

    def describe(self) -> LangfuseTraceDescription:
        """Describe the configured source and most recently loaded corpus."""

        base_url = self._base_url or _resolved_base_url(self.config.base_url)
        return _describe_live_source(self.config, base_url, "v3", self._snapshot)
