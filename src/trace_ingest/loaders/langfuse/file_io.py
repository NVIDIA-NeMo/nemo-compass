# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read native Langfuse trace-detail exports."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from trace_ingest.loaders.helper.merge import Record, assemble
from trace_ingest.loaders.langfuse.config import LangfuseTraceLoadError
from trace_ingest.loaders.langfuse.normalization import _ORCHESTRATION
from trace_ingest.models import Trace


def _parse_trace_export(contents: bytes, path: Path) -> list[dict[str, object]]:
    try:
        text = contents.decode("utf-8")
        payloads = (
            [json.loads(line) for line in text.splitlines() if line.strip()]
            if path.suffix == ".jsonl"
            else [json.loads(text)]
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LangfuseTraceLoadError(f"Invalid Langfuse JSON export {path}: {error}") from error
    if not payloads:
        raise LangfuseTraceLoadError(f"Langfuse export {path} contains no traces")
    records: list[dict[str, object]] = []
    for index, payload in enumerate(payloads, start=1):
        if isinstance(payload, dict) and "body" in payload:
            if payload.get("status") != 200:
                raise LangfuseTraceLoadError(
                    f"Langfuse export {path}, record {index}: unsuccessful CLI response"
                )
            payload = payload["body"]
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("observations"), list)
            or not all(isinstance(obs, dict) for obs in payload["observations"])
        ):
            raise LangfuseTraceLoadError(
                f"Langfuse export {path}, record {index}: expected a complete v3 trace-detail "
                "response with embedded observations; trace-list, UI table, and CSV exports "
                "are not supported. Export with langfuse --api-version 3 api traces get <id> --json"
            )
        records.append(payload)
    return records


def _newest_conversations(
    records: Sequence[tuple[datetime, Record]], max_traces: int
) -> tuple[list[Trace], int]:
    """The newest ``max_traces`` merged traces, oldest first, and how many were merged.

    Each record comes with the time live selection ranks it by; as there, a
    conversation is as new as its newest record.
    """
    try:
        traces = assemble([record for _, record in records], orchestration_subtypes=_ORCHESTRATION)
    except ValueError as error:
        raise LangfuseTraceLoadError(str(error)) from error
    newest: dict[str | int, datetime] = {}
    for index, (time, record) in enumerate(records):
        key = record.trace.session_id or index
        newest[key] = max(time, newest.get(key, time))
    # assemble returns one trace per key, in order of each key's first record.
    keyed = sorted(
        zip(newest.values(), traces, strict=True), key=lambda item: (item[0], item[1].id)
    )
    return [trace for _, trace in keyed[-max_traces:]], len(traces)
