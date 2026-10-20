# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read native Langfuse trace-detail exports."""

from __future__ import annotations

import json
from pathlib import Path

from trace_ingest.loaders.langfuse.config import LangfuseTraceLoadError


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
