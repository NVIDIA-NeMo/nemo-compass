---
name: trace-loader
description: Build or update a deterministic Compass TraceLoader for a concrete external trace source. Use when onboarding a trace provider, dataset, SDK, or export format that is not already canonical Trace JSONL.
---
<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Build an Compass trace loader

A loader is the source boundary. It owns every source-specific decision needed
to turn one concrete provider format into the canonical `Trace` representation.
Compass Signals receive only a `TraceSnapshot`; they must not contain provider
field aliases, source-shape detection, or compatibility mappings.

## Start from the real source

Inspect representative source records, the source SDK or schema, and existing
fixtures before writing code. Determine how the source represents trace and
span identity, nesting, ordering, tool calls and results, timestamps, status,
metrics, and pagination. Do not guess fields or accept several speculative
shapes. Reject ambiguous or incomplete input with a source-specific error.

Read these contracts in `src/trace_ingest/`:

- `loaders/trace_loaders.py` defines `TraceLoader` and `TraceDescription`.
- `models.py` defines `Trace`, `Span`, and `TraceSnapshot`.
- `loaders/mlflow.py` is the reference architecture for
  a provider loader, including configuration, SDK/file ingestion,
  normalization, deterministic ordering, diagnostics, and source description.
- `loaders/fs.py` is only for input already serialized
  as canonical Trace JSONL. Do not add source normalization to it.

Treat the MLflow implementation as a structural example, not as a field-mapping
template. The mapping must come from the actual source being added.

## Implement the boundary

Within that package, add `loaders/<source>.py` and export its public types from
`loaders/__init__.py`.

Use typed configuration and a source-specific load error. `load()` should:

1. Resolve and fetch a bounded source selection.
2. Parse it using the source's authoritative SDK or documented schema.
3. Deterministically construct canonical `Trace` and recursive `Span` models.
4. Reconstruct parent/child relationships and stable ordering when the source
   stores spans flat.
5. Return a `TraceSnapshot`. Canonical model validation rejects duplicate span
   IDs, and the snapshot rejects duplicate trace IDs.

Construct the Pydantic models directly. Do not create a second canonical dict
schema, a permissive alias layer, or a generic translation abstraction. Preserve
raw source values when the canonical model can represent them. In particular,
preserve the canonical distinction between an omitted value and explicit JSON
null rather than manufacturing a result.

Record each matched tool result as a `ToolResult` in `ToolCall.results`, so
`len(results) == result_count`; set `results=[]` when the source recorded no
result, and leave it unset only when the source cannot say. Set `is_error` only
from an authoritative source flag or status; leave it `None` otherwise, and never
map a tracer's default success status to `False`. Attach declared tools as
`ToolDefinition`s (with the verbatim source item in `raw`) to the AGENT span
they belong to, or to the LLM span they were offered to when no AGENT span
encloses its TOOL spans. `trace_ingest.catalog` resolves them for Compass Signals.

`describe()` should report the loaded corpus counts and enough source identity
and selection configuration to reproduce the run. Provider-specific diagnostics
belong in a typed report or extended description, as in the MLflow loader.

Keep provider behavior inside the loader: pagination, SDK imports, export
parsing, provider status translation, source pointers, and malformed parent
handling must not leak into Compass Signals.

## Integrate explicitly

Add an explicit CLI/API source selection path. Do not infer the provider from
arbitrary JSON keys. If the input already conforms to `Trace`, use
`FSDataLoader`; a new loader would add no value.

## Verify the deterministic path

Add realistic source fixtures and focused tests for:

- Exact normalized `Trace` output, including recursive span ordering.
- Missing versus explicit-null inputs and outputs.
- Duplicate IDs, invalid parents, malformed source data, and empty results.
- Pagination, limits, and deterministic ordering when the source is remote.
- `describe()` and provider diagnostics after loading.
- A complete CLI run through the Compass Signals.

Run the relevant focused tests, then `uv run ruff check .` and
`uv run pytest -q`. Inspect the emitted intermediate evidence artifacts; loader
work is complete only when the source produces stable canonical traces and the
downstream signals contain no source-specific logic.
