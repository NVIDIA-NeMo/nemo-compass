<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Analyze trace files

Use this guide for canonical JSONL or ATIF trajectories.
For native exports, follow the [LangSmith](langsmith.md#use-an-export),
[Langfuse](langfuse.md#use-an-export), [MLflow](mlflow.md#use-an-export), or
[NeMo Gym](gym.md) guide.

## Run a file

With [uv installed](../../README.md#start-here), install the CLI:

```bash
uv tool install nemo-compass
```

[Configure your inference model and key](../model-access.md#choose-a-model), then run:

```bash
compass --trace.filesystem.path traces.jsonl --max-tokens 16384
```

For ATIF, use `--trace.atif.path trajectories.jsonl` instead.
Both loaders read the whole file and do not support `trace.max_traces`.
Paths resolve from your working directory.

[Read your results](../results.md) when the run finishes.

## Canonical JSONL

Each line must contain one complete [Trace](../../src/trace_ingest/models.py).
This small example shows a recorded tool failure and evaluation score:

```json
{"id":"run-12","root_spans":[{"id":"tool-1","kind":"TOOL","tool_name":"lookup_order","input":{"order_id":"unknown"},"output":{"error":"Order not found"},"error":"Order not found"}],"aggregate":{},"evaluator_results":{"task_success":0}}
```

Preserve stable trace and span IDs, nested spans, tool inputs and results, user messages,
and recorded evaluation signals. Record a conversation as ordered `turns` on its AGENT
span, pointing at the LLM and TOOL spans each turn produced. Omit values that were not
recorded; `null` means the source explicitly recorded null. Output links back to this file
and its physical line number; whether the viewer jumps to that line depends on its support
for URL fragments. Repeated problems need supporting evidence across multiple traces.

One trace is one conversation. Set `case_id` when several traces are attempts at the same
task, so they count as one independent case. Set `session_id` with `session_basis` (the
source field it came from) when the trace carries a provider conversation key; without
`case_id`, it identifies the case.

Put `tool_definitions` on the owning AGENT span or the LLM span offered those tools.
Keep every entry in order, with its original value in `raw`. Add `name` and `input_schema`
when understood to enable tool-schema checks. Omit an unavailable list; `[]` means the
source explicitly recorded no tools.

Record a tool's results in `tool_call.results`, one per result, with `is_error` only when the
source recorded it. Files written before schema version 2 that carry `attributes.tool_catalog`
still work when no span declares definitions.

## ATIF

Each nonblank line must contain one complete ATIF trajectory with a `trajectory_id` or
`session_id`. The loader accepts Harbor's `Trajectory` fields for schema versions `ATIF-v1.0`
through `ATIF-v1.8`, validated strictly: values are not coerced, and unknown keys are rejected.

ATIF steps become ordered conversation turns linked to recorded model and tool calls. Include
subagent trajectories in the export; external trajectory files are not read. Missing
references and other known gaps are recorded in `Trace.losses`. The
[loader](../../src/trace_ingest/loaders/atif.py) documents the exact mapping, and
[`Loss`](../../src/trace_ingest/models.py) the loss fields.

The `session_id` becomes the trace's `session_id`, with `session_basis` `atif.session_id`.
Include `agent.tool_definitions` to enable tool-schema checks; an observation result with
`is_error: true` (on the result or in its `extra`) marks the tool call as failed.

## A different trace system

Follow [Analyze traces from another system](custom.md) to use a coding agent to
convert an export or fetch and convert traces through your store's API.
