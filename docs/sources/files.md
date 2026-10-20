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
and recorded evaluation signals. Omit values that were not recorded; `null` means the source
explicitly recorded null. Output links back to this file and its physical line number;
whether the viewer jumps to that line depends on its support for URL fragments. Repeated problems need supporting evidence across multiple traces.

## ATIF

Each nonblank line must contain one complete ATIF trajectory. A path ending in `.json`, such
as Harbor's `agent/trajectory.json`, is read as one whole-file trajectory instead. The loader
accepts schema versions `ATIF-v1.0` through `ATIF-v1.8`. Keep tool calls, observations, and
nested subagent trajectories in the exported record.

A trajectory without `trajectory_id` or `session_id` gets the ID `atif-` followed by the first
16 hex digits of the SHA-256 of its record. `agent.tool_definitions` enable tool-schema checks,
and an observation result with `is_error: true` (on the result or in its `extra`) marks the
tool call as failed.

## A different trace system

Follow [Analyze traces from another system](custom.md) to use a coding agent to
convert an export or fetch and convert traces through your store's API.
