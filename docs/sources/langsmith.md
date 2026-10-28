<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Analyze LangSmith traces

Choose a project whose agent behavior you want to understand. Recorded feedback on
root and child runs is included automatically in grader failure analysis.

The live adapter supports the v1 query API, tested with self-hosted LangSmith 0.15.
The SmithDB-backed v2 query API is not yet supported.

## Connect and run

With [uv installed](../../README.md#start-here), install the CLI with LangSmith support:

```bash
uv tool install 'nemo-compass[langsmith]'
```

[Configure your inference model and key](../model-access.md#choose-a-model).
Add your LangSmith key to `.env`:

```dotenv
LANGSMITH_API_KEY=your-langsmith-key
```

Save this as `config.yaml`, replacing the project name:

```yaml
max_tokens: 16384
trace:
  max_traces: 100
  langsmith:
    project: my-agent
```

```bash
compass --config config.yaml
```

Open `insights.yml` if the run produced insights. [Read your results](../results.md)
for help with findings or skipped Compass Signals.

When an insight identifies specific supporting spans, links open their individual
LangSmith runs. If a run URL is unavailable, the report retains the trace link
and span ID.

## Narrow the input

Add fields under `trace.langsmith` as needed:

| Field | Example | What it selects |
| --- | --- | --- |
| `start_time` | `2026-09-01T00:00:00Z` | Traces starting at or after a timezone-aware timestamp. |
| `filter` | `'eq(status, "error")'` | Matching root runs. |
| `tree_filter` | `'eq(run_type, "tool")'` | Complete traces with a matching run anywhere in the tree. |
| `api_url` | `https://langsmith.example.com` | A regional or self-hosted API endpoint. |

`api_url` overrides `LANGSMITH_ENDPOINT`. Set `LANGSMITH_WORKSPACE_ID` in `.env`
if your key needs an explicit workspace.

Compass loads one trace per conversation: root runs whose metadata shares a
`thread_id` (else `session_id`, else `conversation_id`) become one trace, in start
order, with spans named `<trace id>/<run id>` and turns joined across the thread.
A root run without that metadata stays its own trace; a single `chain` root is its
agent run, with turns read from its inputs, LLM outputs, and outputs. A thread has turns
only when every one of its root runs is an `agent` or `chain`.
The default limit is 100 conversations. When roots remain past the limit, each loaded
thread is marked `merge_incomplete`, since unread roots may belong to it.

## Use an export

With the separate LangSmith CLI installed and authenticated, export complete traces:

```bash
langsmith trace export exports/langsmith --project my-agent --limit 100 --full
```

Replace the `trace` section with:

```yaml
trace:
  max_traces: 100
  langsmith_trace_export_file:
    path: exports/langsmith
```

Rerun the same analysis command. Local exports need inference credentials only.
Use `--full` to include child runs. The loader accepts one `.jsonl` file or a directory
of immediate `.jsonl` files, validates and merges the complete export, then keeps the
`max_traces` conversations with the newest root runs, the ones a live load selects,
so a thread is never split.
Both paths use Python SDK `>=0.12,<0.13`, installed by the extra.
