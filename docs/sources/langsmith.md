<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Analyze LangSmith traces

Choose a project whose agent behavior you want to understand. Recorded feedback on
root and child runs is included automatically in grader failure analysis. Each model
call keeps the tools it was offered and its token usage, including cache reads and
writes, so tool-argument and prompt-caching checks can run.

Compass reads with the v2 query API where the server offers it (LangSmith Cloud, and
self-hosted servers with SmithDB queries enabled), else with the v1 query API, which
LangSmith removes on Cloud on 2027-01-31 and on self-hosted in v0.19.
Tested with LangSmith Cloud 0.18 (v1 and v2 give the same traces) and self-hosted 0.15 (v1).

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

`api_url` overrides `LANGSMITH_ENDPOINT`. For a self-hosted server, set either to its
API base URL, `https://<your-langsmith-host>/api/v1`; Compass still uses the v2 query
API when the server offers it. Set `LANGSMITH_WORKSPACE_ID` in `.env`
if your key needs an explicit workspace.

On the v2 API, Compass reads root runs back from now, a window at a time, until it
has `max_traces` conversations, and goes back no further than a day before the
project was created. Set `start_time` to load runs imported with older timestamps.

Compass loads one trace per conversation: root runs whose metadata shares a
`thread_id` (else `session_id`, else `conversation_id`) become one trace, in start
order, with spans named `<trace id>/<run id>` and turns joined across the thread.
A root run without that metadata stays its own trace; a single `chain` root is its
agent run, with turns read from its inputs, LLM outputs, and outputs. A thread has turns
only when every one of its root runs is an `agent` or `chain`.
The default limit is 100 conversations. When roots remain past the limit, each selected
thread's roots are queried again with the same `filter`, `tree_filter` and `start_time`,
restricted to
`and(in(metadata_key, ["thread_id", "session_id", "conversation_id"]), eq(metadata_value, "<id>"))`,
so the thread loads whole. A selected root missing from that query is kept and the thread
is marked `merge_incomplete`.

## Use an export

With the separate LangSmith CLI installed and authenticated, export complete traces:

```bash
langsmith traces export exports/langsmith --project my-agent --limit 100 --full
```

Replace the `trace` section with:

```yaml
trace:
  max_traces: 100
  langsmith_trace_export_file:
    path: exports/langsmith
```

Rerun the same analysis command. Local exports need inference credentials only.
Use `--full` to include child runs. Exports keep token totals but not cache reads
and writes or the tools offered to each model call; load from the API for
tool-argument and prompt-caching checks. The loader accepts one `.jsonl` file or a directory
of immediate `.jsonl` files, validates and merges the complete export, then keeps the
`max_traces` conversations with the newest root runs, the ones a live load selects,
so a thread is never split.
The API path uses Python SDK `>=0.14.10,<0.15`, installed by the extra.
