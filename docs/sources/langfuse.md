<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Analyze Langfuse traces

Choose a time window containing the conversations you want to understand.
Recorded scores, including scores attached to observations, are included automatically.

## Connect and run

With [uv installed](../../README.md#start-here), install the CLI with Langfuse support:

```bash
uv tool install 'nemo-compass[langfuse]'
```

[Configure your inference model and key](../model-access.md#choose-a-model).
Add your deployment URL and project’s Langfuse keys to `.env`:

```dotenv
LANGFUSE_BASE_URL=https://langfuse.example.com
LANGFUSE_PUBLIC_KEY=your-public-key
LANGFUSE_SECRET_KEY=your-secret-key
```

Save this as `config.yaml`. Replace the URL and dates with your deployment and trace window:

```yaml
max_tokens: 16384
output_path: insights.yml
trace:
  max_traces: 100
  langfuse:
    base_url: https://langfuse.example.com
    from_timestamp: 2026-09-01T00:00:00Z
    to_timestamp: 2026-09-02T00:00:00Z
```

```bash
compass --config config.yaml
```

Open `insights.yml` if the run produced insights. [Read your results](../results.md)
for help with findings or skipped Compass Signals.

## Select the right traces

Your API keys select the project. Both timestamps are required and must include a timezone;
the start is inclusive and the end is exclusive. `base_url` overrides `LANGFUSE_BASE_URL`
in the environment.

Compass loads one trace per conversation: selected traces that share a Langfuse `sessionId`
become one trace, in start order, with spans named `<trace id>/<observation id>` and the
conversation's turns joined across them. Langfuse keeps times to the millisecond; steps
starting in the same one are ordered by end time, then a model call requesting tools
before those tools. A trace without a session stays on its own; when
its single root is an agent, chain or span observation, it owns the trace's turns. A merged
conversation has turns only when every one of its traces has such a root. A root
observation whose parent was never exported, such as a caller's span in distributed
tracing, is still the trace's root.
`max_traces` (default 100) counts conversations. Live loads select the newest ones; when
selection stops at the limit, each merged trace is marked incomplete (`merge_incomplete`)
because more of its session may exist. Exports keep the newest `max_traces`
conversations after merging every record, ranked as live loads rank them: by the
newest trace timestamp (v3) or observation start (v4) in the conversation.

To filter by environment, add this under `trace.langfuse`:

```yaml
filter: '[{"type":"string","column":"environment","operator":"=","value":"production"}]'
```

- **V3:** filters select traces; Compass fetches their full details.
- **V4:** filters select observations; Compass fetches all observations for their trace IDs
  **within the same window**. Observations outside it are excluded; costs and latency
  reflect only loaded observations. Widen the window when you need complete traces.

Use your version's [v3 trace](https://api.reference.langfuse.com/#tag/trace/GET/api/public/traces)
or [v4 observation](https://api.reference.langfuse.com/#tag/observationsv2/GET/api/public/v2/observations)
filter syntax. For example, `name` means trace name in v3 and observation name in v4;
use `traceName` for v4 trace names. Compass does not translate filters between versions.

V4 fetches observations for up to four trace IDs concurrently and batches score
requests across up to 100 trace IDs, including scores recorded after the window.
These limits are automatic. See [Langfuse compatibility](https://langfuse.com/docs/compatibility)
for v4 data-availability delays.

For tool-schema checks, record tool definitions in `input.tools` on generation observations.
Missing or conflicting catalogs limit those checks.

## Use an export

With the [Langfuse CLI](https://langfuse.com/docs/api-and-data-platform/features/cli)
installed, use the example for your server version below. `--all` defaults to
1,000 items and at most 100 requests: increase `--max-items` / `--limit` or
narrow the query when needed. V3 also requires `jq`; its extracted JSONL cannot
report whether the initial trace list was truncated.

`langfuse_trace_export_file` replaces the earlier `langfuse_export` key;
`api_version` is required.

### Langfuse v3

```bash
langfuse --env .env --api-version 3 api traces list \
  --from-timestamp 2026-09-01T00:00:00Z \
  --to-timestamp 2026-09-02T00:00:00Z \
  --all --json |
jq -r '.body.data[].id' |
while read -r trace_id; do
  langfuse --env .env --api-version 3 api traces get "$trace_id" --json |
    jq -c '.body'
done > langfuse-v3.jsonl
```

Configure Compass with:

```yaml
output_path: insights.yml
trace:
  langfuse_trace_export_file:
    api_version: v3
    path: langfuse-v3.jsonl
```

### Langfuse v4

Export observations and optionally project scores if you plan to use the
[`grader_failure_patterns`](../compass-signals.md#grader-failures) signal:

```bash
langfuse --env .env --api-version 4 api observations list \
  --from-start-time 2026-09-01T00:00:00Z \
  --to-start-time 2026-09-02T00:00:00Z \
  --fields core,basic,io,metadata,model,usage,prompt,trace_context \
  --all --json > observations.json

# Optional
langfuse --env .env --api-version 4 api scores list \
  --fields details,subject,annotation --all --json > scores.json
```

Keep exported responses unchanged. Compass rejects v4 exports marked truncated.
Observation filters and dates can exclude observations belonging to the same trace.

Configure Compass with:

```yaml
output_path: insights.yml
trace:
  langfuse_trace_export_file:
    api_version: v4
    path: observations.json
    scores_path: scores.json  # Optional
    base_url: https://langfuse.example.com
```

`base_url` is optional and enables Langfuse links in Compass' results.
To analyze supplied scores, add this top-level section (alongside any other
signals you want to run):

```yaml
compass_signals:
  grader_failure_patterns: {}
```

Run `compass --config config.yaml` as above.
