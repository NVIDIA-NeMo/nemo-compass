<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Analyze Arize AX traces

Connect Compass to an Arize AX tracing project. This integration uses AX's public
v2 REST API.

## Connect and run

Install the CLI with [uv](../../README.md#start-here):

```bash
uv tool install nemo-compass
```

[Configure your inference model and key](../model-access.md#choose-a-model),
then add your AX API key to `.env`:

```dotenv
ARIZE_AX_API_KEY=your-arize-ax-key
```

Use the project's ID from its AX browser URL (`/models/<project-id>`),
or list your projects with the [Arize AX API](https://arize-client-python.readthedocs.io/en/latest/projects.html).
Save this as `config.yaml`, replacing the project ID and dates:

```yaml
max_tokens: 16384
trace:
  max_traces: 10
  arize_ax:
    project_id: your-project-id
    from_timestamp: 2026-11-01T00:00:00Z
    to_timestamp: 2026-11-02T00:00:00Z
```

```bash
compass --config config.yaml
```

Open `insights.yml` if the run produced insights. [Read your results](../results.md)
for help with findings or skipped Compass Signals.

## Select the right conversations

Both timestamps are required, must include a timezone, and define an inclusive start
and exclusive end. The loader selects newest conversations first. `trace.max_traces`
defaults to 100 conversations. An optional `filter` under `trace.arize_ax` uses AX's filter syntax, for example
`"status_code = 'ERROR'"`. A matching child span selects its entire trace.

AX's [trace-list endpoint](https://arize-client-python.readthedocs.io/en/latest/traces.html)
can cap its embedded span lists, including without setting `spans_truncated`.
Compass uses it for discovery, then reads every page from the
[span endpoint](https://arize-client-python.readthedocs.io/en/latest/spans.html)
with `context.trace_id` equality. The span query drops the selection filter and expands
the time window to cover the trace's reported start and end, retaining children
outside the discovery window. Choose completed runs; the loader does not wait for
in-flight spans. Missing parents, duplicate spans, parent cycles, malformed responses,
and broken pagination fail the load instead of returning a partial tree.

AX [groups sessions using `session.id`](https://arize.com/docs/ax/observe/tracing/sessions-and-users)
on its spans. Compass reads the complete trace before choosing its conversation key.
When the spans contain exactly one distinct nonempty string session ID, that becomes
the trace's `session_id`, with `session_basis: arize_ax.session.id`. Spans without a session
ID do not prevent grouping. Conflicting IDs leave the trace separate and record a
`conflicting_session_ids` loss; a trace without a session ID also stays on its own.

`max_traces` counts conversations.
Compass includes older matching traces from each selected session within the query
window. The session ID and selection filter can match different spans in a trace.
The query window and filter still apply: this is not necessarily a session's full lifetime.

The shared conversation merger orders records by start time and ID, prefixes span IDs
with their original trace IDs, and puts all original span trees under a conversation
agent span. It joins recorded turns and removes repeated copied history. If any record
lacks a single orchestration root or a recoverable conversation input, the merged
conversation has no turns. Original record metadata stays in `attributes.records`.
Outputs retain AX trace and span IDs without generating UI permalinks.

## Connect to another deployment

The API origin defaults to `https://api.arize.com`. Set `ARIZE_AX_API_URL` in `.env`
for a self-hosted AX deployment, or set `api_url` under `trace.arize_ax` to override it.
Supply the API origin without `/v2`. Credentials stay in the environment, never YAML.
For example, add this to the source block for a private deployment:

```yaml
trace:
  max_traces: 10
  arize_ax:
    api_url: https://arize-ax-api.example.com
    project_id: your-project-id
    from_timestamp: 2026-11-01T00:00:00Z
    to_timestamp: 2026-11-02T00:00:00Z
```
