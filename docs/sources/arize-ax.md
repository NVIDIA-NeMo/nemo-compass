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

No Arize AX SDK extra is required. [Configure your inference model and key](../model-access.md#choose-a-model),
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
Selection reads until it finds a conversation beyond the limit, then queries every
selected session again using the original window and filter plus session equality.
This retrieves older matching records that initial selection may have missed. The
same trace's complete spans are fetched once per load. If the session query omits a
record already selected, Compass retains it and marks the conversation `merge_incomplete`.
The query window and filter still apply: this is not necessarily a session's full lifetime.

The shared conversation merger orders records by start time and ID, prefixes span IDs
with their original trace IDs, and puts all original span trees under a conversation
agent span. It joins recorded turns and removes repeated copied history. If any record
lacks a single orchestration root or a recoverable conversation input, the merged
conversation has no turns. Original record metadata stays in `attributes.records`.
Outputs retain AX trace and span IDs without generating UI permalinks.

## Deployment and data mapping

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

Requests use bearer authentication, do not follow redirects, and report HTTP failures
without exposing response bodies or credentials. There are no automatic retries.

The loader preserves span IDs, parents, timestamps, kinds, errors, inputs and outputs.
OpenInference JSON payloads are decoded only when their MIME type is `application/json`.
A missing output remains missing; a recorded JSON null remains a result. Tool spans
become matched calls/results, using the span ID as the call identity. A source error
status marks a failed result; a default success status does not prove a successful result.
LLM model names, recorded prompt/completion counts, invocation parameters, and tool
declarations populate their canonical fields. Recorded `llm.cost.total` becomes span cost.
Other attributes, events, annotations, and evaluation details remain under
`attributes.arize_ax` on each span. Recorded evaluations
and annotations also populate trace evaluation results for grader failure analysis.

Agent spans with recorded conversation inputs receive ordered turns using the same
input/LLM/tool/output projection as the [MLflow loader](mlflow.md#conversation-turns).
An orchestration root of kind `CHAIN` becomes an agent span, keeping `CHAIN` as its subtype.
Without a recorded input, turns remain absent. Span order follows start time, then ID.
Trace latency is the earliest start through latest end. AX timestamps on replayed or
synthetic data describe that recording, so they need not represent production latency.
