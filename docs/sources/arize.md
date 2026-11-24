<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Analyze Arize AX traces

Connect Compass to an Arize AX tracing project. This integration uses AX's public
v2 REST API. Phoenix has a separate API and is not supported by this loader.

## Connect and run

Install the CLI with [uv](../../README.md#start-here):

```bash
uv tool install nemo-compass
```

No Arize SDK extra is required. [Configure your inference model and key](../model-access.md#choose-a-model),
then add your AX API key to `.env`:

```dotenv
ARIZE_AX_API_KEY=your-arize-key
```

Use the project's ID from its AX browser URL (`/models/<project-id>`),
or list your projects with the [Arize API](https://arize-client-python.readthedocs.io/en/latest/projects.html).
Save this as `config.yaml`, replacing the project ID and dates:

```yaml
max_tokens: 16384
trace:
  max_traces: 10
  arize:
    project_id: your-project-id
    from_timestamp: 2026-11-01T00:00:00Z
    to_timestamp: 2026-11-02T00:00:00Z
```

```bash
compass --config config.yaml
```

Open `insights.yml` if the run produced insights. [Read your results](../results.md)
for help with findings or skipped Compass Signals.

## Selection and completeness

Both timestamps are required, must include a timezone, and define an inclusive start
and exclusive end. The loader selects newest traces first. `trace.max_traces` defaults
to 100. An optional `filter` under `trace.arize` uses AX's filter syntax, for example
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

Each AX trace is a separate Compass record, including all its root spans. Session IDs
remain in the source attributes; this loader does not join traces into sessions.
Outputs retain AX trace and span IDs without generating UI permalinks.

## Deployment and data mapping

The API origin defaults to `https://api.arize.com`. Set `ARIZE_AX_API_URL` in `.env`
for a self-hosted AX deployment, or set `api_url` under `trace.arize` to override it.
Supply the API origin without `/v2`. Credentials stay in the environment, never YAML.
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
`attributes.arize` on each span. Recorded evaluations
and annotations also populate trace evaluation results for grader failure analysis.

Agent spans with recorded conversation inputs receive ordered turns using the same
input/LLM/tool/output projection as the [MLflow loader](mlflow.md#conversation-turns).
An orchestration root of kind `CHAIN` becomes an agent span, keeping `CHAIN` as its subtype.
Without a recorded input, turns remain absent. Span order follows start time, then ID.
Trace latency is the earliest start through latest end. AX timestamps on replayed or
synthetic data describe that recording, so they need not represent production latency.
