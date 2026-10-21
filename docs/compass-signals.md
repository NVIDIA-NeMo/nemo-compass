<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Compass Signals

All five Compass Signals are enabled by default. Each runs when its prerequisites are available.

- [Ethos divergence](#ethos-divergence): supply your agent’s business rules.
- [User sentiment](#user-sentiment): configure embeddings for complaint detection.
- [Grader failures](#grader-failures): include recorded scores or feedback.
- [Tool issues](#tool-issues): preserve tool calls and results.
- [Anomalies and patterns](#anomalies-and-patterns): analyze behavior across traces.

## Ethos divergence

[ETHOS.md](https://github.com/NVIDIA/skills/tree/main/skills/ethos-orchestrate) describes
what an agent is for, what it must not do, and what success looks like. This Compass Signal
finds recorded behavior that diverges from those requirements.

Compass looks for `ETHOS.md` in the current working directory.
To use a different file, set `ethos_path`:

```yaml
ethos_path: path/to/ETHOS.md
```

## User sentiment

This Compass Signal looks for recurring user complaints. It needs recorded user messages and
an embedding backend serving **Qwen/Qwen3-Embedding-8B with 4,096-dimensional output**.
The bundled classifier was trained for this model; other embedding models are not supported.
Local embeddings are off by default. Without a remote endpoint or local embeddings enabled,
this Compass Signal is skipped.

For a remote endpoint, add:

```yaml
compass_signals:
  user_sentiment:
    litellm:
      model: openai/your-qwen3-embedding-8b-alias
      api_base: https://embeddings.example.com/v1
      api_key_env: EMBEDDING_API_KEY
      max_concurrency: 4
      max_retries: 3
      request_timeout: 60
      max_retry_wait: 60
```

Set `EMBEDDING_API_KEY` in your [environment file](model-access.md#credentials).
Replace the model alias and endpoint with your provider’s values.

Remote embeddings start with one request and add one concurrent request after each
successful round, up to `max_concurrency` (1 to 32). Each request contains up to
100 messages. Results retain the original message order. Set `max_concurrency: 1`
for sequential requests. The limit applies to one embedding run, not across
separate Compass processes.

Rate limits, connection errors, timeouts, and HTTP 408, 429, 500, 502, 503, or 504
responses halve concurrency, with a minimum of one. After the current round
finishes, all new requests pause for exponential backoff with jitter. Failed
batches retry first; successful batches are kept. Concurrency grows again after
a successful round.

The settings above show the defaults. `max_retries` is the number of additional
attempts per batch (0 to 10); SDK retries are disabled. `request_timeout` is the
per-request SDK timeout in seconds. `max_retry_wait` caps each cooldown in seconds.
Valid `Retry-After` seconds or HTTP dates are honored; a requested wait beyond
that cap fails the run instead of retrying early. Invalid headers use the normal
backoff. Authentication errors, other permanent failures, invalid vectors, and
exhausted retries fail the run without returning partial results. Already-running
requests finish under their SDK timeouts before the error returns.

Concurrency increases are logged at INFO; retries and reductions at WARNING.
These messages contain batch numbers and error types, without user text or keys.

To run the 8B embedding model locally, add `local-embedding` to the extras in your
install command. Keep your source extra too. For LangSmith:

```bash
uv tool install 'nemo-compass[langsmith,local-embedding]'
compass --config config.yaml --compass-signals.user-sentiment.local-embeddings
```

Or enable local embeddings in your configuration:

```yaml
compass_signals:
  user_sentiment:
    local_embeddings: true
```

Local inference downloads model weights and needs enough memory to run them.
Hardware is selected automatically by default. A configured `litellm` endpoint takes precedence.

Both local and remote embeddings still use your main inference model to investigate
complaints. See [data access](model-access.md#where-data-goes).

## Grader failures

Include Braintrust root-span scores, Langfuse scores, MLflow assessments, or LangSmith feedback.
For canonical JSONL traces, populate `evaluator_results`.
If you already record these, check that your selected traces and exports include them.

NeMo Compass uses those signals to investigate recurring failures. It does not run
your evaluation suite or create missing scores. Setup and export details live in your
[source guide](../README.md#start-here).

## Tool issues

Load traces with recorded tool spans, names, arguments, and results.
Include tool schemas when available so argument-validation checks can run.
If the report says “No tool calls,” check that your selected traces and export preserve them.

## Anomalies and patterns

This Compass Signal looks for unusual traces and recurring behavior across the loaded set.
A small or repetitive set may not support trajectory grouping; other analysis can still finish.
Load more varied traces from the behavior you want to investigate.

## Disable a Compass Signal

Set a Compass Signal to `false` to disable it. Disabled Compass Signals produce no report entries or skip messages:

```yaml
compass_signals:
  ethos_divergence: false
  user_sentiment: false
```

The other keys are `grader_failure_patterns`, `tool_issues`, and `anomaly_and_patterns`.
Omit a key, or use `true` or `{}`, for defaults. Keep at least one Compass Signal enabled.
When combining examples, put their settings under a single `compass_signals` key.
