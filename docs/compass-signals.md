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
      batch_size: 100
      max_concurrency: 4
      max_retries: 3
      request_timeout: 60
      retry_timeout: 300
      max_retry_wait: 60
```

Set `EMBEDDING_API_KEY` in your [environment file](model-access.md#credentials).
Replace the model alias and endpoint with your provider’s values.

Remote embeddings start with one request. After a window of clean successes equal
to the current concurrency, the limit increases by one, up to `max_concurrency`
(1 to 32). Free slots take the next batch immediately. `batch_size` controls the
maximum messages per request; results retain input order. Set `max_concurrency: 1`
for sequential requests. Limits apply to one embedding run, not across processes.

A rate limit halves the active limit, with a minimum of one, and pauses new requests.
Already-running requests finish normally. Rate limits from the same generation of
in-flight requests cause only one reduction. Old successes and successful retries
do not drive recovery; fresh successful requests gradually increase the limit again.
Connection errors, timeouts, and HTTP 408, 500, 502, 503, or 504 also trigger backoff
and retries, but do not reduce concurrency. Successful batches are retained.

The YAML above shows the defaults. `max_retries` allows additional attempts per
batch (0 to 10). Compass owns retries and requests that LiteLLM disable its internal
retries, including a compatibility workaround for its OpenAI-compatible embedding
adapter. All provider requests still go through LiteLLM.

`request_timeout` is the per-attempt SDK timeout in seconds. `retry_timeout` is the
elapsed budget for scheduling retries, measured from the batch's first failure;
an attempt already in flight can finish after this budget. It is not a hard deadline
for the entire dataset. `max_retry_wait` caps each cooldown. Backoff uses exponential
delays with jitter and honors valid `Retry-After` seconds or HTTP dates. A server
wait that exceeds either budget fails rather than retrying early. Invalid headers
use normal backoff. Authentication errors, other permanent failures, invalid vectors,
and exhausted retries fail the run without returning partial results. Active
requests finish under their SDK timeouts before the error returns.

YAML values can be overridden individually, for example:

```bash
compass --config config.yaml \
  --compass-signals.user-sentiment.litellm.max-concurrency 8 \
  --compass-signals.user-sentiment.litellm.max-retries 2
```

Concurrency increases are logged at INFO; retries and reductions at WARNING.
Logs contain batch numbers and error types, without user text or keys. This adapts
to observed throttling; it does not enforce a provider's token/request quotas or
guarantee optimal throughput. Tune batch size and the concurrency ceiling for your
endpoint. The model alias must still refer to the required Qwen embedding model.

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
