<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Compass Signals

All built-in Compass Signals are enabled by default. Each runs when its prerequisites are available.

- [Ethos divergence](#ethos-divergence): supply your agent’s business rules.
- [User sentiment](#user-sentiment): configure embeddings for complaint detection.
- [Grader failures](#grader-failures): include recorded scores or feedback.
- [Tool issues](#tool-issues): preserve tool calls and results.
- [Cost efficiency](#cost-efficiency): preserve model-call usage, timing, and tool definitions.

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
```

Set `EMBEDDING_API_KEY` in your [environment file](model-access.md#credentials).
Replace the model alias and endpoint with your provider’s values.

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
Include tool schemas when available so argument-validation checks can run. When a trace
merges a session's records, each call is checked against the tools its own record offered.
If the report says “No tool calls,” check that your selected traces and export preserve them.

## Cost efficiency

This Compass Signal finds recurring prompt-cache misses after tools are added, removed,
changed, or reordered. It uses your main inference model to extract missing evidence from
recorded spans, then applies deterministic checks to compare model calls within a session
and agent sequence.

Preserve per-call input-token counts, cache-read and cache-write counts, cache lifetimes,
start and end times, and the ordered tool definitions sent with each request. Include
session and agent IDs when available. Comparisons are skipped when the evidence cannot
rule out cache expiry or a shorter request; missing evidence limits coverage.

By default, findings must recur in at least three distinct traces to become candidate
problems for insight compilation. To change that threshold:

```yaml
compass_signals:
  cost_efficiency:
    minimum_independent_traces: 3
```

Reported token counts describe previously cached tokens associated with a miss.
They do not establish the savings a change would achieve.

## Disable a Compass Signal

Set a Compass Signal to `false` to disable it. Disabled Compass Signals produce no report entries or skip messages:

```yaml
compass_signals:
  ethos_divergence: false
  user_sentiment: false
```

The other keys are `grader_failure_patterns`, `tool_issues`, and `cost_efficiency`.
Omit a key, or use `true` or `{}`, for defaults. Keep at least one Compass Signal enabled.
When combining examples, put their settings under a single `compass_signals` key.
