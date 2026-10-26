<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Compass Signals

All five Compass Signals are enabled by default. Each runs when its prerequisites are available.

- [Ethos divergence](#ethos-divergence): supply your agent’s business rules.
- [User sentiment](#user-sentiment): configure Decisions API access for complaint detection.
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

This Compass Signal looks for recurring user complaints. Each recorded human message is
classified through the [OpenAI Decisions API](https://developers.openai.com/api/reference/resources/decisions/methods/create)
as `complaint` or `no_complaint`. Any complaint sends its trace to contextual review
by your main inference model. Messages without text are not sent for classification.

Configure a Decisions-capable model and credentials to enable screening:

```yaml
compass_signals:
  user_sentiment:
    model: gpt-6-luna
    api_base: https://api.openai.com/v1
    api_key_env: DECISIONS_API_KEY
    concurrency: 8
    timeout_seconds: 60
    max_retries: 2
```

Set `DECISIONS_API_KEY` in your [environment file](model-access.md#credentials).
For a compatible gateway, use its API base and exact model identifier. This client calls
`/decisions` directly, so do not add a LiteLLM provider prefix to the model identifier.
Chat-completions support alone does not imply Decisions support.

Without `api_key_env`, this signal is skipped. A configured but missing key is an error.
Requests run concurrently up to `concurrency`; transient failures are retried up to
`max_retries` times. Persistent errors and invalid responses fail the signal instead of
being counted as neutral feedback. Messages are sent in full, so provider input limits apply.

Screening preserves the categories used by the previous classifier. Explicit anger and
criticism of the agent count as complaints; ordinary requests and pasted errors alone do not.
The selected category determines routing. Provider probabilities are retained for inspection;
evaluate their calibration and classification quality on your own data before relying on them.

Migration: replace the previous `litellm`, `local_embeddings`, and `device` settings with
the configuration above. Local embedding inference and the `local-embedding` extra have
been removed. See [data access](model-access.md#where-data-goes).

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
