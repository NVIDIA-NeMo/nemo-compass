<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Configuration reference

For your first run, choose a [source guide](../README.md#start-here).
Use this page when you want to adjust an existing setup.

## Settings

Save reusable settings in YAML. Keep [credentials](model-access.md#credentials)
in the environment or `.env`.

| Setting | Purpose |
| --- | --- |
| `trace` | Exactly one source; see the [source guides](../README.md#start-here). |
| `trace.max_traces` | Limit complete traces from a provider or native export. Unsupported for canonical JSONL, ATIF, and Gym rollouts. |
| `ethos_path` | Optional path to [ETHOS.md](https://github.com/NVIDIA/skills/tree/main/skills/ethos-orchestrate), which defines the agent’s purpose and behavioral requirements. Defaults to `ETHOS.md` in the current working directory. Used by the ethos divergence signal and candidate validation. |
| `output_path` | YAML output file; defaults to `insights.yml`. Use `-` for stdout. |
| `model`, `api_base`, `max_tokens` | [Inference settings](model-access.md#choose-a-model). |
| `compass_signals` | [Compass Signals and their prerequisites](compass-signals.md). All five are enabled by default. |
| `code_base` | [Local agent source](#code-aware-review) to consult during candidate validation and confidence review. |
| `existing_insights` | Previous JSON or YAML insight collection to reconcile with this run. |

Relative paths resolve from the directory where you run the command.

## Code-aware review

Set `code_base` to consult your agent's local source using read-only search and file access.
Before compilation, candidate validation examines source code together with ETHOS.md when
available. It cannot edit or execute your agent's code.

When `code_base` is set, an additional review runs after compilation. This review investigates
each final insight's behavior, root cause, scope, and implied remediation, then assigns
`confidence` as `low`, `med`, or `high`. It does not change severity.

Confidence review makes additional model calls per insight: up to 12
investigation rounds and a final response. Each insight's context can include up to 100,000
characters of supporting traces plus tool results. Time and token usage depend on the model,
trace size, and investigation length. This is in addition to candidate validation when `code_base`
is supplied.

## Candidate validation

When `code_base` or `ethos_path` is available (including a discovered ETHOS.md),
Compass validates candidates from every signal before compilation. One investigation
per candidate receives the supporting traces, ETHOS.md when available, and read-only
source-code tools when configured. Different candidates run concurrently. This also
runs when the ethos divergence signal is disabled.

The validator reconciles intended behavior with observed behavior and implementation.
Code that intentionally implements a defect does not excuse it. Permission in ETHOS.md
only applies when its conditions are satisfied; permission to retry does not excuse an
unbounded retry loop. Only concrete contradictions remove candidates; ambiguous and
inconclusive cases are retained.

Decisions include separate code and ETHOS assessments, a combined explanation, and
an evidence basis for rejection. ETHOS-based rejections require an exact document
passage. The terminal report and telemetry include rejection reasons. Each candidate
adds one model investigation, with a limit of 25 iterations and up to 100,000 characters
of supporting traces. Either source can be used alone; without either, validation is skipped.

## One-run overrides

CLI options override individual YAML values. YAML uses underscores; CLI options use hyphens:

```bash
compass --config config.yaml \
  --trace.max-traces 25 \
  --output-path investigation.yml
```

You can also run without YAML:

```bash
compass --trace.filesystem.path traces.jsonl --max-tokens 16384
```

To enable code-aware validation and confidence review, set `code_base` in YAML:

```yaml
code_base: ../my-agent
```

Or set it for a single run:

```bash
compass --config config.yaml --code-base ../my-agent
```

Validation and confidence both add source investigation calls and run by default whenever
`code_base` is set; there is no separate flag. Severity is assigned during final compilation
with or without `code_base`.

Use `compass --help-all` for every option and its default.
Explicit CLI and YAML model settings take priority over [environment defaults](model-access.md#credentials).

## Repeat a run

Keep the configuration and a complete trace export to analyze the same input again.
Model-generated findings can vary between runs. Give each run its own `output_path`
when you want to compare results.

To update a previous collection, add these settings to your configuration:

```yaml
existing_insights: previous-insights.yml
output_path: updated-insights.yml
```

NeMo Compass reconciles old and new findings into a complete collection.
See [output behavior](results.md#saved-output) before using the files in a pipeline.
