<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Analyze NeMo Platform Intake traces

Select a workspace and a time window. Recorded evaluator results for spans in the
selected traces are included automatically.

## Connect and run

With [uv and Git installed](../../README.md#start-here), install the CLI:

```bash
uv tool install \
  'compass @ git+https://github.com/NVIDIA-NeMo/labs-nemo-compass.git@main'
```

[Configure your inference model and key](../model-access.md#choose-a-model).
For authenticated deployments, add `NMP_ACCESS_TOKEN=your-access-token` to `.env`.

Save this as `config.yaml`. Replace the URL, workspace, and dates:

```yaml
max_tokens: 16384
trace:
  max_traces: 100
  intake:
    base_url: https://platform.example.com
    workspace: my-workspace
    query:
      started_at_gte: 2026-09-01T00:00:00Z
      started_at_lte: 2026-09-02T00:00:00Z
```

```bash
compass --config config.yaml
```

Open `insights.yml` if the run produced insights. [Read your results](../results.md)
for help with findings or skipped Compass Signals.

## Narrow the input

Both time bounds are required, inclusive, and timezone-aware.

Under `trace.intake.query`, use `experiment_id` to select an experiment’s traces.
You can also filter by `agent_name`, `evaluation_name`, `test_case_name`, `session_id`,
or `status` (`success`, `error`, `cancelled`, or `unknown`). If you set both
`experiment_id` and `evaluation_name`, the evaluation must belong to that experiment.

For a single run:

```bash
compass --config config.yaml --trace.intake.query.experiment-id my-experiment-id
```

`sort` defaults to `started_at` (oldest first); use `-started_at` for newest first.
`trace.max_traces` overrides `query.max_traces`. Without either limit, all matching traces
in the window are loaded.

Under `trace.intake`, `page_size` controls pagination (1–1,000; default 100), and
`timeout_seconds` sets the HTTP timeout (default 30). Endpoint and workspace have no
environment-variable fallbacks.

Intake supports live queries only. For local input, use [canonical trace files](files.md).
