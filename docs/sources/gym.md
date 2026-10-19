<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Analyze NeMo Gym trajectories

Choose persisted Gym rollout JSONL containing `ng_trajectory` schema `1.0` from a
supported producer path in Gym's
[trajectory capability matrix](https://docs.nvidia.com/nemo/gym/reference/trajectory-capabilities/).
Recorded rewards are included in grader failure analysis.

## Configure and run

With [uv and Git installed](../../README.md#start-here), install the CLI:

```bash
uv tool install \
  'compass @ git+https://github.com/NVIDIA-NeMo/labs-nemo-compass.git@main'
```

No Gym installation or optional extra is required.
[Configure your inference model and key](../model-access.md#choose-a-model).
Loading is offline; analysis uses your configured inference model.

Save this as `config.yaml`, replacing the rollout path:

```yaml
max_tokens: 16384
trace:
  gym:
    path: rollouts.jsonl
```

```bash
compass --config config.yaml
```

Open `insights.yml` if the run produced insights. [Read your results](../results.md)
for help with findings or skipped checks.

## Select supported rollout evidence

Follow Gym's capability matrix for the observability configuration and capabilities
required by your analysis. Coverage depends on the execution path: `V` applies to
the documented path and configuration, partial (`O`) coverage requires inspecting
recorded gaps, and unavailable (`X`) evidence is not inferred.

For captured model calls, enable `observability_enabled`, configure
`model_call_capture_dir`, and route calls through the rollout-prefixed Gym Model
Server endpoint as documented by Gym. Direct-provider calls bypass that capture.

Use persisted rollout files; MLflow/W&B exporter views omit `ng_trajectory`.
Responses-only exports without the attachment are rejected. ATIF import/export and
round-trip conversion are outside this loader's scope; use the [ATIF loader](files.md#atif)
for retained original ATIF trajectories.

## File options

Each JSONL record remains one rollout. For one pretty-printed rollout object, add
`format: json` under `trace.gym`. Relative paths resolve from your working directory.
The loader reads the complete file and does not support `trace.max_traces`.

You can also run without YAML:

```bash
compass --trace.gym.path rollouts.jsonl --max-tokens 16384
```

## What is preserved

Invocation nesting, captured model calls, tool arguments and results, recorded
status and timing, semantic turns, rewards, and evidence gaps are preserved.
Task IDs group repeated rollouts. Missing evidence stays missing, nullable token
counts are not filled with zero, and rewards do not establish execution success.
Duplicate rollout IDs and ambiguous joins are rejected.
