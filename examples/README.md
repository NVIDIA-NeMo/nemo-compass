<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Try NeMo Compass

Run the example customer-service traces to explore the insights NeMo Compass produces.
You only need an inference API key; no trace-platform account is required.

With [uv and Git installed](../README.md#start-here), install the CLI:

```bash
uv tool install \
  'nemo-compass @ git+https://github.com/NVIDIA-NeMo/labs-nemo-compass.git@main'
```

Download the [example traces](tau_bench_traces.jsonl) and
[configuration](config.yaml) into one folder, keeping their filenames.
In that folder, [configure your model and API key](../docs/model-access.md#choose-a-model), then run:

```bash
compass --config config.yaml
```

The terminal reports completed and skipped Compass Signals. If it finds actionable insights,
it saves them to `insights.yml`. Results vary with the model.

[Read your results](../docs/results.md), or [connect your own traces](../README.md#start-here).

## About the example data

The file contains 200 telecom, retail, and airline traces from τ-bench.
Customer names, addresses, and order IDs are synthetic benchmark fixtures.
The data is covered by the [τ-bench MIT license](../third_party/tau-bench-LICENSE.txt).
