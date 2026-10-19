<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Anomaly and pattern Compass Signal

This signal finds statistically unusual traces and recurring behavioral patterns. It projects
the shared `TraceSnapshot` into its native model, performs deterministic feature extraction,
and returns both native analysis artifacts and generic `Problem` values for Insights generation.

## Analysis

The signal:

1. extracts the built-in trace features plus requested numeric metrics;
2. scores unusual traces with `IsolationForest`;
3. clusters trajectory shapes when at least three traces and two distinct feature vectors are available;
4. groups observed verdicts and recurring strict failures; and
5. renders an evidence digest with supporting trace identifiers and source pointers.

An anomaly is evidence for investigation, not proof of a defect. Only statistical anomalies and
recurring same-tool or cross-tool failures become `Problem` values; the complete native analysis
remains available in `AnomalyAndPatternsArtifacts`.

## Configuration

`AnomalyAndPatternsConfig` owns the signal configuration:

| Field | Default | Purpose |
|---|---:|---|
| `contamination` | `0.02` | Expected fraction of statistical outliers; must be between 0 and 0.5 |
| `input_scaling` | `"none"` | Feature scaling mode; `"robust"` is also supported |
| `cluster_candidates` | `2` through `8` | Candidate trajectory-cluster counts |
| `minimum_independent_traces` | `3` | Recurrence threshold for grouped evidence |
| `feature_names` | `None` | Uses the built-in feature set unless explicitly overridden |

## Run it

Start with a [source guide](../../../../README.md#start-here) to create `config.yaml`.
Keep `compass_signals.anomaly_and_patterns` enabled and set the other signal keys to
`false` for an anomaly-only run.

```bash
uv run compass --config config.yaml
```

The CLI saves actionable insights to `insights.yml`. When using the Python API,
`AnomalyAndPatternsCompassSignal.analyze()` returns native analysis in its result's
`artifacts` field, including the evidence digest, features, anomalies, and recurring groups.

## Implementation boundary

- `to_anomaly_and_patterns_trace()` owns projection from the shared normalized trace model.
- `walk_spans()` is the private shared depth-first traversal of nested spans.
- `run_anomaly_and_patterns()` owns the native analysis.
- `AnomalyAndPatternsCompassSignal.analyze()` owns the shared compass-signal handoff.
- `decode_explicit_failure()` is intentionally independent from the tool-issue decoder.
