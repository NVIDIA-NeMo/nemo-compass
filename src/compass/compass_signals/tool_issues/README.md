<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tool-issue Compass Signal

This signal performs deterministic, capability-gated auditing of tool calls. It projects the
shared `TraceSnapshot` into tool-issue records, evaluates nineteen finding types, groups recurring
findings into evidence cards, and returns recurrence-qualified `Problem` values for Insights
generation.

Missing evidence causes a rule to abstain; it does not create a speculative finding. Every
finding retains the original trace, call identifier, and source pointer.

## Detection and promotion

The nineteen rules cover tool contracts and arguments, call/result integrity, explicit outcomes,
retry behavior, trace instrumentation, argument provenance, and prerequisite or state failures.
`detect()` emits individual findings. `build_cards()` groups findings by issue type and mechanism,
selects representative evidence, and marks a card eligible for NeMo Compass after it appears in
the configured number of independent logical cases.

`problems_from_cards()` excludes audit-only cards by default. This keeps one-off observations in
the native artifacts without presenting them as recurring problems.

## Configuration

`ToolIssueConfig` owns the signal configuration:

| Field | Default | Purpose |
|---|---:|---|
| `minimum_independent_cases` | `3` | Cases required before a card is eligible for NeMo Compass |
| `retry_threshold` | `3` | Failed calls required for retry-pattern findings |
| `include_audit_problems` | `false` | Include cards below the recurrence threshold in projected problems |

## Run it

Start with a [source guide](../../../../README.md#start-here) to create `config.yaml`.
Keep `compass_signals.tool_issues` enabled and set the other signal keys to `false`
for a tool-issue-only run.

```bash
uv run compass --config config.yaml
```

The CLI saves actionable insights to `insights.yml`. When using the Python API,
`ToolIssueCompassSignal.analyze()` returns findings, cards, finding-type counts, and
configuration in its result's `artifacts` field.

The adjacent `coverage.py` reports which rules the supplied corpus has enough evidence to
evaluate. This distinguishes a clean corpus from a detector that is silent because required
fields were not captured.

## Implementation boundary

- `to_tool_issue_trace()` owns projection from the shared normalized trace model.
- `walk_spans()` is the private shared depth-first traversal of nested spans.
- `detect()` owns individual rule evaluation.
- `build_cards()` owns recurrence grouping and promotion.
- `ToolIssueCompassSignal.analyze()` owns the shared compass-signal handoff.
- `strict_failure()` is intentionally narrower than the anomaly signal's decoder.
