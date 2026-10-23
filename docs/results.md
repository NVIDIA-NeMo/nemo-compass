<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Read your results

Start with the saved insights. Each highlights your agent’s behavior and links it to
supporting traces you can inspect in your source platform or export.

An illustrative entry in `insights.yml`:

```yaml
- name: Retries repeat an invalid order lookup
  description: >-
    After lookup_order reports an unknown order ID, the agent repeats
    the same request without asking the customer to correct the ID.
  severity: med
  severity_reason: >-
    The agent completes the task only after an unnecessary retry.
  evidence:
    - trace_id: run-12
      url: https://observability.example.com/traces/run-12
      spans:
        - span_id: lookup-3
          url: https://observability.example.com/spans/lookup-3
    - trace_id: run-38
      url: https://observability.example.com/traces/run-38
```

Open a trace link to review the supporting behavior before deciding what to change.
Each `evidence` entry identifies a supporting trace and optionally its relevant spans.
URLs are optional and come from the loader, not the inference model.

Live LangSmith and Langfuse traces use their provider-returned UI locations; live MLflow
traces use the HTTP tracking server’s UI. Braintrust links require your
[organization name](sources/braintrust.md#links-to-braintrust). Intake traces currently
retain trace IDs without viewer links.

Local inputs link to the original file on the machine that ran the analysis. Canonical
JSONL, ATIF JSONL, Gym JSONL, and LangSmith exports include a `#L<number>` fragment for
the physical source line. Opening the file at that line depends on your viewer’s support.
Whole-file ATIF `.json` trajectories and other native exports link to the containing file.
Canonical file links always point to the input file; user-supplied URLs are not retained.

When a provider does not supply a usable URL, or its location cannot be determined
(for example, an MLflow `databricks` or local tracking URI), the ID remains available.
Previously saved links survive reconciliation for traces absent from the current run;
links for traces loaded in this run are resolved from the current source.

Each trace entry can list relevant `spans` at any nesting depth, with a `span_id` and
optional `url`. Braintrust and live LangSmith supply span links; other sources retain
span IDs. An entry without spans refers to the whole trace.

## Terminal report

The terminal lists insight titles, supporting-trace counts, and analysis coverage.
Open the saved YAML for descriptions and complete trace/span IDs and links.

```text
3 insights from 200 traces

1. Agent calls tools absent from the active catalog
   Evidence: 3 supporting traces

2. Retries repeat an invalid request
   Evidence: 8 supporting traces

3. Agent reports success after a failed tool call
   Evidence: 4 supporting traces

Ran:      Anomalies and patterns (5 candidates), tool issues (3 candidates)
Skipped:  Ethos — no document
          Grader failures — no results
          Sentiment — no embedding backend

To run skipped analyses: https://github.com/NVIDIA-NeMo/labs-nemo-compass/blob/main/docs/compass-signals.md

Saved: out/try-next-step/insights.yml
```

Disabled analyses are omitted. Supporting trace counts do not represent a failure rate.

A **candidate issue** is a problem found during analysis. Further review may merge or
discard it, so candidate counts can exceed the number of saved insights.

**No findings** means a Compass Signal ran and found nothing to report in the available data.
**Skipped** means it could not run. A limitation beside a completed Compass Signal describes
missing coverage; it does not mean the whole Compass Signal was skipped.

## A Compass Signal was skipped

Open [Compass Signals](compass-signals.md) for the prerequisite and a setup example.
[Disable Compass Signals](compass-signals.md#disable-a-compass-signal) you don’t need in your configuration.
Disabled Compass Signals are omitted from the report.

If no traces were loaded, check your source, filters, and time window first.

## No insights were produced

The run may have found no actionable patterns, or its candidates may not have passed
review. This does not establish that your agent is free of problems. Check the report
for skipped analyses and limitations.

## Saved output

The output is a YAML list with `name`, `description`, `evidence`, and optional
`updated_date`, `severity`, `severity_reason`, and `confidence` for each insight.
Each insight has at least two supporting traces. Empty spans and absent URLs are omitted.

`severity` describes observed consequence after recovery: `low`, `med`, or `high`.
Final compilation assigns it to new insights by default; it is omitted when the
outcome is unknown. It does not depend on trace count or confidence.
`severity_reason` states the observed consequence behind the rating in one short
sentence. It is omitted when severity is unknown; existing insights retain saved reasons.
Compilation may fill a missing reason for an existing insight with a severity rating.

`confidence` describes how well source investigation supports the insight's
behavior, root cause, scope, and implied remediation. It is set when
[`code_base`](configuration.md#code-aware-review) is configured, is `low`, `med`, or `high`,
and is omitted otherwise. Existing insights retain their saved ratings during compilation;
confidence review updates confidence without changing severity.

`updated_date` is a UTC timestamp set when an insight is created or gains a trace it
didn't already have. It is omitted until that first happens, and an insight that gains
no new trace in a run keeps its prior `updated_date` unchanged. See
[repeat a run](configuration.md#repeat-a-run) for reconciling across runs.

By default, a non-empty collection is written to `insights.yml`.
**An empty result leaves any existing output file untouched.** Use a fresh output path
per run, or capture stdout to receive the current result, including `[]`:

```bash
compass --config config.yaml --output-path - > run-insights.yml
```

Progress and the report go to stderr. Piped stdout contains YAML.
The default output file is still written unless `--output-path -` is selected.
A successful run, including one with skipped Compass Signals or no insights, exits with code 0.
Missing required environment settings exit with code 2; other errors exit nonzero.

To carry findings forward, see [repeat a run](configuration.md#repeat-a-run).
