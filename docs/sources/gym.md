<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Analyze NeMo Gym trajectories

Choose persisted Gym rollout JSONL. Rollouts containing `ng_trajectory` schema `1.0`
from a supported producer path in Gym's
[trajectory capability matrix](https://docs.nvidia.com/nemo/gym/reference/trajectory-capabilities/)
give the most complete evidence. Rollouts without the attachment load from their
Responses envelope with less detail; see [Rollouts without ng_trajectory](#rollouts-without-ng_trajectory).
Recorded rewards are included in grader failure analysis.

## Configure and run

With [uv installed](../../README.md#start-here), install the CLI:

```bash
uv tool install nemo-compass
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
ATIF import/export and round-trip conversion are outside this loader's scope; use the
[ATIF loader](files.md#atif) for retained original ATIF trajectories.

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

## Rollouts without ng_trajectory

A record without `ng_trajectory` (or with `"ng_trajectory": null`) loads from the
rollout envelope (`responses_create_params` plus `response`) or from a standalone
Responses object (`"object": "response"`). Each record becomes one agent span holding
the request messages, the final assistant text (refusal text included) and the declared
tools; one model span for the recorded response, with its model and token usage; and one
tool span per `function_call`, paired with its `function_call_output` by `call_id`. A call
with no recorded output has no result.

These records lack the invocation tree, per-call model grouping and all timing: the
recorded response can cover several model turns, and no span has start or end times.
The tool spans follow the model span as siblings, so the span tree does not show how
messages, model turns and tool calls were interleaved; each tool span's `item_path`
and the model span's output keep the source order. Each record reports the gaps
`ng_trajectory_unavailable` and `responses_output_aggregated`. The trace ID is `gym-`
plus the first 16 hex digits of the SHA-256 of the record bytes (the JSONL line or JSON
file, without surrounding whitespace), so byte-identical records are rejected as
duplicates. `_ng_task_index` groups repeated rollouts of one task.

Multi-turn agents, such as Gym's gymnasium and simulated-user agents, append each
environment observation or user reply to the response output as a user message before
calling the model again; these messages load in order with the rest of the output. When
the request has no input, the output holds the whole conversation. Gym's aviary agent
records `"contains_transitions": true` and a list of transitions as the output, each the
model input after one environment step; the last transition loads as the output, and
a transition that does not extend the previous one is rejected.

Records referencing remote conversation state (`previous_response_id` or
`conversation`, on the request or the response), a request with input whose response
output starts with a user, system or developer message (other than an exact copy of the
request input), namespaced functions, unsupported item types such as built-in tool calls,
and duplicate or unmatched call IDs are rejected.

### Conversation turns

The agent span of a Responses record also holds the conversation as ordered turns:
request `instructions` (unless the input already starts with that system message),
then each request and response message, reasoning summary and group of adjacent
`function_call` items. User, assistant, system and developer messages become user,
agent and system turns (a developer turn keeps its role in `extra.gym.role`).
Text, refusal and `input_image` parts become text and image parts; an image's type
comes from a `data:` URI or the file suffix, and images are never fetched. Assistant
messages, reasoning and function calls generated by the response link the model span
and are attributed to the model; other messages in the response output, such as
environment observations, are of unknown authorship. Every function call turn links
its tool spans; results stay on those spans. Request
history is copied context and links no model span, except the last request user
message, which is of unknown authorship. Reasoning summaries stay in
`extra.gym.reasoning_summary`, not in the turn's reasoning. Each turn's
`extra.gym.item_path` points at its source item; a turn grouping adjacent function
calls points at the first.

Turns are absent when the conversation has no user message (as in a standalone
Responses object), when a message has another role, message content or a function call
output is neither text nor supported parts (a `null` output included), an image has an
unknown type, function arguments are not a string that decodes to a JSON object, a
reasoning summary is not a list of `summary_text` parts, or the record carries
malformed ATIF conversion metadata or metadata for several Harbor trajectories.
Rollouts with `ng_trajectory` have no turns.
