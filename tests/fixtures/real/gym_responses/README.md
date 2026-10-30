<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Published NeMo Gym rollouts

Byte-identical copies of `example_rollouts.jsonl` files published in
NVIDIA-NeMo/Gym at `399e6783e0e879424fc20a23f8e2d46ffa4cfdee` (Apache-2.0). Gym's
rollout collector produced them against real model endpoints. Each line is the
rollout envelope Gym writes (`responses_create_params`, `response` and verifier
fields). Nothing is trimmed or edited.

- `example_single_tool_call.example_rollouts.jsonl`: upstream
  `resources_servers/example_single_tool_call/data/example_rollouts.jsonl`. 5 rows
  from `gpt-4.1-2025-04-14`. 3 rows call `get_weather` once (`function_call` and
  `function_call_output`), 2 ask for clarification without a tool call.
- `example_tool_call_multireward.example_rollouts.jsonl`: upstream
  `resources_servers/example_tool_call_multireward/data/example_rollouts.jsonl`. 5 rows
  from `nvidia/qwen/eccn-qwen3.6-35b-a3b`, with `reasoning` items, one tool call each,
  `_ng_task_index` and per-component rewards (`reward_components`).
- `example_multi_turn_gymnasium.example_rollouts.jsonl`: upstream
  `resources_servers/example_multi_turn_gymnasium/data/example_rollouts.jsonl`. 5 rows
  from `Qwen/Qwen3-4B-Instruct-2507`. Gym's `gymnasium_agent` appends each environment
  follow-up question to `response.output` as a `role: user` message, then calls the model
  again.
- `aviary.example_rollouts.jsonl`: upstream
  `resources_servers/aviary/data/example_rollouts.jsonl`. 5 GSM8K rows from
  `gpt-4.1-2025-04-14` through Gym's `aviary_agent`, which sets
  `"contains_transitions": true` and records `response.output` as a list of transitions,
  each the full model input after one environment step (1-5 `calculator` and
  `submit_answer` calls per row).

To regenerate, in a Gym checkout:

```sh
rev=399e6783e0e879424fc20a23f8e2d46ffa4cfdee
for env in example_single_tool_call example_tool_call_multireward example_multi_turn_gymnasium; do
  git show "$rev:resources_servers/$env/data/example_rollouts.jsonl" > "$env.example_rollouts.jsonl"
done
git show "$rev:resources_servers/aviary/data/example_rollouts.jsonl" > aviary.example_rollouts.jsonl
```
