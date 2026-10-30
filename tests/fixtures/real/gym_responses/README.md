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

To regenerate, in a Gym checkout:

```sh
rev=399e6783e0e879424fc20a23f8e2d46ffa4cfdee
for env in example_single_tool_call example_tool_call_multireward; do
  git show "$rev:resources_servers/$env/data/example_rollouts.jsonl" > "$env.example_rollouts.jsonl"
done
```
