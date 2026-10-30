<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Real ATIF trajectories from Harbor

All four files come from [harbor-framework/harbor](https://github.com/harbor-framework/harbor)
at tag `v0.24.0` (`b53b8134e1241686dca7759af188f987ecc48e8b`), Apache-2.0
(`LICENSE-harbor.txt`, copied from that tag). They are byte-identical copies, not
trimmed or reformatted. `tests/test_real_atif.py` compacts each one to a one-line JSONL
corpus and loads it on its own, and loads the two Qwen Code files together.

Published by Harbor (real Claude Code runs converted by Harbor's claude-code converter,
ATIF-v1.7):

- `claude-code-hello-load-atif.json`: upstream
  `examples/tasks/hello-load-atif-trajectory-task-level/trajectory.json`. One `Write`
  tool call.
- `claude-code-trajectory_pass.json`: upstream
  `packages/harbor-atif2otel/tests/fixtures/trajectory_pass.json`. 11 agent steps,
  12 tool calls.

Generated with Harbor's real producer code, inputs from upstream unit tests (no live
Qwen Code run):

- `qwen-code-nested-subagents.trajectory.json`: output of Harbor's
  `QwenCode.populate_context_post_run` on the session transcripts written by
  `tests/unit/agents/installed/test_qwen_code_trajectory.py::test_nests_subagent_trajectories_using_parent_agent_metadata`.
  ATIF-v1.7 with `subagent_trajectories` embedded two levels deep. The output is
  deterministic.
- `qwen-code-embedded-subagent.trajectory.json`: the same, for
  `test_embeds_subagent_trajectory_and_links_parent_tool_result`. ATIF-v1.7 with one
  embedded subagent. Both Qwen Code files share `session_id` `session-1` and set no root
  `trajectory_id`.

To regenerate, from a Harbor checkout at `v0.24.0`:

```sh
git show v0.24.0:examples/tasks/hello-load-atif-trajectory-task-level/trajectory.json > claude-code-hello-load-atif.json
git show v0.24.0:packages/harbor-atif2otel/tests/fixtures/trajectory_pass.json > claude-code-trajectory_pass.json
uv sync --frozen
PYTHONPATH=. uv run python - <<'EOF'
import shutil
from tempfile import TemporaryDirectory
from pathlib import Path
from harbor.agents.installed.qwen_code import QwenCode
from harbor.models.agent.context import AgentContext
from tests.unit.agents.installed import test_qwen_code_trajectory as upstream

for name, test in {
    "qwen-code-embedded-subagent": upstream.test_embeds_subagent_trajectory_and_links_parent_tool_result,
    "qwen-code-nested-subagents": upstream.test_nests_subagent_trajectories_using_parent_agent_metadata,
}.items():
    with TemporaryDirectory() as temp:
        logs = Path(temp)
        test(logs)
        QwenCode(logs_dir=logs, model_name="qwen3-coder-plus").populate_context_post_run(AgentContext())
        shutil.copyfile(logs / "trajectory.json", f"{name}.trajectory.json")
EOF
```
