<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Release Compass

1. Run [Start release](https://github.com/NVIDIA-NeMo/labs-trace-intel/actions/workflows/start-release.yaml)
   on `main` with the final version (for example, `0.3.0`) and an optional commit SHA.
   The workflow creates the release branch and uploads an RC to a draft GitHub release.
2. Submit fixes as PRs to the release branch; each merge prepares
   another RC and automatically opens a forward-port PR to main. It auto-merges when
   checks and approvals permit; the fix author resolves blocked forward-ports.
   Leave version updates to automation and do not merge main into the release branch.
3. Once a candidate is ready, run
   [Publish release](https://github.com/NVIDIA-NeMo/labs-trace-intel/actions/workflows/publish-release.yaml)
   on `main` with its RC tag. This rebuilds and validates the final package, stages its
   wheel in NGC, and publishes the GitHub release.

For failures, use **Re-run failed jobs** on the original run. Keep existing tags and
assets. Do not rerun successful workflows.
