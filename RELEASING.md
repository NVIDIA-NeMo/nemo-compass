<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Release Compass

1. Run [Start release](https://github.com/NVIDIA-NeMo/labs-trace-intel/actions/workflows/start-release.yaml)
   on `main` with the final version (for example, `0.3.0`) and an optional commit SHA.
   The workflow creates the release branch and uploads an RC to a draft GitHub release.
2. QA validates that RC. Submit fixes as PRs to the release branch; each merge prepares
   another RC and automatically opens a forward-port PR to main. It auto-merges when
   checks and approvals permit; the fix author resolves blocked forward-ports.
   Leave version updates to automation and do not merge main into the release branch.
3. Once QA accepts the current candidate, run
   [Publish release](https://github.com/NVIDIA-NeMo/labs-trace-intel/actions/workflows/publish-release.yaml)
   on `main` with its RC tag. This rebuilds and validates the final package, stages its
   wheel in NGC, and publishes the GitHub release. Use the summary's NGC details for
   nSpect and the final wheel URL with the Slack PyPI publisher. These remain separate
   operations; the workflow does not wait for nSpect or publish to PyPI.

Publication checks that the accepted SHA is still current. A later merge cannot
change the artifacts being published and needs a subsequent release.

For failures, use **Re-run failed jobs** on the original run. Keep existing tags and
assets; investigate NGC upload failures before retrying. Do not rerun successful
workflows. If the source changes, QA must accept the new candidate.
