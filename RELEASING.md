<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Release Compass

Normal development continues on `main` while QA tests a release branch.
Compass and Eval Author release independently.

1. **Start:** Open [Actions → Release](https://github.com/NVIDIA-NeMo/labs-trace-intel/actions/workflows/release.yaml), select
   `main`, choose **start**, and enter a final version such as `0.3.0`.
   Optionally supply a main commit SHA. Automation validates it, creates
   `release/0.3.0`, and stages the first RC in a draft GitHub release and NGC.
2. **Validate:** Wait for the workflow, including NGC staging, to succeed. QA records
   acceptance of the exact RC tag and source SHA in the existing QA tracker.
   Fixes go through ordinary PRs to the release branch. Each merge prepares a new
   RC and automatically opens a forward-port PR to main; it auto-merges when checks
   and approvals permit. The fix author handles blocked forward-ports.
   Do not manually bump RC versions or merge main into the release branch.
3. **Publish:** Run **publish** on `main` with the accepted RC tag and QA evidence URL.
   Confirm engineering readiness, resolved blockers, and product readiness for major
   customer-facing changes. Automation rebuilds and validates the final version,
   then publishes the GitHub release and notes. Give the final wheel URL from the
   summary to the independent Slack publisher for PyPI. RC preparation never invokes it.

A newer candidate needs new QA acceptance. Publication stops if a branch change is
detected before publishing; a merge after the final check is excluded from the
accepted SHA's artifacts and needs a subsequent release.

## If a run fails

Choose **Re-run failed jobs** on the original run to reuse validated artifacts.
Matching uploads are reused; different bytes stop the release. Keep existing tags
and assets. If the source changed, validate its new RC instead.
The draft release links the source and checks; `candidate.json` records build identity
and checksums. PyPI retries belong to the Slack publisher.

## Release planning

The rotating captain aims to cut Friday and publish Wednesday every two weeks,
confirming holidays and QA availability. Defer incomplete features or skip a release.
QA and engineering agree on CLI startup, representative trace ingestion, and
analysis/export regression coverage; daily regression against main is follow-up work.
For urgent fixes, the captain and engineering owner record reduced QA coverage and
follow-up ownership in the QA evidence, involving product for customer-facing risk.
CI and artifact validation still apply.

## Repository setup (once)

Require **Quality** and up-to-date PRs on main and release branches; restrict direct
and force pushes, protect release tags, and enable auto-merge. Configure a GitHub App
with contents and pull-request write access using `RELEASE_APP_ID` and
`RELEASE_APP_PRIVATE_KEY`, so forward-port PRs receive normal CI. Keep
`AIRE_NGC_SERVICE_KEY` for staging. Land the workflows on main before use and review
workflow changes as privileged code.
