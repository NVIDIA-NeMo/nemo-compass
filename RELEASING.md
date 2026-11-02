<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Release Compass

1. In **Actions → Release**, select `main`, choose **start**, and enter the final
   version (for example `0.3.0`). Optionally select a full commit SHA from main.
   Select a commit containing these release workflows. CI validates the source before creating `release/0.3.0` and its first RC.
2. QA downloads the candidate from its **draft GitHub release** or NGC. The release
   links the checks and source SHA; `candidate.json` records the version, source,
   build run, and artifact checksums. Wait for the complete workflow, including staging.
   Record smoke/regression results and blockers in your existing QA tracker.
3. Submit ordinary fix PRs against the release branch. Merging prepares another RC
   and opens an author-assigned forward-port PR to main. Clean PRs auto-merge when
   repository rules permit; the author resolves conflicts/check failures/approval needs.
   Do not routinely merge main into the release branch or edit release versions.
4. In **Actions → Release** on `main`, choose **publish**, enter the accepted RC tag
   and its QA evidence URL, and confirm readiness. This records your attestation that
   engineering is ready, blockers are resolved, and product has approved major
   customer-facing changes. A changed branch requires QA to accept its newer candidate.
5. Use the final wheel URL in the workflow summary with the existing Slack publisher
   to publish to PyPI. Candidate preparation never invokes that publisher.

RC numbers derive from first-parent commits since the release cut; retries keep the
same number; busy merges may coalesce into the newest candidate. Versions are applied only in the build checkout (including `uv.lock`),
so the tag identifies source and the manifest identifies the package version.
Final artifacts are rebuilt, run through CI/distribution checks and an isolated
wheel installation smoke test, then uploaded. An RC wheel is never renamed.

Retry with **Re-run failed jobs** on the original run: validated artifacts are reused,
matching uploads are skipped, and differing bytes stop the release. Never delete or
move release tags or overwrite assets to repair a retry. If source changes, use its
new RC. The release branch must remain current through publication; avoid merging
fixes during the final publishing step. Already-published PyPI retry behavior belongs
to the independent Slack publisher.

Aim to cut Friday and publish the following Wednesday every two weeks. The rotating
captain confirms holidays and QA availability, and defers features or skips a release
when needed. Initially QA should exercise CLI startup, one representative trace from
supported sources, and analysis/export against a fixed regression corpus; QA and
engineering agree on the corpus before the first release. Daily main regression is
follow-up work. Compass releases are independent of Eval Author.

For critical fixes, the captain and engineering owner agree on reduced validation;
product joins for customer-facing risk. Record the decision, exact candidate, smoke
results, and follow-up regression owner in the QA evidence. Use the same workflow:
exceptions do not bypass CI, identity checks, or existing licensing checks.

## Repository setup (once)

Require **Quality** on main and release-branch PRs;
require PR branches to be up to date, restrict direct/force pushes, and protect release
tags. Enable auto-merge.
Install a repository-scoped GitHub App with contents and pull-request write access;
set `RELEASE_APP_ID` and `RELEASE_APP_PRIVATE_KEY` for forward-port PRs to trigger
normal CI. No branch-protection bypass is needed. Keep `AIRE_NGC_SERVICE_KEY` for
staging. The new trusted workflows/helpers must land on main before use. Review
workflow changes as privileged code.
