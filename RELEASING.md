<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Prepare a Compass release

The distribution is `nemo-compass`. The CLI and Python imports use `compass`.

1. Set the version in `pyproject.toml`, run `uv lock`, and merge both files into `main`.
2. Open [Actions → Prepare release](https://github.com/NVIDIA-NeMo/nemo-compass/actions/workflows/prepare-release.yaml)
   and select **Run workflow**. Choose `main`, optionally enter a specific commit
   SHA, and start the run.
3. Wait for CI to pass. The action tags the checked commit and creates a draft
   release with the wheel, source archive, and checksums. Version `0.1.0`, for
   example, produces tag `v0.1.0`.
4. Review the draft's notes and files before publishing the GitHub release.

The action does not publish to PyPI.

If a run fails, open it in Actions and choose **Re-run failed jobs**. If it left an
incomplete draft, delete that draft before retrying. Keep the tag. If the tag
already points to a different commit, use a new version.
