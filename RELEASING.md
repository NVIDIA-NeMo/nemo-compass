<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Prepare a Compass release

A release ships two distributions at the same version:

- `nemo-compass-trace-ingest` (`packages/trace-ingest/`): the `trace_ingest` models and
  loaders, with only `httpx` and `pydantic` as dependencies.
- `nemo-compass`: the `compass` CLI and Python package. It depends on
  `nemo-compass-trace-ingest` pinned to the same version.

1. Set the same version in `pyproject.toml` and `packages/trace-ingest/pyproject.toml`,
   update the `nemo-compass-trace-ingest==<version>` pins in the root `pyproject.toml`,
   run `uv lock`, and merge these files into `main`. `tools/check_release.py` rejects
   mismatched versions or pins.
2. Open [Actions → Prepare release](https://github.com/NVIDIA-NeMo/nemo-compass/actions/workflows/prepare-release.yaml)
   and select **Run workflow**. Choose `main`, optionally enter a specific commit
   SHA, and start the run.
3. Wait for CI to pass. The action tags the checked commit and creates a draft
   release with both wheels, both source archives, and checksums. Version `0.2.0`,
   for example, produces tag `v0.2.0`.
4. Review the draft's notes and files before publishing the GitHub release.

The action does not publish to PyPI. Publish the release's files manually,
`nemo-compass-trace-ingest` first and then `nemo-compass`, so that the
`nemo-compass` pin always resolves:

```bash
uv publish dist/nemo_compass_trace_ingest-<version>*
uv publish dist/nemo_compass-<version>*
```

If a run fails, open it in Actions and choose **Re-run failed jobs**. If it left an
incomplete draft, delete that draft before retrying. Keep the tag. If the tag
already points to a different commit, use a new version.
