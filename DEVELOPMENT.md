<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Development

Models and loaders live in [trace_ingest](src/trace_ingest/), independent of
`compass`. Both Python packages ship in the `nemo-compass` distribution.
`uv sync` installs the project locally.

## Validation

When adding NVIDIA-authored files or changing dependencies, update the tracked
licensing artifacts first:

```bash
make update-copyright-headers
make update-licenses
```

For dependency changes, run `uv add <package>` and `make update-licenses`, review
the generated diff, and commit it. Generation requires `osv-scanner` 2.3.3 on `PATH`
and automatically collects license and attribution texts for locked runtime
dependencies and all extras into `third_party/NOTICES.txt`, with indexes in
`THIRD_PARTY_LICENSES.md` and `third_party/licenses.jsonl`.

Only exceptions require manual work: if a distribution omits its license, add a
version-specific, checksum-pinned upstream document in `third_party/license_exceptions.yaml`.
Unknown license terms require a shared text under `third_party/license_texts/` and,
where necessary, a verified metadata correction in `third_party/license_overrides.yaml`.
The standard texts were sourced from SPDX license-list-data commit
`16f3aa6c3bdd62e50f8b1cf618f32d2a510250ee`; original package notices are retained separately.

Dependency PRs receive a **Dependency licensing** report covering core, optional
extras, and development tooling, including transitive dependencies and their immediate
introducers. Missing metadata or failed analysis leaves the check unresolved.
For new dependencies or changed license evidence, a maintainer with merge access
posts the acknowledgment line supplied by the report as a separate PR comment.
The check records the author and evidence fingerprint; unrelated code edits preserve
acknowledgment. Changed license/attribution documents also require review when the
expression is unchanged: the report identifies this reason, rather than assuming
that unchanged SPDX metadata proves unchanged terms. Unchanged-license version updates and removals do not require it.

Acknowledgment is not legal approval. Maintainers decide whether to escalate to legal
or OSRB; use a linked ticket/comment recording the owner, affected changes, submission
date, and outcome. Allow for the four-week review SLA. Record merge and shipping
restrictions separately in that ticket and the release's QA evidence. Metadata overrides
and missing-document exceptions are not approval records.

`make check-licenses` verifies the generated disclosures. Scanner intermediates and
download caches live in ignored `tmp/`; a cold cache requires network access.
These notices cover this source/Python distribution;
shipping containers or bundled native environments requires collecting notices from
those actual artifacts too, since platform wheels can bundle additional libraries.

Ruff requires annotations outside tests. ty uses its defaults and checks all Python
files, including tests.

Run the same read-only checks used by CI:

```bash
uv lock --check
make check-copyright-headers
make check-licenses
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked ty check
uv run --locked pytest
uv build --sdist --wheel --no-sources
uv run --locked python tools/check_distribution_licenses.py
```

The distribution includes the repository-root license and attribution files.
The artifact check verifies their contents in the wheel, source distribution,
and a wheel rebuilt outside the repository.

CI also installs the application wheel into a clean environment and checks
imports and the CLI.

## Publish an internal release candidate

The `nemo-compass` version must be a PEP 440 release candidate such as `0.1.0rc1`.
Put the internal Artifactory upload endpoint
and your access or identity token in the gitignored `.env`:

```dotenv
ARTIFACTORY_PYPI_URL=<internal-artifactory-pypi-upload-url>
ARTIFACTORY_TOKEN=<token>
```

Use the full token from Artifactory. The accompanying reference token is a shorter alias and is
not needed by this publisher.

Build and validate without uploading:

```bash
uv run tools/publish_artifactory.py --dry-run
```

Publish the Compass wheel to internal Artifactory:

```bash
uv run tools/publish_artifactory.py
```

The script builds the Compass wheel, submits SHA-1 and SHA-256 checksums,
verifies the downloaded bytes, and prints
the direct URL to register with nSpect. Matching existing artifacts are reused; different bytes
require a version bump. It rejects environment files and non-NVIDIA upload endpoints and disables
trusted/public publishing.

## Dependency review setup (maintainers)

Require **Dependency licensing** and up-to-date PR branches. Land the generator and
regenerated inventory on each target branch before enabling the required check;
missing baseline evidence is unresolved. The workflow/helper must land on the default
branch before use. The privileged workflow reads PR data and never executes PR code.
