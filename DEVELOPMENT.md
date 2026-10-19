<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Development

Models and loaders live in [trace_ingest](src/trace_ingest/), independent of
`compass`. Both Python packages ship in the `compass` distribution.
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
where necessary, a reviewed compatibility entry in `third_party/license_overrides.yaml`.
The standard texts were sourced from SPDX license-list-data commit
`16f3aa6c3bdd62e50f8b1cf618f32d2a510250ee`; original package notices are retained separately.

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

The `compass` version must be a PEP 440 release candidate such as `0.1.0rc1`.
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

## Profile user sentiment

Run the real sentiment signal in isolation against a canonical JSONL dataset:

```bash
uv run --locked python tools/profile_user_sentiment.py \
  --dataset /path/to/analyst-evals/data/leo.jsonl \
  --output tmp/sentiment-leo-baseline \
  --model openai/azure/openai/gpt-5.6-luna \
  --api-base https://inference-api.nvidia.com/v1 \
  --embedding-model openai/nvidia/qwen/qwen3-embedding-8b \
  --embedding-api-base https://inference-api.nvidia.com/v1
```

The tool loads `.env` and reads `COMPASS_API_KEY`; use `--api-key-env` to select
another environment variable. It makes real inference and embedding requests.
`--max-traces N` selects the first N traces after loading the file. The output
directory must be new, preventing accidental overwrites.

`profile.json` contains dataset and message checksums, revision, model settings,
input-size statistics, stage wall times, per-batch client latency, and embedding
throughput. Stage times are nested: `sentiment_total` includes extraction,
embedding generation, classification, and issue detection; `generate_embeddings`
includes endpoint calls and vector compression. Its remaining time includes
cold tokenizer initialization, tokenization, validation, and client-side work.
Client-call latency includes SDK overhead and any internal retries; the profiler
does not claim server-only latency or count SDK retries as separate requests.
The current production batching, concurrency, and retry behavior are unchanged.
Small request counts make latency percentiles descriptive, not robust estimates.

The tool checkpoints after stages and requests and preserves failure types and
partial measurements if an exception occurs. `messages.json` saves the extracted
user turns before embedding, and `result.json` saves the completed signal result.
These two files contain source text: keep the output in ignored local storage.
The timing summary omits message text and API keys. Extraction is model-generated;
inspect its output before treating the extracted corpus as a sentiment benchmark.

Repeat with `--messages tmp/sentiment-leo-baseline/messages.json` and a new output
directory to bypass extraction and reuse the same ordered inputs. Use the same
dataset and trace limit; replay rejects mismatched trace IDs. Repeated user turns
are preserved. Record whether runs used a warm tokenizer/download cache when
comparing them. The isolated developer tool temporarily wraps production methods
within its own process and restores them afterward; ordinary Compass runs are
unaffected.
