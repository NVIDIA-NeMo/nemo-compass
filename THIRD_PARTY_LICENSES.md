<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Third-Party Software Licenses

This repository uses the dependencies listed below. Each package link identifies the exact
version resolved in `uv.lock`. Complete collected license and attribution texts are
preserved in [third_party/NOTICES.txt](third_party/NOTICES.txt), grouped by package and
version, with the original archive URLs, checksums, and file paths. SPDX expressions
are inventory metadata only; upstream texts retain their own terms and notices.
Shared standard license texts are also included under `third_party/license_texts/`.

The inventory covers runtime dependencies and all optional extras, including transitive
and platform-specific dependencies. Tau Bench example data is MIT-licensed; its copyright
and license are preserved in [third_party/tau-bench-LICENSE.txt](third_party/tau-bench-LICENSE.txt).

Run `make update-licenses` after changing dependencies to collect texts automatically.
`make check-licenses` verifies the generated inventory and collected texts.
See DEVELOPMENT.md for collection scope and exception handling.

| Package | SPDX license expression | License and attribution documents |
| --- | --- | --- |
| [`aiohappyeyeballs 2.7.1`](https://pypi.org/project/aiohappyeyeballs/2.7.1/) | `PSF-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [PSF-2.0.txt](third_party/license_texts/PSF-2.0.txt) |
| [`aiohttp 3.14.3`](https://pypi.org/project/aiohttp/3.14.3/) | `APACHE-2.0 AND MIT` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`aiosignal 1.4.0`](https://pypi.org/project/aiosignal/1.4.0/) | `APACHE-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt) |
| [`annotated-doc 0.0.5`](https://pypi.org/project/annotated-doc/0.0.5/) | `MIT` | [NOTICES.txt](third_party/NOTICES.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`annotated-types 0.8.0`](https://pypi.org/project/annotated-types/0.8.0/) | `MIT` | [NOTICES.txt](third_party/NOTICES.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`anyio 4.14.2`](https://pypi.org/project/anyio/4.14.2/) | `MIT` | [NOTICES.txt](third_party/NOTICES.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`attrs 26.1.0`](https://pypi.org/project/attrs/26.1.0/) | `MIT` | [NOTICES.txt](third_party/NOTICES.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`backoff 2.2.1`](https://pypi.org/project/backoff/2.2.1/) | `MIT` | [NOTICES.txt](third_party/NOTICES.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`boto3 1.43.81`](https://pypi.org/project/boto3/1.43.81/) | `APACHE-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt) |
| [`botocore 1.43.81`](https://pypi.org/project/botocore/1.43.81/) | `APACHE-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt) |
| [`cachetools 7.1.7`](https://pypi.org/project/cachetools/7.1.7/) | `MIT` | [NOTICES.txt](third_party/NOTICES.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`certifi 2026.7.22`](https://pypi.org/project/certifi/2026.7.22/) | `MPL-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [MPL-2.0.txt](third_party/license_texts/MPL-2.0.txt) |
| [`cffi 2.1.1`](https://pypi.org/project/cffi/2.1.1/) | `MIT-0` | [NOTICES.txt](third_party/NOTICES.txt), [MIT-0.txt](third_party/license_texts/MIT-0.txt) |
| [`charset-normalizer 3.5.1`](https://pypi.org/project/charset-normalizer/3.5.1/) | `MIT` | [NOTICES.txt](third_party/NOTICES.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`click 8.5.0`](https://pypi.org/project/click/8.5.0/) | `BSD-3-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt) |
| [`cloudpickle 3.1.2`](https://pypi.org/project/cloudpickle/3.1.2/) | `BSD-3-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt) |
| [`colorama 0.4.6`](https://pypi.org/project/colorama/0.4.6/) | `BSD-3-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt) |
| [`cryptography 50.0.1`](https://pypi.org/project/cryptography/50.0.1/) | `APACHE-2.0 OR BSD-3-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt) |
| [`databricks-sdk 0.133.0`](https://pypi.org/project/databricks-sdk/0.133.0/) | `APACHE-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt) |
| [`distro 1.9.0`](https://pypi.org/project/distro/1.9.0/) | `APACHE-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt) |
| [`fastapi 0.141.1`](https://pypi.org/project/fastapi/0.141.1/) | `MIT` | [NOTICES.txt](third_party/NOTICES.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`fastuuid 0.14.0`](https://pypi.org/project/fastuuid/0.14.0/) | `BSD-3-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt) |
| [`filelock 3.32.4`](https://pypi.org/project/filelock/3.32.4/) | `MIT` | [NOTICES.txt](third_party/NOTICES.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`frozenlist 1.8.0`](https://pypi.org/project/frozenlist/1.8.0/) | `APACHE-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt) |
| [`fsspec 2026.7.0`](https://pypi.org/project/fsspec/2026.7.0/) | `BSD-3-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt) |
| [`gitdb 4.0.12`](https://pypi.org/project/gitdb/4.0.12/) | `BSD-3-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt) |
| [`gitpython 3.1.60`](https://pypi.org/project/gitpython/3.1.60/) | `BSD-3-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt) |
| [`google-auth 2.57.0`](https://pypi.org/project/google-auth/2.57.0/) | `APACHE-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt) |
| [`googleapis-common-protos 1.75.3`](https://pypi.org/project/googleapis-common-protos/1.75.3/) | `APACHE-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt) |
| [`h11 0.16.0`](https://pypi.org/project/h11/0.16.0/) | `MIT` | [NOTICES.txt](third_party/NOTICES.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`hf-xet 1.6.0`](https://pypi.org/project/hf-xet/1.6.0/) | `APACHE-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt) |
| [`httpcore 1.0.9`](https://pypi.org/project/httpcore/1.0.9/) | `BSD-3-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt) |
| [`httpcore2 2.12.0`](https://pypi.org/project/httpcore2/2.12.0/) | `BSD-3-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt) |
| [`httpx 0.28.1`](https://pypi.org/project/httpx/0.28.1/) | `BSD-3-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt) |
| [`httpx2 2.12.0`](https://pypi.org/project/httpx2/2.12.0/) | `BSD-3-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt) |
| [`httpx2-jsfetch 1.0`](https://pypi.org/project/httpx2-jsfetch/1.0/) | `BSD-3-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt) |
| [`huggingface-hub 1.29.0`](https://pypi.org/project/huggingface-hub/1.29.0/) | `APACHE-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt) |
| [`idna 3.19`](https://pypi.org/project/idna/3.19/) | `BSD-3-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt) |
| [`importlib-metadata 8.9.0`](https://pypi.org/project/importlib-metadata/8.9.0/) | `APACHE-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt) |
| [`jinja2 3.1.6`](https://pypi.org/project/jinja2/3.1.6/) | `BSD-3-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt) |
| [`jiter 0.16.0`](https://pypi.org/project/jiter/0.16.0/) | `MIT` | [NOTICES.txt](third_party/NOTICES.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`jmespath 1.1.0`](https://pypi.org/project/jmespath/1.1.0/) | `MIT` | [NOTICES.txt](third_party/NOTICES.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`joblib 1.5.3`](https://pypi.org/project/joblib/1.5.3/) | `BSD-3-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt) |
| [`jsonschema 4.26.0`](https://pypi.org/project/jsonschema/4.26.0/) | `MIT` | [NOTICES.txt](third_party/NOTICES.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`jsonschema-specifications 2025.9.1`](https://pypi.org/project/jsonschema-specifications/2025.9.1/) | `MIT` | [NOTICES.txt](third_party/NOTICES.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`langfuse 4.17.0`](https://pypi.org/project/langfuse/4.17.0/) | `MIT` | [NOTICES.txt](third_party/NOTICES.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`langsmith 0.12.2`](https://pypi.org/project/langsmith/0.12.2/) | `MIT` | [NOTICES.txt](third_party/NOTICES.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`litellm 1.98.0`](https://pypi.org/project/litellm/1.98.0/) | `MIT` | [NOTICES.txt](third_party/NOTICES.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`markdown-it-py 4.2.0`](https://pypi.org/project/markdown-it-py/4.2.0/) | `MIT` | [NOTICES.txt](third_party/NOTICES.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`markupsafe 3.0.3`](https://pypi.org/project/markupsafe/3.0.3/) | `BSD-3-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt) |
| [`mdurl 0.1.2`](https://pypi.org/project/mdurl/0.1.2/) | `MIT` | [NOTICES.txt](third_party/NOTICES.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`mlflow-skinny 3.15.2`](https://pypi.org/project/mlflow-skinny/3.15.2/) | `APACHE-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt) |
| [`mpmath 1.3.0`](https://pypi.org/project/mpmath/1.3.0/) | `BSD-3-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt) |
| [`multidict 6.7.1`](https://pypi.org/project/multidict/6.7.1/) | `APACHE-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt) |
| [`narwhals 2.25.0`](https://pypi.org/project/narwhals/2.25.0/) | `MIT` | [NOTICES.txt](third_party/NOTICES.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`networkx 3.6.1`](https://pypi.org/project/networkx/3.6.1/) | `BSD-3-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt) |
| [`nooa 0.0.10`](https://pypi.org/project/nooa/0.0.10/) | `APACHE-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt) |
| [`numpy 2.5.2`](https://pypi.org/project/numpy/2.5.2/) | `BSD-3-CLAUSE AND 0BSD AND MIT AND ZLIB AND CC0-1.0` | [NOTICES.txt](third_party/NOTICES.txt), [0BSD.txt](third_party/license_texts/0BSD.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt), [CC0-1.0.txt](third_party/license_texts/CC0-1.0.txt), [MIT.txt](third_party/license_texts/MIT.txt), [Zlib.txt](third_party/license_texts/Zlib.txt) |
| [`openai 2.54.0`](https://pypi.org/project/openai/2.54.0/) | `APACHE-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt) |
| [`openinference-instrumentation 0.1.61`](https://pypi.org/project/openinference-instrumentation/0.1.61/) | `APACHE-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt) |
| [`openinference-instrumentation-litellm 0.1.41`](https://pypi.org/project/openinference-instrumentation-litellm/0.1.41/) | `APACHE-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt) |
| [`openinference-semantic-conventions 0.1.35`](https://pypi.org/project/openinference-semantic-conventions/0.1.35/) | `APACHE-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt) |
| [`opentelemetry-api 1.45.1`](https://pypi.org/project/opentelemetry-api/1.45.1/) | `APACHE-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt) |
| [`opentelemetry-exporter-http-transport 0.66b1`](https://pypi.org/project/opentelemetry-exporter-http-transport/0.66b1/) | `APACHE-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt) |
| [`opentelemetry-exporter-otlp-common 0.66b1`](https://pypi.org/project/opentelemetry-exporter-otlp-common/0.66b1/) | `APACHE-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt) |
| [`opentelemetry-exporter-otlp-proto-common 1.45.1`](https://pypi.org/project/opentelemetry-exporter-otlp-proto-common/1.45.1/) | `APACHE-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt) |
| [`opentelemetry-exporter-otlp-proto-http 1.45.1`](https://pypi.org/project/opentelemetry-exporter-otlp-proto-http/1.45.1/) | `APACHE-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt) |
| [`opentelemetry-instrumentation 0.66b1`](https://pypi.org/project/opentelemetry-instrumentation/0.66b1/) | `APACHE-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt) |
| [`opentelemetry-proto 1.45.1`](https://pypi.org/project/opentelemetry-proto/1.45.1/) | `APACHE-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt) |
| [`opentelemetry-sdk 1.45.1`](https://pypi.org/project/opentelemetry-sdk/1.45.1/) | `APACHE-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt) |
| [`opentelemetry-semantic-conventions 0.66b1`](https://pypi.org/project/opentelemetry-semantic-conventions/0.66b1/) | `APACHE-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt) |
| [`orjson 3.12.0`](https://pypi.org/project/orjson/3.12.0/) | `MPL-2.0 AND (APACHE-2.0 OR MIT)` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt), [MIT.txt](third_party/license_texts/MIT.txt), [MPL-2.0.txt](third_party/license_texts/MPL-2.0.txt) |
| [`packaging 25.0`](https://pypi.org/project/packaging/25.0/) | `APACHE-2.0 OR BSD-2-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt), [BSD-2-Clause.txt](third_party/license_texts/BSD-2-Clause.txt) |
| [`propcache 0.5.2`](https://pypi.org/project/propcache/0.5.2/) | `APACHE-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt) |
| [`protobuf 7.36.1`](https://pypi.org/project/protobuf/7.36.1/) | `BSD-3-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt) |
| [`pyasn1 0.6.4`](https://pypi.org/project/pyasn1/0.6.4/) | `BSD-2-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-2-Clause.txt](third_party/license_texts/BSD-2-Clause.txt) |
| [`pyasn1-modules 0.4.2`](https://pypi.org/project/pyasn1-modules/0.4.2/) | `BSD-3-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt) |
| [`pycparser 3.0`](https://pypi.org/project/pycparser/3.0/) | `BSD-3-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt) |
| [`pydantic 2.13.4`](https://pypi.org/project/pydantic/2.13.4/) | `MIT` | [NOTICES.txt](third_party/NOTICES.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`pydantic-core 2.46.4`](https://pypi.org/project/pydantic-core/2.46.4/) | `MIT` | [NOTICES.txt](third_party/NOTICES.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`pydantic-settings 2.15.0`](https://pypi.org/project/pydantic-settings/2.15.0/) | `MIT` | [NOTICES.txt](third_party/NOTICES.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`pygments 2.21.0`](https://pypi.org/project/pygments/2.21.0/) | `BSD-2-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-2-Clause.txt](third_party/license_texts/BSD-2-Clause.txt) |
| [`python-dateutil 2.9.0.post0`](https://pypi.org/project/python-dateutil/2.9.0.post0/) | `APACHE-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt) |
| [`python-dotenv 1.2.3`](https://pypi.org/project/python-dotenv/1.2.3/) | `BSD-3-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt) |
| [`pyyaml 6.0.3`](https://pypi.org/project/pyyaml/6.0.3/) | `MIT` | [NOTICES.txt](third_party/NOTICES.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`referencing 0.37.0`](https://pypi.org/project/referencing/0.37.0/) | `MIT` | [NOTICES.txt](third_party/NOTICES.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`regex 2026.7.19`](https://pypi.org/project/regex/2026.7.19/) | `APACHE-2.0 AND CNRI-PYTHON` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt), [CNRI-Python.txt](third_party/license_texts/CNRI-Python.txt) |
| [`requests 2.34.2`](https://pypi.org/project/requests/2.34.2/) | `APACHE-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt) |
| [`requests-toolbelt 1.0.0`](https://pypi.org/project/requests-toolbelt/1.0.0/) | `APACHE-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt) |
| [`rich 15.0.0`](https://pypi.org/project/rich/15.0.0/) | `MIT` | [NOTICES.txt](third_party/NOTICES.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`rpds-py 2026.6.3`](https://pypi.org/project/rpds-py/2026.6.3/) | `MIT` | [NOTICES.txt](third_party/NOTICES.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`s3transfer 0.19.2`](https://pypi.org/project/s3transfer/0.19.2/) | `APACHE-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt) |
| [`safetensors 0.8.0`](https://pypi.org/project/safetensors/0.8.0/) | `APACHE-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt) |
| [`scikit-learn 1.9.0`](https://pypi.org/project/scikit-learn/1.9.0/) | `BSD-3-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt) |
| [`scipy 1.18.1`](https://pypi.org/project/scipy/1.18.1/) | `BSD-3-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt) |
| [`setuptools 84.0.0`](https://pypi.org/project/setuptools/84.0.0/) | `MIT` | [NOTICES.txt](third_party/NOTICES.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`shellingham 1.5.4`](https://pypi.org/project/shellingham/1.5.4/) | `ISC` | [NOTICES.txt](third_party/NOTICES.txt), [ISC.txt](third_party/license_texts/ISC.txt) |
| [`six 1.17.0`](https://pypi.org/project/six/1.17.0/) | `MIT` | [NOTICES.txt](third_party/NOTICES.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`smmap 5.0.3`](https://pypi.org/project/smmap/5.0.3/) | `BSD-3-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt) |
| [`sniffio 1.3.1`](https://pypi.org/project/sniffio/1.3.1/) | `APACHE-2.0 OR MIT` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`sqlparse 0.6.0`](https://pypi.org/project/sqlparse/0.6.0/) | `BSD-3-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt) |
| [`starlette 1.6.0`](https://pypi.org/project/starlette/1.6.0/) | `BSD-3-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt) |
| [`sympy 1.14.0`](https://pypi.org/project/sympy/1.14.0/) | `BSD-3-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt) |
| [`threadpoolctl 3.6.0`](https://pypi.org/project/threadpoolctl/3.6.0/) | `BSD-3-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt) |
| [`tiktoken 0.14.0`](https://pypi.org/project/tiktoken/0.14.0/) | `MIT` | [NOTICES.txt](third_party/NOTICES.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`tokenizers 0.23.1`](https://pypi.org/project/tokenizers/0.23.1/) | `APACHE-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt) |
| [`torch 2.13.0`](https://pypi.org/project/torch/2.13.0/) | `APACHE-2.0 AND APACHE-2.0 WITH LLVM-EXCEPTION AND BSD-2-CLAUSE AND BSD-3-CLAUSE AND BSL-1.0 AND MIT` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt), [BSD-2-Clause.txt](third_party/license_texts/BSD-2-Clause.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt), [BSL-1.0.txt](third_party/license_texts/BSL-1.0.txt), [LLVM-exception.txt](third_party/license_texts/LLVM-exception.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`torch 2.13.0+cpu`](https://pypi.org/project/torch/2.13.0+cpu/) | `APACHE-2.0 AND APACHE-2.0 WITH LLVM-EXCEPTION AND BSD-2-CLAUSE AND BSD-3-CLAUSE AND BSL-1.0 AND MIT` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt), [BSD-2-Clause.txt](third_party/license_texts/BSD-2-Clause.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt), [BSL-1.0.txt](third_party/license_texts/BSL-1.0.txt), [LLVM-exception.txt](third_party/license_texts/LLVM-exception.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`tqdm 4.70.0`](https://pypi.org/project/tqdm/4.70.0/) | `MIT AND MPL-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [MIT.txt](third_party/license_texts/MIT.txt), [MPL-2.0.txt](third_party/license_texts/MPL-2.0.txt) |
| [`transformers 5.17.0`](https://pypi.org/project/transformers/5.17.0/) | `APACHE-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt) |
| [`truststore 0.10.4`](https://pypi.org/project/truststore/0.10.4/) | `MIT` | [NOTICES.txt](third_party/NOTICES.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`typer 0.27.2`](https://pypi.org/project/typer/0.27.2/) | `MIT` | [NOTICES.txt](third_party/NOTICES.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`typing-extensions 4.16.0`](https://pypi.org/project/typing-extensions/4.16.0/) | `PSF-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [PSF-2.0.txt](third_party/license_texts/PSF-2.0.txt) |
| [`typing-inspection 0.4.4`](https://pypi.org/project/typing-inspection/0.4.4/) | `MIT` | [NOTICES.txt](third_party/NOTICES.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`urllib3 2.7.0`](https://pypi.org/project/urllib3/2.7.0/) | `MIT` | [NOTICES.txt](third_party/NOTICES.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`uuid-utils 0.17.1`](https://pypi.org/project/uuid-utils/0.17.1/) | `BSD-3-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt) |
| [`uvicorn 0.52.4`](https://pypi.org/project/uvicorn/0.52.4/) | `BSD-3-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt) |
| [`websockets 17.1`](https://pypi.org/project/websockets/17.1/) | `BSD-3-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt) |
| [`wrapt 1.17.3`](https://pypi.org/project/wrapt/1.17.3/) | `BSD-2-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-2-Clause.txt](third_party/license_texts/BSD-2-Clause.txt) |
| [`xxhash 4.0.1`](https://pypi.org/project/xxhash/4.0.1/) | `BSD-2-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-2-Clause.txt](third_party/license_texts/BSD-2-Clause.txt) |
| [`yarl 1.24.5`](https://pypi.org/project/yarl/1.24.5/) | `APACHE-2.0` | [NOTICES.txt](third_party/NOTICES.txt), [Apache-2.0.txt](third_party/license_texts/Apache-2.0.txt) |
| [`zipp 4.1.0`](https://pypi.org/project/zipp/4.1.0/) | `MIT` | [NOTICES.txt](third_party/NOTICES.txt), [MIT.txt](third_party/license_texts/MIT.txt) |
| [`zstandard 0.25.0`](https://pypi.org/project/zstandard/0.25.0/) | `BSD-3-CLAUSE` | [NOTICES.txt](third_party/NOTICES.txt), [BSD-3-Clause.txt](third_party/license_texts/BSD-3-Clause.txt) |
