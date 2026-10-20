<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# nemo-compass-trace-ingest

The `trace_ingest` package from [NeMo Compass](https://github.com/NVIDIA-NeMo/nemo-compass):
the canonical agent `Trace` model and its provider loaders, without the Compass
application and its analysis dependencies.

```bash
pip install nemo-compass-trace-ingest
pip install "nemo-compass-trace-ingest[mlflow]"     # MLflow loaders
pip install "nemo-compass-trace-ingest[langfuse]"   # Langfuse v3/v4 loaders
pip install "nemo-compass-trace-ingest[langsmith]"  # LangSmith loaders
```

```python
from trace_ingest import Trace
from trace_ingest.loaders.atif import ATIFTraceLoader
```

The source lives in the Compass repository under `src/trace_ingest`. The
`nemo-compass` distribution depends on this one at the same version.
