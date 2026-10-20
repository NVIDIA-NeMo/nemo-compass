# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Print the canonical ``Trace`` JSON Schema committed as ``schemas/trace.schema.json``."""

from __future__ import annotations

import json

from trace_ingest.models import Trace


def main() -> None:
    print(json.dumps(Trace.model_json_schema(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
