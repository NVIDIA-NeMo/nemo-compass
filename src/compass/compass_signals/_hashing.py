# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable JSON serialization and digests shared by Compass Signals."""

from __future__ import annotations

import hashlib
import json


def stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: object) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


__all__ = ["digest", "stable_json"]
