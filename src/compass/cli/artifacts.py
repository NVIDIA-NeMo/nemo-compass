# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write CLI artifacts as JSON-safe structures.

Normalize Pydantic models, dataclasses, NumPy values, and the tool-issue
signal's ``MISSING`` sentinel before JSON serialization.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, JsonValue

from compass.compass_signals.tool_issues.signal import MISSING

__all__ = ["dump_json", "jsonable", "write_json"]

MISSING_MARKER = "<missing>"


def jsonable(value: object) -> JsonValue:
    """Recursively convert ``value`` into JSON-safe primitives."""

    if value is MISSING:
        return MISSING_MARKER
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        # JSON has no NaN/Infinity; emit null rather than invalid JSON.
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(value, BaseModel):
        return jsonable(value.model_dump(mode="python"))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=repr) if isinstance(value, (set, frozenset)) else value
        return [jsonable(item) for item in items]
    item = getattr(value, "item", None)
    if callable(item) and hasattr(value, "dtype"):  # numpy scalar
        return jsonable(item())
    tolist = getattr(value, "tolist", None)
    if callable(tolist):  # numpy array
        return jsonable(tolist())
    return str(value)


def dump_json(value: object, *, indent: int | None = 2) -> str:
    return json.dumps(jsonable(value), indent=indent, ensure_ascii=False, sort_keys=False) + "\n"


def write_json(path: str | Path, value: object, *, indent: int | None = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_json(value, indent=indent), encoding="utf-8")
    return path
