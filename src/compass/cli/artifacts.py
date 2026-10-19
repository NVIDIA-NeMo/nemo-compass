# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write CLI artifacts as JSON-safe structures.

Two things in the engine outputs are not natively serialisable:

* ``run_anomaly_and_patterns`` returns ``PreparedTrace`` dataclasses under ``"prepared"``, which
  carry extracted features and evidence.
* The tool-issue signal's ``MISSING`` sentinel is a bare ``object()``. It should never reach
  ``json.dumps``, and if it somehow does we want ``"<missing>"`` in the output
  rather than a crash three hours into a batch run.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, JsonValue

from compass.compass_signals.anomaly_and_patterns.signal import PreparedTrace
from compass.compass_signals.tool_issues.signal import MISSING

__all__ = ["dump_json", "jsonable", "prepared_features", "write_json"]

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


def prepared_features(prepared: Iterable[PreparedTrace]) -> list[dict[str, Any]]:
    """Flatten ``run_anomaly_and_patterns``'s ``prepared`` list into per-trace feature rows.

    Keeps the numeric vector, the trajectory tokens and the source pointer —
    everything an analyst needs to see why a trace was scored the way it was —
    without echoing the entire input corpus back into the output directory.
    """

    rows = []
    for item in prepared:
        features = item.features
        rows.append(
            {
                "trace_id": features.trace_id,
                "numeric": jsonable(features.numeric),
                "sequence_tokens": list(features.sequence_tokens),
                "tool_names": list(item.tool_names),
                "failure_event_count": len(item.failure_events),
                "last_agent_excerpt": item.last_agent_excerpt,
                "source_pointer": jsonable(features.source_pointer),
            }
        )
    return rows
