# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit, in-process registration for Compass Signals."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from pydantic import BaseModel

from compass.compass_signals.compass_signals import CompassSignal, CompassSignalResult
from compass.traces import TraceSnapshot


class CompassSignalRegistry:
    """Compass Signals keyed by their public names.

    Registration is explicit and preserves insertion order. A signal owns its
    configuration validation; the registry only enforces the shared contract.
    """

    def __init__(self) -> None:
        self._signals: dict[str, CompassSignal] = {}
        self._configurations: dict[str, BaseModel | None] = {}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._signals)

    def configuration(self, name: str) -> BaseModel | None:
        """Return the configuration explicitly associated with a registered signal."""
        return self._configurations[name]

    def register(self, signal: CompassSignal, *, config: BaseModel | None = None) -> None:
        name = signal.name.strip()
        if not name:
            raise ValueError("Compass Signal name must not be empty")
        if name in self._signals:
            raise ValueError(f"Compass Signal {name!r} is already registered")
        self._signals[name] = signal
        self._configurations[name] = config

    async def analyze(
        self,
        name: str,
        snapshot: TraceSnapshot,
        *,
        on_start: Callable[[CompassSignal], None] | None = None,
    ) -> CompassSignalResult:
        try:
            signal = self._signals[name]
        except KeyError as exc:
            available = ", ".join(self.names) or "none"
            raise KeyError(
                f"Compass Signal {name!r} is not registered; available: {available}"
            ) from exc
        skip_reason = await asyncio.to_thread(signal.check_prerequisites, snapshot)
        if not len(snapshot):
            skip_reason = "No traces loaded"
        if skip_reason is not None:
            return CompassSignalResult(signal_name=name, problems=(), skip_reason=skip_reason)
        if on_start is not None:
            on_start(signal)
        result = await signal.analyze(snapshot)
        if result.signal_name != name:
            raise ValueError(f"Compass Signal {name!r} returned result for {result.signal_name!r}")
        return result

    async def analyze_all(self, snapshot: TraceSnapshot) -> tuple[CompassSignalResult, ...]:
        return tuple(
            await asyncio.gather(*(self.analyze(name, snapshot) for name in self._signals))
        )


__all__ = ["CompassSignalRegistry"]
