# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run-local service ownership and instrumentation for analysis agents."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Protocol

from nooa.context_blocks import EventBase
from nooa.runtime.event_manager import EventManager
from nooa.unifiedllm import CacheBoundary, LLMResponse, Tool, UnifiedLLM
from pydantic import BaseModel

InstrumentAgent = Callable[[EventManager], AbstractAsyncContextManager[object]]


class BorrowedModelClient(UnifiedLLM):
    """Delegate inference without entering or closing a host-owned client.

    The host must keep the underlying client open for the complete analysis run.
    A factory can return a new wrapper for every consumer of that shared client.
    """

    def __init__(self, client: UnifiedLLM) -> None:
        # Delegate metadata as well as inference. Reinitializing UnifiedLLM here
        # would replace the host's reasoning, media and context-window settings.
        self.client = client
        self._http = None

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 -- Transparent read-only metadata forwarding.
        return getattr(self.client, name)

    def call(
        self,
        messages: list[dict[str, Any] | LLMResponse | CacheBoundary],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs: Any,  # noqa: ANN401 -- Forward provider-specific inference parameters.
    ) -> LLMResponse:
        return self.client.call(messages, tools=tools, output_model=output_model, **kwargs)

    async def acall(
        self,
        messages: list[dict[str, Any] | LLMResponse | CacheBoundary],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs: Any,  # noqa: ANN401 -- Forward provider-specific inference parameters.
    ) -> LLMResponse:
        return await self.client.acall(messages, tools=tools, output_model=output_model, **kwargs)


class _OwnedSignalClient(BorrowedModelClient):
    """Close a created signal client once, including when its signal is skipped."""

    def __init__(self, client: UnifiedLLM) -> None:
        super().__init__(client)
        self._closed = False

    async def __aenter__(self) -> _OwnedSignalClient:
        await self.client.__aenter__()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if not self._closed:
            self._closed = True
            await self.client.__aexit__(*exc_info)

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            await self.client.aclose()


@asynccontextmanager
async def owned_signal_clients(
    factory: Callable[[], UnifiedLLM] | None,
) -> AsyncIterator[Callable[[], UnifiedLLM] | None]:
    """Track eager construction separately from each signal's execution context."""
    async with AsyncExitStack() as stack:
        if factory is None:
            yield None
            return

        def create() -> UnifiedLLM:
            client = _OwnedSignalClient(factory())
            stack.push_async_callback(client.aclose)
            return client

        yield create


@dataclass
class _RunInstrumentation:
    instrument: InstrumentAgent | None
    handler: Callable[[EventBase], None] | None


_current: ContextVar[_RunInstrumentation | None] = ContextVar("compass_analysis", default=None)


@asynccontextmanager
async def analysis_scope(
    instrument: InstrumentAgent | None,
    handler: Callable[[EventBase], None] | None,
) -> AsyncIterator[None]:
    """Keep hooks isolated between concurrent runs and release them on failure."""
    token = _current.set(_RunInstrumentation(instrument, handler))
    try:
        yield
    finally:
        _current.reset(token)


class InstrumentedAgent(Protocol):
    event_manager: EventManager


@asynccontextmanager
async def agent_scope(agent: InstrumentedAgent) -> AsyncIterator[None]:
    """Enter and exit instrumentation in the task that executes this agent."""
    scope = _current.get()
    if scope is None or (scope.instrument is None and scope.handler is None):
        yield
        return
    manager = agent.event_manager
    async with AsyncExitStack() as stack:
        if scope.instrument is not None:
            await stack.enter_async_context(scope.instrument(manager))
        if scope.handler is not None:
            for event in ("LLMResponse", "PythonOutput"):
                stack.callback(manager.on(event, scope.handler))
        yield
