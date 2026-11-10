# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run-local service ownership and instrumentation for analysis agents."""

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
        super().__init__(client.model, **client.config)
        self.client = client

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
