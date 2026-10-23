# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from email.utils import formatdate
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import numpy as np
import pytest
from litellm import APIConnectionError, APIError, EmbeddingResponse, RateLimitError, Timeout
from pydantic import ValidationError
from tokenizers import Tokenizer, models

from compass.compass_signals.user_embedding import embedding as module
from compass.compass_signals.user_embedding.embedding import (
    LiteLLMEmbeddingConfig,
    UserEmbedding,
    UserEmbeddingGenerator,
)


@pytest.fixture
def generator():
    gen = UserEmbeddingGenerator(
        litellm=LiteLLMEmbeddingConfig(model="openai/qwen", batch_size=1, max_concurrency=1)
    )
    gen.remote_tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    return gen


def result(batch):
    return [UserEmbedding(bytes([batch.index]), batch.counts[0])]


def limited(value=None):
    headers = {} if value is None else {"Retry-After": str(value)}
    return RateLimitError(
        "limited", "openai", "qwen", response=httpx.Response(429, headers=headers)
    )


def test_retry_after_dates_invalid_values_and_wait_limit(monkeypatch):
    monkeypatch.setattr(module.time, "time", lambda: 1000)
    assert module._retry_delay(limited(formatdate(1010, usegmt=True)), 1, 60) == 10
    for value in ("invalid", -4, "inf", "nan"):
        delay = module._retry_delay(limited(value), 1, 60)
        assert delay is not None and 0.5 <= delay <= 1
    assert module._retry_delay(limited(61), 1, 60) is None
    delay = module._retry_delay(limited(), 10, 60)
    assert delay is not None and 30 <= delay <= 60
    header_only = RateLimitError("limited", "openai", "qwen", headers={"Retry-After": "8"})
    assert module._retry_delay(header_only, 1, 60) == 8


@pytest.mark.parametrize(
    "field,value",
    [
        ("batch_size", 0),
        ("retry_timeout", 0),
        ("retry_timeout", float("nan")),
        ("max_concurrency", 0),
        ("max_concurrency", 33),
        ("max_retries", -1),
        ("max_retries", 11),
        ("request_timeout", 0),
        ("request_timeout", float("inf")),
        ("max_retry_wait", -1),
        ("max_retry_wait", float("nan")),
    ],
)
def test_config_limits(field, value):
    with pytest.raises(ValidationError):
        LiteLLMEmbeddingConfig(model="qwen", **{field: value})


@pytest.mark.parametrize("status", [429, 503])
@pytest.mark.parametrize("retries", [0, 1])
def test_real_transport_obeys_retry_budget(generator, monkeypatch, retries, status):
    import json
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    import litellm

    default_retries = litellm.DEFAULT_MAX_RETRIES
    monkeypatch.setattr(litellm, "num_retries", 2)
    received = []

    class Endpoint(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            if len(received) == 1:
                response_status = status
                body = {
                    "error": {
                        "message": "try again",
                        "type": "rate_limit_error" if status == 429 else "server_error",
                    }
                }
            else:
                response_status = 200
                body = {
                    "object": "list",
                    "model": "qwen",
                    "data": [
                        {
                            "object": "embedding",
                            "index": 0,
                            "embedding": np.eye(1, 4096)[0].tolist(),
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }
            encoded = json.dumps(body).encode()
            self.send_response(response_status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Retry-After", "0")
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("TEST_EMBEDDING_KEY", "test")
    generator.litellm = LiteLLMEmbeddingConfig(
        model="openai/qwen",
        api_base=f"http://127.0.0.1:{server.server_port}/v1",
        api_key_env="TEST_EMBEDDING_KEY",
        max_retries=retries,
        max_retry_wait=0.001,
    )
    try:
        if retries:
            assert len(generator.generate_batch(["hello"])) == 1
        else:
            with pytest.raises((litellm.RateLimitError, litellm.ServiceUnavailableError)) as exc:
                generator.generate_batch(["hello"])
            assert exc.value.status_code == status
        assert len(received) == retries + 1
        assert litellm.DEFAULT_MAX_RETRIES == default_retries
        assert litellm.num_retries == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize(
    "milliseconds,seconds,expected",
    [
        ("1500", None, 1.5),
        ("1500.5", None, 1.5005),
        ("1500", "9", 1.5),
        ("0", "9", 0.5),
        ("invalid", "9", 9),
        ("-1", "9", 9),
        ("inf", "9", 9),
        ("nan", "9", 9),
        ("invalid", None, 0.5),
        ("61000", None, None),
    ],
)
def test_millisecond_retry_header(monkeypatch, milliseconds, seconds, expected):
    monkeypatch.setattr(module.random, "uniform", lambda low, high: low)
    headers = {"Retry-After-Ms": milliseconds}
    if seconds is not None:
        headers["Retry-After"] = seconds
    error = RateLimitError("limited", "openai", "qwen", headers=headers)
    assert module._retry_delay(error, 1, 60) == expected


def test_router_limits_requests_refills_and_preserves_order(generator, monkeypatch):
    """Exercise the real Router semaphore, replacing only the provider call."""
    import litellm

    generator.litellm.max_concurrency = 2
    active = peak = 0
    calls = []
    next_started = None

    async def endpoint(**kwargs):
        nonlocal active, peak, next_started
        if next_started is None:
            next_started = asyncio.Event()
        index = int(kwargs["input"][0][-1])
        active += 1
        peak = max(peak, active)
        try:
            if index == 0:
                await asyncio.wait_for(next_started.wait(), 5)
            if index == 2:
                next_started.set()
            await asyncio.sleep(0)
            calls.append(index)
            return EmbeddingResponse(
                data=[{"index": 0, "embedding": np.eye(1, 4096, index)[0].tolist()}]
            )
        finally:
            active -= 1

    monkeypatch.setattr(litellm, "aembedding", endpoint)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    actual = generator.generate_batch([str(i) for i in range(6)])
    assert peak == 2
    assert calls.index(2) < calls.index(0)
    expected = [generator.projection.compress(np.eye(1, 4096, i)[0]) for i in range(6)]
    assert [item.data for item in actual] == expected


@pytest.mark.parametrize(
    "error",
    [
        limited(),
        Timeout("timeout", "qwen", "openai"),
        APIConnectionError("connection", "qwen", "openai"),
        APIError(503, "unavailable", "openai", "qwen"),
    ],
)
def test_retry_attempt_limit(generator, monkeypatch, error):
    endpoint = AsyncMock(side_effect=error)
    sleep = AsyncMock()
    monkeypatch.setattr(generator, "_remote_batch", endpoint)
    monkeypatch.setattr(module.asyncio, "sleep", sleep)
    batch = next(generator._remote_batches(["hello"], 1))
    with pytest.raises(type(error)):
        asyncio.run(generator._remote_batch_with_retries(batch, generator.litellm, None))
    assert endpoint.await_count == 4
    assert sleep.await_count == 3


@pytest.mark.parametrize(
    "error",
    [APIError(401, "unauthorized", "openai", "qwen"), ValueError("invalid vector"), limited(61)],
)
def test_unrecoverable_failure_does_not_retry(generator, monkeypatch, error):
    endpoint = AsyncMock(side_effect=error)
    monkeypatch.setattr(generator, "_remote_batch", endpoint)
    batch = next(generator._remote_batches(["hello"], 1))
    with pytest.raises(type(error)):
        asyncio.run(generator._remote_batch_with_retries(batch, generator.litellm, None))
    assert endpoint.await_count == 1


def test_retry_budget_checked_after_sleep(generator, monkeypatch):
    generator.litellm.retry_timeout = 2
    clock = [100.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    endpoint = AsyncMock(side_effect=limited(1))

    async def sleep(delay):
        clock[0] += 3

    monkeypatch.setattr(generator, "_remote_batch", endpoint)
    monkeypatch.setattr(module.asyncio, "sleep", sleep)
    batch = next(generator._remote_batches(["hello"], 1))
    with pytest.raises(RateLimitError):
        asyncio.run(generator._remote_batch_with_retries(batch, generator.litellm, None))
    assert endpoint.await_count == 1


def test_failure_cancels_other_batches(generator, monkeypatch):
    cancelled = []

    async def endpoint(batch, config, router):
        if batch.index == 0:
            await asyncio.sleep(0)
            raise ValueError("invalid")
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(batch.index)

    monkeypatch.setattr(generator, "_remote_batch", endpoint)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    with pytest.raises(ValueError, match="invalid"):
        generator.generate_batch(["a", "b", "c"])
    assert sorted(cancelled) == [1, 2]


def test_request_timeout_is_explicit(generator, monkeypatch):
    seen = []

    async def endpoint(**kwargs):
        seen.append(kwargs)
        return EmbeddingResponse(data=[{"index": 0, "embedding": np.eye(1, 4096)[0].tolist()}])

    monkeypatch.setattr(
        "compass.compass_signals.user_embedding.embedding._request_embeddings", endpoint
    )
    generator.generate_batch(["hello"])
    assert seen[0]["timeout"] == 60


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"index": 1, "embedding": [1.0] * 4096}],
        [{"index": 0, "embedding": [0.0] * 4096}],
        [{"index": 0, "embedding": [float("nan")] * 4096}],
        [{"index": 0, "embedding": [1.0] * 10}],
    ],
)
def test_bad_response_fails_without_retry(generator, monkeypatch, rows):
    calls = []

    async def endpoint(**kwargs):
        calls.append(kwargs)
        return EmbeddingResponse(data=rows)

    monkeypatch.setattr(
        "compass.compass_signals.user_embedding.embedding._request_embeddings", endpoint
    )
    with pytest.raises(ValueError):
        generator.generate_batch(["first"])
    assert len(calls) == 1


def test_retry_wait_honors_milliseconds_and_elapsed_budget(generator, monkeypatch):
    generator.litellm.retry_timeout = 2
    clock = [100.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    error = RateLimitError("limited", "openai", "qwen", headers={"retry-after-ms": "1500"})
    endpoint = AsyncMock(side_effect=error)
    delays = []

    async def sleep(delay):
        delays.append(delay)
        clock[0] += delay

    monkeypatch.setattr(generator, "_remote_batch", endpoint)
    monkeypatch.setattr(module.asyncio, "sleep", sleep)
    batch = next(generator._remote_batches(["hello"], 1))
    with pytest.raises(RateLimitError):
        asyncio.run(generator._remote_batch_with_retries(batch, generator.litellm, None))
    assert delays == [1.5]
    assert endpoint.await_count == 2


def test_router_cleanup_preserves_other_callbacks(generator, monkeypatch):
    import litellm

    def other_callback(*args, **kwargs):
        pass

    monkeypatch.setattr(litellm, "success_callback", [other_callback])
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    endpoint = AsyncMock(
        return_value=EmbeddingResponse(
            data=[{"index": 0, "embedding": np.eye(1, 4096)[0].tolist()}]
        )
    )
    monkeypatch.setattr(litellm, "aembedding", endpoint)
    for _ in range(2):
        assert len(generator.generate_batch(["hello"])) == 1
        assert litellm.success_callback == [other_callback]
