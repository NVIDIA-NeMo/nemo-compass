# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from collections import Counter
from email.utils import formatdate

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


def test_rolling_queue_refills_before_slow_request_finishes(generator, monkeypatch):
    generator.litellm.max_concurrency = 2
    next_started = threading.Event()
    calls = []

    def worker(batch, config, key):
        if batch.index == 1:
            assert next_started.wait(5)
        if batch.index == 3:
            next_started.set()
        calls.append(batch.index)
        return result(batch)

    monkeypatch.setattr(generator, "_remote_batch", worker)
    actual = generator.generate_batch([str(i) for i in range(6)])
    assert calls.index(3) < calls.index(1)
    assert [x.data for x in actual] == [bytes([i]) for i in range(6)]


@pytest.mark.parametrize("concurrency", [None, 1, 2])
def test_fixed_pool_starts_at_configured_limit(generator, monkeypatch, concurrency):
    generator.litellm = LiteLLMEmbeddingConfig(
        model="qwen",
        batch_size=1,
        **({} if concurrency is None else {"max_concurrency": concurrency}),
    )
    concurrency = 8 if concurrency is None else concurrency
    barrier = threading.Barrier(concurrency, timeout=5)
    lock = threading.Lock()
    active = peak = 0

    def worker(batch, config, key):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            barrier.wait()
            return result(batch)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(generator, "_remote_batch", worker)
    actual = generator.generate_batch(["a"] * 16)
    assert peak == concurrency
    assert [x.data for x in actual] == [bytes([i]) for i in range(16)]


def test_default_batch_size(generator, monkeypatch):
    generator.litellm = LiteLLMEmbeddingConfig(model="qwen")
    sizes = []

    def worker(batch, config, key):
        sizes.append(len(batch.inputs))
        return [UserEmbedding(b"", count) for count in batch.counts]

    monkeypatch.setattr(generator, "_remote_batch", worker)
    assert len(generator.generate_batch(["a"] * 129)) == 129
    assert sorted(sizes) == [1, 128]


@pytest.mark.parametrize("error", [limited(), APIError(503, "unavailable", "openai", "qwen")])
def test_retry_only_failed_batch_while_other_workers_continue(generator, monkeypatch, error):
    generator.litellm.max_concurrency = 2
    sleeping = threading.Event()
    next_finished = threading.Event()
    calls = Counter()
    sleeps = []

    def worker(batch, config, key):
        calls[batch.index] += 1
        if batch.index == 0 and calls[0] == 1:
            raise error
        if batch.index == 1:
            assert sleeping.wait(5)
        if batch.index == 2:
            next_finished.set()
        return result(batch)

    def sleep(delay):
        sleeps.append(delay)
        sleeping.set()
        assert next_finished.wait(5)

    monkeypatch.setattr(generator, "_remote_batch", worker)
    monkeypatch.setattr(module.time, "sleep", sleep)
    actual = generator.generate_batch(["a"] * 6)
    assert [x.data for x in actual] == [bytes([i]) for i in range(6)]
    assert calls == Counter({0: 2, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1})
    assert len(sleeps) == 1


@pytest.mark.parametrize(
    "error",
    [
        limited(),
        Timeout("timeout", "qwen", "openai"),
        APIConnectionError("connection", "qwen", "openai"),
        APIError(503, "unavailable", "openai", "qwen"),
    ],
)
def test_transient_failures_have_bounded_attempts(generator, monkeypatch, error):
    calls = []
    sleeps = []

    def worker(batch, config, key):
        calls.append(batch.index)
        raise error

    monkeypatch.setattr(generator, "_remote_batch", worker)
    clock = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])

    def sleep(delay):
        sleeps.append(delay)
        clock[0] += delay

    monkeypatch.setattr(module.time, "sleep", sleep)
    with pytest.raises(type(error)):
        generator.generate_batch(["first", "never submitted"])
    assert calls == [0] * 4
    assert len(sleeps) == 3
    assert all(2**i / 2 <= delay <= 2**i for i, delay in enumerate(sleeps))


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_errors_do_not_retry(generator, monkeypatch, status):
    calls = []

    def worker(batch, config, key):
        calls.append(batch.index)
        raise APIError(status, "permanent", "openai", "qwen")

    monkeypatch.setattr(generator, "_remote_batch", worker)
    with pytest.raises(APIError):
        generator.generate_batch(["first", "never submitted"])
    assert calls == [0]


def test_zero_retries_and_sequential_configuration(monkeypatch, generator):
    generator.litellm = LiteLLMEmbeddingConfig(
        model="qwen", batch_size=1, max_concurrency=1, max_retries=0
    )
    calls = []

    def worker(batch, config, key):
        calls.append(batch.index)
        if batch.index == 2:
            raise limited()
        return result(batch)

    monkeypatch.setattr(generator, "_remote_batch", worker)
    with pytest.raises(RateLimitError):
        generator.generate_batch(["a"] * 5)
    assert calls == [0, 1, 2]


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


def test_long_retry_after_aborts_without_early_retry(generator, monkeypatch):
    calls = []

    def worker(batch, config, key):
        calls.append(batch.index)
        raise limited(61)

    monkeypatch.setattr(generator, "_remote_batch", worker)
    with pytest.raises(RateLimitError):
        generator.generate_batch(["first", "never submitted"])
    assert calls == [0]


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


def test_request_timeout_is_explicit(generator, monkeypatch):
    seen = []

    def endpoint(**kwargs):
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

    def endpoint(**kwargs):
        calls.append(kwargs)
        return EmbeddingResponse(data=rows)

    monkeypatch.setattr(
        "compass.compass_signals.user_embedding.embedding._request_embeddings", endpoint
    )
    with pytest.raises(ValueError):
        generator.generate_batch(["first", "never submitted"])
    assert len(calls) == 1


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


def test_retry_budget_does_not_retry_before_server_delay(generator, monkeypatch):
    generator.litellm.retry_timeout = 5
    calls = []

    def worker(batch, config, key):
        calls.append(batch.index)
        raise limited(7)

    monkeypatch.setattr(generator, "_remote_batch", worker)
    with pytest.raises(RateLimitError):
        generator.generate_batch(["a"])
    assert calls == [0]


def test_retry_budget_includes_elapsed_retries(generator, monkeypatch):
    generator.litellm.retry_timeout = 2
    clock = [100.0]
    calls = []

    def worker(batch, config, key):
        calls.append(batch.index)
        if len(calls) > 1:
            clock[0] += 2
        raise limited(1)

    monkeypatch.setattr(generator, "_remote_batch", worker)
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))
    with pytest.raises(RateLimitError):
        generator.generate_batch(["a"])
    assert calls == [0, 0]


def test_expired_retry_budget_after_sleep_does_not_resubmit(generator, monkeypatch):
    generator.litellm.retry_timeout = 2
    clock = [100.0]
    calls = []

    def worker(batch, config, key):
        calls.append(batch.index)
        raise limited(1)

    monkeypatch.setattr(generator, "_remote_batch", worker)
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + 3))
    with pytest.raises(RateLimitError):
        generator.generate_batch(["a"])
    assert calls == [0]


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


def test_millisecond_retry_header_delays_retry(generator, monkeypatch):
    clock = [100.0]
    starts = []
    sleeps = []

    def worker(batch, config, key):
        starts.append((batch.index, clock[0]))
        if len(starts) == 1:
            raise RateLimitError(
                "limited",
                "openai",
                "qwen",
                response=httpx.Response(429, headers={"Retry-After-Ms": "1500"}),
            )
        return result(batch)

    def sleep(delay):
        sleeps.append(delay)
        clock[0] += delay

    monkeypatch.setattr(generator, "_remote_batch", worker)
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.time, "sleep", sleep)
    assert len(generator.generate_batch(["first", "second"])) == 2
    assert sleeps == [1.5]
    assert starts == [(0, 100.0), (0, 101.5), (1, 101.5)]
