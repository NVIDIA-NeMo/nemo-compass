# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
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
def generator(monkeypatch):
    gen = UserEmbeddingGenerator(litellm=LiteLLMEmbeddingConfig(model="openai/qwen", batch_size=1))
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


def test_rate_limit_reduces_once_per_generation_and_recovers(generator, monkeypatch, caplog):
    calls = Counter()
    barrier = threading.Barrier(2, timeout=5)
    clock = [100.0]
    sleeps = []

    def worker(batch, config, key):
        calls[batch.index] += 1
        if batch.index in (1, 2) and calls[batch.index] == 1:
            barrier.wait()
            raise limited(7)
        return result(batch)

    def sleep(delay):
        sleeps.append(delay)
        clock[0] += delay

    monkeypatch.setattr(generator, "_remote_batch", worker)
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.time, "sleep", sleep)
    with caplog.at_level(logging.INFO, logger=module.__name__):
        actual = generator.generate_batch([str(i) for i in range(12)])
    assert [x.data for x in actual] == [bytes([i]) for i in range(12)]
    assert calls[1] == calls[2] == 2
    assert sum(calls.values()) == 14
    assert sleeps == [7]
    assert caplog.text.count("after rate limit") == 1
    assert "concurrency 2 -> 1" in caplog.text
    assert caplog.text.count("concurrency 1 -> 2") == 2
    assert "concurrency 3 -> 4" in caplog.text


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


def test_server_error_retries_without_reducing_concurrency(generator, monkeypatch, caplog):
    clock = [100.0]
    calls = Counter()

    def worker(batch, config, key):
        calls[batch.index] += 1
        if batch.index == 1 and calls[1] == 1:
            raise APIError(503, "unavailable", "openai", "qwen")
        return result(batch)

    monkeypatch.setattr(generator, "_remote_batch", worker)
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))
    with caplog.at_level(logging.INFO, logger=module.__name__):
        generator.generate_batch(["a"] * 8)
    assert calls[1] == 2
    assert "after rate limit" not in caplog.text


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


def test_multiple_rate_limits_at_cap_halve_once(generator, monkeypatch, caplog):
    groups = [(0,), (1, 2), (3, 4, 5), (6, 7, 8, 9)]
    barriers = {group: threading.Barrier(len(group), timeout=5) for group in groups}
    calls = Counter()
    clock = [100.0]
    lock = threading.Lock()
    active = peak = 0

    def worker(batch, config, key):
        nonlocal active, peak
        with lock:
            calls[batch.index] += 1
            active += 1
            peak = max(peak, active)
        try:
            if calls[batch.index] == 1 and batch.index < 10:
                group = next(g for g in groups if batch.index in g)
                barriers[group].wait()
                if batch.index >= 6:
                    raise limited(2)
            return result(batch)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(generator, "_remote_batch", worker)
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))
    with caplog.at_level(logging.INFO, logger=module.__name__):
        result_rows = generator.generate_batch(["a"] * 10)
    assert peak == 4
    assert [x.data for x in result_rows] == [bytes([i]) for i in range(10)]
    assert all(calls[i] == 2 for i in range(6, 10))
    assert caplog.text.count("after rate limit") == 1
    assert "concurrency 4 -> 2" in caplog.text


def test_late_success_from_before_rate_limit_does_not_accelerate_recovery(
    generator, monkeypatch, caplog
):
    generator.litellm.max_retry_wait = 0.001
    groups = [(0,), (1, 2), (3, 4, 5), (6, 7, 8, 9)]
    barriers = {group: threading.Barrier(len(group), timeout=5) for group in groups}
    retry_started = threading.Event()
    rate_limit_observed = threading.Event()
    calls = Counter()

    def worker(batch, config, key):
        calls[batch.index] += 1
        if batch.index == 6 and calls[6] == 2:
            retry_started.set()
        elif batch.index < 10:
            group = next(g for g in groups if batch.index in g)
            barriers[group].wait()
            if batch.index == 6:
                raise limited(0)
            if batch.index == 9:
                # Keep an old request alive until work resumes after the reduction.
                assert retry_started.wait(5)
            if batch.index in (7, 8):
                assert rate_limit_observed.wait(5)
        if batch.index == 10:
            assert retry_started.wait(5)
        return result(batch)

    retry_delay = module._retry_delay

    def observe_rate_limit(error, attempt, max_wait):
        delay = retry_delay(error, attempt, max_wait)
        rate_limit_observed.set()
        return delay

    monkeypatch.setattr(module, "_retry_delay", observe_rate_limit)
    monkeypatch.setattr(generator, "_remote_batch", worker)
    with caplog.at_level(logging.INFO, logger=module.__name__):
        actual = generator.generate_batch(["a"] * 11)
    assert [x.data for x in actual] == [bytes([i]) for i in range(11)]
    assert calls[6] == 2
    assert "concurrency 4 -> 2 after rate limit" in caplog.text
    # Only batch 10 is a fresh success after backoff. The late batch 9 and
    # recovered batch 6 must not provide the second success needed to grow.
    recovery_log = caplog.text.split("concurrency 4 -> 2 after rate limit", 1)[1]
    assert "concurrency 2 -> 3" not in recovery_log
