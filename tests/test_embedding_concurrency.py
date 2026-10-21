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
    monkeypatch.setattr(module, "_BATCH_SIZE", 1)
    gen = UserEmbeddingGenerator(litellm=LiteLLMEmbeddingConfig(model="openai/qwen"))
    gen.remote_tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    return gen


def result(batch):
    return [UserEmbedding(bytes([batch.index]), batch.counts[0])]


def limited(value=None):
    headers = {} if value is None else {"Retry-After": str(value)}
    return RateLimitError(
        "limited", "openai", "qwen", response=httpx.Response(429, headers=headers)
    )


def test_ramps_to_cap_with_real_overlap_and_preserves_order(generator, monkeypatch):
    # Clean rounds have 1, 2, 3, 4, 4 requests. Barriers would time out if
    # execution were sequential or a round exceeded its limit.
    groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]]
    barriers = {group[0]: threading.Barrier(len(group), timeout=5) for group in groups}
    done = {i: threading.Event() for i in range(14)}
    completion_order = []

    def worker(batch, config, key):
        group = next(group for group in groups if batch.index in group)
        barriers[group[0]].wait()
        if batch.index != group[-1]:
            assert done[batch.index + 1].wait(5)
        completion_order.append(batch.index)
        done[batch.index].set()
        return result(batch)

    monkeypatch.setattr(generator, "_remote_batch", worker)
    actual = generator.generate_batch([str(i) for i in range(14)])
    assert completion_order == [i for group in groups for i in reversed(group)]
    assert [x.data for x in actual] == [bytes([i]) for i in range(14)]


def test_backoff_drains_round_keeps_successes_and_recovers(generator, monkeypatch, caplog):
    calls = Counter()
    cooldowns = []
    finished = set()
    barrier = threading.Barrier(3, timeout=5)

    def worker(batch, config, key):
        calls[batch.index] += 1
        if batch.index in (3, 4, 5) and calls[batch.index] == 1:
            barrier.wait()
            if batch.index == 3:
                raise limited(7)
        finished.add(batch.index)
        return result(batch)

    def sleep(delay):
        assert {4, 5} <= finished
        assert calls[3] == 1
        assert calls[6] == 0
        cooldowns.append(delay)

    monkeypatch.setattr(generator, "_remote_batch", worker)
    monkeypatch.setattr(module.time, "sleep", sleep)
    with caplog.at_level(logging.INFO, logger=module.__name__):
        actual = generator.generate_batch([str(i) for i in range(10)])
    assert [x.data for x in actual] == [bytes([i]) for i in range(10)]
    assert calls == Counter({**dict.fromkeys(range(10), 1), 3: 2})
    assert cooldowns == [7]
    assert "concurrency 3 -> 1" in caplog.text
    assert caplog.text.count("concurrency 1 -> 2") == 2


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
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
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
    generator.litellm = LiteLLMEmbeddingConfig(model="qwen", max_concurrency=1, max_retries=0)
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
