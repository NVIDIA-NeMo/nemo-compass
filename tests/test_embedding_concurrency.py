# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading

import numpy as np
import pytest
from litellm import EmbeddingResponse
from tokenizers import Tokenizer, models

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


@pytest.mark.parametrize("concurrency", [1, 3])
def test_fixed_pool_starts_at_configured_limit(generator, monkeypatch, concurrency):
    generator.litellm = LiteLLMEmbeddingConfig(
        model="qwen",
        batch_size=1,
        max_concurrency=concurrency,
    )
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
    actual = generator.generate_batch(["a"] * (concurrency * 2))
    assert peak == concurrency
    assert [x.data for x in actual] == [bytes([i]) for i in range(concurrency * 2)]


def test_invalid_response_stops_new_batches(generator, monkeypatch):
    calls = []

    def endpoint(**kwargs):
        calls.append(kwargs)
        return EmbeddingResponse(data=[])

    monkeypatch.setattr(
        "compass.compass_signals.user_embedding.embedding._request_embeddings", endpoint
    )
    with pytest.raises(ValueError):
        generator.generate_batch(["first", "never submitted"])
    assert len(calls) == 1


@pytest.mark.parametrize(
    "failure,retries,recover,headers,expected_attempts",
    [
        (429, 0, True, {}, 1),
        (429, 2, False, {}, 3),
        (503, 1, True, {}, 2),
        (401, 3, True, {}, 1),
        ("timeout", 1, True, {}, 2),
        (429, 1, True, {"Retry-After": "1"}, 2),
        (429, 1, True, {"retry-after-ms": "100"}, 2),
    ],
)
def test_litellm_retries_over_http(
    generator, monkeypatch, failure, retries, recover, headers, expected_attempts
):
    import json
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    import litellm

    received = []
    default_retries = litellm.DEFAULT_MAX_RETRIES
    monkeypatch.setattr(litellm, "num_retries", 2)

    class Endpoint(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            received.append(time.monotonic())
            failing = len(received) == 1 or not recover
            if failing and failure == "timeout":
                time.sleep(0.3)
                return
            status = failure if failing else 200
            body = (
                {"error": {"message": "test failure", "type": "test_error"}}
                if failing
                else {
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
            )
            encoded = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            for name, value in headers.items():
                self.send_header(name, value)
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
        request_timeout=0.1 if failure == "timeout" else 5,
    )
    try:
        if retries and recover and failure != 401:
            assert len(generator.generate_batch(["hello"])) == 1
        else:
            with pytest.raises((litellm.RateLimitError, litellm.AuthenticationError)):
                generator.generate_batch(["hello"])
        assert len(received) == expected_attempts
        if headers:
            minimum_wait = 1 if "Retry-After" in headers else 0.1
            assert received[1] - received[0] >= minimum_wait
        assert litellm.DEFAULT_MAX_RETRIES == default_retries
        assert litellm.num_retries == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
