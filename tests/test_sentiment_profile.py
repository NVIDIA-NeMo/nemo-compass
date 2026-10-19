# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib.util
import json
from contextlib import ExitStack
from pathlib import Path

import litellm
import numpy as np
import pytest
from tokenizers import Tokenizer, models

import compass.compass_signals.user_sentiment.signal as sentiment_module
from compass.compass_signals.user_embedding.embedding import (
    LiteLLMEmbeddingConfig,
    UserEmbeddingGenerator,
)
from trace_ingest.models import Trace, TraceAggregate, TraceSnapshot

spec = importlib.util.spec_from_file_location(
    "sentiment_profile", Path(__file__).resolve().parents[1] / "tools/profile_user_sentiment.py"
)
assert spec is not None and spec.loader is not None
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def test_profile_preserves_batches_and_restores_endpoint(monkeypatch, tmp_path):
    calls = []

    def endpoint(**kwargs):
        calls.append(kwargs["input"])
        return litellm.EmbeddingResponse(
            data=[
                {"index": i, "embedding": np.eye(1, 4096)[0].tolist()}
                for i in reversed(range(len(kwargs["input"])))
            ]
        )

    monkeypatch.setattr(litellm, "embedding", endpoint)
    generator = UserEmbeddingGenerator(litellm=LiteLLMEmbeddingConfig(model="openai/qwen"))
    generator.remote_tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    measurements = profile.Measurements(tmp_path)
    with ExitStack() as stack:
        profile.instrument(measurements, stack, None)
        results = generator.generate_batch(["hello"] * 101)
    assert litellm.embedding is endpoint
    assert list(map(len, calls)) == [100, 1]
    assert len(results) == 101
    saved = json.loads((tmp_path / "profile.json").read_text())
    assert [r["inputs"] for r in saved["requests"]] == [100, 1]
    assert saved["seconds"]["compress_vectors"]["count"] == 101
    assert saved["tokens"]["count"] == 101
    assert saved["embedding_messages_per_second"] > 0
    assert "hello" not in (tmp_path / "profile.json").read_text()


def test_failed_request_is_recorded_and_raised(monkeypatch, tmp_path):
    def fail(**kwargs):
        raise RuntimeError("sensitive provider details")

    monkeypatch.setattr(litellm, "embedding", fail)
    measurements = profile.Measurements(tmp_path)
    with ExitStack() as stack:
        profile.instrument(measurements, stack, None)
        with pytest.raises(RuntimeError):
            litellm.embedding(
                model="openai/qwen",
                api_base=None,
                api_key=None,
                input=["private text"],
                encoding_format="float",
            )
    saved = (tmp_path / "profile.json").read_text()
    assert json.loads(saved)["requests"][0]["status"] == "failed"
    assert "private text" not in saved
    assert "sensitive provider details" not in saved


def test_replay_checks_coverage_and_retains_repeated_turns(tmp_path):
    replay = tmp_path / "replay.json"
    replay.write_text(json.dumps({"a": ["repeat", "repeat"], "b": []}))
    measurements = profile.Measurements(tmp_path)
    with ExitStack() as stack:
        profile.instrument(measurements, stack, replay)
        snapshot = TraceSnapshot(
            [
                Trace(id="a", root_spans=[], aggregate=TraceAggregate()),
                Trace(id="b", root_spans=[], aggregate=TraceAggregate()),
            ]
        )
        recipe = asyncio.run(sentiment_module.UserMessageExtractor().build_extractor(snapshot))
        assert recipe.extract(snapshot) == {"a": ["repeat", "repeat"], "b": []}
        assert measurements.report["inputs"]["embeddable_messages"] == 2
        with pytest.raises(ValueError, match="exactly"):
            recipe.extract(
                TraceSnapshot([Trace(id="a", root_spans=[], aggregate=TraceAggregate())])
            )
