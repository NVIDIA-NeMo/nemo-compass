# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from litellm import EmbeddingResponse
from tokenizers import Tokenizer, models, pre_tokenizers

from compass.compass_signals.user_embedding.embedding import (
    LiteLLMEmbeddingConfig,
    UserEmbeddingGenerator,
)
from compass.compass_signals.user_sentiment.classifier import ComplaintClassifier
from compass.compass_signals.user_sentiment.signal import screen_user_messages


def test_local_batch_preserves_character_limit(monkeypatch):
    generator = UserEmbeddingGenerator()
    encoded = []

    def embed(text):
        encoded.append(text)
        return np.eye(1, 4096)[0]

    monkeypatch.setattr(generator, "embed", embed)
    monkeypatch.setattr(generator, "token_count", len)
    batch = generator.generate_batch(["hello", "x" * 20001])
    assert encoded == ["hello", "x" * 20000]
    assert [item.token_count for item in batch] == [5, 20000]
    assert generator.generate_batch([]) == []


def test_remote_batches_preserve_trace_order_and_skip_empty_messages(monkeypatch):
    monkeypatch.setattr("compass.compass_signals.user_embedding.embedding._BATCH_SIZE", 4)
    generator = UserEmbeddingGenerator(litellm=LiteLLMEmbeddingConfig(model="openai/qwen"))
    generator.remote_tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    calls = []

    def embed(**kwargs):
        calls.append(kwargs["input"])
        # Reversed response order and non-unit vectors exercise both requirements.
        return EmbeddingResponse(
            data=[
                {"index": i, "embedding": (np.eye(1, 4096, i)[0] * 2).tolist()}
                for i in reversed(range(len(kwargs["input"])))
            ]
        )

    monkeypatch.setattr("litellm.embedding", embed)
    messages = {"a": ["message"] * 8, "b": ["", "last"], "c": []}
    results = screen_user_messages(messages, generator, ComplaintClassifier(generator.projection))
    assert list(map(len, calls)) == [4, 4, 1]
    assert all(
        text.startswith(generator.projection.metadata["encoder"]["prefix"])
        for call in calls
        for text in call
    )
    assert len(results["a"].message_results) == 8
    assert results["b"].message_results[0].label == "no_user_messages"
    assert results["b"].message_results[1].token_count == 1
    assert results["c"].message_results == []
    assert "model" not in generator.__dict__


def test_remote_rejects_oversized_input_before_request_and_propagates_failures(monkeypatch):
    generator = UserEmbeddingGenerator(litellm=LiteLLMEmbeddingConfig(model="openai/qwen"))
    generator.remote_tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    generator.remote_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    def fail(**kwargs):
        raise RuntimeError("Provider unavailable")

    monkeypatch.setattr("litellm.embedding", fail)
    with pytest.raises(ValueError, match="8192-token"):
        generator.generate_batch(["a " * 9000])
    with pytest.raises(RuntimeError, match="Provider unavailable"):
        screen_user_messages({"a": ["hello"]}, generator, ComplaintClassifier(generator.projection))
