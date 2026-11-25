# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from litellm import EmbeddingResponse
from tokenizers import Tokenizer, models, pre_tokenizers

from compass.compass_signals.user_embedding.embedding import (
    LiteLLMEmbeddingConfig,
    UserEmbedding,
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


def test_screening_reuses_exact_inputs_preserving_occurrences(monkeypatch):
    generator = UserEmbeddingGenerator(
        litellm=LiteLLMEmbeddingConfig(model="openai/qwen", batch_size=4)
    )
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

    monkeypatch.setattr(
        "compass.compass_signals.user_embedding.embedding._request_embeddings", embed
    )
    long = "x" * 20000
    messages: dict[str, list[str]] = {
        "a": ["message"] * 8 + [" spaced ", "spaced", long + "a"],
        "b": ["", "last", long + "b", "message"],
        "c": [],
        "d": [" " * 20000 + "ignored"],
    }
    results = screen_user_messages(messages, generator, ComplaintClassifier(generator.projection))
    assert sorted(map(len, calls)) == [1, 4]
    prefix = generator.projection.metadata["encoder"]["prefix"]
    assert {text for call in calls for text in call} == {
        prefix + text for text in ["message", " spaced ", "spaced", long, "last"]
    }
    assert len(results["a"].message_results) == 11
    assert results["b"].message_results[0].label == "no_user_messages"
    assert results["b"].message_results[1].token_count == 1
    assert results["c"].message_results == []
    assert results["d"].message_results[0].label == "no_user_messages"
    assert results["a"].message_results[-1] == results["b"].message_results[2]
    assert results["a"].message_results[0] == results["b"].message_results[-1]
    results["a"].message_results[0].scores.clear()
    assert results["b"].message_results[-1].scores  # Occurrences don't share mutable state.
    assert messages["a"][-1] == long + "a"  # Evidence remains verbatim.
    assert "model" not in generator.__dict__


def test_remote_rejects_oversized_input_before_request_and_propagates_failures(monkeypatch):
    generator = UserEmbeddingGenerator(
        litellm=LiteLLMEmbeddingConfig(model="openai/qwen", batch_size=4)
    )
    generator.remote_tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    generator.remote_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    def fail(**kwargs):
        raise RuntimeError("Provider unavailable")

    monkeypatch.setattr(
        "compass.compass_signals.user_embedding.embedding._request_embeddings", fail
    )
    with pytest.raises(ValueError, match="8192-token"):
        generator.generate_batch(["a " * 9000])
    with pytest.raises(RuntimeError, match="Provider unavailable"):
        screen_user_messages({"a": ["hello"]}, generator, ComplaintClassifier(generator.projection))


@pytest.mark.parametrize("remote", [False, True])
def test_embedding_cache_persists_and_encodes_only_misses(tmp_path, monkeypatch, remote):
    path = tmp_path / "nested" / "embeddings.sqlite3"
    calls = []

    def generate(self, texts):
        calls.append(texts)
        return [UserEmbedding(bytes([len(text) % 256]) * 132, len(text)) for text in texts]

    def batches(self, texts):
        results = generate(self, texts)
        for i in reversed(range(len(results))):
            yield i, [results[i]]

    monkeypatch.setattr(UserEmbeddingGenerator, "generate_local", generate)
    monkeypatch.setattr(UserEmbeddingGenerator, "_generate_remote_batches", batches)

    def fresh():
        return UserEmbeddingGenerator(
            litellm=LiteLLMEmbeddingConfig(model="openai/qwen") if remote else None,
            cache_path=path,
        )

    first = fresh().generate_batch(["one", "three"])
    second = fresh().generate_batch(["three", "new!", "one", "one"])
    assert second == [first[1], UserEmbedding(bytes([4]) * 132, 4), first[0], first[0]]
    assert [text for batch in calls for text in batch] == ["one", "three", "new!"]
    warm = fresh()
    assert warm.generate_batch(["one"]) == [first[0]]
    assert "model" not in warm.__dict__ and "remote_tokenizer" not in warm.__dict__
    assert not path.with_name(path.name + "-journal").exists()
    long = "x" * 20000
    fresh().generate_batch([long + "a", " spaced "])
    fresh().generate_batch([long + "b", " spaced ", "spaced"])
    assert [text for batch in calls for text in batch][-3:] == [long + "a", " spaced ", "spaced"]


def test_embedding_cache_returns_the_stored_winner(tmp_path, monkeypatch):
    winner = UserEmbedding(b"w" * 132, 1)
    path = tmp_path / "shared.sqlite3"
    calls = 0

    def generate(self, texts):
        nonlocal calls
        calls += 1
        if calls == 1:
            # A competing run commits after this run's lookup, before its insert.
            assert UserEmbeddingGenerator(cache_path=path).generate_batch(texts) == [winner]
            return [UserEmbedding(b"l" * 132, 2)]
        return [winner]

    monkeypatch.setattr(UserEmbeddingGenerator, "generate_local", generate)
    assert UserEmbeddingGenerator(cache_path=path).generate_batch(["same"]) == [winner]
    assert UserEmbeddingGenerator(cache_path=path).generate_batch(["same"]) == [winner]
    assert calls == 2


@pytest.mark.parametrize(
    "change", ["metadata", "weights", "model", "endpoint", "backend", "operations", "version"]
)
def test_embedding_cache_representation_identity(tmp_path, monkeypatch, change):
    import compass.compass_signals.user_embedding.embedding as module

    calls = []

    def generate(self, texts):
        calls.append(texts)
        return [UserEmbedding(b"x" * 132, 1) for _ in texts]

    monkeypatch.setattr(
        UserEmbeddingGenerator,
        "_generate_remote_batches",
        lambda self, texts: ((0, generate(self, texts)) for _ in range(1)),
    )
    monkeypatch.setattr(UserEmbeddingGenerator, "generate_local", generate)
    config = LiteLLMEmbeddingConfig(model="openai/qwen", api_base="https://example.com")
    path = tmp_path / "embeddings.sqlite3"
    UserEmbeddingGenerator(litellm=config, cache_path=path).generate_batch(["hello"])
    modified = UserEmbeddingGenerator(litellm=config.model_copy(), cache_path=path)
    assert modified.litellm is not None
    if change == "metadata":
        modified.projection.metadata["encoder"]["prefix"] += "changed"
    elif change == "weights":
        folder = tmp_path / "models"
        folder.mkdir()
        original = (
            module.files("compass.compass_signals.user_embedding")
            .joinpath("models", "embedding.npz")
            .read_bytes()
        )
        (folder / "embedding.npz").write_bytes(original + b"changed")
        monkeypatch.setattr(module, "files", lambda _: tmp_path)
    elif change == "model":
        modified.litellm.model = "openai/other"
    elif change == "endpoint":
        modified.litellm.api_base = "https://other.example.com"
    elif change == "backend":
        modified.litellm = None
    elif change == "version":
        monkeypatch.setattr("compass.version.__version__", "next-release")
    else:
        modified.litellm.batch_size = 1
        modified.litellm.max_concurrency = 1
        modified.litellm.api_key_env = "OTHER_KEY"
    modified.generate_batch(["hello"])
    assert len(calls) == (1 if change in ("operations", "version") else 2)


@pytest.mark.parametrize("remote", [False, True])
def test_embedding_cache_resumes_after_failed_batch(tmp_path, monkeypatch, remote):
    calls = []

    def generate(self, texts):
        calls.extend(texts)
        if texts == ["fail"] and calls.count("fail") == 1:
            raise RuntimeError("Provider unavailable")
        return [UserEmbedding(bytes([len(text)]) * 132, len(text)) for text in texts]

    monkeypatch.setattr(UserEmbeddingGenerator, "generate_local", generate)

    def remote_batch(self, batch, config, key):
        prefix = self.projection.metadata["encoder"]["prefix"]
        return generate(self, [text.removeprefix(prefix) for text in batch.inputs])

    monkeypatch.setattr(UserEmbeddingGenerator, "_remote_batch", remote_batch)

    def fresh():
        generator = UserEmbeddingGenerator(
            litellm=LiteLLMEmbeddingConfig(model="openai/qwen", batch_size=1, max_concurrency=1)
            if remote
            else None,
            cache_path=tmp_path / "embeddings.sqlite3",
        )
        generator.remote_tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        return generator

    with pytest.raises(RuntimeError, match="Provider unavailable"):
        fresh().generate_batch(["one", "fail", "three"])
    expected = [UserEmbedding(bytes([n]) * 132, n) for n in (5, 3, 4)]
    assert fresh().generate_batch(["three", "one", "fail"]) == expected
    assert calls == ["one", "fail", "three", "fail"]
    assert fresh().generate_batch(["three", "one", "fail"]) == expected
    assert calls == ["one", "fail", "three", "fail"]
