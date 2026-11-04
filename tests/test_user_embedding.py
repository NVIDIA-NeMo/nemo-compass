# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from litellm import EmbeddingResponse
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
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


def test_screening_reuses_exact_inputs_preserving_occurrences_and_trace_context(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    monkeypatch.setattr("compass.compass_signals.user_embedding.embedding.tracer", tracer)
    monkeypatch.setattr("compass.compass_signals.user_sentiment.signal.tracer", tracer)
    generator = UserEmbeddingGenerator(
        litellm=LiteLLMEmbeddingConfig(model="openai/qwen", batch_size=4)
    )
    generator.remote_tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    calls = []
    request_parents = []

    def embed(**kwargs):
        calls.append(kwargs["input"])
        request_parents.append(trace.get_current_span().get_span_context().span_id)
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
    with tracer.start_as_current_span("signal.user_sentiment") as parent:
        results = screen_user_messages(
            messages, generator, ComplaintClassifier(generator.projection)
        )
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
    spans = exporter.get_finished_spans()
    embeddings = next(span for span in spans if span.name == "sentiment.embeddings")
    batches = [span for span in spans if span.name == "sentiment.embedding_batch"]
    assert embeddings.parent is not None
    assert embeddings.attributes is not None
    assert embeddings.parent.span_id == parent.get_span_context().span_id
    assert embeddings.attributes["unique_input_count"] == 5
    assert len(batches) == 2
    assert all(
        span.parent is not None and span.parent.span_id == embeddings.context.span_id
        for span in batches
    )
    assert set(request_parents) == {span.context.span_id for span in batches}
    provider.shutdown()


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
