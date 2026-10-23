# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable user-text encoding, PCA projection, and four-bit compression."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from functools import cached_property
from importlib.resources import files
from importlib.util import find_spec
from typing import TYPE_CHECKING

import httpx
import numpy as np
from litellm import APIConnectionError, EmbeddingResponse, Router
from pydantic import BaseModel, ConfigDict, Field
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)


class _NoRetries(int):
    """Zero retries, including in LiteLLM adapters that use ``value or default``."""

    def __new__(cls, value: int = 0) -> _NoRetries:
        return super().__new__(cls, 0)

    def __bool__(self) -> bool:
        # LiteLLM 1.98.0's embedding adapter replaces plain zero with its default.
        # Keep numeric zero for the SDK without changing LiteLLM's global state.
        return True


async def _request_embeddings(
    *, router: Router, input: list[str], timeout: float
) -> EmbeddingResponse:
    return await router.aembedding(
        model="user-sentiment",
        input=input,
        encoding_format="float",
        timeout=timeout,
        max_retries=_NoRetries(),
        num_retries=0,
    )


def _retry_delay(error: Exception, attempt: int, max_wait: float) -> float | None:
    """Return a bounded cooldown, or None when this failure must propagate."""
    if not isinstance(error, APIConnectionError) and getattr(error, "status_code", None) not in {
        408,
        429,
        500,
        502,
        503,
        504,
    }:
        return None
    response = getattr(error, "response", None)
    headers = httpx.Headers(getattr(error, "headers", None))
    if response is not None:
        headers.update(response.headers)
    server_delay = _server_retry_delay(headers)
    # Never retry earlier than a valid server instruction to fit our budget.
    if server_delay > max_wait:
        logger.warning(
            "Embedding server retry delay exceeds the %.2fs wait limit; stopping", max_wait
        )
        return None
    ceiling = min(max_wait, 2.0 ** min(attempt - 1, 30))
    return max(0.0, server_delay, random.uniform(ceiling / 2, ceiling))


def _server_retry_delay(headers: httpx.Headers) -> float:
    """Prefer millisecond precision, as the OpenAI SDK does, then standard Retry-After."""
    for name, divisor in (("retry-after-ms", 1000), ("retry-after", 1)):
        value = headers.get(name)
        if value is None:
            continue
        try:
            delay = float(value) / divisor
        except ValueError:
            if name == "retry-after-ms":
                continue
            try:
                delay = parsedate_to_datetime(value).timestamp() - time.time()
            except (TypeError, ValueError, OverflowError):
                continue
        if math.isfinite(delay) and delay >= 0:
            return delay
    return 0.0


@dataclass
class _RemoteBatch:
    index: int
    inputs: list[str]
    counts: list[int]


if TYPE_CHECKING:
    from transformers import (  # ty: ignore[unresolved-import] -- Optional encoder extra.
        BatchEncoding,
        PreTrainedModel,
        PreTrainedTokenizerBase,
    )


@dataclass(frozen=True)
class UserEmbedding:
    """Compressed vector and token count for the recorded encoder convention."""

    data: bytes
    token_count: int


class UserEmbeddingProjection:
    """Project and compress normalized user embeddings into reusable 132-byte vectors."""

    def __init__(self) -> None:
        folder = files("compass.compass_signals.user_embedding").joinpath("models")
        self.metadata = json.loads(folder.joinpath("embedding.json").read_text())
        with folder.joinpath("embedding.npz").open("rb") as handle:
            with np.load(handle, allow_pickle=False) as arrays:
                self.arrays = {key: arrays[key] for key in arrays.files}

    def compress(self, embedding: np.ndarray) -> bytes:
        x = np.asarray(embedding, dtype="float32")
        if x.shape != (4096,) or not np.isfinite(x).all():
            raise ValueError("Expected one finite 4096-dimensional Qwen embedding")
        if not np.isclose(np.linalg.norm(x), 1, atol=1e-4):
            raise ValueError("Qwen embedding must be L2-normalized")
        a = self.arrays
        # Same projection order as the saved sklearn PCA transform.
        z = x @ a["pca_components"].T - a["pca_mean"] @ a["pca_components"].T
        norm = np.float32(np.linalg.norm(z))
        rotated = (z / norm if norm else z) @ a["rotation"]
        centers = a["centers"]
        codes = np.searchsorted((centers[:-1] + centers[1:]) / 2, rotated).astype("uint8")
        packed = codes[::2] | (codes[1::2] << 4)
        return packed.tobytes() + np.asarray(norm, dtype="<f4").tobytes()

    def decompress(self, packed: bytes) -> np.ndarray:
        """Reconstruct the 256 PCA features for a compatible downstream head."""
        if len(packed) != 132:
            raise ValueError("Expected one 132-byte four-bit MSE vector")
        data = np.frombuffer(packed, dtype="uint8", count=128)
        codes = np.empty(256, dtype="uint8")
        codes[::2], codes[1::2] = data & 15, data >> 4
        norm = np.frombuffer(packed, dtype="<f4", count=1, offset=128)[0]
        if not np.isfinite(norm) or norm < 0:
            raise ValueError("Quantized vector norm must be finite and nonnegative")
        a = self.arrays
        return (a["centers"][codes] @ a["rotation"].T) * norm


def validate_embedding_dependencies() -> None:
    if find_spec("torch") is None or find_spec("transformers") is None:
        raise ValueError(
            'User embeddings require: uv pip install "nemo-compass[local-embedding]" '
            "(from a source checkout: uv sync --extra local-embedding)"
        )


class LiteLLMEmbeddingConfig(BaseModel):
    """Remote endpoint serving Qwen/Qwen3-Embedding-8B with 4096 dimensions."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1, description="LiteLLM model alias for Qwen/Qwen3-Embedding-8B")
    api_base: str | None = None
    api_key_env: str | None = Field(default=None, description="API key environment variable")
    batch_size: int = Field(default=128, ge=1, description="Maximum messages per request")
    max_concurrency: int = Field(
        default=8, ge=1, le=32, description="Maximum concurrent embedding requests"
    )
    max_retries: int = Field(
        default=3, ge=0, le=10, description="Retries per failed embedding batch"
    )
    request_timeout: float = Field(
        default=60, gt=0, allow_inf_nan=False, description="Per-request SDK timeout in seconds"
    )
    retry_timeout: float = Field(
        default=300,
        gt=0,
        allow_inf_nan=False,
        description="Time budget in seconds for scheduling retries after the first failure",
    )
    max_retry_wait: float = Field(
        default=60,
        gt=0,
        allow_inf_nan=False,
        description="Maximum retry delay in seconds; longer Retry-After fails the run",
    )


class UserEmbeddingGenerator:
    """Encode locally or through LiteLLM using the pinned Qwen instruction."""

    def __init__(
        self, device: str | None = None, litellm: LiteLLMEmbeddingConfig | None = None
    ) -> None:
        self.device = device
        self.litellm = litellm
        self.projection = UserEmbeddingProjection()

    @cached_property
    def remote_tokenizer(self) -> Tokenizer:
        encoder = self.projection.metadata["encoder"]
        return Tokenizer.from_pretrained(encoder["model"], revision=encoder["revision"])

    @cached_property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        validate_embedding_dependencies()
        from transformers import AutoTokenizer  # ty: ignore[unresolved-import]

        encoder = self.projection.metadata["encoder"]
        tokenizer = AutoTokenizer.from_pretrained(
            encoder["model"], revision=encoder["revision"], padding_side="left"
        )
        if tokenizer is None:
            raise ValueError("The encoder checkpoint did not provide a tokenizer")
        return tokenizer

    @cached_property
    def model(self) -> PreTrainedModel:
        validate_embedding_dependencies()
        import torch  # ty: ignore[unresolved-import] -- Optional encoder extra.
        from transformers import AutoModel  # ty: ignore[unresolved-import]

        device = self.device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        dtype = torch.float32 if torch.device(device).type == "cpu" else torch.bfloat16
        encoder = self.projection.metadata["encoder"]
        model = (
            AutoModel.from_pretrained(
                encoder["model"],
                revision=encoder["revision"],
                dtype=dtype,
                attn_implementation="sdpa",
                use_safetensors=True,
            )
            .to(device)
            .eval()
        )
        model.requires_grad_(False)
        model.config.use_cache = False
        return model

    def _tokens(self, text: str) -> BatchEncoding:
        return self.tokenizer(
            self.projection.metadata["encoder"]["prefix"] + text,
            truncation=False,
            return_tensors="pt",
            verbose=False,
        )

    def token_count(self, text: str) -> int:
        return self._tokens(text)["input_ids"].shape[-1]

    def embed(self, text: str) -> np.ndarray:
        """Return one float32 normalized vector, using the pinned input convention."""
        import torch  # ty: ignore[unresolved-import] -- Optional encoder extra.

        tokens = self._tokens(text)
        if tokens["input_ids"].shape[-1] > self.projection.metadata["max_tokens"]:
            raise ValueError("User message exceeds the 8192-token budget; input was not truncated")
        with torch.inference_mode():
            hidden = self.model(**tokens.to(self.model.device)).last_hidden_state
            return torch.nn.functional.normalize(hidden[:, -1].float(), dim=-1)[0].cpu().numpy()

    def generate_local(self, texts: list[str]) -> list[UserEmbedding]:
        """Encode locally, preserving the per-message character limit."""
        return [
            UserEmbedding(
                data=self.projection.compress(self.embed(text[:20000])),
                token_count=self.token_count(text[:20000]),
            )
            for text in texts
        ]

    def _remote_batches(self, texts: list[str], size: int) -> Iterator[_RemoteBatch]:
        # Prepare only the next batch, on the coordinator thread.
        for start in range(0, len(texts), size):
            inputs = [
                self.projection.metadata["encoder"]["prefix"] + text[:20000]
                for text in texts[start : start + size]
            ]
            counts = [len(tokens.ids) for tokens in self.remote_tokenizer.encode_batch(inputs)]
            if any(count > self.projection.metadata["max_tokens"] for count in counts):
                raise ValueError(
                    "User message exceeds the 8192-token budget; input was not truncated"
                )
            yield _RemoteBatch(start, inputs, counts)

    def generate_remote(self, texts: list[str]) -> list[UserEmbedding]:
        """Encode with Router concurrency limits and independently retried batches."""
        config = self.litellm
        if config is None:
            raise ValueError("Remote embeddings require LiteLLM configuration")
        if not texts:
            return []
        return asyncio.run(self._generate_remote_async(texts, config))

    async def _generate_remote_async(
        self, texts: list[str], config: LiteLLMEmbeddingConfig
    ) -> list[UserEmbedding]:
        import litellm

        batches = list(self._remote_batches(texts, config.batch_size))
        router = Router(
            model_list=[
                {
                    "model_name": "user-sentiment",
                    "litellm_params": {
                        "model": config.model,
                        "api_base": config.api_base,
                        "api_key": os.environ[config.api_key_env] if config.api_key_env else None,
                        "max_parallel_requests": config.max_concurrency,
                        "max_retries": _NoRetries(),
                    },
                }
            ],
            num_retries=0,
            disable_cooldowns=True,
        )
        tasks = [
            asyncio.create_task(self._remote_batch_with_retries(batch, config, router))
            for batch in batches
        ]
        try:
            results = await asyncio.gather(*tasks)
            return [item for batch in results for item in batch]
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            # Router.reset() clears unrelated applications' callbacks too.
            litellm.logging_callback_manager.remove_callback_from_all_lists(
                router, require_self=True
            )

    async def _remote_batch_with_retries(
        self, batch: _RemoteBatch, config: LiteLLMEmbeddingConfig, router: Router
    ) -> list[UserEmbedding]:
        attempt = 0
        deadline = None
        while True:
            attempt += 1
            try:
                return await self._remote_batch(batch, config, router)
            except Exception as error:
                if attempt > config.max_retries:
                    raise
                delay = _retry_delay(error, attempt, config.max_retry_wait)
                now = time.monotonic()
                if deadline is None:
                    deadline = now + config.retry_timeout
                if delay is None or now + delay >= deadline:
                    raise
                logger.warning(
                    "Embedding batch %d retry %d/%d: %s; waiting %.2fs",
                    batch.index // config.batch_size,
                    attempt,
                    config.max_retries,
                    type(error).__name__,
                    delay,
                )
                await asyncio.sleep(delay)
                if time.monotonic() >= deadline:
                    raise

    async def _remote_batch(
        self, batch: _RemoteBatch, config: LiteLLMEmbeddingConfig, router: Router
    ) -> list[UserEmbedding]:
        response = await _request_embeddings(
            router=router,
            input=batch.inputs,
            timeout=config.request_timeout,
        )
        rows = sorted(response.data, key=lambda row: row["index"])
        if [row["index"] for row in rows] != list(range(len(batch.inputs))):
            raise ValueError("Embedding response must contain one vector per input")
        results = []
        for row, count in zip(rows, batch.counts, strict=True):
            vector = np.asarray(row["embedding"], dtype="float32")
            norm = np.linalg.norm(vector)
            if not np.isfinite(norm) or norm == 0:
                raise ValueError("Embedding response must contain finite, nonzero vectors")
            results.append(UserEmbedding(self.projection.compress(vector / norm), count))
        return results

    def generate_batch(self, texts: list[str]) -> list[UserEmbedding]:
        """Batch remote requests while preserving input order and local token limits."""
        if self.litellm is None:
            return self.generate_local(texts)
        return self.generate_remote(texts)
