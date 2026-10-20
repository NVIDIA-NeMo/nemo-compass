# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Complaint scoring from reusable compressed user embeddings."""

import json
from importlib.resources import files
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from compass.compass_signals.user_embedding.embedding import (
    UserEmbedding,
    UserEmbeddingProjection,
)


class ScreeningResult(BaseModel):
    """One user message's label, or why scoring was skipped; scores are uncalibrated."""

    label: Literal["complaint", "no_complaint", "no_user_messages"]
    token_count: int = 0
    scores: dict[str, float] = Field(default_factory=dict)


class ComplaintClassifier:
    """Apply the trained logistic head to a shared embedding representation."""

    def __init__(self, projection: UserEmbeddingProjection) -> None:
        self.projection = projection
        folder = files("compass.compass_signals.user_sentiment").joinpath("models")
        self.metadata = json.loads(folder.joinpath("complaint.json").read_text())
        with folder.joinpath("complaint.npz").open("rb") as handle:
            with np.load(handle, allow_pickle=False) as arrays:
                self.weight = arrays["weight"]
                self.bias = arrays["bias"]

    def classify(self, embedding: UserEmbedding) -> ScreeningResult:
        features = self.projection.decompress(embedding.data)
        logit = features @ self.weight + self.bias
        score = float(np.exp(-np.logaddexp(0, -logit)))
        return ScreeningResult(
            label="complaint" if score >= self.metadata["threshold"] else "no_complaint",
            token_count=embedding.token_count,
            scores={"complaint": score, "no_complaint": 1 - score},
        )
