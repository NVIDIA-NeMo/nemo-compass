# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile the real sentiment signal without changing its batching or retry behavior."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import subprocess
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterator
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any, ParamSpec, TypeVar
from unittest.mock import patch

import litellm
import numpy as np
from dotenv import load_dotenv
from nooa.unifiedllm import CompletionClient, UnifiedLLM
from pydantic import TypeAdapter

import compass.compass_signals.user_sentiment.signal as sentiment_module
from compass.compass_signals.issue_detector import IssueDetector
from compass.compass_signals.user_embedding.embedding import (
    LiteLLMEmbeddingConfig,
    UserEmbedding,
    UserEmbeddingGenerator,
    UserEmbeddingProjection,
)
from compass.compass_signals.user_sentiment.classifier import ComplaintClassifier
from compass.compass_signals.user_sentiment.signal import (
    UserMessageExtraction,
    UserMessageExtractor,
    UserSentimentCompassSignal,
    UserSentimentConfig,
)
from trace_ingest.loaders.fs import FSDataLoader
from trace_ingest.models import TraceSnapshot

P = ParamSpec("P")
T = TypeVar("T")


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "total": sum(values),
        "min": min(values),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": max(values),
    }


class Measurements:
    """Nested wall-clock timings; parent durations include their children."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.durations: dict[str, list[float]] = defaultdict(list)
        self.report: dict[str, Any] = {"status": "running", "requests": [], "active_stages": []}

    def save(self) -> None:
        self.report["seconds"] = {k: distribution(v) for k, v in self.durations.items()}
        temporary = self.directory / "profile.json.tmp"
        temporary.write_text(json.dumps(self.report, indent=2) + "\n")
        temporary.replace(self.directory / "profile.json")

    @contextmanager
    def measure(self, name: str, *, checkpoint: bool = True) -> Iterator[None]:
        if checkpoint:
            print(f"Starting {name}", flush=True)
            self.report["active_stages"].append(name)
            self.save()
        start = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - start
            self.durations[name].append(elapsed)
            if checkpoint:
                self.report["active_stages"].remove(name)
                self.save()
                print(f"{name}: {elapsed:.3f}s", flush=True)

    def sync(self, name: str, function: Callable[P, T]) -> Callable[P, T]:
        @wraps(function)
        def measured(*args: P.args, **kwargs: P.kwargs) -> T:
            with self.measure(name, checkpoint=False):
                return function(*args, **kwargs)

        return measured

    def asynchronous(
        self, name: str, function: Callable[P, Awaitable[T]]
    ) -> Callable[P, Awaitable[T]]:
        @wraps(function)
        async def measured(*args: P.args, **kwargs: P.kwargs) -> T:
            with self.measure(name):
                return await function(*args, **kwargs)

        return measured


def instrument(measurements: Measurements, stack: ExitStack, replay: Path | None) -> None:
    original_extract = UserMessageExtractor.build_extractor
    original_embedding = litellm.embedding
    original_generate = UserEmbeddingGenerator.generate_batch

    async def extract(
        agent: UserMessageExtractor | None, snapshot: TraceSnapshot
    ) -> UserMessageExtraction:
        if replay is None:
            assert agent is not None
            with measurements.measure("build_extractor"):
                recipe = await original_extract(agent, snapshot)
            function = recipe.extract
        else:
            messages = TypeAdapter(dict[str, list[str]]).validate_json(
                replay.read_bytes(), strict=True
            )

            def function(_: TraceSnapshot) -> dict[str, list[str]]:
                return messages

        def execute(snapshot: TraceSnapshot) -> dict[str, list[str]]:
            with measurements.measure("extract_messages"):
                messages = function(snapshot)
                messages = TypeAdapter(dict[str, list[str]]).validate_python(messages, strict=True)
                if messages.keys() != snapshot.traces_by_id.keys():
                    raise ValueError("Saved messages must cover exactly the selected trace IDs")
            path = measurements.directory / "messages.json"
            path.write_text(json.dumps(messages, ensure_ascii=False) + "\n")
            texts = [
                text for values in messages.values() for text in values if text[:20000].strip()
            ]
            measurements.report["inputs"] = {
                "messages_sha256": digest(path),
                "traces": len(messages),
                "traces_with_messages": sum(bool(v) for v in messages.values()),
                "embeddable_messages": len(texts),
                "distinct_texts": len(set(texts)),
                "characters": distribution([float(len(t)) for t in texts]),
                "messages_over_character_cap": sum(len(t) > 20000 for t in texts),
            }
            measurements.save()
            return messages

        return UserMessageExtraction(extract=execute)

    def embedding(
        *,
        model: str,
        api_base: str | None,
        api_key: str | None,
        input: list[str],
        encoding_format: str,
    ) -> litellm.EmbeddingResponse:
        record: dict[str, Any] = {"inputs": len(input), "status": "running"}
        measurements.report["requests"].append(record)
        measurements.save()
        start = perf_counter()
        try:
            result = original_embedding(
                model=model,
                api_base=api_base,
                api_key=api_key,
                input=input,
                encoding_format=encoding_format,
            )
            record["status"] = "success"
            return result
        except Exception as error:
            record.update(
                status="failed",
                error_type=type(error).__name__,
                status_code=getattr(error, "status_code", None),
            )
            raise
        finally:
            record["seconds"] = perf_counter() - start
            measurements.durations["embedding_client_call"].append(record["seconds"])
            measurements.save()
            print(
                f"Embedding batch {len(measurements.report['requests'])}: {record['status']} ({record['seconds']:.3f}s)",
                flush=True,
            )

    def generate(generator: UserEmbeddingGenerator, texts: list[str]) -> list[UserEmbedding]:
        with measurements.measure("generate_embeddings"):
            result = original_generate(generator, texts)
        measurements.report["tokens"] = distribution([float(item.token_count) for item in result])
        seconds = measurements.durations["generate_embeddings"][-1]
        measurements.report["embedding_messages_per_second"] = (
            len(result) / seconds if seconds else None
        )
        measurements.save()
        return result

    def extractor_factory(*, llm: UnifiedLLM | None = None) -> SimpleNamespace:
        agent = None
        if replay is None:
            assert llm is not None
            agent = UserMessageExtractor(llm=llm)
        return SimpleNamespace(build_extractor=lambda snapshot: extract(agent, snapshot))

    def detector_factory(*, llm: UnifiedLLM) -> SimpleNamespace:
        detector = IssueDetector(llm=llm)
        return SimpleNamespace(
            detect_issues=measurements.asynchronous("detect_issues", detector.detect_issues)
        )

    stack.enter_context(patch.object(sentiment_module, "UserMessageExtractor", extractor_factory))
    stack.enter_context(patch.object(UserEmbeddingGenerator, "generate_batch", generate))
    stack.enter_context(patch.object(litellm, "embedding", embedding))
    for cls, method, name in [
        (UserEmbeddingProjection, "compress", "compress_vectors"),
        (ComplaintClassifier, "classify", "classify_messages"),
    ]:
        stack.enter_context(
            patch.object(cls, method, measurements.sync(name, getattr(cls, method)))
        )
    stack.enter_context(patch.object(sentiment_module, "IssueDetector", detector_factory))


async def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    measurements = Measurements(args.output)
    measurements.report.update(
        started_at=datetime.now(UTC).isoformat(),
        profiler_sha256=digest(Path(__file__)),
        dataset_sha256=digest(args.dataset),
        dataset=str(args.dataset.resolve()),
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        working_tree_dirty=bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True)
        ),
        python=platform.python_version(),
        model=args.model,
        embedding_model=args.embedding_model,
        api_base=args.api_base,
        embedding_api_base=args.embedding_api_base,
        replay_messages_sha256=digest(args.messages) if args.messages else None,
        max_traces=args.max_traces,
        timing_note="Nested wall times; embedding_client_call includes SDK overhead and any internal retries, not server-only latency.",
    )
    measurements.save()
    try:
        with measurements.measure("load_dataset"):
            snapshot = FSDataLoader(args.dataset).load()
            if args.max_traces:
                snapshot = TraceSnapshot(list(snapshot.traces_by_id.values())[: args.max_traces])
        llm = CompletionClient(
            model=args.model,
            api_base=args.api_base,
            api_key=os.environ[args.api_key_env],
            max_tokens=16384,
            reasoning_effort="medium",
            drop_params=True,
        )
        signal = UserSentimentCompassSignal(
            UserSentimentConfig(
                litellm=LiteLLMEmbeddingConfig(
                    model=args.embedding_model,
                    api_base=args.embedding_api_base,
                    api_key_env=args.api_key_env,
                )
            ),
            llm,
        )
        signal.check_prerequisites(snapshot)
        with ExitStack() as stack:
            instrument(measurements, stack, args.messages)
            with measurements.measure("sentiment_total"):
                result = await signal.analyze(snapshot)
        (args.output / "result.json").write_text(result.model_dump_json(indent=2) + "\n")
        measurements.report.update(
            status="success",
            candidate_traces=result.finding_count,
            problems=len(result.problems),
            skip_reason=result.skip_reason,
        )
    except BaseException as error:
        measurements.report.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        measurements.save()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory for local artifacts"
    )
    parser.add_argument("--messages", type=Path, help="Replay messages.json from an earlier run")
    parser.add_argument("--max-traces", type=int)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-api-base", required=True)
    parser.add_argument("--api-key-env", default="COMPASS_API_KEY")
    args = parser.parse_args()
    if args.max_traces is not None and args.max_traces < 1:
        parser.error("--max-traces must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
