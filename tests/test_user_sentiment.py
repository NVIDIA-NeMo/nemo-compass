# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import builtins
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from nooa.unifiedllm import FakeLLMClient
from pydantic import ValidationError

from compass.compass_signals.user_sentiment import signal
from compass.compass_signals.user_sentiment.classifier import (
    ComplaintClassifier,
    DecisionsConfig,
    UserSentimentConfig,
)
from compass.traces import Trace, TraceAggregate, TraceSnapshot


def decision(label="complaint"):
    return {
        "answers": [
            {
                "type": "choice",
                "name": "user_sentiment",
                "choice": label,
                "probabilities": [
                    {"value": "complaint", "probability": 0.7 if label == "complaint" else 0.2},
                    {"value": "no_complaint", "probability": 0.3 if label == "complaint" else 0.8},
                ],
            }
        ],
        "usage": {"input_tokens": 30, "output_tokens": 0},
    }


def configure_client(monkeypatch, handler):
    real_client = httpx.AsyncClient
    monkeypatch.setenv("DECISIONS_API_KEY", "test-key")
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: real_client(**kwargs, transport=httpx.MockTransport(handler)),
    )
    return DecisionsConfig(
        api_key_env="DECISIONS_API_KEY",
        api_base="https://gateway.example/v1",
        model="provider/model",
        concurrency=2,
    )


def test_signal_routes_complaints_and_preserves_messages(monkeypatch):
    active = peak = 0
    requests = []
    long_message = "Please help. " * 2000

    async def handler(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        payload = json.loads(request.content)
        requests.append(payload)
        assert str(request.url) == "https://gateway.example/v1/decisions"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert payload["model"] == "provider/model"
        assert payload["questions"][0]["choices"] == [
            {"value": "complaint"},
            {"value": "no_complaint"},
        ]
        await asyncio.sleep(0.01 if payload["input"] == "You ignored my instructions." else 0)
        active -= 1
        label = (
            "complaint" if payload["input"] == "You ignored my instructions." else "no_complaint"
        )
        return httpx.Response(200, json=decision(label))

    config = configure_client(monkeypatch, handler)
    messages = {
        "flagged": ["You ignored my instructions.", "Thanks", " "],
        "neutral": [long_message],
        "empty": [],
    }
    snapshot = TraceSnapshot(
        [Trace(id=trace_id, root_spans=[], aggregate=TraceAggregate()) for trace_id in messages]
    )
    extractor = AsyncMock(
        return_value=signal.UserMessageExtraction(extract=lambda snapshot: messages)
    )
    monkeypatch.setattr(
        signal, "UserMessageExtractor", lambda **kwargs: SimpleNamespace(build_extractor=extractor)
    )
    detector = AsyncMock(return_value=[])
    monkeypatch.setattr(
        signal, "IssueDetector", lambda **kwargs: SimpleNamespace(detect_issues=detector)
    )
    result = asyncio.run(
        signal.UserSentimentCompassSignal(
            UserSentimentConfig(decisions=config), FakeLLMClient()
        ).analyze(snapshot)
    )
    assert peak == 2
    assert sorted(p["input"] for p in requests) == sorted(
        ["You ignored my instructions.", "Thanks", long_message]
    )
    screening = result.artifacts["screening"]
    assert [r["label"] for r in screening["flagged"]["message_results"]] == [
        "complaint",
        "no_complaint",
        "no_user_messages",
    ]
    assert screening["flagged"]["message_results"][0]["scores"]["complaint"] == 0.7
    assert screening["neutral"]["flagged"] is False
    assert screening["empty"]["message_results"] == []
    assert result.finding_count == 1
    assert result.artifacts["coverage"]["traces_with_user_messages"] == 2
    assert result.artifacts["classifier"]["backend"] == "decisions"
    assert set(detector.call_args.args[0].traces_by_id) == {"flagged"}


@pytest.mark.parametrize("failure", [429, 503, "timeout"])
def test_transient_failures_retry(monkeypatch, failure):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            if failure == "timeout":
                raise httpx.ReadTimeout("timed out", request=request)
            return httpx.Response(failure)
        return httpx.Response(200, json=decision())

    config = configure_client(monkeypatch, handler)
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)
    results = asyncio.run(ComplaintClassifier(config).classify_batch(["Complaint"]))
    assert results[0].label == "complaint"
    assert calls == 2
    sleep.assert_awaited_once_with(1)


@pytest.mark.parametrize("status,expected_calls", [(401, 1), (400, 1), (503, 3)])
def test_request_failures_never_become_neutral(monkeypatch, status, expected_calls):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(status)

    config = configure_client(monkeypatch, handler)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    with pytest.raises(builtins.ExceptionGroup) as exc:
        asyncio.run(ComplaintClassifier(config).classify_batch(["Message"]))
    assert isinstance(exc.value.exceptions[0], httpx.HTTPStatusError)
    assert calls == expected_calls


@pytest.mark.parametrize("invalid", ["unknown_label", "missing_probability", "bad_sum"])
def test_invalid_responses_fail_classification(monkeypatch, invalid):
    response = decision()
    answer = response["answers"][0]
    if invalid == "unknown_label":
        answer["choice"] = "positive"
    elif invalid == "missing_probability":
        answer["probabilities"].pop()
    else:
        answer["probabilities"][0]["probability"] = 0.1
    config = configure_client(monkeypatch, lambda request: httpx.Response(200, json=response))
    with pytest.raises(builtins.ExceptionGroup) as exc:
        asyncio.run(ComplaintClassifier(config).classify_batch(["Message"]))
    assert isinstance(exc.value.exceptions[0], ValidationError)
