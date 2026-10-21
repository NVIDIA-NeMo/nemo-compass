# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json

import nooa.unifiedllm.unifiedllm as unifiedllm
from litellm import ModelResponse
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from compass.cli.main import _build_llm, _run_compass_signals, get_config
from compass.compass_signals.compass_signals import Problem
from compass.config import RunConfig
from compass.traces import Span, SpanKind, Trace, TraceAggregate, TraceSnapshot


def _response(name, arguments, call_id):
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=json.dumps(arguments))],
        finish_reason="tool_calls",
        assistant_message={},
    )


def test_grader_model_request_honors_cli_token_limit(monkeypatch, select_signals):
    requests = []

    async def completion(params):
        requests.append(params)
        return ModelResponse(
            choices=[
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "final",
                                "type": "function",
                                "function": {
                                    "name": "return_result",
                                    "arguments": '{"result":{"problems":[]}}',
                                },
                            }
                        ],
                    },
                }
            ]
        )

    monkeypatch.setattr(unifiedllm, "_litellm_acompletion", completion)
    config = get_config(
        [
            "--trace.filesystem.path",
            "unused.jsonl",
            "--compass-signals",
            json.dumps(select_signals(grader_failure_patterns={})),
            "--model",
            "openai/gpt-4o-mini",
            "--max-tokens",
            "32768",
        ]
    )
    snapshot = TraceSnapshot(
        [
            Trace(
                id="failed",
                root_spans=[],
                aggregate=TraceAggregate(),
                evaluator_results={"score": 0},
            )
        ]
    )
    asyncio.run(
        _run_compass_signals(
            config.compass_signals,
            snapshot,
            lambda: _build_llm(config, "test-key"),
        )
    )
    assert requests
    assert all(request["max_tokens"] == 32768 for request in requests)


def test_configured_signal_fetches_traces_before_reporting(select_signals):
    trace = Trace(
        id="failed",
        aggregate=TraceAggregate(),
        evaluator_results={"score": 0.1},
        root_spans=[Span(id="tool", kind=SpanKind.TOOL, input="search request", error="timeout")],
    )
    problem = Problem(description="Search times out; score=0.1.", supporting_trace_ids=(trace.id,))
    report = {"result": {"problems": [problem.model_dump()]}}
    llm = FakeLLMClient(
        [
            _response("return_result", report, "early"),
            _response("execute_python", {"code": "print(self.fetch_traces(['failed']))"}, "fetch"),
            _response("return_result", report, "final"),
        ]
    )
    config = RunConfig(
        trace={"filesystem": {"path": "unused.jsonl"}},
        compass_signals=select_signals(grader_failure_patterns={"max_tool_rounds": 1}),
    )

    results = asyncio.run(
        _run_compass_signals(config.compass_signals, TraceSnapshot([trace]), lambda: llm)
    )

    assert results[0].problems == (problem,)
    assert llm.call_count == 3
    assert "search request" in str(llm.last_messages)
