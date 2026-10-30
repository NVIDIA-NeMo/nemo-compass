# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rollouts NVIDIA-NeMo/Gym publishes at 399e6783e0e8, see fixtures/real/gym_responses/README.md."""

import json
from pathlib import Path

from trace_ingest.loaders import GymTraceConfig, GymTraceLoader
from trace_ingest.models import UNSET, Authorship, SpanKind, TextPart, TokenCounts, TurnRole

FIXTURES = Path(__file__).parent / "fixtures" / "real" / "gym_responses"
SINGLE = FIXTURES / "example_single_tool_call.example_rollouts.jsonl"
MULTIREWARD = FIXTURES / "example_tool_call_multireward.example_rollouts.jsonl"
MULTI_TURN = FIXTURES / "example_multi_turn_gymnasium.example_rollouts.jsonl"
AVIARY = FIXTURES / "aviary.example_rollouts.jsonl"


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def load(path):
    return list(GymTraceLoader(GymTraceConfig(path=path)).load())


def output_items(row, kind):
    return [item for item in row["response"]["output"] if item["type"] == kind]


def assert_tool_span_matches_source(trace, row):
    """Each function_call becomes a TOOL span linked to its function_call_output by call_id."""
    (root,) = trace.root_spans
    tools = [span for span in root.children if span.kind is SpanKind.TOOL]
    calls = output_items(row, "function_call")
    outputs = {
        item["call_id"]: item["output"] for item in output_items(row, "function_call_output")
    }
    assert len(tools) == len(calls)
    for tool, call in zip(tools, calls, strict=True):
        assert tool.tool_name == call["name"]
        assert tool.input == json.loads(call["arguments"])
        assert tool.tool_call.call_id == tool.tool_call.result_id == call["call_id"]
        assert tool.output == outputs[call["call_id"]]
        (turn,) = [turn for turn in root.turns if tool.id in turn.tool_call_span_ids]
        assert (turn.role, turn.authorship) == (TurnRole.AGENT, Authorship.MODEL)
        assert turn.extra["gym"]["id"] == call["id"]


def assert_model_span_matches_source(trace, row):
    (root,) = trace.root_spans
    (model,) = [span for span in root.children if span.kind is SpanKind.LLM]
    usage = row["response"]["usage"]
    assert model.model == row["response"]["model"]
    assert model.token_counts == TokenCounts(
        input_total=usage["input_tokens"],
        cached_input_tokens=usage["input_tokens_details"]["cached_tokens"],
        output_tokens=usage["output_tokens"],
    )
    (message,) = output_items(row, "message")
    assert root.output == message["content"][0]["text"]
    assert root.turns[-1].content == [TextPart(text=message["content"][0]["text"])]


def test_single_tool_call_rollouts():
    source = rows(SINGLE)
    traces = load(SINGLE)

    assert len(traces) == 5
    assert len({trace.id for trace in traces}) == 5
    assert all(trace.id.startswith("gym-") for trace in traces)
    assert [(trace.case_id, trace.session_id) for trace in traces] == [(None, None)] * 5
    assert [trace.evaluator_results for trace in traces] == [
        {"gym.reward": row["reward"]} for row in source
    ]
    assert [trace.evaluator_results["gym.reward"] for trace in traces] == [0.0, 0.0, 1.0, 1.0, 1.0]

    for trace, row in zip(traces, source, strict=True):
        assert_tool_span_matches_source(trace, row)
        assert_model_span_matches_source(trace, row)
        (root,) = trace.root_spans
        assert [definition.name for definition in root.tool_definitions] == ["get_weather"]
        assert root.tool_definitions[0].input_schema["required"] == ["city"]
        system, user = root.turns[:2]
        assert (system.role, system.authorship) == (TurnRole.SYSTEM, Authorship.COPIED_CONTEXT)
        assert system.content == row["responses_create_params"]["input"][0]["content"]
        assert (user.role, user.content) == (
            TurnRole.USER,
            row["responses_create_params"]["input"][1]["content"],
        )
        assert trace.losses == []

    # Rows 0-1 ask for clarification without a tool call; rows 2-4 call get_weather once.
    assert [len(trace.root_spans[0].children) for trace in traces] == [1, 1, 2, 2, 2]
    clarify, _, humidity, _, _ = traces
    assert [turn.role for turn in clarify.root_spans[0].turns] == [
        TurnRole.SYSTEM,
        TurnRole.USER,
        TurnRole.AGENT,
    ]
    (_, user, call, answer) = humidity.root_spans[0].turns
    assert user.content == "humidity in sf"
    assert call.content is UNSET
    assert call.tool_call_span_ids == ["gym/response/tool/0"]
    assert call.llm_span_ids == answer.llm_span_ids == ["gym/response/model"]
    tool = humidity.root_spans[0].children[1]
    assert tool.input == {"city": "San Francisco"}
    assert tool.tool_call.prior_user_text == "humidity in sf"
    assert json.loads(tool.output) == {
        "city": "San Francisco",
        "weather_description": "The weather in San Francisco is cold.",
    }


def test_multireward_rollouts_keep_task_index_and_reward_components():
    source = rows(MULTIREWARD)
    traces = load(MULTIREWARD)

    assert len(traces) == 5
    assert len({trace.id for trace in traces}) == 5
    assert [trace.case_id for trace in traces] == [
        f"gym-task-index-{row['_ng_task_index']}" for row in source
    ]
    assert [trace.case_id for trace in traces] == [
        "gym-task-index-1",
        "gym-task-index-0",
        "gym-task-index-4",
        "gym-task-index-2",
        "gym-task-index-3",
    ]
    assert all(trace.session_id is None for trace in traces)
    for trace, row in zip(traces, source, strict=True):
        assert trace.evaluator_results == {
            "gym.reward": row["reward"],
            "gym.reward_components": row["reward_components"],
        }
        assert trace.evaluator_results == {
            "gym.reward": 2.0,
            "gym.reward_components": {"correctness": 1.0, "schema_valid": 1.0, "format": 0.0},
        }
        assert_tool_span_matches_source(trace, row)
        assert_model_span_matches_source(trace, row)
        (root,) = trace.root_spans
        (definition,) = root.tool_definitions
        assert (definition.name, definition.description) == (
            "get_weather",
            "Get the current weather for a city.",
        )
        # The predicted call the verifier scored is the one the TOOL span records.
        (predicted,) = row["predicted_calls"]
        (tool,) = [span for span in root.children if span.kind is SpanKind.TOOL]
        assert (tool.tool_name, tool.input) == (predicted["name"], predicted["arguments"])
        # Every response output item except the tool result is one model-authored turn.
        agent_turns = root.turns[2:]
        items = [
            item for item in row["response"]["output"] if item["type"] != "function_call_output"
        ]
        assert len(agent_turns) == len(items)
        assert [turn.extra["gym"]["id"] for turn in agent_turns] == [item["id"] for item in items]
        assert {(turn.role, turn.authorship) for turn in agent_turns} == {
            (TurnRole.AGENT, Authorship.MODEL)
        }
        assert trace.losses == []

    # Task 1's output has one empty reasoning item; the others have two.
    assert [len(trace.root_spans[0].turns) for trace in traces] == [5, 6, 6, 6, 6]
    london = traces[4].root_spans[0]
    assert london.turns[1].content == "is it raining in London right now?"
    assert london.children[1].input == {"city": "London"}


def test_multi_turn_gymnasium_rollouts_keep_environment_user_messages():
    source = rows(MULTI_TURN)
    traces = load(MULTI_TURN)

    assert len(traces) == 5
    assert [trace.case_id for trace in traces] == [
        f"gym-task-index-{row['_ng_task_index']}" for row in source
    ]
    assert [trace.evaluator_results for trace in traces] == [{"gym.reward": 1.0}] * 5
    for trace, row in zip(traces, source, strict=True):
        (root,) = trace.root_spans
        output = row["response"]["output"]
        (model,) = root.children
        assert (model.kind, model.model, model.output) == (
            SpanKind.LLM,
            "Qwen/Qwen3-4B-Instruct-2507",
            output,
        )
        assert root.input == row["responses_create_params"]["input"]
        assert root.output == output[-1]["content"][0]["text"]
        # Gym's gymnasium agent appends each environment observation to the output as a
        # user message, then calls the model again: the turns alternate from the request.
        question, *rest = root.turns
        assert (question.role, question.authorship, question.content) == (
            TurnRole.USER,
            Authorship.UNKNOWN,
            row["responses_create_params"]["input"][0]["content"],
        )
        assert len(rest) == len(output)
        for turn, item in zip(rest, output, strict=True):
            if item["role"] == "user":
                assert (turn.role, turn.authorship, turn.llm_span_ids) == (
                    TurnRole.USER,
                    Authorship.UNKNOWN,
                    [],
                )
                assert turn.content == item["content"]
            else:
                assert (turn.role, turn.authorship, turn.llm_span_ids) == (
                    TurnRole.AGENT,
                    Authorship.MODEL,
                    [model.id],
                )
                assert turn.content == [TextPart(text=item["content"][0]["text"])]
        assert trace.losses == []

    assert [len(trace.root_spans[0].turns) for trace in traces] == [4, 4, 4, 6, 4]
    paris = traces[3].root_spans[0]
    assert [turn.content for turn in paris.turns if turn.role is TurnRole.USER] == [
        "What is the capital of France?",
        "What is the approximate population of that city?",
        "Name one famous landmark there.",
    ]


def test_aviary_transitions_load_the_last_transition():
    source = rows(AVIARY)
    traces = load(AVIARY)

    assert len(traces) == 5
    assert [trace.evaluator_results["gym.reward"] for trace in traces] == [0.0, 0.0, 1.0, 0.0, 0.0]
    for trace, row in zip(traces, source, strict=True):
        # Gym's aviary agent records the model input after each environment step; each
        # transition extends the previous one, so the last is the whole conversation.
        transitions = row["response"]["output"]
        assert row["response"]["contains_transitions"] is True
        last = len(transitions) - 1
        conversation = transitions[last]
        (root,) = trace.root_spans
        model, *tools = root.children
        assert (model.kind, model.model, model.output) == (
            SpanKind.LLM,
            "gpt-4.1-2025-04-14",
            conversation,
        )
        calls = [item for item in conversation if item["type"] == "function_call"]
        outputs = {
            item["call_id"]: item["output"]
            for item in conversation
            if item["type"] == "function_call_output"
        }
        assert [(tool.tool_name, tool.input, tool.output) for tool in tools] == [
            (call["name"], json.loads(call["arguments"]), outputs[call["call_id"]])
            for call in calls
        ]
        assert tools[0].attributes["gym"]["item_path"] == f"$.response.output[{last}][1]"
        assert tools[0].tool_call.prior_user_text == conversation[0]["content"]
        question, *rest = root.turns
        assert (question.role, question.authorship, question.content) == (
            TurnRole.USER,
            Authorship.UNKNOWN,
            conversation[0]["content"],
        )
        assert question.extra["gym"]["item_path"] == f"$.response.output[{last}][0]"
        assert {(turn.role, turn.authorship) for turn in rest} == {
            (TurnRole.AGENT, Authorship.MODEL)
        }
        final = conversation[-1]
        if final["type"] == "message":
            assert root.output == final["content"][0]["text"]
        else:
            assert root.output is UNSET
        assert trace.losses == []
    assert [len(trace.root_spans[0].children) - 1 for trace in traces] == [1, 1, 5, 3, 3]
