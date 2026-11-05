<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Real MLflow trace exports

Byte-for-byte output of the MLflow CLI run against a local SQLite tracking store,
produced by `mlflow==3.15.2` (with `mlflow-skinny==3.15.2`) and `openai==2.47.0`
autologging (`mlflow.openai.autolog()`), CPython 3.12. Nothing was reformatted,
trimmed or edited after export.

- `openai_agent_session.json`: `mlflow traces search --output json` of experiment
  `compass-openai-agent`, 3 traces. Two share session `support-chat-42`
  (turn 1: `AGENT` root, `CHAT_MODEL` requesting `get_order_status`, `TOOL`,
  `CHAT_MODEL` answer; turn 2: `AGENT` root over the full chat history, one
  `CHAT_MODEL`). The third trace has no session.
- `openai_agent_single_get.json`: `mlflow traces get` of the session-less trace above
  (a bare `{"info", "data"}` trace).
- `docs_process_chat_session.json`: `mlflow traces search --output json` of experiment
  `compass-docs-process-chat`, 2 traces of session `web-session-7` recorded with the
  `process_chat` function from MLflow's "Track Users & Sessions" docs, verbatim (default
  `@mlflow.trace` span type, so an `UNKNOWN` root over one `CHAT_MODEL`; a plain-string
  `message`; session set with `mlflow.update_current_trace`).

Span structure, attribute names and encodings (base64 span ids, JSON-encoded
attributes, `mlflow.chat.tokenUsage`), trace metadata (`mlflow.trace.session`,
`mlflow.trace.user`, token and cost rollups) and the export format are MLflow's own.
Only the LLM is faked: the OpenAI client uses an in-process `httpx.MockTransport` that
returns fixed `chat.completion` bodies, which the real OpenAI SDK parses before MLflow
records them. The application code and reply texts are ours, except `process_chat`,
which is MLflow's documentation example (Apache-2.0); the data is Apache-2.0 with
Compass. No network, account or API key is involved.

Trace ids, span ids, timestamps, durations and size stats change on every run. To
regenerate, create a project with `dependencies = ["mlflow==3.15.2", "openai==2.47.0"]`
(Python 3.12), save the script below as `generate_openai_agent.py`, and run from an empty
work directory:

```bash
export USER=compass LOGNAME=compass MLFLOW_TRACKING_URI="sqlite:///$PWD/mlflow.db"
uv run python generate_openai_agent.py
experiment="$(uv run python -c "import mlflow; print(mlflow.get_experiment_by_name('compass-openai-agent').experiment_id)")"
uv run mlflow traces search --experiment-id "$experiment" --max-results 100 --output json > openai_agent_session.json
experiment="$(uv run python -c "import mlflow; print(mlflow.get_experiment_by_name('compass-docs-process-chat').experiment_id)")"
uv run mlflow traces search --experiment-id "$experiment" --max-results 100 --output json > docs_process_chat_session.json
single="$(uv run python -c "import json; print(next(t['info']['trace_id'] for t in json.load(open('openai_agent_session.json'))['traces'] if 'mlflow.trace.session' not in t['info']['trace_metadata']))")"
uv run mlflow traces get --trace-id "$single" > openai_agent_single_get.json
```

```python
import json

import httpx
import mlflow
import openai
from mlflow.entities import SpanType

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Look up the shipping status of an order.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    }
]
SYSTEM = {"role": "system", "content": "You are a concise customer-support agent."}
calls = 0


def fake_llm(request: httpx.Request) -> httpx.Response:
    global calls
    calls += 1
    messages = json.loads(request.content)["messages"]
    last = messages[-1]
    message = {"role": "assistant", "content": None, "refusal": None, "annotations": []}
    if last["role"] == "tool":
        status = json.loads(last["content"])
        message["content"] = (
            f"Order A-1001 has {status['status']} via {status['carrier']} and should "
            f"arrive on {status['eta']}."
        )
    elif "A-1001" in last["content"]:
        message["tool_calls"] = [
            {
                "id": "call_fake_0001",
                "type": "function",
                "function": {"name": "get_order_status", "arguments": '{"order_id": "A-1001"}'},
            }
        ]
    elif "address" in last["content"]:
        message["content"] = (
            "Order A-1001 has already shipped, so I can't change its address. You can "
            "request a redirect from UPS with the tracking number."
        )
    elif "Canada" in last["content"]:
        message["content"] = "Yes, we ship to Canada; delivery takes 7-10 business days."
    else:
        message["content"] = "Standard shipping takes 3-5 business days."
    prompt, completion = len(json.dumps(messages)) // 4, len(json.dumps(message)) // 4
    return httpx.Response(
        200,
        json={
            "id": f"chatcmpl-fake-{calls:04d}",
            "object": "chat.completion",
            "created": 1793318400 + calls,
            "model": "gpt-4o-mini-2024-07-18",
            "system_fingerprint": "fp_fake",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "logprobs": None,
                    "finish_reason": "tool_calls" if "tool_calls" in message else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
            },
        },
    )


client = openai.OpenAI(
    api_key="fake-key",
    base_url="https://fake-llm.invalid/v1",
    http_client=httpx.Client(transport=httpx.MockTransport(fake_llm)),
)


@mlflow.trace(span_type=SpanType.TOOL)
def get_order_status(order_id: str) -> dict:
    return {"A-1001": {"status": "shipped", "carrier": "UPS", "eta": "2026-11-03"}}[order_id]


@mlflow.trace(span_type=SpanType.AGENT)
def support_agent(messages: list[dict]) -> str:
    conversation = [SYSTEM, *messages]
    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=conversation, tools=TOOLS, temperature=0
        )
        choice = response.choices[0].message
        if not choice.tool_calls:
            return choice.content
        conversation.append(choice.model_dump(exclude_none=True))
        for call in choice.tool_calls:
            result = get_order_status(**json.loads(call.function.arguments))
            conversation.append(
                {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)}
            )


@mlflow.trace
def process_chat(message: str, user_id: str, session_id: str):
    mlflow.update_current_trace(session_id=session_id, user=user_id)
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message},
        ],
    )
    return response.choices[0].message.content


mlflow.openai.autolog()
mlflow.set_experiment("compass-openai-agent")
turn1 = [{"role": "user", "content": "Where is my order A-1001?"}]
with mlflow.tracing.context(session_id="support-chat-42", user="user-7"):
    answer1 = support_agent(turn1)
turn2 = [
    *turn1,
    {"role": "assistant", "content": answer1},
    {"role": "user", "content": "Can you change its delivery address?"},
]
with mlflow.tracing.context(session_id="support-chat-42", user="user-7"):
    support_agent(turn2)
support_agent([{"role": "user", "content": "How long does standard shipping take?"}])
mlflow.set_experiment("compass-docs-process-chat")
process_chat("How long does standard shipping take?", "user-9", "web-session-7")
process_chat("Do you ship to Canada?", "user-9", "web-session-7")
mlflow.flush_trace_async_logging()
```
