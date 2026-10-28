<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Analyze MLflow traces

Choose an experiment containing your agent’s traces. Recorded, named assessments
are included automatically in grader failure analysis.

## Connect and run

With [uv installed](../../README.md#start-here), install the CLI with MLflow support:

```bash
uv tool install 'nemo-compass[mlflow]'
```

[Configure your inference model and key](../model-access.md#choose-a-model).
If your tracking server requires authentication, add a bearer token to `.env`:

```dotenv
MLFLOW_TRACKING_TOKEN=your-tracking-token
```

For HTTP Basic authentication, use `MLFLOW_TRACKING_USERNAME` and
`MLFLOW_TRACKING_PASSWORD` instead.

Save this as `config.yaml`, replacing the experiment name and tracking URL:

```yaml
max_tokens: 16384
trace:
  max_traces: 100
  mlflow_experiment:
    experiment: my-agent
    tracking_uri: https://mlflow.example.com
```

```bash
compass --config config.yaml
```

Open `insights.yml` if the run produced insights. [Read your results](../results.md)
for help with findings or skipped Compass Signals.

## Narrow the input

Add `filter: "trace.status = 'ERROR'"` under `trace.mlflow_experiment` to select failed traces.
`tracking_uri` overrides `MLFLOW_TRACKING_URI` in the environment.

Keep an explicit limit for a first run. Without one, the live loader defaults to 10,000 traces,
fetched in pages of up to 500. Memory use grows with the number and size of traces.

## Use an export

Set `MLFLOW_TRACKING_URI` and any tracking-server credentials in `.env`, then export complete traces:

```bash
uvx --env-file .env --from 'mlflow-skinny>=3.6,<4' mlflow traces search \
  --experiment-id YOUR_EXPERIMENT_ID --max-results 100 --output json > mlflow-traces.json
```

Replace the `trace` section with:

```yaml
trace:
  max_traces: 100
  mlflow_export:
    path: mlflow-traces.json
```

Rerun the same analysis command. Local loading needs inference credentials only.
Exports must include spans. The loader accepts native trace objects, arrays, JSONL,
`mlflow traces get` output, and `Trace.to_json()` or `Trace.to_dict()` output.
It validates the entire file before applying the limit; export continuation tokens are not followed.

The extra installs `mlflow-skinny` SDK `>=3.6,<4`. Exports must be readable by that SDK;
tracking-server compatibility depends on your deployment. Duplicate active assessment names
are rejected, so each assessment must have an unambiguous name.

## Conversation turns

When the trace has a single root span of type `AGENT`, `CHAIN`, `TASK` or `WORKFLOW`, it
loads as the trace's agent span, keeping its MLflow type in `attributes.subtype` and
`attributes.mlflow.attributes`, and holds the conversation as ordered turns:

1. The trace input, read from the root span (`mlflow.spanInputs`, `input.value` or
   `gen_ai.input.messages`), else the trace metadata or request preview. A message list
   (OpenAI or LangChain messages, bare, under `messages`, or in a one-item batch) becomes
   one turn per message: `assistant`, `ai`, `agent` and `model` map to agent, `user` and
   `human` to user, and `system` and `developer` to system. A `tool` or `function` message
   becomes an unlinked result on the preceding agent turn. Every message except the last
   user message is copied context. Any other input is one user turn holding its text or
   JSON.
2. One model turn per LLM span in tree order, holding its output text and linking the LLM
   span. TOOL spans join the latest model turn; a tool call before any model call starts
   its own turn. Orchestration, retriever, embedding, reranker and guardrail spans stay
   spans only.
3. A closing agent turn for the trace output when it differs from the last agent message.

Message text comes from string content, a `content`/`message`/`text`/`response`/`answer`/`output`
field, a nested message, the first choice or the first LangChain generation; anything else is
kept as JSON text. Input content made only of OpenAI `text` and `image_url` parts becomes text
and image parts; an image's type comes from a `data:` URI or the file suffix, and images are
never fetched.

Turns are absent when the trace has several root spans or its root is not an orchestration
span, no input is recorded, or the input messages are empty, malformed, have an unknown role
or have no user message. Spans owned by a nested agent are left out of the root agent's turns
and listed in a `steps_outside_agent_not_in_turns` loss.
