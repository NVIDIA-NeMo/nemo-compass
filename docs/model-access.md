<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Data and model access

NeMo Compass needs an inference model to investigate traces and write insights.
Its API key is separate from your Braintrust, LangSmith, Langfuse, MLflow, or Intake credentials.

## Choose a model

NeMo Compass uses the LiteLLM library to connect to supported inference providers,
including OpenAI and Anthropic. Choose a model that supports tool calling
and structured output, and save its settings in `.env`.

For OpenAI:

```dotenv
COMPASS_MODEL=openai/gpt-5.2
COMPASS_API_KEY=your-openai-api-key
```

For Anthropic:

```dotenv
COMPASS_MODEL=anthropic/claude-sonnet-4-6
COMPASS_API_KEY=your-anthropic-api-key
```

These are full LiteLLM model strings: the `openai/` or `anthropic/` prefix selects
the provider. Direct access uses the provider's default endpoint; remove any old
`COMPASS_API_BASE` or YAML/CLI `api_base` override when switching providers.
For OpenAI directly, also unset inherited `OPENAI_API_BASE` and `OPENAI_BASE_URL`
gateway overrides.

You can also use an OpenAI-compatible gateway by setting its model name and API base:

```dotenv
COMPASS_MODEL=openai/your-model-name
COMPASS_API_KEY=your-gateway-key
COMPASS_API_BASE=https://gateway.example.com/v1
```

Replace the example values with your provider’s settings. Use the documented API base,
without appending a resource path such as `/chat/completions`.
The first `openai/` selects the protocol; the rest is the gateway’s model name.
If no model is set, the package defaults to `openai/azure/openai/gpt-5.6-luna`,
which requires the corresponding gateway configuration.

The source guides set `max_tokens: 16384` to limit each model response.
Adjust that YAML setting to a value your endpoint supports.
You can also set `model` and `api_base` in YAML or override them on the CLI.

## Recommended models

We recommend the following models for NeMo Compass:

- **NVIDIA Nemotron 3.5 Super VL**. Avoid Nemotron Lightning models: they have known
  tool-calling failures in Compass. The CLI warns when a selected model name identifies
  Nemotron Lightning, but lets analysis continue without changing the model.
- **Claude Sonnet 5.5**
- **GPT-6 Luna**

## Credentials

Keep `.env` in the folder where you run the command to load it automatically:

```bash
compass --config config.yaml
```

Or select an environment file explicitly with uv:

```bash
uv tool run --from nemo-compass --env-file /path/to/.env compass --config config.yaml
```

Exported variables take priority over values in either file.
Keep credentials out of configuration YAML and source control.

`OPENAI_API_KEY` and `ANTHROPIC_API_KEY` are accepted as fallback inference keys,
in that order. Set `COMPASS_API_KEY` explicitly when credentials for multiple
providers are present. For OpenRouter, put its key in `COMPASS_API_KEY`;
`OPENROUTER_API_KEY` alone does not satisfy the CLI's credential check.

`OPENAI_API_BASE` and `OPENAI_BASE_URL` are fallback endpoint settings, except for
models using the `anthropic/` or `openrouter/` prefixes. Explicit
`COMPASS_API_BASE` and YAML/CLI `api_base` overrides still apply to those models.
Choose the matching model explicitly. The CLI requires an inference key before loading traces.

Each source guide names the credentials needed for live access.
Loading a saved export does not require credentials for that trace platform.

Return to your [source guide](../README.md#start-here) to finish setup.

## Where data goes

| When you use… | Data access |
| --- | --- |
| A live trace source | NeMo Compass reads traces and recorded evaluation signals from that service. |
| Model-based analysis | Trace content, evaluation signals, and candidate findings may be sent to your configured inference provider. |
| ETHOS.md | Its contents are included in model-based analysis. |
| Code validation | Selected source excerpts are sent to the inference provider. The validator has read-only file tools. |
| Remote sentiment embeddings | Extracted user messages are sent to the configured embedding endpoint. |
| Local sentiment embeddings | The embedding model runs on your machine; model and tokenizer files may download on first use. |

Remote sentiment also loads a tokenizer, which may need an initial download.
Using local trace files still requires inference access. There is no automatic redaction
of trace content before analysis; choose inputs and endpoints appropriate for your data.

The CLI writes insights to a local YAML file or stdout. With telemetry disabled it
does not publish findings back to the trace platform. [Output details](results.md#saved-output).
