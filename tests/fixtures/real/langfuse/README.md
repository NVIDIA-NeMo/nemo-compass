<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Real Langfuse server fixtures

These files were exported from real self-hosted Langfuse servers. The traces
were ingested with the real Langfuse Python SDK. Only the LLM was fake: an
in-process `httpx.MockTransport` answered the OpenAI client with fixed
chat-completion bodies. No paid API was called and no account was created. The
conversation content (a weather agent and `2 + 2`) was written for these
fixtures, so the data is covered by this repository's Apache-2.0 license. No
Langfuse source is included.

Components:

- Servers, from the upstream `docker-compose.yml` at each tag with image tags pinned:
  - v3: `langfuse/langfuse:3.225.15`
    (`sha256:dfae5abcdacdf2c868f9a3850616899260e8a379817ef5678e4021d15d446883`) and
    `langfuse/langfuse-worker:3.225.15`
    (`sha256:1249cc8b38bc5b3e64339f9eda7c75ae6db50e12de6ee2ce3478020ff7dbd5c5`),
    from <https://github.com/langfuse/langfuse/blob/v3.225.15/docker-compose.yml>.
  - v4: `langfuse/langfuse:4.72.0`
    (`sha256:c311fd9db4b69f6514fdcb1b86a2a105d5a4ed51fbf0ded033ff2722459c804a`) and
    `langfuse/langfuse-worker:4.72.0`
    (`sha256:aa1724b989a00f1e4ca9930709202489c73deaaf2f1116bbd3d71c77edb8fedb`),
    from <https://github.com/langfuse/langfuse/blob/v4.72.0/docker-compose.yml>.
- Ingestion: `langfuse==4.17.0` (OTel exporter and the `langfuse.openai` wrapper),
  `openai==2.54.0`, `httpx==0.28.1`, `opentelemetry-sdk==1.45.2`.
- Exports: `@langfuse/cli@1.2.4`.
- Cassettes: this repo's unmodified `LangfuseTraceLoader` at
  `50d920f187a298749d573f5df20b2fc760f149d4`, run against the live server.
  An httpx response hook stored each response body verbatim.

Each server holds the same scenario. Two `weather-chat` traces share
`sessionId=weather-session-001`. Each has an AGENT `weather-agent` with a
GENERATION that returns a `get_weather` tool call, a TOOL `get_weather`
observation, and an answering GENERATION. Turn 2's input repeats turn 1's
history and adds "And tomorrow?". The `arithmetic` trace has no session: an AGENT
with one GENERATION. There are three scores: `helpfulness` (NUMERIC, on turn 1),
`tool_args_valid` (BOOLEAN, on turn 1's TOOL observation) and `resolution`
(CATEGORICAL, on turn 2). The fake LLM waits 40 ms per call and the tool waits
5 ms, so no two observations share a start millisecond.

| file | content | loader |
|---|---|---|
| `v3/langfuse-v3.jsonl` | 3 `GET /api/public/traces/{id}` bodies (`langfuse api traces get <id> --json \| jq -c .body`) | `LangfuseV3FileTraceLoader` |
| `v3-envelope/single-trace.json` | unprocessed `langfuse api traces get <id> --json` envelope for the sessionless trace | `LangfuseV3FileTraceLoader` |
| `v3/live-v3-http.json` | cassette: health, `GET /api/public/traces` (core), 3× `GET /api/public/traces/{id}` | `LangfuseTraceLoader` (v3) |
| `v4/observations.json` | `langfuse --api-version 4 api observations list --fields core,basic,io,metadata,model,usage,prompt,trace_context --all --json` | `LangfuseV4FileTraceLoader` |
| `v4/scores.json` | `langfuse --api-version 4 api scores list --fields details,subject,annotation --all --json` | `LangfuseV4FileTraceLoader` |
| `v4/live-v4-http.json` | cassette: health, `GET /api/public/v2/observations` (selection plus one fetch per trace), `GET /api/public/v3/scores` | `LangfuseTraceLoader` (v4) |

Each live cassette comes from the same server run as the export next to it, so
the ids match. A cassette records `base_url`, `from_timestamp` and `to_timestamp`.
Replaying with that window reproduces the loader's exact requests. Nothing was
trimmed or edited. The only processing is the `jq -c .body` step in the v3 recipe.

Known gap: v4 flattens observation metadata keys (for example `"scope.name"`), so
the loader does not detect the SDK scope there and v4 spans get no token counts.
For that reason, the tests check token counts only for v3.

To regenerate (v3 shown; for v4 use tag `v4.72.0` and `--api-version 4`):

1. Start the server with the upstream compose file at the tag and the image tags
   above. Provision a headless project with `LANGFUSE_INIT_*` env and telemetry off.
2. Ingest the scenario above with the SDK's `start_as_current_observation`
   (`as_type="agent"`/`"tool"`), `propagate_attributes(session_id=...)`,
   `langfuse.openai.OpenAI` (with a `MockTransport` HTTP client) and
   `create_score`. Then flush and wait about 30 s for the worker to write to ClickHouse.
3. Export with the CLI commands in the table, using `--from-timestamp`/`--to-timestamp`
   (v3) or `--from-start-time`/`--to-start-time` (v4) around the ingestion window.
4. Record each cassette by running `LangfuseTraceLoader` with a real `Langfuse`
   client. Pass an `httpx.Client(event_hooks={"response": [...]})` that stores the
   method, path, query, status, content type and `response.text` of every response.
