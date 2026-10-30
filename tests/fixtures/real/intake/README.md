<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Real NeMo Platform Intake responses

`all-sessions.json` holds the 8 HTTP exchanges that `IntakeTraceLoader` made
against a real, local NeMo Helix Intake server: `nemo-helix[all]==0.8.1` from PyPI
([NVIDIA-NeMo/nemo-helix](https://github.com/NVIDIA-NeMo/nemo-helix) tag `0.8.1` =
`d24f2a7718ecd7d9cf388edb749e562f4ce514bc`, Apache-2.0), started with
`nemo services run --services auth,entities,intake` (auth disabled, workspace
`default`, ClickHouse `clickhouse/clickhouse-server:26.3`). Each exchange records
the request URL, query params, status, content type and the response `body`
exactly as served. Nothing was trimmed or edited. Query window:
2026-08-01T00:00Z to 2026-10-31T00:00Z.

The server holds two sessions, ingested through Intake's public APIs:

- `langchain-openinference-smoke` has 2 traces (the merge case). Upstream
  `services/intake/examples/send_langchain_openinference_agent.py` at the tag above
  was run twice, unchanged. It is a real LangChain `create_agent` with a `multiply`
  tool, instrumented by OpenInference and exported over OTLP to
  `/ingest/otlp/v1/traces`. Versions: langchain 1.3.17, langchain-core 1.6.0,
  langchain-openai 1.4.1, langgraph 1.2.11, openai 2.53.0,
  openinference-instrumentation 0.1.56, openinference-instrumentation-langchain
  0.1.72, opentelemetry-sdk and opentelemetry-exporter-otlp-proto-http 1.43.0.
  Only the LLM is fake: a deterministic `/v1/chat/completions` stub that answers
  the first request with a `multiply(a=6, b=7)` tool call (id `call_multiply_1`)
  and, once a tool message is present, with that tool's content (`"42"`). It
  reports 42 prompt and 7 completion tokens.
- `d7d4e19e-608d-44ef-b166-cd050ef274ba` has 1 trace (the single-record case). It is
  a Claude Code ATIF trajectory published by Harbor
  ([harbor-framework/harbor](https://github.com/harbor-framework/harbor) tag
  `v0.24.0` = `b53b8134e1241686dca7759af188f987ecc48e8b`,
  `examples/tasks/hello-load-atif-trajectory-task-level/trajectory.json`,
  Apache-2.0, `LICENSE-harbor.txt`), POSTed unchanged to `/ingest/atif`. The span
  inputs, outputs and `raw_attributes` in these responses carry that Harbor content.

Each session has one evaluator result, posted to `POST /evaluator-results`. The
scores are illustrative; the response shape is the server's:
`answer_correct` (`BOOLEAN`, `value: 1`, comment `6 * 7 == 42`) on root span
`e95366268936a416` of trace `e859c721…`, and `outcome` (`CATEGORICAL`,
`string_value: "pass"`) on ATIF root span `span-e299ea5f59c8bbcf0982e2aba19e4839`.

To regenerate (OTLP ids and all timestamps are random or time-dependent; Intake
keeps 90 days, so move the window):

1. Start the server as above. Start the fake LLM and run the upstream example twice
   with `OPENAI_BASE_URL=<fake>/v1 OPENAI_API_KEY=fake-key` and
   `--endpoint <base>/apis/intake/v2/workspaces/default/ingest/otlp/v1/traces`.
2. POST the Harbor trajectory JSON to `<base>/apis/intake/v2/workspaces/default/ingest/atif`.
3. POST the two evaluator results above. Take the LangChain root span id from
   `GET /traces?filter[session_id]=langchain-openinference-smoke` (`root_span_id`).
4. In this repository, run `IntakeTraceLoader` (`base_url=<base>`, `workspace=default`,
   the window above) with an `httpx.HTTPTransport` subclass whose `handle_request`
   reads each response and appends `{method, url (no query), params, status_code,
   content_type, body}` to a list, then write that list with `json.dump(..., indent=2)`
   plus a trailing newline.
