<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Real Braintrust query responses

Published producer output: HTTP responses recorded by Braintrust's own SDK test suites
against the hosted Braintrust API. Files are byte-identical copies of the upstream files
(or of the recorded body string); nothing is trimmed or re-serialized. Upstream had
already redacted credentials and encrypted web-search content.

- `btql-0196bdc4536f.json`, `btql-a4a38e186326.json`: response bodies of
  `POST https://api.braintrust.dev/btql` for
  `SELECT * FROM project_logs('f1e858a4-…') WHERE root_span_id = '<id>' ORDER BY created ASC LIMIT 1000`,
  recorded 2026-08-31 (WireMock/VCR). From
  [braintrust-sdk-java](https://github.com/braintrustdata/braintrust-sdk-java) at
  `ae1e2c6142592d6888b28a2e0f7162d16b893203`,
  `test-harness/src/testFixtures/resources/cassettes/braintrust/__files/`. Spans were
  produced by braintrust-sdk-java 0.3.22 auto-instrumentation calling real models:
  - `0196bdc4536f`: `task web_search` → `llm responses` (gpt-4o via LangChain, OpenAI
    Responses API) → `tool web_search_call` (built-in web search).
  - `a4a38e186326`: `task reasoning` → two `llm responses` calls (o4-mini, encrypted
    reasoning items); the second call continues the first conversation.
  License: MIT, Copyright (c) 2025 Braintrust Data, Inc. (`LICENSE-braintrust-sdk-java.txt`).
- `trace_spans.btql-response.jsonl`: body (`fmt: jsonl`, 1,878 B) of the `/btql` response in
  `test/fixtures/vcr_cassettes/btql/trace_spans.yml` of
  [braintrust-sdk-ruby](https://github.com/braintrustdata/braintrust-sdk-ruby) at
  `42c2e9db2b5ee0e6476c55842f27ac98ef973fa0`, recorded 2026-03-06 against
  `staging-api.braintrust.dev`: an experiment trace with root `eval` → `task`.
  License: Apache-2.0, Copyright 2025 Braintrust (`LICENSE-braintrust-sdk-ruby.txt`).
- `tool_calls_per_turn.fetch-response.json`, `round_trips.fetch-response.json`: the first
  `POST https://api.braintrust.dev/v1/experiment/<id>/fetch` response body in
  `py/src/braintrust/integrations/harbor/cassettes/latest/test_atif_import_scopes_tool_calls_per_turn_and_reports_bounded_content.yaml`
  and `…/test_atif_import_round_trips_with_real_sdks.yaml` of
  [braintrust-sdk-python](https://github.com/braintrustdata/braintrust-sdk-python) at
  `7a3a6c4f2d9b51c34119b2c7e8468153d621fb25`, recorded 2026-08-26 (vcrpy) after braintrust
  0.34.0 imported an ATIF agent trajectory into an experiment: root `task agent_execution` →
  `llm chat.completions.create` calls with OpenAI `tool_calls`, `tool` spans for those calls
  (`metadata.tool_call_id`), and in `tool_calls_per_turn` a `task trajectory.step.4`. The
  legacy fetch API answers `{"events": [...], "cursor": ...}`; its events have the `/btql`
  experiment row columns. Real quirks kept: the root and step spans have `"metadata": null`;
  timestamps are whole seconds, so an LLM span and the tool span it requested start together
  (only `span_attributes.exec_counter` orders them); `round_trips` has one stale tool row of a
  root absent from the page. License: Apache-2.0, Copyright 2023 Braintrust Data LLC
  (`LICENSE-braintrust-sdk-python.txt`).

Root spans have `input`/`output` `null` in all five traces, so no run-root turns are
recorded. Braintrust has no session key; each trace stands alone.

`tests/test_real_braintrust.py` replays the rows through `httpx.MockTransport`: the
root-selection query gets `{root_span_id, created}` of the `is_root` rows, the per-trace
query gets the verbatim rows of that `root_span_id` inside `{"data": [...]}` (the Ruby JSONL
lines and the fetch `events` are wrapped the same way; recorded cursors are not replayed).

To regenerate, check out the pinned commits and copy the files:

```sh
cp braintrust-sdk-java/test-harness/src/testFixtures/resources/cassettes/braintrust/__files/btql-{0196bdc4536f,a4a38e186326}.json .
cp braintrust-sdk-java/LICENSE LICENSE-braintrust-sdk-java.txt
cp braintrust-sdk-ruby/LICENSE LICENSE-braintrust-sdk-ruby.txt
cp braintrust-sdk-python/LICENSE LICENSE-braintrust-sdk-python.txt
for case in tool_calls_per_turn:test_atif_import_scopes_tool_calls_per_turn_and_reports_bounded_content round_trips:test_atif_import_round_trips_with_real_sdks; do
  uv run --with pyyaml python -c "import yaml, sys; c = yaml.safe_load(open('braintrust-sdk-python/py/src/braintrust/integrations/harbor/cassettes/latest/${case#*:}.yaml')); sys.stdout.write(next(i['response']['body']['string'] for i in c['interactions'] if i['request']['uri'].endswith('/fetch')))" > "${case%%:*}.fetch-response.json"
done
uv run --with pyyaml python -c "import yaml, sys; c = yaml.safe_load(open('braintrust-sdk-ruby/test/fixtures/vcr_cassettes/btql/trace_spans.yml')); sys.stdout.write(next(i['response']['body']['string'] for i in c['http_interactions'] if i['request']['uri'].endswith('/btql')))" > trace_spans.btql-response.jsonl
```
