<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Real LangSmith runs from the LangChain docs

Six real LangSmith traces (30 runs) that LangChain publishes as public share links in
[langchain-ai/docs](https://github.com/langchain-ai/docs) at
`529d160b243aee561329e0d40754fdc10905c57a`, `src/code-samples/trace-links.json`.
LangChain's docs CI writes them to the LangSmith project `docs-code-samples` by running
the docs code samples against real OpenAI and Anthropic models. Fetched 2026-10-30.
License: MIT, Copyright (c) 2025 LangChain (`LICENSE-langchain-docs.txt`, copied from
that commit). The files are byte-identical copies; nothing is trimmed or re-serialized.

- `api/<trace_id>.runs-query.json`: response bodies of
  `POST https://api.smith.langchain.com/public/{share_token}/runs/query` with body
  `{"id": [root id, *root.child_run_ids]}` (the unauthenticated share API). They hold
  the same `RunSchema` objects as `/api/v1/runs/query`: the root and every descendant.
- `export/<trace_id>.jsonl`: written by the real `langsmith` CLI v0.3.12
  (release `langsmith_darwin_arm64.tar.gz`, commit
  `a96b015c320a2fe80a8d0f41bbeafb4b17528b1e`, MIT) with
  `langsmith trace export export --project docs-code-samples --since 2026-01-01T00:00:00Z --limit 100 --full`.
  The CLI read the `api/` runs, unchanged, from a local server that answered
  `/info`, `/sessions` and `/runs/query` in place of a self-hosted 0.15 v1 API. The
  placeholder project id and server version it used do not appear in the export.

| trace id | share token | docs snippet | `thread_id` | runs | notes |
| --- | --- | --- | --- | --- | --- |
| `01a189c7-63c9-72b2-a6c6-7dfb9b6bd47d` | `c5dd94b2-c40b-4677-b8da-7a8da97b9642` | long-term-memory-read-tool-postgres-py | none | 7 | ChatAnthropic `claude-sonnet-4-6`, tool `get_user_info` with null inputs |
| `01a189c9-3719-7fe0-ac21-95a03442a355` | `e885406b-502e-4732-a55f-fc133bdfd936` | tool-runtime-context-thread-py | `01a189c9-3716-7070-b983-2f22abbb0864` | 7 | ChatOpenAI, tool `get_account_info` |
| `01a13c55-2f6c-72de-aeaa-c9803ea1a291` | `b512e509-c58f-4a96-8ee3-9fe060257d62` | short-term-memory-usage-js | `1` | 5 | JS, lc-serialized messages, 2026-10-14 |
| `01a189c8-ee2b-7c03-8f38-5c94a37a9eca` | `6fe1dc0b-6e0f-4911-90a4-4b33a46cff77` | short-term-memory-usage-py | `1` | 3 | Python |
| `01a189c9-5b35-7873-83f6-e1f120fcf1a6` | `865aef4a-3f52-4774-936e-0e21756f2cc1` | langgraph-interrupts-multiple-py | `1` | 3 | non-chat state `{"vals": []}`, runs with `status: "interrupted"` |
| `01a189c9-6b2a-7531-bc88-dd109dec4218` | `98a5d6e1-b3fc-4248-a3b6-48901419f96b` | langgraph-subgraphs-interrupt-v2-py | `1` | 5 | tool `fruit_info` interrupted; AIMessage content blocks (`output_version: "v1"`) |

Several docs samples reuse `thread_id="1"`, so the four `thread_id` `1` roots are a real
thread-id collision in one project; LangSmith's Threads view groups them the same way.
They are separate programs, not one multi-turn conversation: a public share covers one
trace only. The runs contain LangChain secret references (environment variable names),
no key values.

To regenerate, fetch each share's root with `GET /public/{token}/run`, then
`POST /public/{token}/runs/query` with the root and its `child_run_ids`, and run the
CLI command above against a server that serves those runs.
