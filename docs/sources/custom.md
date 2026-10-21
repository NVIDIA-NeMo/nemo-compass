<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Analyze traces from another system

Use a coding agent to write and run a conversion script for your trace store.
The script produces `traces.jsonl`, which NeMo Compass can read directly.

## Choose your input

Start with a small set of completed agent runs. Give your coding agent either:

- **An export:** the path to a file containing complete traces, including child spans,
  messages, and tool calls and results. Summary tables usually omit this detail.
- **API access:** the store's API documentation and URL, the project and time window,
  and the name of the environment variable holding your API key. Keep the key itself
  in your environment or `.env`.

Select the traces during export or retrieval; NeMo Compass reads the entire converted file.

## Ask your coding agent to convert the traces

Give your agent the input details above and this prompt:

```text
Write and run a small script to convert my traces into traces.jsonl for
NeMo Compass: https://github.com/NVIDIA-NeMo/labs-nemo-compass
Use the export or API access details I provided.

Use its current canonical Trace model and write one complete trace per
JSONL line. Preserve trace IDs, span hierarchy, user messages, tool calls
and results, and recorded errors and evaluation scores. Do not invent
missing data. Read any API credentials from the environment or .env.

Validate the output with the canonical filesystem loader. Tell me how many
traces were converted and what data could not be preserved. Leave me the
script and the command to rerun it.
```

The [canonical format](files.md#canonical-jsonl) and
[Trace model](../../src/trace_ingest/models.py) are the conversion references.
Compare a converted trace with its original before analyzing a larger export.

## Run NeMo Compass

With [uv installed](../../README.md#start-here), install the CLI:

```bash
uv tool install nemo-compass
```

[Configure your inference model and key](../model-access.md#choose-a-model),
which is separate from your trace store's key. From the folder containing `traces.jsonl`, run:

```bash
compass --trace.filesystem.path traces.jsonl --max-tokens 16384
```

Open `insights.yml` if the run produced insights. [Read your results](../results.md)
for help with findings or skipped Compass Signals.
