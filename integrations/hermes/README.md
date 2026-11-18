<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NVIDIA NeMo Compass for Hermes

Finds recurring problems in your Hermes sessions and writes a local report.

Compass reads the session traces that NeMo Relay (built into Hermes) records, groups
similar failures across sessions, and turns them into a short list of insights: tools that
keep failing, outlier sessions, wasted tokens, user frustration, and places where the agent
breaks the rules in your `ETHOS.md`. Each insight links the sessions that back it.

Compass is an NVIDIA research preview. Check each finding against its sessions before acting.

## Setup

1. Enable the plugin: `hermes plugins enable nvidia-compass`
2. Turn on Relay's ATIF export. `hermes compass setup` prints the `plugins.toml` lines.
3. Give Compass a model. By default it reuses your Hermes main model and endpoint; you
   only add `COMPASS_API_KEY` (Desktop: Settings > Plugins > nvidia-compass). The model
   must support tool calling. Override `compass_model` / `api_base` in the same settings page.
4. `hermes compass status` checks everything.

## Use

- Ask Hermes: "What keeps going wrong in my recent sessions?" Hermes asks for your approval, then
  starts `compass_analyze` in the background and tells you roughly how long it'll take. Ask again
  later and it reads the results with `compass_latest_report`.
- Or from a terminal: `hermes compass run`, then `hermes compass report` to open the page.

Each run writes `traces.jsonl`, `insights.yaml` and `report.html` to
`<HERMES_HOME>/plugin-data/nvidia-compass/runs/<timestamp>/`.

## What this plugin does on your machine

- **Reads** Relay ATIF files from `<HERMES_HOME>/telemetry/nemo-relay/atif/` (or the folder you
  set), an optional `ETHOS.md` you name, and the non-secret model settings (model, provider,
  `base_url`) from Hermes's `config.yaml` to prefill its own. Nothing else outside its data folder.
- **Sends** the selected sessions' content, and your `ETHOS.md` if set, to the Compass model
  endpoint you configure. With the default settings that is the provider your Hermes main model uses.
- **Stores** a copy of each analyzed session's full conversation text in that run's `traces.jsonl`,
  trimmed of Relay's raw request logs. Runs are kept until you delete them.
- **Asks first.** Every `compass_analyze` call goes through Hermes's approval prompt (a
  `pre_tool_call` hook). Unattended runs follow your `approvals` settings, and are blocked by default.
- **Runs** Compass as a child process (`python -m compass.cli.main`) only when you or the agent
  ask. When the agent asks, the run happens in the background (usually 10 to 30 minutes) and keeps
  going if Hermes exits; the next status check writes its report. A small watchdog process
  enforces the run's time limit even if Hermes exits. One run at a time.
- **Keeps your keys out.** The child gets only `COMPASS_API_KEY`, `COMPASS_MODEL`,
  `COMPASS_API_BASE` and basic system variables. Compass looks for a `.env` file starting in its
  working folder, so the plugin writes an empty `.env` into each run folder; that stops the search
  before it can reach Hermes's own `.env`.
- **Installs** `nemo-compass` from PyPI into Hermes's environment when you enable the plugin.
- **No telemetry.** Compass's optional LangSmith telemetry stays off, and LiteLLM uses its
  bundled model price list instead of downloading one.
- **Opens your browser** only when you run `hermes compass report`.
- Reads `COMPASS_API_KEY` only through Hermes (`get_secret`). Never reads `.env` or `auth.json`.
- No middleware, and nothing running between runs.

## Remove

`hermes plugins remove nvidia-compass`. Reports stay in the plugin data folder until you delete it.
