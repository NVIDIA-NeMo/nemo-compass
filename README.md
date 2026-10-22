<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Compass

![Status: Research Preview](https://img.shields.io/badge/Status-Research%20Preview-orange)

The goal of NeMo Compass is to efficiently find signals in agent traces that lead to meaningful improvements in quality, cost, safety, privacy, and/or user satisfaction. 

A critical input to improving any agent is understanding what it’s doing, especially once you turn it on in production. Evals often capture only a small subset of the tasks an agent actually needs to be good at. Agent developers have discovered that production trace analysis is an essential part of improving an agent, but today there's no standard for how this type of analysis gets validated and measured.

NeMo Compass is an applied research project with a goal of improving the tools and techniques available to developers for trace analysis at scale. It uses a range of techniques to identify different types of issues in a large set of traces, then summarizes findings as actionable insights. We call each of those discovery techniques "Compass Signals."

| Compass Signal | Question It Answers | How It Works |
| --- | --- | --- |
| Anomaly Detection | What outlier behaviors exist in a set of agent traces? | Scores outliers with an Isolation Forest and reports feature-level deviations |
| Tool Issues	| What recurring problems occur when the agent uses tools?	| Checks tool calls and results for invalid arguments, failures, missing results, and unproductive retries |
| User Sentiment | What recurring complaints do users have about the agent? | Screens recorded human messages with a classifier, then reviews recurring complaints in context |
| ETHOS.md Divergence | Where does the agent’s behavior conflict with its stated rules and goals? | Compares traces with the agent’s [ETHOS.md](https://github.com/NVIDIA/skills/blob/main/skills/ethos-orchestrate/README.md#how-to-use) document and investigates possible gaps |
| Grader failures | What behaviors recur in traces with poor evaluation results? | Uses recorded scores and feedback to select traces, then investigates shared failure patterns |

Learn more about each current Compass Signal: [Architecture](docs/architecture.md)

> [!IMPORTANT]
> **Research preview**
>
> NeMo Compass is an early-stage applied research project. 
> Its APIs, configuration, and output formats may change without
> backward compatibility. All findings should be reviewed against their supporting
> traces before you act on them. This project is not intended for production use.

We are sharing the implementation at this stage to encourage feedback and testing by others as we continue to improve on it.

## Start here

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then choose a guide:

| Your starting point | Guide |
| --- | --- |
| Try it before connecting your data | [Run the example](examples/README.md) |
| Braintrust project logs and experiments | [Braintrust](docs/sources/braintrust.md) |
| LangSmith traces and feedback | [LangSmith](docs/sources/langsmith.md) |
| Langfuse traces and scores | [Langfuse](docs/sources/langfuse.md) |
| MLflow traces and assessments | [MLflow](docs/sources/mlflow.md) |
| NeMo Platform Intake | [Intake](docs/sources/intake.md) |
| NeMo Gym rollout trajectories | [NeMo Gym](docs/sources/gym.md) |
| JSONL or ATIF files | [Trace files](docs/sources/files.md) |
| Another trace store | [Convert your traces](docs/sources/custom.md) |

Install from PyPI as [`nemo-compass`](https://pypi.org/project/nemo-compass/); the command is `compass`.
If `compass` is not found after installation, run `uv tool update-shell`
and restart your terminal.

## After your first run

- [Read your results](docs/results.md) — understand insights, skipped Compass Signals, and saved output.
- [Compass Signals](docs/compass-signals.md) — supply business rules, configure sentiment, or check findings against code.
- [Data and model access](docs/model-access.md) — configure your inference endpoint and understand where data goes.
- [Configuration reference](docs/configuration.md) — overrides, limits, and repeatable runs.

[Purpose and expectations](ETHOS.md) · [Development](DEVELOPMENT.md) · [Releasing](RELEASING.md) · [Architecture](docs/architecture.md)

# Research Topics 

## Questions we are exploring

- How can developers efficiently find recurring, fixable agent problems in trace datasets containing
  hundreds of thousands or millions of executions?
- How can developers estimate how often a problem occurs and how much it matters without
  asking an LLM to inspect every trace?
- How can trace evidence, evaluation results, and read-only checks against agent
  code help distinguish real problems from misleading patterns?
- How should we measure whether an insight is correct, useful, and worth a
  developer's time to investigate?

This repository explores these questions. The [architecture
guide](docs/architecture.md) explains what each part does and where judgment or
coverage limits remain.

## Help shape the direction

We are especially interested in feedback from teams building agent observability
or evaluation tools, and from developers investigating their own agents:

- Which types of recurring issues are hardest to identify today?
- How do you decide whether an identified issue matters to your agent?
- What would make a finding more "actionable": examples, frequency, estimated impact, something else?
- How would you ideally use these tools in your own workflows? What would you need to integrate it?
- What types of additional insights or findings would be most valuable to you?

If an NVIDIA contact shared this preview with you, please send your feedback
through that contact and ask them to route it to the NeMo Compass research team.
Otherwise, use the [NVIDIA Developer contact form](https://developer.nvidia.com/contact)
with **NeMo Compass research preview feedback** in the subject. Please avoid sending raw traces,
prompts, credentials, or user data.

## Contributing

**This project is currently not accepting code contributions, pull requests, or
GitHub issues from external contributors.** Please do not submit external pull
requests or open issues, including bug reports and feature requests.

For research feedback, use the channels described below. To report a security
vulnerability, follow [SECURITY.md](SECURITY.md).


## License

Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This project is licensed under the [Apache License, Version 2.0](LICENSE). See
[NOTICE](NOTICE) for project attributions,
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for human-readable dependency license
disclosures, and [third_party/licenses.jsonl](third_party/licenses.jsonl) for the machine-readable
inventory.
