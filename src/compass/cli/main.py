# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analyze your agent's behavior and surface actionable insights from its traces.

After configuring your model and credentials, run:
  compass --config config.yaml

Start with your trace source (Braintrust, LangSmith, Langfuse, MLflow, Intake, or files):
  https://github.com/NVIDIA-NeMo/labs-nemo-compass/blob/main/README.md#start-here

Understand your results:
  https://github.com/NVIDIA-NeMo/labs-nemo-compass/blob/main/docs/results.md

Use --help-all for every setting. CLI options override individual YAML values.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from argparse import SUPPRESS, Action, ArgumentParser
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from nooa.unifiedllm import CompletionClient, UnifiedLLM
from pydantic import ValidationError
from pydantic_settings import CliSettingsSource
from rich.console import Console

from compass import telemetry_phases as telemetry
from compass.cli.output import RunOutput, RunResult, display_name, signal_limitations
from compass.compass_signals.builtins import registered_builtin_signals
from compass.compass_signals.compass_signals import (
    CompassSignalResult,
    Problem,
)
from compass.config import CompassSignalsConfig, RunConfig, TraceConfig
from compass.insight import Insight, load_insights, resolve_trace_links
from compass.insights_generation.confidence import InsightConfidence
from compass.insights_generation.config import (
    ENV_API_BASE,
    ENV_API_KEY,
    ENV_MODEL,
    load_dotenv,
    resolve,
)
from compass.insights_generation.defaults import DEFAULT_MAX_TOKENS, DEFAULT_MODEL
from compass.insights_generation.insight_compilation import InsightCompilation
from compass.insights_generation.validation import (
    CandidateProblemDecision,
    CandidateProblemValidationResult,
    CandidateProblemValidator,
)
from compass.telemetry import record_input_metadata, record_span, telemetry_session
from compass.trace_loaders.atif import ATIFTraceLoader
from compass.trace_loaders.braintrust import (
    BRAINTRUST_DEFAULT_MAX_TRACES,
    BraintrustTraceConfig,
    BraintrustTraceLoader,
)
from compass.trace_loaders.fs import FSDataLoader
from compass.trace_loaders.intake import IntakeTraceLoader
from compass.trace_loaders.langfuse import (
    LANGFUSE_DEFAULT_MAX_TRACES,
    LangfuseFileTraceConfig,
    LangfuseTraceConfig,
    LangfuseTraceLoader,
    LangfuseV3FileTraceLoader,
    LangfuseV4FileTraceConfig,
    LangfuseV4FileTraceLoader,
)
from compass.trace_loaders.langsmith import (
    LANGSMITH_DEFAULT_MAX_TRACES,
    LangSmithTraceConfig,
    LangSmithTraceExportFileConfig,
    LangSmithTraceExportFileLoader,
    LangSmithTraceLoader,
)
from compass.trace_loaders.mlflow import (
    MLFLOW_DEFAULT_MAX_TRACES,
    MLflowFileTraceConfig,
    MLflowFileTraceLoader,
    MLflowTraceConfig,
    MLflowTraceLoader,
)
from compass.trace_loaders.trace_loaders import TraceLoader
from compass.traces import Trace, TraceSnapshot
from trace_ingest.loaders.gym import GymTraceLoader

EXIT_OK = 0
EXIT_SETUP = 2
DEFAULT_REASONING_EFFORT = "high"


class SetupError(ValueError):
    """Required settings are missing from the environment for this run."""


def _check_environment(config: RunConfig) -> str:
    """Check the selected integrations before constructing clients or loading traces."""

    load_dotenv()
    api_key = resolve(ENV_API_KEY)
    required = {f"{ENV_API_KEY} — API key for the configured model": api_key}
    if config.trace.langsmith is not None:
        # Match the LangSmith SDK's legacy alias without importing the optional SDK.
        required["LANGSMITH_API_KEY — API key for LangSmith"] = os.environ.get(
            "LANGSMITH_API_KEY", ""
        ).strip() or os.environ.get("LANGCHAIN_API_KEY")
    if config.trace.braintrust is not None:
        required["BRAINTRUST_API_KEY — API key for Braintrust"] = os.environ.get(
            "BRAINTRUST_API_KEY"
        )
    if config.trace.langfuse is not None:
        for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
            required[f"{name} — Langfuse project credentials"] = os.environ.get(name)
        if not (config.trace.langfuse.base_url or "").strip():
            required["LANGFUSE_BASE_URL — Langfuse URL; or set trace.langfuse.base_url"] = (
                os.environ.get("LANGFUSE_BASE_URL")
            )
    sentiment = config.compass_signals.user_sentiment
    if sentiment is not None and sentiment.litellm is not None and sentiment.litellm.api_key_env:
        name = sentiment.litellm.api_key_env
        hint = "API key named by compass_signals.user_sentiment.litellm.api_key_env"
        required[f"{name} — {hint}"] = os.environ.get(name)

    missing = [f"  {label}" for label, value in required.items() if not (value or "").strip()]
    if missing:
        raise SetupError(
            "Missing required environment settings:\n"
            + "\n".join(missing)
            + "\n\nSet these in .env in your working directory (NAME=value),\n"
            "or export them in your shell, then rerun the command.\n"
            "Setup: https://github.com/NVIDIA-NeMo/labs-nemo-compass/blob/main/docs/model-access.md"
        )
    assert api_key is not None
    return api_key


def _configured_trace_loader(config: TraceConfig) -> TraceLoader:
    """Construct the trace loader selected by the run configuration."""

    if config.filesystem is not None:
        return FSDataLoader(config.filesystem.path)
    if config.gym is not None:
        return GymTraceLoader(config.gym)
    if config.atif is not None:
        return ATIFTraceLoader(config.atif)
    if config.intake is not None:
        source = config.intake
        if config.max_traces is not None:
            query = source.query.model_copy(update={"max_traces": config.max_traces})
            source = source.model_copy(update={"query": query})
        return IntakeTraceLoader(config=source)
    if config.braintrust is not None:
        source = config.braintrust
        return BraintrustTraceLoader(
            BraintrustTraceConfig(
                project_id=source.project_id,
                experiment_id=source.experiment_id,
                org_name=source.org_name,
                app_url=source.app_url,
                api_url=source.api_url,
                from_timestamp=source.from_timestamp,
                to_timestamp=source.to_timestamp,
                max_traces=config.max_traces or BRAINTRUST_DEFAULT_MAX_TRACES,
            )
        )
    if config.langfuse is not None:
        source = config.langfuse
        return LangfuseTraceLoader(
            LangfuseTraceConfig(
                base_url=source.base_url,
                from_timestamp=source.from_timestamp,
                to_timestamp=source.to_timestamp,
                filter_string=source.filter,
                max_traces=config.max_traces or LANGFUSE_DEFAULT_MAX_TRACES,
            )
        )
    if config.langfuse_trace_export_file is not None:
        source = config.langfuse_trace_export_file
        if source.api_version == "v4":
            return LangfuseV4FileTraceLoader(
                LangfuseV4FileTraceConfig(
                    path=source.path,
                    scores_path=source.scores_path,
                    base_url=source.base_url,
                    max_traces=config.max_traces or LANGFUSE_DEFAULT_MAX_TRACES,
                )
            )
        return LangfuseV3FileTraceLoader(
            LangfuseFileTraceConfig(
                path=source.path,
                max_traces=config.max_traces or LANGFUSE_DEFAULT_MAX_TRACES,
            )
        )
    if config.langsmith is not None:
        source = config.langsmith
        return LangSmithTraceLoader(
            LangSmithTraceConfig(
                project_name=source.project,
                api_url=source.api_url,
                filter=source.filter,
                tree_filter=source.tree_filter,
                start_time=source.start_time,
                max_traces=config.max_traces or LANGSMITH_DEFAULT_MAX_TRACES,
            )
        )
    if config.langsmith_trace_export_file is not None:
        return LangSmithTraceExportFileLoader(
            LangSmithTraceExportFileConfig(
                path=config.langsmith_trace_export_file.path,
                max_traces=config.max_traces or LANGSMITH_DEFAULT_MAX_TRACES,
            )
        )
    if config.mlflow_experiment is not None:
        source = config.mlflow_experiment
        return MLflowTraceLoader(
            MLflowTraceConfig(
                experiment_name=source.experiment,
                tracking_uri=source.tracking_uri,
                filter_string=source.filter,
                max_traces=config.max_traces or MLFLOW_DEFAULT_MAX_TRACES,
            )
        )
    assert config.mlflow_export is not None
    return MLflowFileTraceLoader(
        MLflowFileTraceConfig(
            path=config.mlflow_export.path,
            max_traces=config.max_traces or MLFLOW_DEFAULT_MAX_TRACES,
        )
    )


async def _run_compass_signals(
    config: CompassSignalsConfig,
    snapshot: TraceSnapshot,
    llm_factory: Callable[[], UnifiedLLM] | None = None,
    progress: Callable[[Sequence[str]], None] | None = None,
    *,
    ethos_path: Path | None = None,
) -> list[CompassSignalResult]:
    """Run the configured Compass Signals concurrently in registration order."""

    registry = registered_builtin_signals(
        anomaly_and_patterns=config.anomaly_and_patterns,
        tool_issues=config.tool_issues,
        ethos_divergence=config.ethos_divergence,
        ethos_path=ethos_path,
        user_sentiment=config.user_sentiment,
        grader_failure_patterns=config.grader_failure_patterns,
        llm_factory=llm_factory,
    )

    active: set[str] = set()

    def changed() -> None:
        if progress is not None:
            progress(tuple(name for name in registry.names if name in active))

    async def analyze(name: str) -> CompassSignalResult:
        configuration = registry.configuration(name)

        def started() -> None:
            active.add(name)
            changed()

        try:
            with telemetry.signal(name, configuration, len(snapshot)) as record:
                result = await registry.analyze(name, snapshot, on_start=started)
                telemetry.record_signal_result(record, result)
                return result
        finally:
            if name in active:
                active.remove(name)
                changed()

    # Cancel sibling work on failure; no analysis outlives this run.
    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(analyze(name)) for name in registry.names]
    return [task.result() for task in tasks]


async def _validate_candidate_problems(
    signal_results: list[CompassSignalResult],
    snapshot: TraceSnapshot,
    code_base_path: Path | None,
    llm: UnifiedLLM,
    ethos: str | None = None,
) -> CandidateProblemValidationResult:
    """Review the candidate problems in each signal result, preserving their order."""
    validator = CandidateProblemValidator(code_base_path, llm, ethos)

    async def validate_problem(problem: Problem) -> CandidateProblemDecision:
        traces = tuple(snapshot.get_trace_by_id(tid) for tid in problem.supporting_trace_ids)
        with telemetry.problem_validation(problem, code_base_path, ethos) as record:
            decision = await validator.validate(problem, traces)
            telemetry.record_problem_decision(record, decision)
            return decision

    async def validate_signal_problems(
        result: CompassSignalResult,
    ) -> list[CandidateProblemDecision]:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(validate_problem(problem)) for problem in result.problems]
        return [task.result() for task in tasks]

    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(validate_signal_problems(result)) for result in signal_results]
    return CandidateProblemValidationResult(signal_results, [task.result() for task in tasks])


async def _rate_confidence_with_code(
    insights: list[Insight],
    snapshot: TraceSnapshot,
    code_base_path: Path,
    llm: UnifiedLLM,
) -> list[Insight]:
    """Rate each insight's confidence against the supplied codebase."""

    code_base_path = code_base_path.expanduser().resolve()
    if not code_base_path.is_dir():
        raise ValueError(f"code_base must be an existing directory: {code_base_path}")
    rater = InsightConfidence(code_base_path, llm)

    def supporting_trace(ref: str) -> Trace | None:
        try:
            return snapshot.get_trace_by_id(ref)
        except KeyError:
            return None

    async def rate_insight(insight: Insight) -> Insight:
        supporting_traces = tuple(
            trace
            for trace in (supporting_trace(item.trace_id) for item in insight.evidence)
            if trace is not None
        )
        with telemetry.insight_confidence(insight, code_base_path) as record:
            rating = await rater.rate(insight, supporting_traces)
            telemetry.record_rating(record, insight, rating)
            return insight.model_copy(update={"confidence": rating})

    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(rate_insight(insight)) for insight in insights]
    return [task.result() for task in tasks]


def _configured_model(config: RunConfig) -> str:
    return config.model or resolve(ENV_MODEL) or DEFAULT_MODEL


def _build_llm(config: RunConfig, api_key: str) -> CompletionClient:
    """Construct an LLM client with the run's shared model settings."""

    model = _configured_model(config)
    api_base = config.api_base or os.environ.get(ENV_API_BASE)
    # A gateway configured for OpenAI must not redirect native provider routes.
    if not api_base and not model.startswith(("anthropic/", "openrouter/")):
        api_base = resolve(ENV_API_BASE)

    return CompletionClient(
        model=model,
        api_base=api_base or None,
        api_key=api_key,
        max_tokens=config.max_tokens or DEFAULT_MAX_TOKENS,
        reasoning_effort=DEFAULT_REASONING_EFFORT,
        # Anthropic needs LiteLLM's native translation, not raw OpenAI fields.
        allowed_openai_params=[]
        if model.startswith("anthropic/")
        else ["tool_choice", "reasoning_effort"],
        drop_params=True,
    )


async def _generate_insights(config: RunConfig, output: RunOutput) -> RunResult:
    api_key = _check_environment(config)
    async with output.progress():
        with record_span("ingestion") as record:
            record.set_input(config.trace)
            loader = _configured_trace_loader(config.trace)
            snapshot = await asyncio.to_thread(loader.load)
            record.set_output({"trace_count": len(snapshot)})
        record_input_metadata(config, loader)
        output.trace_count = len(snapshot)
        output.activity = "Checking prerequisites"

        def progress(active: Sequence[str]) -> None:
            output.activity = (
                "Analyzing: " + ", ".join(display_name(name).lower() for name in active)
                if active
                else "Checking prerequisites"
            )

        evidence = await _run_compass_signals(
            config.compass_signals,
            snapshot,
            lambda: _build_llm(config, api_key),
            progress,
            ethos_path=config.ethos_path,
        )
        existing = load_insights(config.existing_insights) if config.existing_insights else []
        result = RunResult(
            trace_count=len(snapshot),
            evidence=evidence,
            insights=list(existing),
            existing_insights=existing,
            limitations=signal_limitations(evidence),
        )
        if any(item.problems for item in evidence):
            run_timestamp = datetime.now(timezone.utc)
            result.insights = await _compile_evidence(
                config, api_key, snapshot, result, output, run_timestamp
            )
        result.insights = resolve_trace_links(result.insights, snapshot, existing)
        if config.code_base is not None and result.insights:
            output.activity = "Investigating insight confidence against the codebase"
            with telemetry.confidence(result.insights, config.code_base) as record:
                async with _build_llm(config, api_key) as llm:
                    result.insights = await _rate_confidence_with_code(
                        result.insights, snapshot, config.code_base, llm
                    )
                telemetry.record_confidence(record, result.insights)
        return result


async def _compile_evidence(
    config: RunConfig,
    api_key: str,
    snapshot: TraceSnapshot,
    result: RunResult,
    output: RunOutput,
    run_timestamp: datetime,
) -> list[Insight]:
    signal_results = result.evidence
    async with _build_llm(config, api_key) as llm:
        ethos_path = config.ethos_path
        ethos = ethos_path.read_text(encoding="utf-8") if ethos_path is not None else None
        if config.code_base is not None or ethos is not None:
            output.activity = "Validating candidate problems"
            with telemetry.candidate_problem_validation(
                signal_results, config.code_base, ethos
            ) as record:
                validation = await _validate_candidate_problems(
                    signal_results, snapshot, config.code_base, llm, ethos
                )
                telemetry.record_candidate_problem_validation(record, validation)
            signal_results = validation.retained_signal_results()
            result.validation_rejections = validation.rejections
            if validation.unresolved_count:
                result.limitations.append(
                    f"Candidate problem validation was inconclusive for {validation.unresolved_count} problem(s); these were retained for compilation."
                )
        if not any(item.problems for item in signal_results):
            return result.existing_insights
        output.activity = "Compiling candidate problems into insights"
        with telemetry.compilation(signal_results, result.existing_insights) as record:
            insights = await InsightCompilation(llm=llm).compile_insights(
                signal_results,
                snapshot,
                result.existing_insights,
                run_timestamp,
            )
            telemetry.record_insights(record, insights)
            return insights


def _render_insights(insights: Sequence[Insight]) -> str:
    return yaml.safe_dump(
        [insight.model_dump() for insight in insights],
        allow_unicode=True,
        sort_keys=False,
    )


def _write_insights(output_path: Path, rendered: str) -> None:
    if output_path == Path("-"):
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")


def get_config(argv: Sequence[str] | None = None) -> RunConfig:
    """Load run settings from optional YAML and generated CLI arguments."""

    try:
        return RunConfig(_cli_parse_args=list(sys.argv[1:] if argv is None else argv))
    except ValidationError as error:
        messages = ["Invalid configuration:"]
        for problem in error.errors():
            field = ".".join(str(part) for part in problem["loc"]) or "configuration"
            messages.append(f"  {field}: {problem['msg']}")
        raise SetupError("\n".join(messages)) from error


# Pydantic hooks accept parser/group objects and argparse's heterogeneous keyword arguments.
def _add_common_argument(parser: Any, *args: str, **kwargs: Any) -> Action:  # noqa: ANN401
    if (
        kwargs.get("dest") not in ("config", "output_path", "model")
        and kwargs.get("action") != "help"
    ):
        kwargs["help"] = SUPPRESS
    return parser.add_argument(*args, **kwargs)


def _add_common_group(parser: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    kwargs["description"] = None
    return parser.add_argument_group(**kwargs)


def _print_help(*, full: bool) -> None:
    parser = CliSettingsSource(
        RunConfig,
        add_argument_method=ArgumentParser.add_argument if full else _add_common_argument,
        add_argument_group_method=ArgumentParser.add_argument_group if full else _add_common_group,
    ).root_parser
    parser.description = __doc__
    parser.add_argument("--help-all", action="help", help="Show all configuration options")
    parser.print_help()


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or any(arg in argv for arg in ("-h", "--help", "--help-all")):
        _print_help(full="--help-all" in argv)
        return EXIT_OK

    output = RunOutput(Console(stderr=True, markup=False, highlight=False))
    try:
        config = get_config(argv)
        _check_environment(config)
        model = _configured_model(config)
        if re.search(r"(?:^|/)nvidia/nemotron-3\.5-lightning-30b$", model, re.IGNORECASE):
            output.console.print(
                f"Warning: {model} has known tool-calling failures in Compass. "
                "Try NVIDIA Nemotron 3.5 Super VL instead.",
                soft_wrap=True,
            )
        if config.ethos_path is None and config.compass_signals.ethos_divergence is not None:
            output.console.print(
                "Warning: skipping ethos divergence because no ethos_path was specified and no "
                "ETHOS.md was found in the working directory. "
                "Set ethos_path in YAML or pass --ethos-path to enable it.",
                soft_wrap=True,
            )
        with telemetry_session(config) as record:
            result = asyncio.run(_generate_insights(config, output))
            rendered = _render_insights(result.insights)
            if result.insights:
                _write_insights(config.output_path, rendered)
            if config.output_path == Path("-") or not sys.stdout.isatty():
                print(rendered, end="")
            output.report(result, config.output_path)
            telemetry.record_run(record, result)
        return EXIT_OK
    except SetupError as error:
        output.console.print(str(error), soft_wrap=True)
        return EXIT_SETUP


if __name__ == "__main__":
    raise SystemExit(main())
