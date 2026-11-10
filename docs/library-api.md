# Using Compass as a library

`compass.analysis.analyze` runs the same analysis pipeline as the Compass CLI.
The caller loads a canonical `TraceSnapshot`, supplies model access and optional
resources, and decides how to store the result. Compass selects and orders its
signals, candidate validation, compilation, evidence-link resolution, and
code-based confidence review.

```python
from pathlib import Path

from nooa.unifiedllm import CompletionClient
from compass.analysis import AnalysisOptions, analyze

result = await analyze(
    snapshot,
    llm_factory=lambda: CompletionClient(model=model, api_key=api_key),
    options=AnalysisOptions.model_validate({
        "compass_signals": {"tool_issues": {"minimum_independent_cases": 3}},
    }),
    existing_insights=previous_insights,
    ethos_path=Path("ETHOS.md"),
    code_base=Path("agent-source"),
)
Path("analysis.json").write_text(result.model_dump_json(indent=2))
```

All signals are enabled by default. Missing optional prerequisites produce skip
reasons. Validation runs when code or an ETHOS document is supplied; confidence
review requires code. The library does not discover files in the working
directory, load environment files, configure telemetry exporters, or render a
terminal report. Those responsibilities remain with the caller.

## Configuration and upgrades

`AnalysisOptions` contains analysis settings shared with the CLI's `RunConfig`.
The CLI keeps its existing YAML/flag layout. Platform integrations can accept a
nested JSON object and validate it with `AnalysisOptions.model_validate` instead
of duplicating the signal schema. Save user overrides, using
`model_dump(mode="json", exclude_unset=True)` when starting from a validated
model. Expanding defaults into a saved configuration pins those choices across
future upgrades.

Each `AnalysisResult` records effective analysis options and the installed Compass
version. It contains insights, raw signal evidence, generic signal summaries,
validation rejections, stage statuses, limitations, and common diagnostic notes.
Use `model_dump(mode="json")` or `model_dump_json()` for storage and
`AnalysisResult.model_validate_json()` for readback. Serialized signal artifacts
are plain data; integrations should use the generic summaries and notes for
reporting instead of inspecting signal-specific artifact classes.

Keep the full result artifact even when projecting insights into a platform
entity schema. The platform owns identity, merge policy, and protection of
user-edited fields. New result fields can then be retained before the platform
adds UI support for them.

A dependency upgrade picks up new default signals and stages that use existing
inputs. New external resources or incompatible public API changes still need
integration work.

## Model ownership and instrumentation

`llm_factory` returns a fresh `UnifiedLLM` client whose context Compass enters and
exits. `fast_llm_factory`, when supplied, is used by signals; the primary factory
is used by validation, compilation, and confidence review. Without it, both use
`llm_factory`.

For clients already opened and owned by the host, factories can return
`BorrowedModelClient(client)` wrappers. The wrappers delegate inference without
opening or closing the underlying client. Keep the host client alive for the
entire analysis call, including concurrent signal work.

`progress` receives `AnalysisProgress` with a stage and currently active signals.
`event_handler` receives Nooa `LLMResponse` and `PythonOutput` events from internal
agents. For middleware that needs an event manager, `instrument_agent` accepts an
async context-manager factory taking that manager. Compass applies it around each
internal agent execution and exits that context in the same task when execution
finishes or fails. These
hooks are isolated between concurrent runs. Configure OpenTelemetry exporters
and any outer platform tracing scope in the caller; Compass's own stage spans
inherit the current tracing context.

## Embeddings and fetched Intake records

Pass `embedding_generator` to supply host-managed embeddings for sentiment.
`UserEmbeddingGenerator` and `LiteLLMEmbeddingConfig` are exported from
`compass.analysis`. The generator supports an `embed_batch` callback alongside
remote batching configuration. It calls this synchronous callback from worker
threads with prefixed, token-checked inputs. The callback must return one raw
vector per input in order and handle its own authentication, timeout, and retries.
Compass retains batching, normalization, projection, and compression. The backend
must serve the supported Qwen3-Embedding-8B encoder with 4,096-dimensional vectors;
an arbitrary embedding model is incompatible with the sentiment classifier.
Injected generators are borrowed and are not closed by Compass. Configuration
contains no embedding credential value; pass credentials through host services.

`normalize_intake_trace(trace, spans, evaluator_results, base_url=..., workspace=...)`
accepts already-fetched Intake wire dictionaries and returns a canonical `Trace`.
It validates required fields and ownership using the same normalization as the
Intake loader. Authentication, pagination, and fetching remain with the caller;
private Intake record classes are not part of this interface.
