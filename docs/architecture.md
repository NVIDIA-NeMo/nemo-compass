<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Compass architecture

NeMo Compass analyzes recorded agent executions from production and evaluation runs to identify
recurring problems. It has three stages: ingestion, evidence discovery, and insight compilation.

Five Compass Signals analyze business requirements, user complaints, evaluation results, tool
behavior, and statistical patterns. They use deterministic checks, learned classifiers, and LLM
investigation, with each signal implementing the same
[interface](../src/compass/compass_signals/compass_signals.py): check prerequisites, analyze
the snapshot, and return candidate `Problem` objects containing a description and supporting trace
IDs. Results can also include analysis artifacts, coverage limitations, and reasons for skipping
analysis. Partners can run the NeMo Compass pipeline, reuse individual Compass Signals or trace loaders.

[Insight compilation](../src/compass/insights_generation/insight_compilation.py) investigates
candidates in their original traces, assesses their impact and whether a developer can fix them,
merges duplicates across signals and existing insights, and ranks the results. Its instructions
require each new insight to have support from more than one trace and to merge symptoms only when
they establish the same problem. These decisions rely on model judgment.

When `code_base` or ETHOS.md is available, [candidate validation](../src/compass/insights_generation/validation.py)
runs before compilation. Each candidate gets one investigation with its supporting
traces and all configured context: intended behavior from ETHOS.md and read-only
source-code tools. Candidates are investigated concurrently. The validator reconciles
these sources before deciding; intentional implementation does not excuse a violation
of ETHOS.md, and permissions apply only within their stated conditions.

Concrete contradictions remove candidates; uncertainty retains them. Rejections preserve
an explanation and evidence basis, including an exact passage when relying on ETHOS.md.
Compilation handles priority, severity, and recurrence after validation.

When `code_base` is configured, a [second code-aware
review](../src/compass/insights_generation/confidence.py) runs after compilation. It evaluates
each final insight’s behavior, root cause, scope, and implied remediation, assigning confidence
without changing severity. Validation and confidence are separate NOOA agents with read-only access
to the codebase and separate decision contracts, because they operate on different artifacts at
different pipeline stages. If an agent cannot decide within its step budget, validation retains the
problem and confidence is left unset.

![Architecture overview: a Trace loaders group contains LangSmith, Langfuse, Braintrust, MLflow, and Other, all feeding a normalized snapshot. A Compass Signals group contains the five detectors, which feed candidate problems through optional code validation and insight compilation to ranked insights.](assets/architecture/overview.png)

## Ethos divergence

The ethos-divergence signal compares observed behavior with a document describing the agent’s
business purpose, requirements, and limits. This document, called `ETHOS.md`, defines expected
behavior independently of the implementation. For example, a support agent might be allowed to
explain refund policy but forbidden to issue refunds. The signal would investigate a recorded refund
action as a possible violation, even if the tool call succeeded. `ETHOS.md` can be generated with the
help of the [Ethos skills](https://github.com/NVIDIA/skills/blob/main/skills/ethos-orchestrate/README.md#how-to-use). 

The signal supplies the document and trace snapshot to an LLM issue detector. Its instructions
require reviewing the visible conversation and identifying which requirement was violated and how.
Requirements apply within their stated scope, and the business contract takes precedence when the
agent’s system prompt permits conflicting behavior. The review considers the agent’s response to an
out-of-scope request when determining whether a violation occurred.

Approval requirements, supported workflows, customer commitments, and limits on delegated authority
often need domain-specific evaluation. This signal lets teams assess those requirements directly
from a written specification, including requirements they have yet to cover with formal tests. Clear
requirements and complete conversational evidence reduce ambiguity in the review.

Business requirements also help interpret findings from other signals. A correct refusal may explain
a user’s complaint, while a successful tool call may violate an approval requirement. Comparing
these observations with the documented requirements helps compilation determine whether the behavior
needs to change.

The signal requires a non-empty requirements document and skips analysis when none is configured. An
observability platform could store this document at the project level and display findings against
the relevant requirements. The [ethos-divergence
implementation](../src/compass/compass_signals/ethos_divergence/ethos_divergence_detector.py)
defines configuration and evaluation instructions; the shared [issue
detector](../src/compass/compass_signals/issue_detector.py) defines the output descriptions
and supporting trace references.

## User sentiment

The user-sentiment signal finds recurring complaints in recorded human messages, including
corrections made later in a conversation. It captures reports of ignored instructions, repeated
misunderstandings, and dissatisfaction with the agent’s work. Production conversations can provide
this feedback even when evaluation labels are sparse. The signal screens messages for anger or
criticism, then investigates recurring complaint themes.

An LLM inspects the recorded message formats and constructs an extraction function, which processes
the snapshot in the caller’s process. The extraction instructions identify actual human turns,
preserve later messages, and account for accumulated conversation history. Generated documents and
evaluator inputs can also appear under a `user` role, so extraction must establish authorship from
the source structure. Runtime checks require the result to cover exactly the supplied trace IDs.

Each message is screened independently using Qwen3-Embedding-8B, PCA256, four-bit MSE TurboQuant
compression, and logistic regression. The pipeline projects normalized 4,096-dimensional embeddings
into 256 dimensions and stores each compressed vector in 132 bytes. The classifier reconstructs
those features before scoring them. Any flagged message routes its trace to contextual review,
including when later messages are neutral. Embeddings can run locally or through a remote endpoint;
projection and classification run locally. The model, input convention, projection, and classifier
must remain compatible, so a replacement embedding model requires an appropriately trained head and
validation.

![User-sentiment design: actual human turns pass through embedding and compression, local complaint screening, and contextual review; recurring complaints retain exact trace IDs and user quotations.](assets/architecture/sentiment.png)

An LLM then inspects the flagged conversations and groups complaints by their supported reasons.
Review instructions require separate occurrences in at least two distinct traces and representative
verbatim quotations. They also require identifying whether criticism concerns the agent or an
external subject and distinguishing the user’s stated reason from an inferred cause. Classifier
scores are uncalibrated and used to select conversations for this review.

Screening reduces the set of conversations sent for detailed review. The separate representation and
classification components also allow partners to train and evaluate additional heads for other
tasks. Running the current signal requires preserved human messages and a compatible embedding
backend; its results report coverage for traces lacking user text. The [signal
implementation](../src/compass/compass_signals/user_sentiment/signal.py) handles extraction
and review, with the [embedding
pipeline](../src/compass/compass_signals/user_embedding/embedding.py) and
[classifier](../src/compass/compass_signals/user_sentiment/classifier.py) implemented
separately.

## Grader failures

The grader-failure signal investigates recurring behavior associated with recorded scores and
feedback. It reads `evaluator_results` attached to normalized traces and uses them to select
executions for investigation. Repeated low correctness scores, for example, may lead it to inspect
whether several executions failed at the same step. Evaluation results must already be present in
the input traces.

The signal builds a compact index containing trace IDs, logical case IDs, evaluator results, tool
names, and recorded errors. An LLM investigator uses the index to select and retrieve complete
traces. This allows it to inspect summary information before fetching the detailed executions needed
for an explanation. Configuration limits the number of fetch rounds.

A runtime postcondition requires the investigator to fetch every trace it cites. Investigation
instructions require grouping traces by the observed failure and including the relevant evaluator
names and values in each problem description. Since the same score can result from different
failures, the investigator must examine execution evidence before grouping cases. It must also
distinguish associations in the data from established causes.

Teams with evaluation datasets, human feedback, or model-based graders can use this signal to
investigate score changes and identify repeated failures for developers to address. The analysis can
help explain which behaviors contribute to an aggregate result. Its conclusions depend on evaluator
quality and coverage, and the result reports how many traces lack evaluation signals.

Integration requires preserving evaluator names and values during normalization, along with
execution details that can explain the results. The compact index and trace-retrieval pattern can
also support other investigations over large traces. The [grader-failure
implementation](../src/compass/compass_signals/grader_failure_patterns.py) contains the index,
bounded retrieval tool, citation postcondition, and investigation instructions. The signal skips
when no non-null evaluator results are available.

## Tool issues

The tool-issue signal applies deterministic rules to recorded tool calls and results. It detects
unknown tools, malformed or invalid arguments, missing or mismatched results, explicit failures, and
unsuccessful retry patterns. These checks help diagnose failures caused by API contracts, missing
prerequisites, or orchestration code.

The implementation defines nineteen finding types covering tool contracts, call/result integrity,
explicit outcomes, retries, instrumentation, argument provenance, and prerequisite or state
failures. Each check requires specific recorded evidence. Argument-schema checks need schemas, for
example, while some outcome checks need only calls and results. Rules abstain when required evidence
is unavailable, so diagnostic coverage depends on what the trace records preserve.

Findings retain trace IDs, call identifiers, and source pointers. The signal groups them into
evidence cards by issue type and mechanism, with representative examples and counts of independent
logical cases. By default, a card becomes eligible for insight generation after appearing in three
independent cases. Findings below that threshold remain in the audit artifacts. Counting cases
separately from individual failures limits the effect of repeated errors within one execution.

The rules produce reproducible findings that developers can inspect at the call level. Repeated
omission of a required argument, for example, identifies a specific contract violation to
investigate in the agent’s prompt, tool schema, or calling code. Compilation examines the
surrounding execution to assess impact and recovery before promoting the finding to an insight.
Instrumentation findings can also identify incomplete or inconsistent trace records.

Partners can run the audit independently of LLM analysis, display evidence cards beside tool spans,
or pass eligible problems to NeMo Compass. Integration should preserve tool names, arguments,
results, identifiers, schemas when available, and logical case identity. The [tool-issue
implementation](../src/compass/compass_signals/tool_issues/signal.py) contains normalization,
detection rules, card construction, and problem promotion. Its artifacts retain both individual
findings and the recurring groups used for insight generation.

## Anomalies and patterns

The anomaly-and-pattern signal identifies unusual execution patterns and recurring failures across a
corpus. It can help developers select traces to investigate when evaluation labels and user feedback
are unavailable or when existing checks have left a failure unexplained. An outlier may reflect
either a defect or legitimate variation in the tasks being performed; interpreting it requires
reviewing the execution.

The signal extracts numeric features including tool-call count, trajectory length, tool diversity,
repeated-call rate, explicit-failure rate, output size, and recorded tool duration. It scores
outliers with an Isolation Forest and reports feature-level deviations to help interpret them.
Configuration controls the expected outlier fraction, optional robust scaling, and feature
selection, including supplied numeric metrics. Corpus composition affects the comparison: complex
research tasks may be legitimate outliers in a dataset dominated by simple requests.

The signal also groups execution trajectories using TF-IDF representations of ordered event unigrams
and bigrams, followed by K-means clustering. It compares candidate cluster counts using silhouette
scores. These features capture similarities in local execution sequences that aggregate call counts
can miss. Clustering requires at least three traces and two distinct feature vectors; the other
analyses can complete when clustering is unavailable. Trajectory clusters and observed verdict
groups are retained as descriptive artifacts.

Explicit failures are grouped by normalized signatures within and across tools, with a default
recurrence threshold of three distinct traces. Similar errors can help locate repeated integration
failures or a common upstream dependency, though their causes still require investigation.
Statistical outliers and recurring same-tool or cross-tool failure groups become candidate problems.
Features, clusters, verdict groups, and an evidence digest remain available for inspection.

Partners can reuse feature extraction, trajectory grouping, and failure-signature analysis
separately and configure them for their own trace populations. These methods can surface behavior
that existing rubrics and complaint classifiers do not cover. The [anomaly-and-pattern
implementation](../src/compass/compass_signals/anomaly_and_patterns/signal.py) contains the
analysis methods and shared-signal adapter. In the full NeMo Compass pipeline, compilation reviews these
candidates alongside findings from the other signals to determine which problems have sufficient
support and a practical remedy.

---

For configuration and prerequisites, see [Compass Signals](compass-signals.md).
