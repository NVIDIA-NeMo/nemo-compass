# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read canonical token counts from a provider's own usage object.

Each provider module copies the counts of its API's usage object into the
fields with the same meaning: ``anthropic.token_counts`` (Messages) and
``openai.token_counts`` (Responses). Loaders find the usage object wherever their
source keeps it and pick the reader by the API route, never by guessing from
the keys; a source that renames the provider's keys maps them back first. To
support another API, add a function for its usage object.
"""

from trace_ingest.loaders.helper.llm_span.token_usage import anthropic, openai
from trace_ingest.loaders.helper.llm_span.token_usage._core import token_counts

__all__ = ["anthropic", "openai", "token_counts"]
