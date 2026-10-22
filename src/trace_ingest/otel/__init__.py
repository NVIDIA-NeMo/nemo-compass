# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read canonical span fields from OTel span attributes written to an OTel convention.

Each module reads one convention and has the same functions: ``is_<convention>``,
``token_counts``, ``tools`` and ``params``. ``openinference`` reads any
OpenInference instrumentation. Loaders pass the span attributes wherever their
source keeps them. To support another convention, such as OTel GenAI
(``gen_ai.*``), add a module with those functions. Producers with attributes of
their own, such as NeMo Relay (``trace_ingest.nemo_relay``), have their own
module and may build on these.

Readers use key names, numbers and request structure only, never message content.
"""
