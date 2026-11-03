# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cache misses with a known cause.

``sequences`` orders calls per agent, ``misses`` finds calls that read nothing of
the cache their reference left, and each cause rule (``tool_change``) explains a
miss. ``detector`` ties them into the signal's detector.
"""

from compass.compass_signals.cost_efficiency.prompt_caching_miss.detector import (
    CAUSES,
    SPECS,
    detect,
)

__all__ = ["CAUSES", "SPECS", "detect"]
