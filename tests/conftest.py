# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from compass.config import CompassSignalsConfig


@pytest.fixture
def select_signals():
    """Explicitly disable other signals in tests exercising a single integration."""

    def select(**selected):
        return dict.fromkeys(CompassSignalsConfig.model_fields, False) | selected

    return select
