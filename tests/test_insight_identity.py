# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from compass.insight import Insight, load_insights


def test_existing_identity_survives_artifact_round_trip(tmp_path):
    insight = Insight.model_validate(
        {
            "id": "platform-123",
            "name": "Timeouts",
            "description": "Repeated timeouts",
            "severity": "med",
            "severity_reason": "Agent retries burn tokens but the user still gets a response.",
            "evidence": [{"trace_id": "a"}, {"trace_id": "b"}],
        }
    )
    path = tmp_path / "insights.json"
    path.write_text("[" + insight.model_dump_json() + "]")
    loaded = load_insights(path)[0]
    assert loaded.id == "platform-123"
    assert loaded.severity == "med"
    assert loaded.severity_reason == insight.severity_reason


def test_new_insight_does_not_require_or_serialize_an_id():
    insight = Insight.model_validate(
        {
            "name": "Timeouts",
            "description": "Repeated timeouts",
            "severity": "med",
            "severity_reason": "Agent retries burn tokens but the user still gets a response.",
            "evidence": [{"trace_id": "a"}, {"trace_id": "b"}],
        }
    )
    assert insight.id is None
    assert "id" not in insight.model_dump()


def test_empty_id_is_rejected():
    with pytest.raises(ValidationError):
        Insight.model_validate(
            {
                "id": "",
                "name": "Timeouts",
                "description": "Repeated timeouts",
                "evidence": [{"trace_id": "a"}, {"trace_id": "b"}],
            }
        )


def test_insight_without_severity_loads_for_backward_compatibility(tmp_path):
    """Insights saved without ratings or reasons still load for reconciliation."""
    path = tmp_path / "insights.json"
    path.write_text(
        '[{"name": "Timeouts", "description": "Repeated timeouts", '
        '"evidence": [{"trace_id": "a"}, {"trace_id": "b"}]}]'
    )
    loaded = load_insights(path)[0]
    assert loaded.severity is None
    assert loaded.severity_reason is None
    assert "severity" not in loaded.model_dump()
    assert "severity_reason" not in loaded.model_dump()


def test_invalid_severity_is_rejected():
    for rating in ("medium", "critical"):
        with pytest.raises(ValidationError):
            Insight.model_validate(
                {
                    "name": "Timeouts",
                    "description": "Repeated timeouts",
                    "severity": rating,
                    "severity_reason": "Agent retries burn tokens but the user still gets a response.",
                    "evidence": [{"trace_id": "a"}, {"trace_id": "b"}],
                }
            )


def test_generated_schema_uses_only_unified_evidence():
    schema = Insight.model_json_schema()
    assert "evidence" in schema["required"]
    assert (
        not {"trace_refs", "trace_links", "span_refs", "span_links"} & schema["properties"].keys()
    )
    with pytest.raises(ValidationError):
        Insight.model_validate(
            {"name": "Issue", "description": "Details", "evidence": [{"trace_id": "a"}]}
        )
