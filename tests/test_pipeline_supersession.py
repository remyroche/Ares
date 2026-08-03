from __future__ import annotations

import json
from pathlib import Path

import pytest
import pandas as pd

from extreme_price_movements.pipeline_supersession import (
    SupersededArtifactError,
    assert_artifact_usable,
    load_supersession_manifest,
)
from extreme_price_movements.feature_provenance_gate import (
    FeatureLineageRecord,
    FeatureProvenanceError,
    assert_feature_lineage,
    audit_feature_frame,
    validate_feature_columns,
)


def _manifest(tmp_path: Path) -> Path:
    path = tmp_path / "supersession.json"
    path.write_text(
        json.dumps(
            {
                "schema": "pipeline_supersession_manifest_v1",
                "entries": [
                    {
                        "artifact": str(tmp_path / "revoked"),
                        "status": "REVOKED",
                        "reason": "future target proxy",
                        "allowed_uses": ["audit"],
                        "blocked_uses": ["training", "inference", "promotion"],
                    }
                ],
            }
        )
    )
    return path


def test_operational_use_of_revoked_artifact_fails_closed(tmp_path: Path) -> None:
    manifest = load_supersession_manifest(_manifest(tmp_path))
    with pytest.raises(SupersededArtifactError, match="future target proxy"):
        assert_artifact_usable(tmp_path / "revoked" / "run_manifest.json", purpose="training", manifest=manifest)


def test_audit_use_can_read_revoked_artifact_explicitly(tmp_path: Path) -> None:
    manifest = load_supersession_manifest(_manifest(tmp_path))
    record = assert_artifact_usable(tmp_path / "revoked" / "result.parquet", purpose="audit", manifest=manifest)
    assert record is not None and record["status"] == "REVOKED"


def test_current_stage_d_runner_cannot_reuse_revoked_d2() -> None:
    from scripts.run_stage_d_compact_action_model import D2, require_d2

    if not D2.exists():
        pytest.skip("historical D2 artifact is not materialized in this checkout")
    with pytest.raises(SupersededArtifactError):
        require_d2()


def test_name_gate_rejects_future_cost_names() -> None:
    with pytest.raises(FeatureProvenanceError, match="known_row_cost"):
        validate_feature_columns(["trend_slope", "known_row_cost_bps"])


def test_transitive_lineage_rejects_hidden_future_dependency() -> None:
    records = [
        FeatureLineageRecord("raw_spread", point_in_time_safe=True, live_reproducible=True),
        FeatureLineageRecord("derived_cost_proxy", ("raw_spread", "known_row_cost_bps"), point_in_time_safe=True, live_reproducible=True),
        FeatureLineageRecord("known_row_cost_bps", point_in_time_safe=True, live_reproducible=True),
    ]
    with pytest.raises(FeatureProvenanceError, match="forbidden target/future/cost"):
        assert_feature_lineage(records, ["derived_cost_proxy"])


def test_oof_and_timestamp_requirements_are_enforced() -> None:
    record = FeatureLineageRecord(
        "regime_prediction",
        point_in_time_safe=True,
        live_reproducible=True,
        oof_required=True,
        oof_verified=False,
    )
    with pytest.raises(FeatureProvenanceError, match="OOF"):
        assert_feature_lineage([record], ["regime_prediction"])
    frame = pd.DataFrame({"decision_ts": ["2024-01-01T00:00Z"], "available_ts": ["2024-01-01T01:00Z"]})
    report = audit_feature_frame(
        frame,
        [FeatureLineageRecord("x", point_in_time_safe=True, live_reproducible=True)],
        ["x"],
        decision_column="decision_ts",
        available_column="available_ts",
    )
    assert not report["passed"] and report["timestamp_violations"] == 1
